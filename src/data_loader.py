"""Carga de datos.

Prioridad de fuentes:
1. Base de datos PostgreSQL / Supabase, si existe la variable de entorno DATABASE_URL (o PGHOST...).
2. Archivos CSV de la carpeta ``data/`` (instantánea exportada de la misma base), por ruta relativa.

Las consultas SQL están documentadas aquí para que el informe sea reproducible.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATA_DIR

try:                                    # .env opcional
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:                     # pragma: no cover
    pass

# ----------------------------------------------------------------------------- consultas
QUERIES = {
    # Una fila por día: ventas, ingresos y cuántas ventas incluyen cada categoría.
    "diario": """
        WITH l AS (
            SELECT v.id_venta, v.fecha_hora::date AS f, v.tipo_venta, v.total_venta,
                   MAX((p.categoria = 'Gaseosa')::int)     AS gas,
                   MAX((p.categoria = 'Acompañante')::int) AS aco,
                   MAX((p.categoria = 'Cerveza')::int)     AS cer
            FROM ventas.venta v
            JOIN ventas.detalle_venta d USING (id_venta)
            JOIN ventas.producto p USING (id_producto)
            GROUP BY 1, 2, 3, 4)
        SELECT f AS fecha, COUNT(*) AS ventas, SUM(total_venta) AS ingresos,
               SUM((tipo_venta = 'Domicilio')::int) AS domicilios,
               SUM(gas) AS con_gaseosa, SUM(aco) AS con_acomp, SUM(cer) AS con_cerveza
        FROM l GROUP BY f ORDER BY f""",
    # Ingresos y unidades por día, categoría y producto (para filtros y drill-down).
    "productos_dia": """
        SELECT v.fecha_hora::date AS fecha, p.categoria, p.nombre AS producto,
               SUM(d.cantidad) AS unidades, SUM(d.subtotal) AS ingresos
        FROM ventas.detalle_venta d
        JOIN ventas.venta v USING (id_venta)
        JOIN ventas.producto p USING (id_producto)
        GROUP BY 1, 2, 3 ORDER BY 1""",
    # Costos de producción totales (para el margen bruto).
    "costo_produccion": """
        SELECT COALESCE(SUM(monto), 0) AS costo FROM ventas.costo WHERE tipo_costo = 'Producción'""",
    # Costos operativos mensuales (para el costo fijo por día).
    "operativo_mensual": """
        SELECT TO_CHAR(fecha, 'YYYY-MM') AS mes, SUM(monto) AS costo
        FROM ventas.costo WHERE tipo_costo = 'Operativo' GROUP BY 1 ORDER BY 1""",
}

ARCHIVOS = {k: DATA_DIR / f"{k}.csv" for k in QUERIES}


@dataclass
class Datos:
    """Contenedor de las tablas que usa el dashboard."""
    diario: pd.DataFrame
    productos_dia: pd.DataFrame
    margen_bruto: float            # fracción 0-1
    costo_fijo_dia: float          # COP por día
    fuente: str                    # texto para mostrar en la interfaz


def _url_base_datos() -> str | None:
    """Construye la URL de conexión a partir de variables de entorno (nunca desde el código)."""
    url = os.getenv("DATABASE_URL")
    if url:
        return re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", url)
    if os.getenv("PGHOST"):
        g = os.getenv
        return (f"postgresql+psycopg2://{g('PGUSER', 'postgres')}:{quote_plus(g('PGPASSWORD', ''))}"
                f"@{g('PGHOST')}:{g('PGPORT', '5432')}/{g('PGDATABASE', 'postgres')}")
    return None


def crear_motor():
    """Devuelve un motor SQLAlchemy o None si no hay conexión configurada."""
    url = _url_base_datos()
    if not url:
        return None
    args = {"sslmode": "require"} if "supabase" in url else {}
    return create_engine(url, pool_pre_ping=True, connect_args=args)


def leer_consultas(motor) -> dict[str, pd.DataFrame]:
    """Ejecuta todas las consultas contra la base de datos."""
    with motor.connect() as cn:
        return {k: pd.read_sql(text(q), cn) for k, q in QUERIES.items()}


def leer_csv() -> dict[str, pd.DataFrame]:
    """Lee la instantánea local (data/*.csv)."""
    faltan = [str(p) for p in ARCHIVOS.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError("Faltan archivos en data/: " + ", ".join(faltan)
                                + ". Ejecute scripts/export_snapshot.py con la base de datos conectada.")
    return {k: pd.read_csv(p) for k, p in ARCHIVOS.items()}


def _a_numerico(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass
    return df


def preparar(t: dict[str, pd.DataFrame], fuente: str) -> Datos:
    """ETL: tipos, porcentajes diarios, margen bruto y costo fijo por día."""
    d = _a_numerico(t["diario"].copy())
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d.sort_values("fecha").reset_index(drop=True)
    d["pct_gaseosa"] = 100 * d.con_gaseosa / d.ventas
    d["pct_acomp"] = 100 * d.con_acomp / d.ventas
    d["pct_cerveza"] = 100 * d.con_cerveza / d.ventas

    pr = _a_numerico(t["productos_dia"].copy())
    pr["fecha"] = pd.to_datetime(pr["fecha"])

    costo_prod = float(_a_numerico(t["costo_produccion"]).iloc[0, 0])
    margen = 1 - costo_prod / float(d.ingresos.sum())

    op = _a_numerico(t["operativo_mensual"].copy())
    completos = op.iloc[1:-1] if len(op) > 2 else op           # se descartan meses parciales
    dias = sum(pd.Period(m).days_in_month for m in completos.mes)
    costo_fijo = float(completos.costo.sum()) / dias
    return Datos(d, pr, margen, costo_fijo, fuente)


def cargar_datos(forzar_csv: bool = False) -> Datos:
    """Punto de entrada: BD si hay conexión; si falla o no existe, CSV local."""
    motor = None if forzar_csv else crear_motor()
    if motor is not None:
        try:
            return preparar(leer_consultas(motor), "Base de datos PostgreSQL (consulta en vivo)")
        except Exception as exc:       # sin red, credenciales erróneas, etc.
            return preparar(leer_csv(), f"Instantánea CSV (la base de datos no respondió: {type(exc).__name__})")
    return preparar(leer_csv(), "Instantánea CSV de la base de datos (carpeta data/)")
