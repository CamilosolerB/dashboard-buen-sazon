"""Ingeniería de variables: calendario colombiano y matriz de predictores."""
from __future__ import annotations

import holidays
import pandas as pd

from .config import CALENDARIO, PREDICTORES

COLUMNAS = list(PREDICTORES) + CALENDARIO        # orden fijo de las columnas del modelo


def calendario(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario: fin de semana (sáb/dom) y festivo en Colombia."""
    anios = sorted(set(fechas.year))
    festivos = holidays.country_holidays("CO", years=anios)
    return pd.DataFrame({
        "fin_de_semana": (fechas.dayofweek >= 5).astype(int),
        "festivo": [int(f.date() in festivos) for f in fechas],
    }, index=fechas)


def matriz_historica(diario: pd.DataFrame) -> pd.DataFrame:
    """Predictores históricos (una fila por día) alineados con 'diario'."""
    cal = calendario(pd.DatetimeIndex(diario["fecha"]))
    X = diario[list(PREDICTORES)].copy().reset_index(drop=True)
    return pd.concat([X, cal.reset_index(drop=True)], axis=1)[COLUMNAS]


def matriz_escenario(inicio: pd.Timestamp, dias: int, valores: dict[str, float]) -> pd.DataFrame:
    """Predictores de un escenario: valores asignados por el usuario + calendario real de cada fecha."""
    fechas = pd.date_range(inicio, periods=dias)
    X = pd.DataFrame({k: float(valores[k]) for k in PREDICTORES}, index=fechas)
    return pd.concat([X, calendario(fechas)], axis=1)[COLUMNAS]


def rangos_historicos(diario: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """(mínimo, promedio, máximo) observado de cada predictor, para avisar de extrapolaciones."""
    return {k: (float(diario[k].min()), float(diario[k].mean()), float(diario[k].max())) for k in PREDICTORES}
