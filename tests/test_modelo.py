"""Pruebas básicas: pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import cargar_datos                      # noqa: E402
from src.features import rangos_historicos                    # noqa: E402
from src.model import entrenar, fuera_de_rango, simular       # noqa: E402

D = cargar_datos(forzar_csv=True)
R = rangos_historicos(D.diario)
BASE = {k: v[1] for k, v in R.items()}


def test_datos_completos():
    assert len(D.diario) == 213 and D.diario.ingresos.sum() > 0
    assert 0 < D.margen_bruto < 1


def test_simulacion_base_cercana_al_promedio():
    m = entrenar("rlm", D.diario)
    _, s = simular(m, pd.Timestamp("2026-09-21"), 8, BASE, D.margen_bruto, D.costo_fijo_dia)
    assert len(s) == 8
    assert abs(s.ingresos.mean() / (7 * D.diario.ingresos.mean()) - 1) < 0.02


def test_mas_venta_cruzada_sube_ingresos():
    m = entrenar("rlm", D.diario)
    alto = dict(BASE, pct_gaseosa=BASE["pct_gaseosa"] + 5)
    _, a = simular(m, pd.Timestamp("2026-09-21"), 4, BASE, D.margen_bruto, D.costo_fijo_dia)
    _, b = simular(m, pd.Timestamp("2026-09-21"), 4, alto, D.margen_bruto, D.costo_fijo_dia)
    assert b.ingresos.sum() > a.ingresos.sum()


def test_alerta_de_extrapolacion():
    assert fuera_de_rango(dict(BASE, ventas=70), R) == ["Ventas por día"]
