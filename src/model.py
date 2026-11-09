"""Modelos predictivos y simulación de escenarios.

Modelos permitidos por la asignatura que usa este dashboard:
  * Regresión lineal múltiple (OLS) - interpretable, extrapola con tendencia lineal.
  * Random forest              - no lineal, pero no extrapola fuera del rango histórico.
Variable objetivo: ingreso diario (COP). El pronóstico semanal es la suma de siete días.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

from .config import PREDICTORES, VALIDACION_FRACCION, Z95
from .features import COLUMNAS, matriz_escenario, matriz_historica

MODELOS = {"rlm": "Regresión lineal múltiple", "rf": "Random forest"}


@dataclass
class Validacion:
    """Métricas calculadas sobre el último tramo de días (no usado para entrenar)."""
    r2: float
    mape_dia: float
    mape_semana: float
    mape_base_dia: float           # error de simplemente repetir el promedio de entrenamiento
    mape_base_semana: float
    n_test: int


@dataclass
class ModeloDemanda:
    clave: str
    modelo: object
    sd_residual: float             # desviación del error diario (para el IC)
    validacion: Validacion
    coeficientes: pd.DataFrame | None = None
    columnas: list[str] = field(default_factory=lambda: list(COLUMNAS))

    def predecir_dia(self, X: pd.DataFrame) -> np.ndarray:
        if self.clave == "rlm":
            return np.asarray(self.modelo.predict(sm.add_constant(X[self.columnas], has_constant="add")))
        return self.modelo.predict(X[self.columnas])


def _ajustar(clave: str, X: pd.DataFrame, y: pd.Series):
    if clave == "rlm":
        return sm.OLS(y.values, sm.add_constant(X, has_constant="add")).fit()
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0).fit(X, y)


def _predecir(clave: str, modelo, X: pd.DataFrame) -> np.ndarray:
    if clave == "rlm":
        return np.asarray(modelo.predict(sm.add_constant(X, has_constant="add")))
    return modelo.predict(X)


def _mape(real: np.ndarray, pred: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(real - pred) / real))


def _mape_semanal(real: pd.Series, pred: np.ndarray) -> float:
    sem = np.arange(len(real)) // 7
    r = real.groupby(sem).sum()
    p = pd.Series(pred, index=real.index).groupby(sem).sum()
    completo = real.groupby(sem).size() == 7
    return _mape(r[completo].values, p[completo].values)


def validar(clave: str, diario: pd.DataFrame) -> Validacion:
    """Entrena con los primeros días y mide el error en los últimos (validación cronológica)."""
    X, y = matriz_historica(diario), diario.ingresos.reset_index(drop=True)
    corte = int(len(diario) * (1 - VALIDACION_FRACCION))
    m = _ajustar(clave, X.iloc[:corte], y.iloc[:corte])
    p = _predecir(clave, m, X.iloc[corte:])
    real = y.iloc[corte:]
    base = np.full(len(real), y.iloc[:corte].mean())
    ss_res, ss_tot = ((real - p) ** 2).sum(), ((real - real.mean()) ** 2).sum()
    return Validacion(r2=float(1 - ss_res / ss_tot), mape_dia=_mape(real.values, p),
                      mape_semana=_mape_semanal(real, p), mape_base_dia=_mape(real.values, base),
                      mape_base_semana=_mape_semanal(real, base), n_test=len(real))


def entrenar(clave: str, diario: pd.DataFrame) -> ModeloDemanda:
    """Ajusta el modelo con todos los datos y calcula su validación y coeficientes."""
    X, y = matriz_historica(diario), diario.ingresos.reset_index(drop=True)
    val = validar(clave, diario)
    m = _ajustar(clave, X, y)
    coef = None
    if clave == "rlm":
        ic = m.conf_int()
        ic = np.asarray(ic)
        coef = pd.DataFrame({"coeficiente": np.asarray(m.params), "ic_inf": ic[:, 0], "ic_sup": ic[:, 1],
                             "p_valor": np.asarray(m.pvalues)},
                            index=["constante"] + COLUMNAS)
        sd = float(np.sqrt(m.scale))
    else:
        # El random forest se ajusta casi perfecto en entrenamiento: se usa el error de validación.
        corte = int(len(diario) * (1 - VALIDACION_FRACCION))
        mv = _ajustar(clave, X.iloc[:corte], y.iloc[:corte])
        sd = float(np.std(y.iloc[corte:].values - _predecir(clave, mv, X.iloc[corte:]), ddof=1))
    return ModeloDemanda(clave, m, sd, val, coef)


def simular(modelo: ModeloDemanda, inicio: pd.Timestamp, semanas: int, valores: dict[str, float],
            margen_bruto: float, costo_fijo_dia: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pronostica día a día y por semana para los valores asignados a las variables predictoras.

    Devuelve (diario, semanal). El IC 95 % semanal usa sd_residual * sqrt(7) (errores diarios
    aproximadamente independientes: la autocorrelación de los residuos es despreciable).
    """
    dias = semanas * 7
    X = matriz_escenario(inicio, dias, valores)
    ing = np.clip(modelo.predecir_dia(X), 0, None)
    d = pd.DataFrame({"fecha": X.index, "ingresos": ing, "festivo": X.festivo.values,
                      "fin_de_semana": X.fin_de_semana.values})
    d["semana"] = np.arange(dias) // 7 + 1
    d["utilidad"] = d.ingresos * margen_bruto - costo_fijo_dia
    s = d.groupby("semana").agg(inicio=("fecha", "min"), fin=("fecha", "max"), ingresos=("ingresos", "sum"),
                                utilidad=("utilidad", "sum"), festivos=("festivo", "sum"))
    e = Z95 * modelo.sd_residual * np.sqrt(7)
    s["ic_inf"], s["ic_sup"] = s.ingresos - e, s.ingresos + e
    s["ventas"] = valores["ventas"] * 7
    return d, s.reset_index()


def fuera_de_rango(valores: dict[str, float], rangos: dict[str, tuple[float, float, float]]) -> list[str]:
    """Nombres de predictores cuyo valor queda fuera del rango visto en los datos (extrapolación)."""
    return [PREDICTORES[k]["etiqueta"] for k, v in valores.items() if v < rangos[k][0] - 1e-9 or v > rangos[k][2] + 1e-9]


def predicciones_validacion(clave: str, diario: pd.DataFrame) -> pd.DataFrame:
    """Fecha, ingreso real y predicho en el tramo de validación (para graficar)."""
    X, y = matriz_historica(diario), diario.ingresos.reset_index(drop=True)
    corte = int(len(diario) * (1 - VALIDACION_FRACCION))
    m = _ajustar(clave, X.iloc[:corte], y.iloc[:corte])
    return pd.DataFrame({"fecha": diario.fecha.iloc[corte:].values, "real": y.iloc[corte:].values,
                         "predicho": _predecir(clave, m, X.iloc[corte:])})


def efecto_variables(modelo: ModeloDemanda, valores: dict[str, float], desv: dict[str, float], inicio: pd.Timestamp,
                     margen_bruto: float, costo_fijo_dia: float) -> pd.DataFrame:
    """Cambio en ingresos semanales al mover cada predictor ± 1 desviación estándar histórica (resto fijo)."""
    _, ref = simular(modelo, inicio, 1, valores, margen_bruto, costo_fijo_dia)
    filas = {}
    for k, sd in desv.items():
        fila = {}
        for nombre, signo in (("bajo", -1), ("alto", 1)):
            v = dict(valores); v[k] = max(0.0, v[k] + signo * sd)
            _, s = simular(modelo, inicio, 1, v, margen_bruto, costo_fijo_dia)
            fila[nombre] = float(s.ingresos.iloc[0] - ref.ingresos.iloc[0])
        fila["rango"] = abs(fila["alto"] - fila["bajo"])
        filas[PREDICTORES[k]["etiqueta"]] = fila
    return pd.DataFrame(filas).T
