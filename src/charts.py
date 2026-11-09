"""Gráficos Plotly con la paleta corporativa. Cada función recibe datos ya procesados."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import CAFE, GRIS, MOSTAZA, TERRACOTA, VERDE

FUENTE = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def _base(fig: go.Figure, alto: int = 360, titulo: str | None = None) -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=44 if titulo else 12, b=10), title=dict(text=titulo, x=0.0, font=dict(size=15, color=CAFE)) if titulo else None,
                      font=dict(family=FUENTE, color=CAFE, size=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"), hoverlabel=dict(font_family=FUENTE))
    fig.update_xaxes(showgrid=False, linecolor="#d9cdb8")
    fig.update_yaxes(gridcolor="#e8dcc6", zeroline=False)
    return fig


_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
_DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def fecha_es(f: pd.Timestamp, con_dia: bool = False, anio: bool = False) -> str:
    """'21-sep' (o 'Lun 21-sep', '21-sep-2026') con nombres en español."""
    t = f"{f.day:02d}-{_MESES[f.month - 1]}" + (f"-{f.year}" if anio else "")
    return f"{_DIAS[f.dayofweek]} {t}" if con_dia else t


def _ejes_fecha(fig: go.Figure, fechas, n: int = 7) -> None:
    """Etiquetas de fecha en español (Plotly las muestra en inglés por defecto)."""
    f = pd.DatetimeIndex(pd.to_datetime(list(fechas)))
    pos = f[np.linspace(0, len(f) - 1, min(n, len(f))).astype(int)]
    fig.update_xaxes(tickmode="array", tickvals=list(pos), ticktext=[fecha_es(x) for x in pos])


def cop(v: float) -> str:
    """Formato de pesos colombianos: $1.234.567"""
    return "$" + f"{v:,.0f}".replace(",", ".")


def millones(v: float) -> str:
    return "$" + f"{v/1e6:,.2f}".replace(".", ",") + " M"


def fig_semanal(s: pd.DataFrame, base: pd.DataFrame | None = None, ic: bool = True) -> go.Figure:
    """Ingresos pronosticados por semana (barras) con intervalo de confianza y, opcional, escenario base."""
    x = [f"S{int(w)} · {fecha_es(i)}" for w, i in zip(s.semana, s.inicio)]
    fig = go.Figure()
    fig.add_bar(x=x, y=s.ingresos, name="Escenario simulado", marker_color=TERRACOTA,
                error_y=dict(type="data", symmetric=False, array=(s.ic_sup - s.ingresos) if ic else None,
                             arrayminus=(s.ingresos - s.ic_inf) if ic else None, color=CAFE, thickness=1.2, width=4) if ic else None,
                customdata=np.c_[s.ic_inf, s.ic_sup, s.utilidad],
                hovertemplate="<b>%{x}</b><br>Ingresos: $%{y:,.0f}<br>IC 95 %: $%{customdata[0]:,.0f} – $%{customdata[1]:,.0f}<br>Utilidad: $%{customdata[2]:,.0f}<extra></extra>")
    if base is not None:
        fig.add_scatter(x=x, y=base.ingresos, name="Escenario base", mode="lines+markers", line=dict(color=MOSTAZA, width=2.5, dash="dash"),
                        marker=dict(size=7), hovertemplate="Base: $%{y:,.0f}<extra></extra>")
    lo = float(min(s.ic_inf.min(), base.ingresos.min() if base is not None else 1e18))
    hi = float(max(s.ic_sup.max(), base.ingresos.max() if base is not None else 0))
    fig.update_yaxes(range=[lo * 0.9, hi * 1.04], tickprefix="$", tickformat=",.0f", title="Ingresos semanales (COP, eje recortado)")
    return _base(fig, 380)


def fig_diario(d: pd.DataFrame) -> go.Figure:
    """Detalle día a día de una semana (drill-down)."""
    et = [fecha_es(f, con_dia=True) for f in d.fecha]
    fig = go.Figure(go.Bar(x=et, y=d.ingresos, marker_color=[MOSTAZA if f else TERRACOTA for f in d.festivo],
                           hovertemplate="%{x}<br>Ingresos: $%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", rangemode="tozero")
    return _base(fig, 300)


def fig_comparacion(tabla: pd.DataFrame) -> go.Figure:
    """Ingresos y utilidad total del horizonte por escenario guardado."""
    fig = go.Figure()
    fig.add_bar(x=tabla.index, y=tabla["Ingresos"], name="Ingresos", marker_color=TERRACOTA, hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>")
    fig.add_bar(x=tabla.index, y=tabla["Utilidad"], name="Utilidad", marker_color=VERDE, hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group"); fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _base(fig, 340)


def fig_tornado(efectos: pd.DataFrame) -> go.Figure:
    """Efecto en los ingresos semanales de mover cada variable ± una desviación estándar histórica."""
    e = efectos.sort_values("rango")
    fig = go.Figure()
    fig.add_bar(y=e.index, x=e.bajo, orientation="h", name="− 1 desv. est.", marker_color=GRIS, hovertemplate="%{y}: %{x:+,.0f}<extra></extra>")
    fig.add_bar(y=e.index, x=e.alto, orientation="h", name="+ 1 desv. est.", marker_color=TERRACOTA, hovertemplate="%{y}: %{x:+,.0f}<extra></extra>")
    fig.update_layout(barmode="relative"); fig.update_xaxes(tickformat="+,.0f", title="Cambio semanal de ingresos (COP)")
    fig = _base(fig, 300)
    fig.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"))
    return fig


def fig_historico(d: pd.DataFrame) -> go.Figure:
    """Ingreso diario y promedio móvil de 7 días dentro del rango de fechas elegido."""
    fig = go.Figure()
    fig.add_scatter(x=d.fecha, y=d.ingresos, mode="lines", name="Ingreso diario", line=dict(color="#d6bfa0", width=1))
    fig.add_scatter(x=d.fecha, y=d.ingresos.rolling(7, min_periods=3).mean(), mode="lines", name="Promedio móvil 7 días", line=dict(color=TERRACOTA, width=2.5))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    _ejes_fecha(fig, d.fecha)
    return _base(fig, 320)


def fig_categorias(df: pd.DataFrame) -> go.Figure:
    """Ingresos por categoría (barras horizontales)."""
    df = df.sort_values("ingresos")
    fig = go.Figure(go.Bar(y=df.categoria, x=df.ingresos, orientation="h", marker_color=TERRACOTA, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return _base(fig, 300)


def fig_productos(df: pd.DataFrame, categoria: str) -> go.Figure:
    """Drill-down: productos de una categoría."""
    df = df.sort_values("ingresos")
    fig = go.Figure(go.Bar(y=df.producto, x=df.ingresos, orientation="h", marker_color=MOSTAZA, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return _base(fig, max(240, 32 * len(df) + 60))


def fig_validacion(fechas, real, pred) -> go.Figure:
    """Real vs predicho en los días de validación (no usados para entrenar)."""
    fig = go.Figure()
    fig.add_scatter(x=fechas, y=real, name="Real", mode="lines", line=dict(color=CAFE, width=1.6))
    fig.add_scatter(x=fechas, y=pred, name="Predicho", mode="lines", line=dict(color=TERRACOTA, width=2))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    _ejes_fecha(fig, fechas)
    return _base(fig, 320)
