"""Dashboard interactivo - Simulando predicciones (El Buen Sazón).

Ejecutar:  streamlit run app.py
La interfaz solo orquesta: los datos vienen de src.data_loader, el modelo de src.model
y los gráficos de src.charts.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.config import CAFE, CREMA, NOMBRE_RESTAURANTE, PREDICTORES, PREGUNTA, TERRACOTA
from src.data_loader import cargar_datos
from src.features import rangos_historicos
from src.model import MODELOS, efecto_variables, entrenar, fuera_de_rango, predicciones_validacion, simular

st.set_page_config(page_title=f"{NOMBRE_RESTAURANTE} · Simulador predictivo", page_icon="🍽️", layout="wide")

st.markdown(f"""
<style>
  h1, h2, h3 {{ color:{CAFE}; letter-spacing:-0.01em; }}
  .pregunta {{ background:#F1E6D3; border-left:5px solid {TERRACOTA}; padding:.7rem 1rem; border-radius:6px; font-size:.95rem; }}
  div[data-testid="stMetric"] {{ background:white; border:1px solid #e8dcc6; border-radius:10px; padding:.8rem 1rem; }}
  .aviso {{ font-size:.85rem; color:#6b5e4f; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- datos y modelos (con caché)
@st.cache_data(show_spinner="Cargando datos…", ttl=600)
def _datos():
    return cargar_datos()


@st.cache_resource(show_spinner="Entrenando modelo…")
def _modelo(clave: str, huella: int):
    return entrenar(clave, _datos().diario)


D = _datos()
RANGOS = rangos_historicos(D.diario)
DESV = {k: float(D.diario[k].std()) for k in PREDICTORES}
ULTIMA = D.diario.fecha.max()

# Escenarios predefinidos (valores asignados a las variables predictoras)
BASE = {k: round(RANGOS[k][1], 1) for k in PREDICTORES}
PRESETS = {
    "Base (promedio histórico)": BASE,
    "Optimista (venta cruzada)": {"ventas": 55.0, "pct_gaseosa": 35.0, "pct_acomp": 22.0, "pct_cerveza": 5.5},
    "Conservador (demanda a la baja)": {"ventas": 51.0, "pct_gaseosa": 24.0, "pct_acomp": 13.0, "pct_cerveza": 3.0},
}

# ----------------------------------------------------------------------------- estado de sesión
for k in PREDICTORES:
    st.session_state.setdefault(f"v_{k}", BASE[k])
st.session_state.setdefault("escenarios", {n: dict(v) for n, v in PRESETS.items()})


def aplicar_preset(nombre: str) -> None:
    for k, v in st.session_state.escenarios[nombre].items():
        st.session_state[f"v_{k}"] = float(v)


def guardar_escenario() -> None:
    nombre = st.session_state.get("nombre_esc", "").strip()
    if nombre:
        st.session_state.escenarios[nombre] = {k: st.session_state[f"v_{k}"] for k in PREDICTORES}


# ----------------------------------------------------------------------------- barra lateral (controles)
with st.sidebar:
    st.header("Parámetros")
    clave = st.radio("Técnica predictiva", list(MODELOS), format_func=MODELOS.get, key="modelo",
                     help="La regresión lineal es interpretable y extrapola; el random forest no sale del rango histórico.")
    inicio = pd.Timestamp(st.date_input("Inicio del pronóstico", value=(ULTIMA + pd.Timedelta(days=1)).date(), key="inicio",
                                        help="Primer día a pronosticar."))
    semanas = st.slider("Semanas a pronosticar", 1, 12, 8, key="semanas")

    st.subheader("Escenarios rápidos")
    for nombre in PRESETS:
        st.button(nombre, on_click=aplicar_preset, args=(nombre,), use_container_width=True, key=f"btn_{nombre}")

    st.subheader("Variables predictoras")
    valores = {}
    for k, m in PREDICTORES.items():
        lo, mu, hi = RANGOS[k]
        valores[k] = st.slider(f"{m['etiqueta']} ({m['unidad']})", m["minimo"], m["maximo"], step=0.1 if k != "ventas" else 0.5,
                               key=f"v_{k}", help=f"{m['ayuda']} Histórico: mín {lo:.1f} · prom {mu:.1f} · máx {hi:.1f}.")
        st.caption(f"Histórico: {lo:.1f} – {hi:.1f} (promedio {mu:.1f})")
    st.caption("Los festivos y fines de semana se toman del calendario real de las fechas elegidas.")

    st.subheader("Guardar para comparar")
    st.text_input("Nombre del escenario", key="nombre_esc", placeholder="Ej.: Promoción gaseosa")
    st.button("Guardar escenario actual", on_click=guardar_escenario, use_container_width=True, type="primary")

modelo = _modelo(clave, len(D.diario))
margen, fijo = D.margen_bruto, D.costo_fijo_dia
diario_sim, semanal = simular(modelo, inicio, semanas, valores, margen, fijo)
_, semanal_base = simular(modelo, inicio, semanas, BASE, margen, fijo)

# ----------------------------------------------------------------------------- encabezado
st.title(f"🍽️ {NOMBRE_RESTAURANTE}: simulador de ingresos")
st.markdown(f'<div class="pregunta"><b>Pregunta predictiva.</b> {PREGUNTA}</div>', unsafe_allow_html=True)
st.caption(f"Fuente de datos: {D.fuente} · {len(D.diario)} días ({charts.fecha_es(D.diario.fecha.min(), anio=True)} a {charts.fecha_es(ULTIMA, anio=True)}) · "
           f"margen bruto {margen*100:.1f} % · costo fijo {charts.cop(fijo)}/día (costos simulados para el proyecto)")

alertas = fuera_de_rango(valores, RANGOS)
if alertas:
    st.warning("Extrapolación: " + ", ".join(alertas) + " quedan fuera del rango visto en los datos. "
               "El resultado es menos confiable" + (" (el random forest repite su valor extremo)." if clave == "rf" else "."))

tab_sim, tab_comp, tab_hist, tab_mod = st.tabs(["Simulador", "Comparar escenarios", "Histórico y drill-down", "Modelo y validación"])

# ----------------------------------------------------------------------------- 1. simulador
with tab_sim:
    ing, uti, ing_b, uti_b = semanal.ingresos.sum(), semanal.utilidad.sum(), semanal_base.ingresos.sum(), semanal_base.utilidad.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Ingresos en {semanas} semanas", charts.millones(ing), f"{100*(ing/ing_b-1):+.1f} % vs base")
    k2.metric("Ingreso semanal promedio", charts.millones(ing / semanas))
    k3.metric("Utilidad estimada", charts.millones(uti), f"{100*(uti/uti_b-1):+.1f} % vs base" if uti_b else None)
    k4.metric("Ventas totales", f"{valores['ventas']*7*semanas:,.0f}".replace(",", "."))

    izq, der = st.columns([1.55, 1])
    with izq:
        st.subheader("Ingresos por semana")
        ver_ic = st.toggle("Mostrar intervalo de confianza 95 %", value=True, key="ver_ic")
        ver_base = st.toggle("Comparar con el escenario base", value=True, key="ver_base")
        st.plotly_chart(charts.fig_semanal(semanal, semanal_base if ver_base else None, ver_ic), use_container_width=True)
    with der:
        st.subheader("¿Qué variable pesa más?")
        ef = efecto_variables(modelo, valores, DESV, inicio, margen, fijo)
        st.plotly_chart(charts.fig_tornado(ef), use_container_width=True)
        st.caption("Cambio semanal en ingresos al mover cada variable ± 1 desviación estándar histórica.")

    st.subheader("Drill-down: detalle de una semana")
    opciones = {f"Semana {int(r.semana)} ({charts.fecha_es(r.inicio)} a {charts.fecha_es(r.fin)})": int(r.semana) for r in semanal.itertuples()}
    elegida = opciones[st.selectbox("Semana a detallar", list(opciones), key="semana_det")]
    dd = diario_sim[diario_sim.semana == elegida]
    a, b = st.columns([1.3, 1])
    a.plotly_chart(charts.fig_diario(dd), use_container_width=True)
    tabla = pd.DataFrame({"Día": [charts.fecha_es(f, con_dia=True) for f in dd.fecha], "Ingresos": dd.ingresos.map(charts.cop).values,
                          "Utilidad": dd.utilidad.map(charts.cop).values, "Festivo": dd.festivo.map({1: "Sí", 0: ""}).values})
    b.dataframe(tabla, hide_index=True, use_container_width=True)
    st.caption("Barras mostaza = días festivos en Colombia.")
    st.download_button("Descargar pronóstico semanal (CSV)", semanal.round(0).to_csv(index=False).encode("utf-8"),
                       file_name="pronostico_semanal.csv", mime="text/csv")

# ----------------------------------------------------------------------------- 2. comparar escenarios
with tab_comp:
    st.subheader("Escenarios guardados")
    filas = {}
    for nombre, v in st.session_state.escenarios.items():
        _, s = simular(modelo, inicio, semanas, v, margen, fijo)
        filas[nombre] = {**{PREDICTORES[k]["etiqueta"]: v[k] for k in PREDICTORES}, "Ingresos": s.ingresos.sum(), "Utilidad": s.utilidad.sum()}
    comp = pd.DataFrame(filas).T
    base_ing = comp["Ingresos"].iloc[0]
    comp["Δ ingresos vs base"] = 100 * (comp["Ingresos"] / base_ing - 1)
    st.plotly_chart(charts.fig_comparacion(comp), use_container_width=True)
    vista = comp.copy()
    for c in vista.columns[:4]:
        vista[c] = vista[c].map(lambda x: f"{x:.1f}".replace(".", ","))
    vista["Ingresos"], vista["Utilidad"] = comp["Ingresos"].map(charts.cop), comp["Utilidad"].map(charts.cop)
    vista["Δ ingresos vs base"] = comp["Δ ingresos vs base"].map(lambda x: f"{x:+.1f} %".replace(".", ","))
    st.dataframe(vista, use_container_width=True)
    quitar = st.multiselect("Eliminar escenarios guardados", [n for n in st.session_state.escenarios if n not in PRESETS], key="quitar")
    if quitar and st.button("Eliminar seleccionados"):
        for n in quitar:
            st.session_state.escenarios.pop(n, None)
        st.rerun()
    st.caption(f"Horizonte: {semanas} semanas desde {charts.fecha_es(inicio, anio=True)} · técnica: {MODELOS[clave]}. Guarde sus propios escenarios desde la barra lateral.")

# ----------------------------------------------------------------------------- 3. histórico
with tab_hist:
    st.subheader("Comportamiento histórico")
    f1, f2 = st.columns([1, 1.2])
    rango = f1.date_input("Rango de fechas", value=(D.diario.fecha.min().date(), ULTIMA.date()),
                          min_value=D.diario.fecha.min().date(), max_value=ULTIMA.date(), key="rango_hist")
    cats = sorted(D.productos_dia.categoria.unique())
    sel = f2.multiselect("Categorías", cats, default=cats, key="cats_hist")
    if isinstance(rango, tuple) and len(rango) == 2:
        ini, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
    else:
        ini, fin = D.diario.fecha.min(), ULTIMA
    dh = D.diario[(D.diario.fecha >= ini) & (D.diario.fecha <= fin)]
    ph = D.productos_dia[(D.productos_dia.fecha >= ini) & (D.productos_dia.fecha <= fin) & D.productos_dia.categoria.isin(sel)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Ingresos del rango", charts.millones(dh.ingresos.sum()))
    m2.metric("Ventas", f"{dh.ventas.sum():,.0f}".replace(",", "."))
    m3.metric("Ticket promedio", charts.cop(dh.ingresos.sum() / max(dh.ventas.sum(), 1)))
    st.plotly_chart(charts.fig_historico(dh), use_container_width=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Ingresos por categoría**")
        por_cat = ph.groupby("categoria", as_index=False).ingresos.sum()
        if len(por_cat):
            st.plotly_chart(charts.fig_categorias(por_cat), use_container_width=True)
        else:
            st.info("Seleccione al menos una categoría.")
    with g2:
        st.markdown("**Drill-down: productos de una categoría**")
        if len(por_cat):
            cat = st.selectbox("Categoría", por_cat.categoria.tolist(), key="cat_drill")
            por_prod = ph[ph.categoria == cat].groupby("producto", as_index=False).ingresos.sum()
            st.plotly_chart(charts.fig_productos(por_prod, cat), use_container_width=True)

# ----------------------------------------------------------------------------- 4. modelo y validación
with tab_mod:
    v = modelo.validacion
    st.subheader(f"{MODELOS[clave]}: ¿qué tan confiable es?")
    a, b, c, d_ = st.columns(4)
    a.metric("R² en validación", f"{v.r2:.2f}".replace(".", ","))
    b.metric("Error diario (MAPE)", f"{v.mape_dia:.1f} %", f"promedio simple: {v.mape_base_dia:.1f} %", delta_color="off")
    c.metric("Error semanal (MAPE)", f"{v.mape_semana:.1f} %", f"promedio simple: {v.mape_base_semana:.1f} %", delta_color="off")
    d_.metric("Días de validación", v.n_test)
    pv = predicciones_validacion(clave, D.diario)
    st.plotly_chart(charts.fig_validacion(pv.fecha, pv.real, pv.predicho), use_container_width=True)
    st.markdown(f"""
**Lectura honesta del modelo.** La validación usa el último 25 % de los días, que el modelo no vio al entrenar.
El error semanal ronda {v.mape_semana:.1f} %, similar al de repetir el promedio ({v.mape_base_semana:.1f} %): la demanda diaria de este restaurante es muy estable,
así que **el valor del modelo está en cuantificar el efecto de las variables (simulación «qué pasa si»), no en superar al promedio histórico**.
Los costos, tipos de evento y 10 productos de la base fueron simulados para el proyecto; úselo como ejercicio académico y valide con ventas reales.
""")
    if modelo.coeficientes is not None:
        st.markdown("**Coeficientes de la regresión (efecto en COP por día de +1 unidad de la variable)**")
        nombres = {"constante": "Constante", "fin_de_semana": "Fin de semana (sí=1)", "festivo": "Festivo (sí=1)",
                   **{k: m["etiqueta"] for k, m in PREDICTORES.items()}}
        co = modelo.coeficientes.rename(index=nombres)
        st.dataframe(co, use_container_width=True, column_config={
            "coeficiente": st.column_config.NumberColumn("Coeficiente", format="%.0f"), "ic_inf": st.column_config.NumberColumn("IC 95 % inf.", format="%.0f"),
            "ic_sup": st.column_config.NumberColumn("IC 95 % sup.", format="%.0f"), "p_valor": st.column_config.NumberColumn("p-valor", format="%.3f")})
        st.caption("Fin de semana y festivo no son significativos (p > 0,05): se mantienen como controles de calendario.")
