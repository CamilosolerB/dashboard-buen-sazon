# El Buen Sazón · Dashboard "Simulando predicciones"

Proyecto de aula de **Business Analytics / Business Intelligence** (Universidad de Cundinamarca).
Dashboard interactivo en Python (**Streamlit + Plotly**) que responde la pregunta predictiva del proyecto:

> ¿Cuántos ingresos y qué utilidad bruta generará El Buen Sazón en cada una de las próximas semanas según el volumen de
> ventas diarias y la proporción de ventas que incluyen gaseosa, acompañante o cerveza?

El usuario asigna valores a las **variables predictoras** y la técnica predictiva (regresión lineal múltiple o random forest)
recalcula al instante los ingresos y la utilidad semanal con su intervalo de confianza del 95 %.

## Ejecutar en local (3 comandos)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                    # abre http://localhost:8501
```

Sin configurar nada, el dashboard lee los **CSV de `data/`** (instantánea de la base de datos, rutas relativas).

### Conectarlo a la base de datos en vivo (opcional, Supabase / PostgreSQL)

1. Copie `.env.example` a `.env` y escriba `DATABASE_URL=postgresql://usuario:clave@host:5432/postgres`
   (o las variables `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`). El archivo `.env` está en `.gitignore`: **nunca suba credenciales**.
2. Reinicie la app. La barra superior indica la fuente: «Base de datos PostgreSQL (consulta en vivo)». Si la conexión falla, vuelve a los CSV.
3. Para actualizar la instantánea local: `python scripts/export_snapshot.py`.

Esquema esperado: `ventas.venta`, `ventas.detalle_venta`, `ventas.producto`, `ventas.costo` (ver `src/data_loader.py`).

## Estructura (código modular)

| Ruta | Responsabilidad |
|---|---|
| `app.py` | Interfaz Streamlit: controles, pestañas e interacciones (solo orquesta) |
| `src/config.py` | Paleta corporativa, textos, definición de las variables predictoras |
| `src/data_loader.py` | Carga y ETL: consultas SQL documentadas, variables de entorno, respaldo CSV |
| `src/features.py` | Calendario colombiano (festivos, fin de semana) y matrices de predictores |
| `src/model.py` | Regresión lineal múltiple, random forest, validación cronológica, simulación de escenarios |
| `src/charts.py` | Gráficos Plotly |
| `scripts/export_snapshot.py` | Exporta la base a `data/*.csv` |
| `tests/` | Pruebas (`pytest -q`) |

## Variables predictoras

| Variable | Tipo | Descripción |
|---|---|---|
| Ventas por día | asignable | Número de facturas diarias |
| % ventas con gaseosa | asignable | Venta cruzada de gaseosas |
| % ventas con acompañante | asignable | Venta cruzada de acompañantes |
| % ventas con cerveza | asignable | Venta cruzada de cerveza |
| Fin de semana | calendario | Se calcula con la fecha |
| Festivo | calendario | Festivos de Colombia, se calcula con la fecha |

Variable objetivo: ingreso diario (COP). El pronóstico semanal suma siete días; la utilidad = ingresos × margen bruto − costo fijo diario.

## Interacciones disponibles

1. Selector de técnica predictiva (regresión lineal múltiple / random forest).
2. Fecha de inicio del pronóstico y número de semanas (1–12).
3. Tres escenarios rápidos (base, optimista, conservador) y deslizadores para las 4 variables predictoras.
4. Aviso automático cuando un valor queda fuera del rango histórico (extrapolación).
5. Interruptores para ver el intervalo de confianza y compararlo con el escenario base.
6. Gráfico «¿qué variable pesa más?» (sensibilidad ± 1 desviación estándar).
7. Drill-down: semana → detalle diario → tabla, y descarga del pronóstico en CSV.
8. Guardar escenarios propios y compararlos (ingresos, utilidad, variación vs base).
9. Histórico con filtro de fechas, selector de categorías y drill-down categoría → productos.
10. Pestaña de validación con R², errores y coeficientes con IC 95 %.

## Alcance y limitaciones (leer antes de usar los resultados)

* Los datos de ventas cubren 213 días (20-feb a 20-sep de 2026). Los **costos, tipos de evento y 10 productos fueron simulados** para el proyecto.
* La demanda diaria es muy estable: en validación el modelo **no supera por mucho al promedio histórico** (R² ≈ 0,08).
  Su valor es la simulación «qué pasa si»: cuantifica el efecto de cada variable, no una predicción perfecta.
* Los efectos de las variables son **asociaciones históricas**, no causalidad: subir la venta cruzada requiere una acción comercial que debe validarse con ventas reales.

