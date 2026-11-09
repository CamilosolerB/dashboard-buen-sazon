"""Configuración central: paleta corporativa, textos y parámetros del modelo."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"          # ruta relativa al proyecto (funciona en cualquier equipo)

# Paleta corporativa (cocina colombiana: terracota, mostaza y crema)
TERRACOTA = "#B5442C"
MOSTAZA = "#D9A441"
CAFE = "#2B2118"
CREMA = "#FBF6EE"
ARENA = "#F1E6D3"
GRIS = "#8C8173"
VERDE = "#4F7A5A"

NOMBRE_RESTAURANTE = "El Buen Sazón"
PREGUNTA = ("¿Cuántos ingresos y qué utilidad bruta generará El Buen Sazón en cada una de las próximas "
            "semanas según el volumen de ventas diarias y la proporción de ventas que incluyen gaseosa, "
            "acompañante o cerveza?")

# Variables predictoras asignables por el usuario (clave -> metadatos de la interfaz).
# 'paso' y límites de los deslizadores; el rango histórico real se calcula con los datos.
PREDICTORES = {
    "ventas":      dict(etiqueta="Ventas por día", unidad="ventas", paso=1.0, minimo=35.0, maximo=75.0,
                        ayuda="Número de ventas (facturas) promedio por día."),
    "pct_gaseosa": dict(etiqueta="Ventas con gaseosa", unidad="%", paso=0.5, minimo=0.0, maximo=60.0,
                        ayuda="Porcentaje de las ventas que incluyen al menos una gaseosa (venta cruzada)."),
    "pct_acomp":   dict(etiqueta="Ventas con acompañante", unidad="%", paso=0.5, minimo=0.0, maximo=45.0,
                        ayuda="Porcentaje de las ventas que incluyen al menos un acompañante."),
    "pct_cerveza": dict(etiqueta="Ventas con cerveza", unidad="%", paso=0.5, minimo=0.0, maximo=25.0,
                        ayuda="Porcentaje de las ventas que incluyen al menos una cerveza."),
}
# Variables de calendario (se calculan solas a partir de las fechas elegidas).
CALENDARIO = ["fin_de_semana", "festivo"]

FRANJAS_PICO = (12, 15)          # horas pico de almuerzo
Z95 = 1.96
VALIDACION_FRACCION = 0.25       # último 25 % de los días se reserva para validar
