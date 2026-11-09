"""Exporta a data/*.csv el resultado de las consultas (requiere conexión a la base de datos).

Uso:  DATABASE_URL=postgresql://... python scripts/export_snapshot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR                       # noqa: E402
from src.data_loader import ARCHIVOS, crear_motor, leer_consultas  # noqa: E402

if __name__ == "__main__":
    motor = crear_motor()
    if motor is None:
        sys.exit("Defina DATABASE_URL (o PGHOST/PGUSER/PGPASSWORD/PGDATABASE) antes de exportar.")
    DATA_DIR.mkdir(exist_ok=True)
    for nombre, df in leer_consultas(motor).items():
        df.to_csv(ARCHIVOS[nombre], index=False)
        print(f"{nombre}: {len(df)} filas -> {ARCHIVOS[nombre].name}")
