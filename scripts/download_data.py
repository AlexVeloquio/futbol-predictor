"""
download_data.py — Descarga el dataset de Liga MX desde Kaggle.

Requiere la CLI de Kaggle instalada y configurada:
  pip install kaggle
  # Colocar kaggle.json en ~/.kaggle/

Alternativamente, descarga manualmente desde:
  https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022
"""

import os
import subprocess
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATASET = "gerardojaimeescareo/ligamx-matches-2016-2022"


def check_kaggle_cli():
    """Verifica que la CLI de Kaggle esté instalada."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Verificar si ya hay CSVs
    existing = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if existing:
        print(f"⏭️  Ya hay {len(existing)} archivo(s) CSV en {RAW_DIR}:")
        for f in existing:
            print(f"   - {f}")
        print("Elimínalos si quieres descargar de nuevo.")
        return

    if not check_kaggle_cli():
        print("⚠️  Kaggle CLI no encontrada.")
        print("   Instala con: pip install kaggle")
        print("   Configura con: ~/.kaggle/kaggle.json")
        print(f"\n   O descarga manualmente desde:")
        print(f"   https://www.kaggle.com/datasets/{DATASET}")
        print(f"   y coloca el CSV en: {RAW_DIR}")
        sys.exit(1)

    print(f"📥 Descargando dataset: {DATASET}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR, "--unzip"],
        check=True,
    )
    print(f"✅ Dataset descargado en {RAW_DIR}")


if __name__ == "__main__":
    main()
