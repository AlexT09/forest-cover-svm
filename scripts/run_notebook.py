"""Ejecuta el notebook completo y guarda las salidas dentro del mismo .ipynb.

Uso:  python scripts/run_notebook.py            (perfil completo)
      FOREST_PROFILE=smoke python scripts/run_notebook.py   (prueba rápida de punta a punta)
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "notebooks" / "clasificacion_cobertura_forestal.ipynb"
profile = os.environ.get("FOREST_PROFILE", "full")
dst = src if profile == "full" else src.with_name(f"_{profile}_{src.name}")

nb = nbformat.read(src, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}},
                        allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb, dst)
    print(f"[{profile}] {dst.name} escrito en {(time.time() - t0) / 60:.1f} min", file=sys.stderr)
