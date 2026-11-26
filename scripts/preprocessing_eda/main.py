"""Punto de entrada TDSP — preprocessing/EDA.

Preferir:
  - Notebook: Proyecto_Fase2_ThreatIntel_EDA.ipynb
  - Hardening CTI: python scripts/preprocessing_eda/harden_fase2_cti.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("harden_fase2_cti.py")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
