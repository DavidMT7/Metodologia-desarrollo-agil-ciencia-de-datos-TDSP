# -*- coding: utf-8 -*-
"""Evalúa / acepta el checkpoint Fase IV (demo S1–S5)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    runpy.run_module("threatintel.rag.run_fase4", run_name="__main__")
