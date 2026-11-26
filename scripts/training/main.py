# -*- coding: utf-8 -*-
"""Entrypoint TDSP de fine-tuning Fase IV.

Uso (desde raíz Fase_IV, PYTHONPATH=src):

  python scripts/training/main.py list
  python scripts/training/main.py science
  python scripts/training/main.py expand
  python scripts/training/main.py improve
  python scripts/training/main.py stripped
  python scripts/training/main.py ttp
  python scripts/training/main.py accept   # demo S1-S5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


COMMANDS = {
    "science": "threatintel.rag.run_fase4_science",
    "expand": "threatintel.rag.run_fase4_expand",
    "improve": "threatintel.rag.run_fase4_improve",
    "stripped": "threatintel.rag.run_fase4_stripped_ft",
    "ttp": "threatintel.rag.run_fase4_ttp_pack",
    "accept": "threatintel.rag.run_fase4",
}


def list_cmds() -> None:
    print("Fine-tuning Fase IV — comandos disponibles:\n")
    print("  list       muestra esta ayuda")
    for key, mod in COMMANDS.items():
        print(f"  {key:<10} python -m {mod}")
    print()
    print("Notebook:", ROOT / "scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb")
    print("Docs FT: ", ROOT / "docs/modeling/fine_tuning")
    print("Artifacts:", ROOT / "artifacts/fase4")


def run_module(dotted: str) -> None:
    import runpy

    print(f"→ ejecutando {dotted}", flush=True)
    runpy.run_module(dotted, run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDSP scripts/training — Fase IV fine-tuning")
    parser.add_argument("command", nargs="?", default="list", choices=["list", *COMMANDS.keys()])
    args = parser.parse_args(argv)
    if args.command == "list":
        list_cmds()
        return 0
    run_module(COMMANDS[args.command])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
