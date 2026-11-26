# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Ancestro más cercano con configs/retrieval.yaml y docs/data/."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (cur / "configs" / "retrieval.yaml").is_file() and (cur / "docs" / "data").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("No se encontró configs/retrieval.yaml + docs/data/ en ancestros")


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or find_project_root()
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "figures": root / "docs" / "data" / "figures",
        "docs_data": root / "docs" / "data",
        "eval": root / "docs" / "data" / "evaluation",
        "attack_cache": root / "data" / "raw" / "mitre_attack",
    }
