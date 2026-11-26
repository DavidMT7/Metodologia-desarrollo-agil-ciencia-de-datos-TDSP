# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Ancestro más externo que contiene docs/data/."""
    cur = (start or Path.cwd()).resolve()
    candidates: list[Path] = []
    for _ in range(12):
        if (cur / "docs" / "data").is_dir():
            candidates.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    if not candidates:
        raise FileNotFoundError("No se encontró docs/data/ en ancestros")
    return candidates[-1]


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
