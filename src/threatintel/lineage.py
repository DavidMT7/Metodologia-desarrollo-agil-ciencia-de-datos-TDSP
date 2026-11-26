# -*- coding: utf-8 -*-
"""Linaje de datos: hashes, snapshot y export de instructions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(raw_dir: Path, entries: list[dict]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fase 2 hardening — linaje reproducible para Fase 3",
        "artifacts": entries,
    }
    path = raw_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_instructions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: Path,
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    p_train = processed_dir / "instructions_train_clean.parquet"
    p_test = processed_dir / "instructions_test_clean.parquet"
    train.to_parquet(p_train, index=False)
    test.to_parquet(p_test, index=False)
    return p_train, p_test
