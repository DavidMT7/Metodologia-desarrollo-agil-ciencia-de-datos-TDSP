# -*- coding: utf-8 -*-
"""Contratos mínimos de calidad (sin dependencia Pandera)."""
from __future__ import annotations

import pandas as pd


class SchemaError(ValueError):
    pass


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: faltan columnas {missing}")


def validate_cve(df: pd.DataFrame) -> None:
    _require_cols(
        df,
        [
            "cve_id",
            "description",
            "cvss_score",
            "in_cisa_kev",
            "exploit_available",
            "likely_mitre_tactic",
            "soc_priority_score",
            "likely_mitre_tactic_enriched",
        ],
        "cve",
    )
    if df["cve_id"].duplicated().any():
        raise SchemaError("cve: cve_id duplicados")
    if df["description"].isna().any():
        raise SchemaError("cve: description nula")
    if not pd.api.types.is_numeric_dtype(df["soc_priority_score"]):
        raise SchemaError("cve: soc_priority_score no numérico")


def validate_mitre(df: pd.DataFrame) -> None:
    _require_cols(df, ["technique_id", "description", "detection", "detection_source"], "mitre")
    if df["technique_id"].duplicated().any():
        raise SchemaError("mitre: technique_id duplicados")
    # Tras enriquecimiento, detection no debería ser 100% nulo
    null_rate = df["detection"].isna().mean()
    if null_rate > 0.25:
        raise SchemaError(f"mitre: detection aún {null_rate:.1%} nulo tras enrich")


def validate_kaggle(df: pd.DataFrame) -> None:
    _require_cols(df, ["incident_id", "severity_level"], "kaggle")
    if df.isna().any().any():
        # permitir NaN solo en columnas auxiliares opcionales
        crit = df[["incident_id", "severity_level"]]
        if crit.isna().any().any():
            raise SchemaError("kaggle: nulos en columnas críticas")


def validate_instructions(df: pd.DataFrame, name: str) -> None:
    _require_cols(df, ["instruction", "input", "output"], name)
    if len(df) == 0:
        raise SchemaError(f"{name}: vacío")
