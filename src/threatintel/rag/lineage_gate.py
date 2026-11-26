# -*- coding: utf-8 -*-
"""Control E18: la Fase III no entrena si clean y hardened se vuelven a mezclar."""
from __future__ import annotations

import pandas as pd


class LineageError(RuntimeError):
    pass


def _unknown_rate(series: pd.Series) -> float:
    return float((series.astype(str) == "Unknown").mean())


def assert_phase3_lineage(
    cve_clean: pd.DataFrame,
    cve_hard: pd.DataFrame,
    mitre_clean: pd.DataFrame,
    mitre_hard: pd.DataFrame,
) -> dict:
    """Falla si el ETL limpio volvió a quedar igual que el producto endurecido."""
    if cve_clean.equals(cve_hard) or mitre_clean.equals(mitre_hard):
        raise LineageError("E18: *_clean y *_hardened son iguales. No se entrena.")
    if "soc_priority_score" in cve_clean.columns or "likely_mitre_tactic_enriched" in cve_clean.columns:
        raise LineageError("E18: el parquet CVE limpio todavía tiene columnas de hardening.")
    if "soc_priority_score" not in cve_hard.columns or "likely_mitre_tactic_enriched" not in cve_hard.columns:
        raise LineageError("El parquet CVE endurecido no trae score ni táctica enriquecida.")
    if "detection" not in mitre_clean.columns or "detection" not in mitre_hard.columns:
        raise LineageError("Falta la columna detection en MITRE.")

    det_clean = float(mitre_clean["detection"].isna().mean())
    det_hard = float(mitre_hard["detection"].isna().mean())
    if det_clean != 1.0:
        raise LineageError(f"E18: detection limpio no está 100 % nulo ({det_clean:.1%}).")
    if det_hard != 0.0:
        raise LineageError(f"detection endurecido no está completo ({det_hard:.1%} nulo).")

    unknown_tactic = _unknown_rate(cve_hard["likely_mitre_tactic"])
    unknown_enriched = _unknown_rate(cve_hard["likely_mitre_tactic_enriched"])
    if unknown_tactic < 0.3:
        raise LineageError("La columna original likely_mitre_tactic ya no muestra el 39,5 % Unknown.")
    if not 0.10 < unknown_enriched < 0.25:
        raise LineageError("likely_mitre_tactic_enriched no está en el residual esperado (~16,5 %).")

    return {
        "cve_clean_shape": [int(cve_clean.shape[0]), int(cve_clean.shape[1])],
        "cve_hard_shape": [int(cve_hard.shape[0]), int(cve_hard.shape[1])],
        "mitre_clean_shape": [int(mitre_clean.shape[0]), int(mitre_clean.shape[1])],
        "mitre_hard_shape": [int(mitre_hard.shape[0]), int(mitre_hard.shape[1])],
        "detection_null_clean": det_clean,
        "detection_null_hardened": det_hard,
        "unknown_tactic": unknown_tactic,
        "unknown_enriched": unknown_enriched,
        "index_source": "hardened",
    }
