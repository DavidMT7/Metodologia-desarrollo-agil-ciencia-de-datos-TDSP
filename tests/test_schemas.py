# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatintel.schemas import SchemaError, validate_cve, validate_kaggle, validate_mitre

PROCESSED = ROOT / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "cve_data_hardened.parquet").exists(), reason="parquet missing")
def test_cve_schema():
    """Contrato de producto CTI = hardened (score SOC + táctica enriquecida)."""
    df = pd.read_parquet(PROCESSED / "cve_data_hardened.parquet")
    validate_cve(df)


@pytest.mark.skipif(not (PROCESSED / "mitre_attack_hardened.parquet").exists(), reason="parquet missing")
def test_mitre_detection_enriched():
    df = pd.read_parquet(PROCESSED / "mitre_attack_hardened.parquet")
    validate_mitre(df)
    assert df["detection"].isna().mean() < 0.25


@pytest.mark.skipif(not (PROCESSED / "kaggle_cti_2026_clean.parquet").exists(), reason="parquet missing")
def test_kaggle_schema():
    validate_kaggle(pd.read_parquet(PROCESSED / "kaggle_cti_2026_clean.parquet"))


@pytest.mark.skipif(
    not (PROCESSED / "cve_data_clean.parquet").exists()
    or not (PROCESSED / "cve_data_hardened.parquet").exists(),
    reason="parquet missing",
)
def test_clean_hardened_lineage_separated():
    """E18: clean ETL no debe ser idéntico al hardened."""
    cve_c = pd.read_parquet(PROCESSED / "cve_data_clean.parquet")
    cve_h = pd.read_parquet(PROCESSED / "cve_data_hardened.parquet")
    mit_c = pd.read_parquet(PROCESSED / "mitre_attack_clean.parquet")
    mit_h = pd.read_parquet(PROCESSED / "mitre_attack_hardened.parquet")
    assert not cve_c.equals(cve_h)
    assert not mit_c.equals(mit_h)
    assert "soc_priority_score" not in cve_c.columns
    assert "soc_priority_score" in cve_h.columns
    assert mit_c["detection"].isna().mean() == 1.0
    assert mit_h["detection"].isna().mean() < 0.25
    assert (cve_c["likely_mitre_tactic"] == "Unknown").mean() == pytest.approx(0.3954, abs=1e-4)
    assert (cve_h["likely_mitre_tactic_enriched"] == "Unknown").mean() == pytest.approx(0.1648, abs=1e-4)


def test_cve_rejects_duplicate_ids():
    df = pd.DataFrame(
        {
            "cve_id": ["CVE-1", "CVE-1"],
            "description": ["a", "b"],
            "cvss_score": [9.0, 8.0],
            "in_cisa_kev": [False, False],
            "exploit_available": [True, False],
            "likely_mitre_tactic": ["Execution", "Unknown"],
            "soc_priority_score": [1.0, 2.0],
            "likely_mitre_tactic_enriched": ["Execution", "Execution"],
        }
    )
    with pytest.raises(SchemaError):
        validate_cve(df)
