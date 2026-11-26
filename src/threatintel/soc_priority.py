# -*- coding: utf-8 -*-
"""Score compuesto de priorización SOC (documentado, auditable)."""
from __future__ import annotations

import pandas as pd

# Pesos fijos — KEV domina; CVSS solo matiza. Suma teórica máx ≈ 100.
W_KEV = 50.0
W_RANSOMWARE = 15.0
W_EXPLOIT = 15.0
W_CVSS = 15.0  # (cvss/10)*15
W_RISK = 5.0  # (risk_score/10)*5


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def soc_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    kev = _as_bool(out["in_cisa_kev"])
    if "ransomware_use" in out.columns:
        ransom = _as_bool(out["ransomware_use"])
    else:
        ransom = pd.Series(False, index=out.index)
    exploit = _as_bool(out["exploit_available"])
    cvss = pd.to_numeric(out["cvss_score"], errors="coerce").fillna(0.0).clip(0, 10)
    risk = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0).clip(0, 10)

    score = (
        kev.astype(float) * W_KEV
        + ransom.astype(float) * W_RANSOMWARE
        + exploit.astype(float) * W_EXPLOIT
        + (cvss / 10.0) * W_CVSS
        + (risk / 10.0) * W_RISK
    )
    out["soc_priority_score"] = score.round(2)

    band = pd.Series("P4-monitor", index=out.index, dtype=object)
    band = band.mask(score >= 20, "P3-plan")
    band = band.mask(score >= 40, "P2-patch-soon")
    band = band.mask(score >= 60, "P1-immediate")
    # Política: todo KEV es al menos P2; KEV + ransomware → P1
    band = band.mask(kev, "P2-patch-soon")
    band = band.mask(kev & ransom, "P1-immediate")
    band = band.mask(kev & (score >= 60), "P1-immediate")
    out["soc_priority_band"] = band
    return out
