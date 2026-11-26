# -*- coding: utf-8 -*-
"""Puente CVE ↔ MITRE: CWE heuristics, keywords y equivalencias de taxonomía."""
from __future__ import annotations

import re

import pandas as pd

# Equivalencias CVE.likely_mitre_tactic → etiquetas ATT&CK / dataset MITRE
TAXONOMY_EQUIVALENCE = [
    {"cve_label": "Execution", "mitre_like": "Execution", "notes": "Directo"},
    {"cve_label": "Initial Access", "mitre_like": "Initial Access", "notes": "Directo"},
    {"cve_label": "Credential Access", "mitre_like": "Credential Access", "notes": "Directo"},
    {"cve_label": "Collection", "mitre_like": "Collection", "notes": "Directo"},
    {"cve_label": "Defense Evasion", "mitre_like": "Defense Impairment", "notes": "CVE usa Defense Evasion; corpus MITRE HF usa Defense Impairment"},
    {"cve_label": "Privilege Escalation", "mitre_like": "Privilege Escalation", "notes": "Directo"},
    {"cve_label": "Unknown", "mitre_like": None, "notes": "Requiere inferencia CWE/keywords"},
    {"cve_label": None, "mitre_like": "Stealth", "notes": "Dominante en catálogo MITRE HF; poco presente como etiqueta CVE"},
    {"cve_label": None, "mitre_like": "Persistence", "notes": "Presente en MITRE HF; raro/ausente en likely_mitre_tactic CVE"},
    {"cve_label": None, "mitre_like": "Reconnaissance", "notes": "MITRE HF; no aparece en top CVE"},
]

# CWE primario → táctica ATT&CK probable (heurística SOC, documentada)
CWE_TO_TACTIC: dict[str, str] = {
    "CWE-89": "Initial Access",  # SQLi
    "CWE-78": "Execution",  # OS command injection
    "CWE-77": "Execution",
    "CWE-94": "Execution",  # code injection
    "CWE-95": "Execution",
    "CWE-502": "Execution",  # deserialization
    "CWE-119": "Execution",  # buffer
    "CWE-120": "Execution",
    "CWE-125": "Collection",  # out-of-bounds read / info
    "CWE-787": "Execution",  # OOB write
    "CWE-416": "Execution",  # UAF
    "CWE-190": "Execution",
    "CWE-20": "Initial Access",  # improper input → often entry
    "CWE-287": "Credential Access",
    "CWE-306": "Credential Access",
    "CWE-307": "Credential Access",
    "CWE-798": "Credential Access",  # hard-coded creds
    "CWE-522": "Credential Access",
    "CWE-200": "Collection",  # info exposure
    "CWE-264": "Privilege Escalation",
    "CWE-269": "Privilege Escalation",
    "CWE-284": "Privilege Escalation",
    "CWE-22": "Initial Access",  # path traversal
    "CWE-352": "Initial Access",  # CSRF
    "CWE-79": "Initial Access",  # XSS → often Initial Access / Execution in web
    "CWE-918": "Initial Access",  # SSRF
    "CWE-611": "Collection",  # XXE
    "CWE-434": "Initial Access",  # unrestricted upload
}

KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(rce|remote code execution|command injection|arbitrary code)\b", re.I), "Execution"),
    (re.compile(r"\b(sql injection|sqli|authentication bypass|ssrf|path traversal)\b", re.I), "Initial Access"),
    (re.compile(r"\b(hard.?coded password|credential|bruteforce|password reset)\b", re.I), "Credential Access"),
    (re.compile(r"\b(privilege escalation|gain elevated|administrator privileges)\b", re.I), "Privilege Escalation"),
    (re.compile(r"\b(information disclosure|sensitive data|leak)\b", re.I), "Collection"),
    (re.compile(r"\b(bypass (antivirus|security|detection)|evade|obfuscat)\b", re.I), "Defense Evasion"),
]


def primary_cwe(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    m = re.search(r"CWE-\d+", str(value))
    return m.group(0) if m else None


def infer_tactic_from_text(text: str) -> str | None:
    if not text:
        return None
    for pat, tactic in KEYWORD_RULES:
        if pat.search(text):
            return tactic
    return None


def map_cve_tactics(df_cve: pd.DataFrame) -> pd.DataFrame:
    out = df_cve.copy()
    out["cwe_primary"] = out["cwe_ids"].map(primary_cwe)
    out["tactic_from_cwe"] = out["cwe_primary"].map(CWE_TO_TACTIC)
    out["tactic_from_keywords"] = out["description"].fillna("").map(infer_tactic_from_text)

    orig = out["likely_mitre_tactic"].astype(str)
    inferred = out["tactic_from_cwe"].fillna(out["tactic_from_keywords"])
    out["likely_mitre_tactic_enriched"] = orig.where(orig.ne("Unknown"), inferred)
    out["likely_mitre_tactic_enriched"] = out["likely_mitre_tactic_enriched"].fillna("Unknown")

    out["tactic_enrichment_source"] = "original"
    used_cwe = (orig.eq("Unknown")) & out["tactic_from_cwe"].notna()
    used_kw = (orig.eq("Unknown")) & out["tactic_from_cwe"].isna() & out["tactic_from_keywords"].notna()
    still = out["likely_mitre_tactic_enriched"].eq("Unknown")
    out.loc[used_cwe, "tactic_enrichment_source"] = "cwe_heuristic"
    out.loc[used_kw, "tactic_enrichment_source"] = "keyword_heuristic"
    out.loc[still, "tactic_enrichment_source"] = "unresolved"

    # map to MITRE HF taxonomy label for RAG joins
    evasion_fix = {"Defense Evasion": "Defense Impairment"}
    out["mitre_tactic_join_key"] = out["likely_mitre_tactic_enriched"].replace(evasion_fix)
    return out


def taxonomy_frame() -> pd.DataFrame:
    return pd.DataFrame(TAXONOMY_EQUIVALENCE)
