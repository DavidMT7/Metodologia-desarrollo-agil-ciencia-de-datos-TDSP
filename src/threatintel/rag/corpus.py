# -*- coding: utf-8 -*-
"""Corpus Fase III. Solo debe construirse con los dataframes endurecidos."""
from __future__ import annotations

import re

import pandas as pd

from threatintel.rag.ids import strip_ids

_ANALYTIC = re.compile(r"(?=Analytic\s+\d+\s*:)", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _clean(text: object) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    value = _WS.sub(" ", str(text)).strip()
    if value.lower() == "nan":
        return ""
    return value


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(value)


def chunk_detection(text: object) -> list[str]:
    """Parte la detección STIX por analítica y conserva el preámbulo en cada chunk."""
    raw = _clean(text)
    if not raw:
        return []
    parts = [part.strip() for part in _ANALYTIC.split(raw) if part.strip()]
    if len(parts) <= 1:
        return [raw]
    preamble = ""
    chunks: list[str] = []
    for part in parts:
        if re.match(r"Analytic\s+\d+\s*:", part, flags=re.IGNORECASE):
            chunks.append(f"{preamble} {part}".strip() if preamble else part)
        else:
            preamble = part
    return chunks or [raw]


def build_corpus(df_cve: pd.DataFrame, df_mitre: pd.DataFrame, max_chars: int = 1800) -> pd.DataFrame:
    """Índice de producto. `df_cve` y `df_mitre` tienen que ser los parquets hardened."""
    if "likely_mitre_tactic_enriched" not in df_cve.columns or "soc_priority_score" not in df_cve.columns:
        raise ValueError("El corpus CVE no es el parquet hardened (faltan columnas de enriquecimiento).")
    if "detection_source" not in df_mitre.columns:
        raise ValueError("El corpus MITRE no es el parquet hardened (falta detection_source).")
    if float(df_mitre["detection"].isna().mean()) > 0:
        raise ValueError("detection endurecido tiene nulos: no se indexa un MITRE vacío.")

    rows: list[dict] = []
    for rec in df_cve.itertuples(index=False):
        cve_id = str(rec.cve_id)
        product = _clean(getattr(rec, "primary_vendor_product", "")) or "producto no especificado"
        cwe = _clean(getattr(rec, "cwe_primary", "")) or "CWE no indicado"
        tactic = _clean(getattr(rec, "likely_mitre_tactic_enriched", "")) or "Unknown"
        band = _clean(getattr(rec, "soc_priority_band", "")) or "P4-monitor"
        action = _clean(getattr(rec, "kev_required_action", ""))
        description = _clean(getattr(rec, "description", ""))
        kev = _as_bool(getattr(rec, "in_cisa_kev", False))
        score = float(getattr(rec, "soc_priority_score", 0.0) or 0.0)
        cvss = getattr(rec, "cvss_score", None)
        try:
            cvss_val = float(cvss) if cvss is not None and not pd.isna(cvss) else None
        except (TypeError, ValueError):
            cvss_val = None
        prefix = (
            f"{cve_id}. KEV={'si' if kev else 'no'}. banda={band}. score={score:.1f}. "
            f"CVSS={cvss_val if cvss_val is not None else 'na'}. CWE={cwe}. "
            f"tactica={tactic}. producto={product}. "
        )
        text = (prefix + description)[:max_chars]
        rows.append(
            {
                "chunk_id": f"{cve_id}#c0",
                "doc_id": cve_id,
                "doc_type": "cve",
                "chunk_index": 0,
                "text": text,
                "tactic": tactic,
                "soc_priority_band": band,
                "soc_priority_score": score,
                "in_cisa_kev": kev,
                "cvss_score": cvss_val,
                "cwe": cwe,
                "product": product,
                "kev_required_action": action,
                "technique_name": "",
                "truncated": len(prefix + description) > max_chars,
            }
        )

    for rec in df_mitre.itertuples(index=False):
        technique_id = str(rec.technique_id)
        name = _clean(getattr(rec, "technique_name", ""))
        tactic = _clean(getattr(rec, "tactic_primary", "")) or _clean(getattr(rec, "tactics", ""))
        chunks = chunk_detection(getattr(rec, "detection", "")) or [_clean(getattr(rec, "description", ""))]
        for index, chunk in enumerate(chunks):
            prefix = f"{technique_id} {name}. tactica={tactic}. "
            body = _clean(chunk)
            text = (prefix + body)[:max_chars]
            rows.append(
                {
                    "chunk_id": f"{technique_id}#c{index}",
                    "doc_id": technique_id,
                    "doc_type": "mitre",
                    "chunk_index": index,
                    "text": text,
                    "tactic": tactic or "Unknown",
                    "soc_priority_band": "",
                    "soc_priority_score": None,
                    "in_cisa_kev": False,
                    "cvss_score": None,
                    "cwe": "",
                    "product": "",
                    "kev_required_action": "",
                    "technique_name": name,
                    "truncated": len(prefix + body) > max_chars,
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def snippet(text: object, limit: int = 160) -> str:
    cleaned = strip_ids(_clean(text))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()
