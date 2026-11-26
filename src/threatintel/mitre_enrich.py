# -*- coding: utf-8 -*-
"""Enriquecimiento de MITRE ATT&CK desde STIX oficial (enterprise-attack)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


def download_enterprise_attack(cache_path: Path, force: bool = False) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force and cache_path.stat().st_size > 1_000_000:
        return cache_path
    r = requests.get(ATTACK_URL, timeout=180)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return cache_path


def _external_id(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return str(ref["external_id"])
    return None


def parse_attack_techniques(stix_path: Path) -> pd.DataFrame:
    """Parsea técnicas y reconstruye texto de detección vía detection-strategy + analytics.

    El STIX moderno ya no incluye `x_mitre_detection` en `attack-pattern`.
    """
    data = json.loads(stix_path.read_text(encoding="utf-8"))
    objects = data.get("objects", [])

    by_id = {o["id"]: o for o in objects if "id" in o}
    attack_patterns = [
        o
        for o in objects
        if o.get("type") == "attack-pattern"
        and not o.get("revoked")
        and not o.get("x_mitre_deprecated")
    ]
    analytics = {
        o["id"]: o
        for o in objects
        if o.get("type") == "x-mitre-analytic" and not o.get("x_mitre_deprecated")
    }
    strategies = {
        o["id"]: o
        for o in objects
        if o.get("type") == "x-mitre-detection-strategy" and not o.get("x_mitre_deprecated")
    }

    # strategy_id -> list[attack-pattern id]
    detects_map: dict[str, list[str]] = {}
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "detects":
            continue
        src, tgt = rel.get("source_ref"), rel.get("target_ref")
        if src in strategies and tgt in by_id and by_id[tgt].get("type") == "attack-pattern":
            detects_map.setdefault(tgt, []).append(src)

    # mitigations course-of-action related via mitigates
    mitigates_map: dict[str, list[str]] = {}
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "mitigates":
            continue
        src, tgt = rel.get("source_ref"), rel.get("target_ref")
        if tgt in by_id and by_id[tgt].get("type") == "attack-pattern":
            mitigates_map.setdefault(tgt, []).append(src)

    rows = []
    for obj in attack_patterns:
        tid = _external_id(obj)
        if not tid or not re.match(r"^T\d+", tid):
            continue

        det_parts: list[str] = []
        # legacy field if somehow present
        legacy = (obj.get("x_mitre_detection") or "").strip()
        if legacy:
            det_parts.append(legacy)

        for sid in detects_map.get(obj["id"], []):
            strat = strategies.get(sid) or {}
            sname = (strat.get("name") or "").strip()
            if sname:
                det_parts.append(f"Strategy: {sname}")
            for aref in strat.get("x_mitre_analytic_refs") or []:
                an = analytics.get(aref) or {}
                adesc = (an.get("description") or "").strip()
                aname = (an.get("name") or "").strip()
                if adesc:
                    det_parts.append(f"{aname}: {adesc}" if aname else adesc)
                elif aname:
                    det_parts.append(aname)

        # fallback: data components / sources names on technique (older field)
        for ds in obj.get("x_mitre_data_sources") or []:
            det_parts.append(f"Data source: {ds}" if isinstance(ds, str) else f"Data source: {ds}")

        detection = "\n\n".join(dict.fromkeys([p for p in det_parts if p])).strip() or None

        platforms = obj.get("x_mitre_platforms") or []
        tactics = []
        for phase in obj.get("kill_chain_phases") or []:
            if phase.get("kill_chain_name") == "mitre-attack":
                tactics.append(str(phase.get("phase_name", "")).replace("-", " ").title())

        mit_names = []
        for mid in mitigates_map.get(obj["id"], [])[:5]:
            mobj = by_id.get(mid) or {}
            if mobj.get("name"):
                mit_names.append(mobj["name"])

        rows.append(
            {
                "technique_id": tid,
                "official_name": obj.get("name"),
                "official_description": obj.get("description"),
                "official_detection": detection,
                "official_platforms": "|".join(platforms) if platforms else None,
                "official_tactics": "|".join(tactics) if tactics else None,
                "official_mitigations": "|".join(mit_names) if mit_names else None,
            }
        )
    return pd.DataFrame(rows)


def enrich_mitre_frame(df_mitre: pd.DataFrame, attack_df: pd.DataFrame) -> pd.DataFrame:
    out = df_mitre.copy()
    # quitar columnas de un enrich previo para evitar sufijos _x/_y
    drop_cols = [
        c
        for c in out.columns
        if c.startswith("official_")
        or c in {"detection_enriched", "detection_source"}
    ]
    out = out.drop(columns=drop_cols, errors="ignore")
    # reset detection to original empties if previously filled from bad enrich
    if "detection" in out.columns and out["detection"].notna().any():
        # keep as-is; merge will fill remaining nulls
        pass
    base = attack_df.drop_duplicates("technique_id")
    out = out.merge(base, on="technique_id", how="left", suffixes=("", "_attack"))

    missing = out["official_detection"].isna()
    if missing.any():
        parents = out.loc[missing, "technique_id"].astype(str).str.split(".").str[0]
        parent_map = base.set_index("technique_id")["official_detection"].to_dict()
        out.loc[missing, "official_detection"] = parents.map(parent_map)

    det = out["detection"]
    empty = det.isna() | (det.astype(str).str.strip().isin(["", "nan", "None", "null"]))
    out.loc[empty, "detection"] = out.loc[empty, "official_detection"]
    out["detection_enriched"] = empty & out["detection"].notna()
    out["detection_source"] = "dataset"
    out.loc[empty & out["detection"].notna(), "detection_source"] = "mitre_official_stix"
    out.loc[out["detection"].isna(), "detection_source"] = "unavailable"
    return out
