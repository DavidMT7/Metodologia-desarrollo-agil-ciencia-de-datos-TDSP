# -*- coding: utf-8 -*-
"""Identificadores CTI canónicos para indexación y filtro de citas."""
from __future__ import annotations

import re

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)


def extract_ids(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for rx in (CVE_RE, TECH_RE):
        for match in rx.finditer(text or ""):
            value = match.group(0).upper()
            if value not in seen:
                seen.add(value)
                found.append(value)
    return found


def strip_ids(text: str) -> str:
    cleaned = CVE_RE.sub(" ", text or "")
    cleaned = TECH_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
