# -*- coding: utf-8 -*-
"""Métricas de retrieval con la misma definición que el baseline de Fase II."""
from __future__ import annotations

import numpy as np


def ranking_metrics(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    mrrs: list[float] = []
    recalls = {k: [] for k in ks}
    for row in rows:
        relevant = set(row.get("relevant") or [])
        if not relevant:
            continue
        ranked = list(row.get("ranked") or [])
        reciprocal = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                reciprocal = 1.0 / rank
                break
        mrrs.append(reciprocal)
        for k in ks:
            recalls[k].append(float(any(doc_id in relevant for doc_id in ranked[:k])))
    return {
        "n": int(len(mrrs)),
        "mrr": float(np.mean(mrrs)) if mrrs else None,
        "recall_at_k": {str(k): (float(np.mean(v)) if v else None) for k, v in recalls.items()},
    }
