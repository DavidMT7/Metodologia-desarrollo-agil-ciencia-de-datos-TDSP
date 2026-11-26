# -*- coding: utf-8 -*-
"""Estadísticos de retrieval. No eligen el checkpoint."""
from __future__ import annotations

import re

import numpy as np

from threatintel.eval.metrics import ranking_metrics
from threatintel.rag.gate import should_abstain

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def aligned_reciprocal_ranks(items: list[dict], rankings: list[list[str]]) -> list[float]:
    values: list[float] = []
    for item, ranked in zip(items, rankings):
        relevant = {str(doc_id) for doc_id in item.get("relevant_doc_ids") or []}
        if not relevant:
            continue
        values.append(reciprocal_rank([str(doc_id) for doc_id in ranked], relevant))
    return values


def hit_ranks(items: list[dict], rankings: list[list[str]]) -> list[int | None]:
    ranks: list[int | None] = []
    for item, ranked in zip(items, rankings):
        relevant = {str(doc_id) for doc_id in item.get("relevant_doc_ids") or []}
        found = None
        for rank, doc_id in enumerate(ranked, start=1):
            if str(doc_id) in relevant:
                found = rank
                break
        ranks.append(found)
    return ranks


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "low": None, "high": None, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.sort(arr[draws].mean(axis=1))
    lo = int(np.floor((alpha / 2) * n_boot))
    hi = int(np.ceil((1 - alpha / 2) * n_boot)) - 1
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "low": float(means[lo]),
        "high": float(means[hi]),
        "n_boot": int(n_boot),
    }


def bootstrap_paired_delta(
    baseline: list[float],
    challenger: list[float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    base = np.asarray(baseline, dtype=float)
    other = np.asarray(challenger, dtype=float)
    if len(base) != len(other) or len(base) == 0:
        raise ValueError("Las dos listas de RR tienen que estar alineadas y no vacías")
    delta = other - base
    ci = bootstrap_ci(delta.tolist(), n_boot=n_boot, seed=seed, alpha=alpha)
    ci["excludes_zero"] = bool(ci["low"] > 0 or ci["high"] < 0)
    return ci


def group_metrics(items: list[dict], rankings: list[list[str]], key: str = "family") -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for item, ranked in zip(items, rankings):
        label = str(item.get(key) or "sin_grupo")
        buckets.setdefault(label, []).append(
            {"relevant": [str(doc_id) for doc_id in item.get("relevant_doc_ids") or []], "ranked": [str(doc_id) for doc_id in ranked]}
        )
    return {label: ranking_metrics(rows) for label, rows in sorted(buckets.items())}


def recall_curve(items: list[dict], rankings: list[list[str]], max_k: int = 10) -> dict:
    packed = [
        {"relevant": [str(doc_id) for doc_id in item.get("relevant_doc_ids") or []], "ranked": [str(doc_id) for doc_id in ranked]}
        for item, ranked in zip(items, rankings)
    ]
    metrics = ranking_metrics(packed, ks=tuple(range(1, max_k + 1)))
    return {"k": list(range(1, max_k + 1)), "recall": [metrics["recall_at_k"][str(k)] for k in range(1, max_k + 1)], "n": metrics["n"]}


def jaccard_tokens(left: str, right: str) -> float:
    a = set(_TOKEN.findall((left or "").lower()))
    b = set(_TOKEN.findall((right or "").lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_jaccard(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return float(np.mean([jaccard_tokens(left, right) for left, right in pairs]))


def threshold_curve(rows: list[dict], grid: list[float] | None = None) -> list[dict]:
    candidates = grid or [round(i / 100, 2) for i in range(5, 81, 5)]
    curve = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for row in rows:
            pred = should_abstain(row["score"], threshold, row["explicit_ids_missing"])
            expect = bool(row["expect_abstain"])
            if pred and expect:
                tp += 1
            elif pred and not expect:
                fp += 1
            elif (not pred) and expect:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        curve.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return curve
