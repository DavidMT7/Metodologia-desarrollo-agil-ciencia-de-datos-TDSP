# -*- coding: utf-8 -*-
"""Compuerta de abstención: identificador ausente o similitud bajo el umbral."""
from __future__ import annotations


def should_abstain(dense_top_score: float, threshold: float, explicit_ids_missing: bool) -> bool:
    if explicit_ids_missing:
        return True
    return float(dense_top_score) < float(threshold)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def calibrate_threshold(rows: list[dict], grid: list[float] | None = None) -> dict:
    candidates = grid or [round(i / 100, 2) for i in range(5, 81, 5)]
    best: dict | None = None
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for row in rows:
            pred = should_abstain(row["score"], threshold, row["explicit_ids_missing"])
            expect = bool(row["expect_abstain"])
            if pred and expect:
                tp += 1
            elif pred and not expect:
                fp += 1
            elif not pred and expect:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _f1(precision, recall)
        record = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        if best is None or f1 > best["f1"] + 1e-12 or (
            abs(f1 - best["f1"]) <= 1e-12 and threshold < best["threshold"]
        ):
            best = record
    if best is None:
        raise ValueError("No hay filas para calibrar el umbral")
    return best
