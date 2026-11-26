# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatintel.rag.diagnostics import (
    aligned_reciprocal_ranks,
    bootstrap_ci,
    bootstrap_paired_delta,
    group_metrics,
    jaccard_tokens,
    threshold_curve,
)


def test_reciprocal_rank_and_groups():
    items = [
        {"family": "kev_priority", "relevant_doc_ids": ["A"]},
        {"family": "kev_priority", "relevant_doc_ids": ["B"]},
        {"family": "mitre_detection", "relevant_doc_ids": ["T"]},
    ]
    rankings = [["A", "Z"], ["Z", "B"], ["T"]]
    ranks = aligned_reciprocal_ranks(items, rankings)
    assert ranks == [1.0, 0.5, 1.0]
    grouped = group_metrics(items, rankings)
    assert grouped["kev_priority"]["n"] == 2
    assert grouped["mitre_detection"]["mrr"] == 1.0


def test_bootstrap_delta_excludes_zero_when_the_shift_is_constant():
    base = [0.2] * 30
    other = [0.8] * 30
    delta = bootstrap_paired_delta(base, other, n_boot=400, seed=1)
    assert delta["excludes_zero"]
    assert delta["low"] > 0
    same = bootstrap_ci([0.5] * 20, n_boot=200, seed=1)
    assert same["low"] == same["high"] == 0.5


def test_jaccard_ignores_case_and_punctuation():
    score = jaccard_tokens("Adobe Reader, CVE", "adobe reader")
    assert score > 0.5


def test_threshold_curve_peaks_where_classes_separate():
    rows = [
        {"score": 0.1, "explicit_ids_missing": False, "expect_abstain": True},
        {"score": 0.9, "explicit_ids_missing": False, "expect_abstain": False},
    ]
    curve = threshold_curve(rows, grid=[0.05, 0.5, 0.95])
    best = max(curve, key=lambda row: row["f1"])
    assert best["threshold"] == 0.5
    assert best["f1"] == 1.0
