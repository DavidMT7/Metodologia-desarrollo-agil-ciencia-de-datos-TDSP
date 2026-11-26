# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatintel.rag.classical import TfidfIndex, choose_on_dev, fuse_score_maps, strip_excerpt


def test_strip_excerpt_removes_the_pasted_english():
    query = (
        "¿Qué banda corresponde a esta vulnerabilidad en cisco, asociada a CWE-20: "
        "«Cisco Prime allows remote attackers to execute code»?"
    )
    cleaned = strip_excerpt(query)
    assert "Cisco Prime" not in cleaned
    assert "cisco" in cleaned
    assert "CWE-20" in cleaned
    assert strip_excerpt("pregunta sin extracto") == "pregunta sin extracto"


def test_tfidf_ranks_the_matching_document_first():
    corpus = pd.DataFrame(
        {
            "doc_id": ["A", "B"],
            "text": ["alpha beta vulnerability", "gamma delta protocol"],
        }
    )
    index = TfidfIndex(corpus, analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    ranking, _scores, _chunks = index.rank("alpha beta")
    assert ranking[0] == "A"


def test_linear_fusion_respects_the_weight():
    left = {"A": 1.0, "B": 0.0}
    right = {"A": 0.0, "B": 1.0}
    assert fuse_score_maps(left, right, 1.0)[0] == "A"
    assert fuse_score_maps(left, right, 0.0)[0] == "B"


def test_choose_on_dev_rejects_gold_rows():
    rows = [
        {"split": "dev", "family": "biencoder", "system": "biencoder_ft", "mrr": 0.50},
        {"split": "gold", "family": "bm25", "system": "bm25", "mrr": 0.90},
    ]
    try:
        choose_on_dev(rows)
    except ValueError as exc:
        assert "desarrollo" in str(exc)
    else:
        raise AssertionError("una fila de gold no puede elegir el modelo")


def test_choose_on_dev_keeps_the_neural_system_inside_the_margin():
    rows = [
        {"split": "dev", "family": "biencoder", "system": "biencoder_ft", "mrr": 0.50},
        {"split": "dev", "family": "rrf", "system": "hybrid_rrf", "mrr": 0.55},
        {"split": "dev", "family": "bm25", "system": "bm25", "mrr": 0.56},
        {"split": "dev", "family": "tfidf", "system": "tfidf_word", "mrr": 0.40},
        {"split": "dev", "family": "linear", "system": "linear", "mrr": 0.56},
    ]
    decision = choose_on_dev(rows, margin=0.02)
    assert decision["deploy"] == "biencoder_rrf"
    assert decision["classical_wins"] is False
    assert decision["best_neural_system"] == "hybrid_rrf"


def test_choose_on_dev_switches_when_the_lexical_model_wins_by_more_than_the_margin():
    rows = [
        {"split": "dev", "family": "biencoder", "system": "biencoder_ft", "mrr": 0.40},
        {"split": "dev", "family": "rrf", "system": "hybrid_rrf", "mrr": 0.50},
        {"split": "dev", "family": "tfidf", "system": "tfidf_char", "mrr": 0.53},
    ]
    decision = choose_on_dev(rows, margin=0.02)
    assert decision["deploy"] == "classical"
    assert decision["winner_system"] == "tfidf_char"
    assert decision["gap"] > 0.02
