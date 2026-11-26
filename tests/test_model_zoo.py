# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatintel.rag.model_zoo import Doc2VecIndex, LsaIndex


def test_lsa_ranks_matching_document_first():
    corpus = pd.DataFrame(
        {
            "doc_id": ["A", "B", "C"],
            "text": [
                "remote code execution vulnerability in network manager",
                "cooking recipes and pasta sauce",
                "travel guide to mountain hiking",
            ],
        }
    )
    index = LsaIndex(corpus, n_components=2, random_state=42)
    ranking, _scores, _ = index.rank("remote code execution vulnerability")
    assert ranking[0] == "A"


def test_doc2vec_builds_and_ranks():
    corpus = pd.DataFrame(
        {
            "doc_id": ["A", "A", "B"],
            "text": ["alpha beta gamma vulnerability", "alpha beta longer text", "unrelated kitchen pasta"],
        }
    )
    index = Doc2VecIndex(corpus, vector_size=20, epochs=5, min_count=1, seed=42)
    ranking, scores, _ = index.rank("alpha beta vulnerability")
    assert "A" in ranking
    assert len(ranking) == 2
    assert scores[0] >= scores[-1]
