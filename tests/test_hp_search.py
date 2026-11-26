# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatintel.rag.hp_search import choose_deployment
from threatintel.rag.retrieve import BM25Index
from threatintel.rag.train import warmup_step_count


def test_warmup_zero_is_really_zero():
    assert warmup_step_count(4000, 32, 1, 0.0) == 0
    assert warmup_step_count(4000, 32, 1, 0.1) == 12


def test_bm25_k1_and_b_rank_a_tiny_corpus():
    corpus = pd.DataFrame({"doc_id": ["A", "B"], "text": ["alpha beta", "gamma delta"]})
    ranking, scores, _chunks = BM25Index(corpus, k1=0.9, b=0.4).rank("alpha beta")
    assert ranking[0] == "A"
    assert scores[0] >= scores[1]


def test_deployment_keeps_the_published_checkpoint():
    rows = [
        {"tag": "batch_16", "epochs": 1, "dev_mrr_without_id": 0.70},
        {"tag": "ep3", "epochs": 3, "dev_mrr_without_id": 0.70},
    ]
    decision = choose_deployment(0.688, rows, margin=0.01)
    assert decision["overwrite_published"] is False
    assert decision["challenger_tag"] == "ep3"
    assert decision["best_one_epoch_tag"] == "batch_16"

    quiet = choose_deployment(0.688, [{"tag": "wd", "epochs": 1, "dev_mrr_without_id": 0.66}], margin=0.01)
    assert quiet["challenger_tag"] is None
