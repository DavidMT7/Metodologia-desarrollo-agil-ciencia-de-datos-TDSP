# -*- coding: utf-8 -*-
"""Baseline de retrieval semántico (sentence-transformers) para Fase 3 RAG."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def build_corpus(df_cve: pd.DataFrame, df_mitre: pd.DataFrame) -> pd.DataFrame:
    cve_docs = pd.DataFrame(
        {
            "doc_id": df_cve["cve_id"].astype(str),
            "doc_type": "cve",
            "text": (
                df_cve["cve_id"].astype(str)
                + " "
                + df_cve["description"].fillna("").astype(str)
                + " tactic="
                + df_cve.get("likely_mitre_tactic_enriched", df_cve["likely_mitre_tactic"]).astype(str)
            ),
        }
    )
    mitre_docs = pd.DataFrame(
        {
            "doc_id": df_mitre["technique_id"].astype(str),
            "doc_type": "mitre",
            "text": (
                df_mitre["technique_id"].astype(str)
                + " "
                + df_mitre.get("technique_name", "").fillna("").astype(str)
                + " "
                + df_mitre["description"].fillna("").astype(str)
                + " detection="
                + df_mitre["detection"].fillna("").astype(str)
            ),
        }
    )
    mitre_docs["text"] = mitre_docs["text"].str.slice(0, 4000)
    return pd.concat([cve_docs, mitre_docs], ignore_index=True)


def run_baseline(
    corpus: pd.DataFrame,
    gold: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    from sentence_transformers import SentenceTransformer

    np.random.seed(SEED)
    model = SentenceTransformer(model_name)
    emb = model.encode(
        corpus["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    id_to_idx = {d: i for i, d in enumerate(corpus["doc_id"].tolist())}

    recalls = {k: [] for k in ks}
    mrrs: list[float] = []

    for item in gold:
        q = item["question_es"]
        relevant = item.get("relevant_doc_ids") or []
        if not relevant:
            continue
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = emb @ q_emb
        order = np.argsort(-sims)
        ranked_ids = [corpus.iloc[i]["doc_id"] for i in order]
        # MRR
        rr = 0.0
        for rank, did in enumerate(ranked_ids, start=1):
            if did in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        for k in ks:
            top = set(ranked_ids[:k])
            hit = len(top.intersection(relevant)) > 0
            recalls[k].append(float(hit))

    metrics = {
        "model": model_name,
        "n_corpus": int(len(corpus)),
        "n_queries_eval": int(len(mrrs)),
        "mrr": float(np.mean(mrrs)) if mrrs else None,
        "recall_at_k": {str(k): float(np.mean(v)) if v else None for k, v in recalls.items()},
        "seed": SEED,
    }
    return metrics


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
