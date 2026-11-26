# -*- coding: utf-8 -*-
"""Búsqueda densa, BM25, fusión RRF y re-ranking."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from threatintel.rag.ids import extract_ids

_TOKEN = re.compile(r"cve-\d{4}-\d{4,}|t\d{4}(?:\.\d{3})?|[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    tokens = [tok.lower() for tok in _TOKEN.findall(text or "")]
    return tokens or ["vacio"]


def rrf_fuse(rankings: list[list[str]], k: int = 60, top_n: int = 100) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[:top_n]


def best_parent_order(doc_ids: np.ndarray, scores: np.ndarray) -> tuple[list[str], list[float], dict[str, int]]:
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranked_ids: list[str] = []
    ranked_scores: list[float] = []
    chunk_of: dict[str, int] = {}
    for position in order:
        idx = int(position)
        doc_id = str(doc_ids[idx])
        if doc_id in chunk_of:
            continue
        chunk_of[doc_id] = idx
        ranked_ids.append(doc_id)
        ranked_scores.append(float(scores[idx]))
    return ranked_ids, ranked_scores, chunk_of


def hydrate_parents(corpus: pd.DataFrame, doc_ids: list[str], chunk_of: dict[str, int], scores: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    for doc_id in doc_ids:
        rec = corpus.iloc[chunk_of[doc_id]]
        score_soc = rec["soc_priority_score"]
        cvss = rec["cvss_score"]
        rows.append(
            {
                "chunk_id": rec["chunk_id"],
                "doc_id": doc_id,
                "doc_type": rec["doc_type"],
                "text": rec["text"],
                "tactic": rec["tactic"],
                "soc_priority_band": rec["soc_priority_band"] or "",
                "soc_priority_score": None if pd.isna(score_soc) else float(score_soc),
                "in_cisa_kev": bool(rec["in_cisa_kev"]),
                "cvss_score": None if pd.isna(cvss) else float(cvss),
                "cwe": rec["cwe"] or "",
                "product": rec["product"] or "",
                "kev_required_action": rec["kev_required_action"] or "",
                "technique_name": rec["technique_name"] or "",
                "score": float(scores.get(doc_id, 0.0)),
            }
        )
    return rows


class BM25Index:
    def __init__(self, corpus: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus.reset_index(drop=True)
        self.k1 = float(k1)
        self.b = float(b)
        self._bm25 = BM25Okapi(
            [tokenize(text) for text in self.corpus["text"].tolist()],
            k1=self.k1,
            b=self.b,
        )

    def rank(self, query: str) -> tuple[list[str], list[float], dict[str, int]]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        return best_parent_order(self.corpus["doc_id"].to_numpy(), scores)


class DenseIndex:
    def __init__(self, corpus: pd.DataFrame, embeddings: np.ndarray, model_name: str):
        self.corpus = corpus.reset_index(drop=True)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.model_name = model_name
        if len(self.embeddings) != len(self.corpus):
            raise ValueError("Embeddings y corpus tienen distinto largo")

    def rank(self, query_embedding: np.ndarray) -> tuple[list[str], list[float], dict[str, int]]:
        scores = self.embeddings @ np.asarray(query_embedding, dtype=np.float32)
        return best_parent_order(self.corpus["doc_id"].to_numpy(), scores)


def explicit_id_status(query: str, known_ids: set[str]) -> tuple[list[str], bool]:
    ids = extract_ids(query)
    if not ids:
        return [], False
    return ids, all(doc_id not in known_ids for doc_id in ids)


def encode_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def rerank_parents(cross_encoder, query: str, parents: list[dict], top_n: int = 30) -> list[dict]:
    pool = parents[:top_n]
    if not pool:
        return []
    pairs = [(query, str(row["text"])[:1500]) for row in pool]
    scores = np.asarray(cross_encoder.predict(pairs, show_progress_bar=False), dtype=float)
    order = np.argsort(-scores)
    ranked = []
    for position in order:
        row = dict(pool[int(position)])
        row["rerank_score"] = float(scores[int(position)])
        ranked.append(row)
    ranked.extend(parents[top_n:])
    return ranked


def ids_of(parents: list[dict]) -> list[str]:
    return [str(row["doc_id"]) for row in parents]
