# -*- coding: utf-8 -*-
"""Punto de entrada Fase IV: responder(pregunta). No reentrena. Sin re-ranker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from threatintel.rag.experiment import build_retrieval_for_query, load_config
from threatintel.rag.generate import answer_query
from threatintel.rag.lineage_gate import assert_phase3_lineage
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts
from threatintel.rag.run_fase3 import load_encoder
from threatintel.rag.corpus import build_corpus

# Compuerta publicada en Fase III / entrada congelada
DEPLOY_THRESHOLD = 0.40
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class ResponderService:
    root: Path
    cfg: dict
    corpus: pd.DataFrame
    known_ids: set[str]
    model: object
    dense_index: DenseIndex
    bm25: BM25Index
    threshold: float = DEPLOY_THRESHOLD

    def retrieve(self, question: str) -> dict | None:
        from threatintel.rag.generate import route_query

        if route_query(question) != "retrieval":
            return None
        vector = encode_texts(self.model, [question], batch_size=1)[0]
        dense_ids, dense_scores, dense_chunks = self.dense_index.rank(vector)
        bm25_ids, bm25_scores, bm25_chunks = self.bm25.rank(question)
        retrieval, _final = build_retrieval_for_query(
            question,
            known_ids=self.known_ids,
            corpus=self.corpus,
            dense_ids=dense_ids,
            dense_scores=dense_scores,
            dense_chunks=dense_chunks,
            bm25_ids=bm25_ids,
            bm25_scores=bm25_scores,
            bm25_chunks=bm25_chunks,
            cfg=self.cfg,
            cross_encoder=None,  # fuera de la entrega
        )
        return retrieval

    def __call__(self, pregunta: str) -> dict:
        retrieval = self.retrieve(pregunta)
        return answer_query(pregunta, retrieval, threshold=self.threshold)


_SERVICE: ResponderService | None = None


def load_responder(root: Path | None = None, *, force: bool = False) -> ResponderService:
    global _SERVICE
    if _SERVICE is not None and not force:
        return _SERVICE

    root, cfg = load_config(root)
    print("responder: linaje + corpus…", flush=True)
    cve_clean = pd.read_parquet(root / cfg["paths"]["cve_clean"])
    mitre_clean = pd.read_parquet(root / cfg["paths"]["mitre_clean"])
    cve = pd.read_parquet(root / cfg["paths"]["cve_parquet"])
    mitre = pd.read_parquet(root / cfg["paths"]["mitre_parquet"])
    assert_phase3_lineage(cve_clean, cve, mitre_clean, mitre)
    corpus = build_corpus(cve, mitre, max_chars=int(cfg["max_chars"]))
    if corpus.loc[corpus["doc_type"] == "mitre", "text"].str.contains("Strategy:", regex=False).mean() < 0.5:
        raise RuntimeError("Índice MITRE sin detection endurecido.")

    model_path = root / cfg["paths"]["best_model"]
    print(f"responder: carga {model_path} en GPU si hay CUDA…", flush=True)
    model = load_encoder(str(model_path), int(cfg["max_seq_length"]))
    batch = int(cfg["encode_batch_size"])
    print(f"responder: encode índice (batch={batch})…", flush=True)
    embeddings = encode_texts(model, corpus["text"].astype(str).tolist(), batch_size=batch)
    dense_index = DenseIndex(corpus, embeddings, "biencoder_best")
    bm25 = BM25Index(corpus, k1=BM25_K1, b=BM25_B)
    _SERVICE = ResponderService(
        root=root,
        cfg=cfg,
        corpus=corpus.reset_index(drop=True),
        known_ids=set(corpus["doc_id"].astype(str)),
        model=model,
        dense_index=dense_index,
        bm25=bm25,
        threshold=DEPLOY_THRESHOLD,
    )
    print("responder: listo (RRF k=60, sin re-ranker, umbral 0,40)", flush=True)
    return _SERVICE


def responder(pregunta: str) -> dict:
    """API pública Fase IV."""
    return load_responder()(pregunta)
