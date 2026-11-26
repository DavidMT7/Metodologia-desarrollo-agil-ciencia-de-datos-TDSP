# -*- coding: utf-8 -*-
"""Modelos léxicos. La selección mira el desarrollo sin el extracto pegado."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from threatintel.rag.retrieve import best_parent_order

_EXCERPT = re.compile(r"«[^»]*»")
_SPACES = re.compile(r"\s+")

BM25_K1 = (0.6, 0.9, 1.2, 1.5, 2.0)
BM25_B = (0.3, 0.5, 0.75, 0.9)
WORD_MIN_DF = (1, 2)
WORD_SUBLINEAR = (False, True)
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
NEURAL_FAMILIES = {"biencoder", "rrf"}
CLASSICAL_FAMILIES = {"bm25", "tfidf", "linear"}


def strip_excerpt(query: str) -> str:
    """Quita el fragmento oficial pegado entre comillas angulares."""
    text = _EXCERPT.sub(" ", query or "")
    return _SPACES.sub(" ", text).strip()


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fuse_score_maps(left: dict[str, float], right: dict[str, float], weight: float) -> list[str]:
    """`weight` va al mapa izquierdo. El resto, al derecho. Ambos se normalizan por consulta."""
    keys = list(left)
    if not keys:
        return []
    combined = weight * minmax(np.array([left[key] for key in keys])) + (1.0 - weight) * minmax(
        np.array([right.get(key, 0.0) for key in keys])
    )
    order = np.argsort(-combined, kind="mergesort")
    return [keys[int(pos)] for pos in order]


class TfidfIndex:
    def __init__(
        self,
        corpus: pd.DataFrame,
        *,
        analyzer: str,
        ngram_range: tuple[int, int],
        min_df: int,
        sublinear_tf: bool,
    ):
        self.corpus = corpus.reset_index(drop=True)
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = int(min_df)
        self.sublinear_tf = bool(sublinear_tf)
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
        )
        self.matrix = normalize(self.vectorizer.fit_transform(self.corpus["text"].astype(str).tolist()))

    def scores(self, query: str) -> np.ndarray:
        query_vec = normalize(self.vectorizer.transform([query or "vacio"]))
        return np.asarray((self.matrix @ query_vec.T).toarray()).ravel()

    def rank(self, query: str) -> tuple[list[str], list[float], dict[str, int]]:
        return best_parent_order(self.corpus["doc_id"].to_numpy(), self.scores(query))


def choose_on_dev(rows: list[dict], margin: float = 0.02) -> dict:
    """Elige con el MRR de desarrollo. Rechaza cualquier fila que no sea de desarrollo."""
    if not rows:
        raise ValueError("La selección no tiene filas de desarrollo")
    for row in rows:
        split = str(row.get("split", ""))
        if split != "dev":
            raise ValueError(f"La selección solo acepta desarrollo, llegó {split!r}")
    neural = [row for row in rows if row["family"] in NEURAL_FAMILIES]
    classical = [row for row in rows if row["family"] in CLASSICAL_FAMILIES]
    if not neural or not classical:
        raise ValueError("Faltan el bi-encoder, el RRF o un modelo léxico")
    best_neural = max(neural, key=lambda row: float(row["mrr"]))
    best_classical = max(classical, key=lambda row: float(row["mrr"]))
    gap = float(best_classical["mrr"]) - float(best_neural["mrr"])
    classical_wins = gap > margin
    if classical_wins:
        reason = (
            f"{best_classical['system']} supera por {gap:.3f} a {best_neural['system']} "
            f"en el desarrollo sin extracto. La Fase IV responde con ese modelo léxico. "
            "biencoder_best no se borra."
        )
        deploy = "classical"
        winner = best_classical
    else:
        reason = (
            f"El mejor léxico ({best_classical['system']}, MRR {float(best_classical['mrr']):.3f}) "
            f"no supera por {margin:.2f} a {best_neural['system']} "
            f"(MRR {float(best_neural['mrr']):.3f}) en el desarrollo sin extracto. "
            "La Fase IV sigue con el bi-encoder de dos épocas y la fusión RRF."
        )
        deploy = "biencoder_rrf"
        winner = best_neural
    return {
        "deploy": deploy,
        "classical_wins": classical_wins,
        "winner_system": winner["system"],
        "winner_family": winner["family"],
        "winner_mrr": float(winner["mrr"]),
        "best_neural_system": best_neural["system"],
        "best_neural_mrr": float(best_neural["mrr"]),
        "best_classical_system": best_classical["system"],
        "best_classical_mrr": float(best_classical["mrr"]),
        "gap": gap,
        "margin": margin,
        "reason": reason,
    }
