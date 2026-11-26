# -*- coding: utf-8 -*-
"""Baselines NLP y un segundo encoder. El gold no elige."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from threatintel.rag.retrieve import best_parent_order


class LsaIndex:
    """LSA = SVD sobre TF-IDF (M4U5). Coseno en el espacio reducido."""

    def __init__(self, corpus: pd.DataFrame, n_components: int = 100, random_state: int = 42):
        self.corpus = corpus.reset_index(drop=True)
        self.n_components = int(n_components)
        texts = self.corpus["text"].astype(str).tolist()
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        self.matrix = normalize(self.svd.fit_transform(tfidf))

    def rank(self, query: str) -> tuple[list[str], list[float], dict[str, int]]:
        q = normalize(self.svd.transform(self.vectorizer.transform([query or "vacio"])))
        scores = np.asarray(self.matrix @ q.T).ravel()
        return best_parent_order(self.corpus["doc_id"].to_numpy(), scores)


class Doc2VecIndex:
    """Doc2Vec (gensim) sobre el corpus hardened. Baseline didáctico M4U3."""

    def __init__(
        self,
        corpus: pd.DataFrame,
        *,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        seed: int = 42,
    ):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        self.corpus = corpus.reset_index(drop=True)
        # Un documento por padre: el chunk más largo.
        parents: dict[str, str] = {}
        lengths: dict[str, int] = {}
        for doc_id, text in zip(self.corpus["doc_id"].astype(str), self.corpus["text"].astype(str)):
            if doc_id not in parents or len(text) > lengths[doc_id]:
                parents[doc_id] = text
                lengths[doc_id] = len(text)
        self.parent_ids = list(parents)
        docs = [
            TaggedDocument(words=text.lower().split(), tags=[doc_id])
            for doc_id, text in parents.items()
        ]
        self.model = Doc2Vec(
            documents=docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=1,
            epochs=epochs,
            seed=seed,
            dm=1,
        )
        matrix = np.vstack([self.model.dv[doc_id] for doc_id in self.parent_ids])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (matrix / norms).astype(np.float32)

    def rank(self, query: str) -> tuple[list[str], list[float], dict[str, int]]:
        vector = self.model.infer_vector(str(query or "vacio").lower().split())
        norm = float(np.linalg.norm(vector)) or 1.0
        vector = (vector / norm).astype(np.float32)
        scores = self.matrix @ vector
        order = np.argsort(-scores)
        ranked_ids = [self.parent_ids[int(i)] for i in order]
        ranked_scores = [float(scores[int(i)]) for i in order]
        chunk_of = {doc_id: idx for idx, doc_id in enumerate(ranked_ids)}
        return ranked_ids, ranked_scores, chunk_of
