# -*- coding: utf-8 -*-
"""Splits de entrenamiento, dev y paráfrasis. El gold no entra a los pares."""
from __future__ import annotations

import numpy as np
import pandas as pd

from threatintel.rag.corpus import snippet
from threatintel.rag.ids import extract_ids, strip_ids

FACTUAL_FAMILIES = {"kev_priority", "cve_tactic", "mitre_detection"}


def gold_doc_ids(gold: list[dict]) -> set[str]:
    banned: set[str] = set()
    for item in gold:
        for doc_id in item.get("relevant_doc_ids") or []:
            banned.add(str(doc_id))
    return banned


def assert_no_leakage(train_doc_ids: set[str], banned: set[str]) -> None:
    leaked = sorted(train_doc_ids & banned)
    if leaked:
        raise RuntimeError(f"Fuga de gold hacia entrenamiento: {leaked[:8]}")


def _product(row: pd.Series) -> str:
    return snippet(row.get("primary_vendor_product"), 80) or "el producto afectado"


def _cwe(row: pd.Series) -> str:
    return snippet(row.get("cwe_primary"), 40) or "CWE no indicado"


def cve_query_without_id(row: pd.Series) -> str:
    text = (
        f"¿Qué banda de prioridad SOC y estado CISA KEV corresponden a esta vulnerabilidad "
        f"en {_product(row)}, asociada a {_cwe(row)}: «{snippet(row.get('description'), 160)}»?"
    )
    return strip_ids(text)


def cve_query_with_id(row: pd.Series) -> str:
    return (
        f"¿La vulnerabilidad {row['cve_id']} está en CISA KEV y qué prioridad SOC tiene? "
        "Resume el riesgo en una frase."
    )


def mitre_query_without_id(row: pd.Series) -> str:
    name = strip_ids(str(row.get("technique_name") or "técnica sin nombre"))
    return (
        f"¿Cómo detectar la actividad «{name}»? Resume la estrategia y al menos una analítica, "
        "y cita el identificador ATT&CK."
    )


def mitre_query_with_id(row: pd.Series) -> str:
    return (
        f"¿Cómo detectar la técnica {row['technique_id']} ({row.get('technique_name', '')})? "
        "Cita fuentes de datos si aplica."
    )


def _positive_text(corpus: pd.DataFrame) -> dict[str, str]:
    texts: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for rec in corpus.itertuples(index=False):
        length = len(str(rec.text))
        if rec.doc_id not in texts or length > lengths[rec.doc_id]:
            texts[rec.doc_id] = str(rec.text)
            lengths[rec.doc_id] = length
    return texts


def _sample_ids(ids: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if n <= 0 or not ids:
        return []
    take = min(n, len(ids))
    picked = rng.choice(np.array(ids, dtype=object), size=take, replace=False)
    return [str(x) for x in picked]


def prepare_splits(
    corpus: pd.DataFrame,
    df_cve: pd.DataFrame,
    df_mitre: pd.DataFrame,
    gold: list[dict],
    *,
    seed: int = 42,
    n_dev_cve: int = 40,
    n_dev_mitre: int = 20,
    n_dev_unknown: int = 15,
    n_dev_nonsense: int = 10,
    max_train_pairs: int = 4000,
) -> dict:
    rng = np.random.default_rng(seed)
    banned = gold_doc_ids(gold)
    known_ids = set(corpus["doc_id"].astype(str))
    positives = _positive_text(corpus)
    cve = df_cve.copy()
    cve["cve_id"] = cve["cve_id"].astype(str)
    mitre = df_mitre.copy()
    mitre["technique_id"] = mitre["technique_id"].astype(str)
    cve_pool = [i for i in cve["cve_id"].tolist() if i not in banned and i in known_ids]
    mitre_pool = [i for i in mitre["technique_id"].tolist() if i not in banned and i in known_ids]
    dev_cve = _sample_ids(cve_pool, n_dev_cve, rng)
    dev_mitre = _sample_ids(mitre_pool, n_dev_mitre, rng)
    dev_ids = set(dev_cve) | set(dev_mitre)
    train_cve = [i for i in cve_pool if i not in dev_ids]
    train_mitre = [i for i in mitre_pool if i not in dev_ids]
    assert_no_leakage(set(train_cve) | set(train_mitre), banned)
    assert_no_leakage(dev_ids, banned)
    cve_by_id = cve.set_index("cve_id", drop=False)
    mitre_by_id = mitre.set_index("technique_id", drop=False)

    dev_items: list[dict] = []
    for index, doc_id in enumerate(dev_cve):
        row = cve_by_id.loc[doc_id]
        with_id = index % 2 == 0
        question = cve_query_with_id(row) if with_id else cve_query_without_id(row)
        dev_items.append(
            {
                "id": f"D-CVE-{index:02d}",
                "family": "dev_cve",
                "question_es": question,
                "relevant_doc_ids": [doc_id],
                "with_id": with_id,
                "expect_abstain": False,
                "slice": "dev",
            }
        )
    for index, doc_id in enumerate(dev_mitre):
        row = mitre_by_id.loc[doc_id]
        with_id = index % 2 == 0
        question = mitre_query_with_id(row) if with_id else mitre_query_without_id(row)
        dev_items.append(
            {
                "id": f"D-MITRE-{index:02d}",
                "family": "dev_mitre",
                "question_es": question,
                "relevant_doc_ids": [doc_id],
                "with_id": with_id,
                "expect_abstain": False,
                "slice": "dev",
            }
        )
    unknown_items: list[dict] = []
    cursor = 1
    while len(unknown_items) < n_dev_unknown:
        fake = f"CVE-2099-{cursor:04d}"
        cursor += 1
        if fake in known_ids:
            continue
        unknown_items.append(
            {
                "id": f"D-UNK-{len(unknown_items):02d}",
                "family": "abstain",
                "question_es": f"¿Qué CVSS, estado KEV y prioridad SOC tiene {fake}?",
                "relevant_doc_ids": [],
                "with_id": True,
                "expect_abstain": True,
                "slice": "dev_abstain",
            }
        )
    for index in range(n_dev_nonsense):
        unknown_items.append(
            {
                "id": f"D-NON-{index:02d}",
                "family": "abstain_semantic",
                "question_es": (
                    "¿Qué prioridad SOC tiene el incidente del unicornio púrpura "
                    f"número {index} dentro del refrigerador de la oficina?"
                ),
                "relevant_doc_ids": [],
                "with_id": False,
                "expect_abstain": True,
                "slice": "dev_nonsense",
            }
        )
    dev_items.extend(unknown_items)

    pair_rows: list[dict] = []
    n_cve_sem = int(max_train_pairs * 0.62)
    n_cve_id = int(max_train_pairs * 0.18)
    n_mitre_sem = int(max_train_pairs * 0.14)
    n_mitre_id = max_train_pairs - n_cve_sem - n_cve_id - n_mitre_sem

    def _add_cve(doc_ids: list[str], with_id: bool) -> None:
        for doc_id in doc_ids:
            row = cve_by_id.loc[doc_id]
            query = cve_query_with_id(row) if with_id else cve_query_without_id(row)
            if not with_id and str(doc_id) in query:
                continue
            pair_rows.append(
                {
                    "query": query,
                    "positive_text": positives[doc_id],
                    "positive_doc_id": doc_id,
                    "with_id": with_id,
                }
            )

    def _add_mitre(doc_ids: list[str], with_id: bool) -> None:
        for doc_id in doc_ids:
            row = mitre_by_id.loc[doc_id]
            query = mitre_query_with_id(row) if with_id else mitre_query_without_id(row)
            if not with_id and str(doc_id).upper() in query.upper():
                continue
            pair_rows.append(
                {
                    "query": query,
                    "positive_text": positives[doc_id],
                    "positive_doc_id": doc_id,
                    "with_id": with_id,
                }
            )

    _add_cve(_sample_ids(train_cve, n_cve_sem, rng), False)
    _add_cve(_sample_ids(train_cve, n_cve_id, rng), True)
    _add_mitre(_sample_ids(train_mitre, n_mitre_sem, rng), False)
    _add_mitre(_sample_ids(train_mitre, n_mitre_id, rng), True)
    train_pairs = pd.DataFrame(pair_rows).reset_index(drop=True)
    assert_no_leakage(set(train_pairs["positive_doc_id"].astype(str)), banned)

    paraphrases: list[dict] = []
    for item in gold:
        if item.get("family") not in FACTUAL_FAMILIES:
            continue
        relevant = [str(x) for x in item.get("relevant_doc_ids") or []]
        if len(relevant) != 1 or relevant[0] not in known_ids:
            continue
        doc_id = relevant[0]
        if doc_id in cve_by_id.index:
            question = cve_query_without_id(cve_by_id.loc[doc_id])
        elif doc_id in mitre_by_id.index:
            question = mitre_query_without_id(mitre_by_id.loc[doc_id])
        else:
            continue
        if doc_id.upper() in question.upper() or extract_ids(question):
            question = strip_ids(question)
        paraphrases.append(
            {
                "id": f"P-{item['id']}",
                "family": item["family"],
                "question_es": question,
                "relevant_doc_ids": [doc_id],
                "must_cite": item.get("must_cite") or [],
                "with_id": False,
                "expect_abstain": False,
                "slice": "paraphrase",
                "paraphrase_of": item["id"],
                "difficulty": "dificil",
                "requires_hybrid": True,
                "origin": "plantilla_es_mas_extracto_oficial_sin_id",
            }
        )

    gold_f3: list[dict] = []
    for item in gold:
        copied = dict(item)
        relevant = [str(x) for x in item.get("relevant_doc_ids") or []]
        copied["with_id"] = bool(extract_ids(item.get("question_es", "")))
        copied["expect_abstain"] = item.get("family") == "abstain"
        copied["slice"] = "gold_f2"
        copied["paraphrase_of"] = None
        copied["difficulty"] = "facil" if copied["with_id"] and relevant else "politica"
        copied["requires_hybrid"] = False
        gold_f3.append(copied)
    gold_f3.extend(paraphrases)
    return {
        "train_pairs": train_pairs,
        "dev_items": dev_items,
        "paraphrases": paraphrases,
        "gold_f3": gold_f3,
        "banned_ids": banned,
        "dev_doc_ids": dev_ids,
    }
