# -*- coding: utf-8 -*-
"""Pasos del experimento Fase III. El índice se arma solo con parquets hardened."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from threatintel.rag.corpus import build_corpus
from threatintel.rag.experiment import (
    build_retrieval_for_query,
    checklist_score,
    dense_rankings,
    dev_answerable,
    dev_gate_rows,
    dump_json,
    fmt_num,
    load_config,
    official_abstain_queries,
    pack_metrics,
    SCENARIO_SPECS,
)
from threatintel.rag.generate import answer_query
from threatintel.rag.lineage_gate import assert_phase3_lineage
from threatintel.rag.pairs import FACTUAL_FAMILIES, prepare_splits
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts, rrf_fuse
from threatintel.rag.train import describe_architecture, set_seed, train_biencoder
from threatintel.retrieval_baseline import build_corpus as build_f2_corpus
from threatintel.retrieval_baseline import run_baseline


def prepare(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    set_seed(int(cfg["seed"]))
    cve_clean = pd.read_parquet(root / cfg["paths"]["cve_clean"])
    mitre_clean = pd.read_parquet(root / cfg["paths"]["mitre_clean"])
    cve = pd.read_parquet(root / cfg["paths"]["cve_parquet"])
    mitre = pd.read_parquet(root / cfg["paths"]["mitre_parquet"])
    lineage = assert_phase3_lineage(cve_clean, cve, mitre_clean, mitre)
    gold = json.loads((root / cfg["paths"]["gold"]).read_text(encoding="utf-8"))
    corpus = build_corpus(cve, mitre, max_chars=int(cfg["max_chars"]))
    if corpus.loc[corpus["doc_type"] == "mitre", "text"].str.contains("Strategy:", regex=False).mean() < 0.5:
        raise RuntimeError("El índice MITRE no está usando el detection endurecido.")
    splits = prepare_splits(
        corpus,
        cve,
        mitre,
        gold,
        seed=int(cfg["seed"]),
        n_dev_cve=int(cfg["splits"]["n_dev_cve"]),
        n_dev_mitre=int(cfg["splits"]["n_dev_mitre"]),
        n_dev_unknown=int(cfg["splits"]["n_dev_unknown"]),
        n_dev_nonsense=int(cfg["splits"]["n_dev_nonsense"]),
        max_train_pairs=int(cfg["max_train_pairs"]),
    )
    chunks_path = root / cfg["paths"]["chunks"]
    corpus_path = root / cfg["paths"]["corpus"]
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.loc[corpus["doc_type"] == "mitre"].to_parquet(chunks_path, index=False)
    corpus.to_parquet(corpus_path, index=False)
    dump_json(root / cfg["paths"]["gold_f3"], splits["gold_f3"])
    return {
        "root": root,
        "cfg": cfg,
        "cve": cve,
        "mitre": mitre,
        "gold": gold,
        "corpus": corpus,
        "splits": splits,
        "lineage": lineage,
        "known_ids": set(corpus["doc_id"].astype(str)),
    }


def load_encoder(model_name: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    return model


def embed_corpus(model, corpus: pd.DataFrame, batch_size: int) -> np.ndarray:
    return encode_texts(model, corpus["text"].astype(str).tolist(), batch_size=batch_size)


def evaluate_dense(model, corpus, embeddings, items, batch_size: int, name: str) -> dict:
    index = DenseIndex(corpus, embeddings, name)
    rankings, scores = dense_rankings(model, index, [item["question_es"] for item in items], batch_size)
    packed = pack_metrics(name, items, rankings)
    packed["top_scores"] = scores
    return packed


def evaluate_bm25(corpus: pd.DataFrame, items: list[dict], name: str = "bm25") -> dict:
    index = BM25Index(corpus)
    rankings = [index.rank(item["question_es"])[0] for item in items]
    return pack_metrics(name, items, rankings)


def train_grid(state: dict) -> dict:
    import torch

    cfg = state["cfg"]
    root = state["root"]
    corpus = state["corpus"]
    train_pairs = state["splits"]["train_pairs"]
    dev_items = dev_answerable(state["splits"]["dev_items"])
    rows = []
    curves = {}
    best = None
    best_embeddings = None
    run_root = root / "artifacts" / "fase3" / "runs"
    for lr in cfg["grid"]["learning_rates"]:
        for epochs in cfg["grid"]["epochs"]:
            tag = f"lr{lr:.0e}_ep{epochs}"
            print(f"Entrenando {tag} sobre {len(train_pairs)} pares del índice hardened…")
            model, history = train_biencoder(
                cfg["models"]["biencoder"],
                train_pairs,
                lr=float(lr),
                epochs=int(epochs),
                batch_size=int(cfg["batch_size"]),
                seed=int(cfg["seed"]),
                output_dir=run_root / tag,
                max_seq_length=int(cfg["max_seq_length"]),
                warmup_ratio=float(cfg["warmup_ratio"]),
            )
            embeddings = embed_corpus(model, corpus, int(cfg["encode_batch_size"]))
            metrics = evaluate_dense(model, corpus, embeddings, dev_items, int(cfg["encode_batch_size"]), tag)
            last_loss = None
            for event in reversed(history):
                if "loss" in event:
                    last_loss = float(event["loss"])
                    break
            record = {
                "tag": tag,
                "lr": float(lr),
                "epochs": int(epochs),
                "train_loss_last": last_loss,
                "dev_mrr": metrics["all"]["mrr"],
                "dev_r5": metrics["all"]["recall_at_k"]["5"],
                "dev_mrr_without_id": metrics["without_id"]["mrr"],
                "dev_r5_without_id": metrics["without_id"]["recall_at_k"]["5"],
                "dev_mrr_with_id": metrics["with_id"]["mrr"],
                "n_dev": metrics["all"]["n"],
                "n_dev_without_id": metrics["without_id"]["n"],
            }
            rows.append(record)
            curves[tag] = history
            print(
                f"  dev MRR sin id={fmt_num(record['dev_mrr_without_id'])} "
                f"R@5 sin id={fmt_num(record['dev_r5_without_id'])} loss={fmt_num(last_loss)}"
            )
            score = record["dev_mrr_without_id"] if record["dev_mrr_without_id"] is not None else -1.0
            better = best is None or score > best["score"] + 1e-12
            if best is not None and abs(score - best["score"]) <= 1e-12:
                better = (record["dev_r5_without_id"] or 0) > (best["record"]["dev_r5_without_id"] or 0)
            if better:
                best_dir = root / cfg["paths"]["best_model"]
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(best_dir))
                np.savez_compressed(best_dir / "embeddings.npz", embeddings=embeddings)
                best = {"score": score, "record": record, "dir": str(best_dir)}
                best_embeddings = embeddings
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if best is None or best_embeddings is None:
        raise RuntimeError("La grilla no produjo un modelo")
    dump_json(root / "artifacts" / "fase3" / "grid_rows.json", {"rows": rows, "best": best["record"]})
    return {"rows": rows, "curves": curves, "best": best, "best_embeddings": best_embeddings}


def frozen_reference(state: dict, model_name: str, items: list[dict]):
    cfg = state["cfg"]
    model = load_encoder(model_name, int(cfg["max_seq_length"]))
    embeddings = embed_corpus(model, state["corpus"], int(cfg["encode_batch_size"]))
    metrics = evaluate_dense(model, state["corpus"], embeddings, items, int(cfg["encode_batch_size"]), model_name)
    del model
    return metrics, embeddings


def dense_query_details(model, corpus, embeddings, queries: list[str], batch_size: int):
    index = DenseIndex(corpus, embeddings, "query")
    encoded = encode_texts(model, queries, batch_size=batch_size)
    return [{"ids": ids, "scores": scores, "chunks": chunks} for ids, scores, chunks in (index.rank(v) for v in encoded)]


def hybrid_rankings_for(details_dense, bm25: BM25Index, queries: list[str], cfg: dict, corpus: pd.DataFrame, cross_encoder=None):
    rrf_rankings, rerank_rankings, boosted_rankings = [], [], []
    kev_ids = set(corpus.loc[corpus["in_cisa_kev"].astype(bool), "doc_id"].astype(str))
    bonus = float(cfg["retrieval"]["kev_boost"])
    known = set(corpus["doc_id"].astype(str))
    for query, dense in zip(queries, details_dense):
        b_ids, b_scores, b_chunks = bm25.rank(query)
        fused = rrf_fuse(
            [dense["ids"][: int(cfg["retrieval"]["dense_top_k"])], b_ids[: int(cfg["retrieval"]["bm25_top_k"])]],
            k=int(cfg["retrieval"]["rrf_k"]),
            top_n=int(cfg["retrieval"]["rrf_top_n"]),
        )
        rrf_rankings.append([doc_id for doc_id, _score in fused])
        boosted_rankings.append(
            [doc_id for doc_id, _score in sorted(fused, key=lambda item: item[1] + (bonus if item[0] in kev_ids else 0.0), reverse=True)]
        )
        if cross_encoder is None:
            rerank_rankings.append(rrf_rankings[-1])
            continue
        _retrieval, final_ids = build_retrieval_for_query(
            query,
            known_ids=known,
            corpus=corpus,
            dense_ids=dense["ids"],
            dense_scores=dense["scores"],
            dense_chunks=dense["chunks"],
            bm25_ids=b_ids,
            bm25_scores=b_scores,
            bm25_chunks=b_chunks,
            cfg=cfg,
            cross_encoder=cross_encoder,
        )
        rerank_rankings.append(final_ids)
    return rrf_rankings, rerank_rankings, boosted_rankings


def answer_items(model, embeddings, bm25, cross_encoder, state, items, threshold: float) -> list[dict]:
    cfg = state["cfg"]
    corpus = state["corpus"]
    queries = [item["question_es"] if "question_es" in item else item["question"] for item in items]
    details = dense_query_details(model, corpus, embeddings, queries, int(cfg["encode_batch_size"]))
    answers = []
    for item, dense, query in zip(items, details, queries):
        b_ids, b_scores, b_chunks = bm25.rank(query)
        retrieval, _ids = build_retrieval_for_query(
            query,
            known_ids=state["known_ids"],
            corpus=corpus,
            dense_ids=dense["ids"],
            dense_scores=dense["scores"],
            dense_chunks=dense["chunks"],
            bm25_ids=b_ids,
            bm25_scores=b_scores,
            bm25_chunks=b_chunks,
            cfg=cfg,
            cross_encoder=cross_encoder,
        )
        answer = answer_query(query, retrieval, threshold=threshold)
        answer["id"] = item.get("id")
        answers.append(answer)
    return answers


def citation_summary(answers: list[dict], items: list[dict]) -> dict:
    in_context, gold_match = [], []
    answered = 0
    for answer, item in zip(answers, items):
        if item.get("family") not in FACTUAL_FAMILIES:
            continue
        if answer.get("abstained"):
            continue
        answered += 1
        cited = set(answer.get("cited") or [])
        in_context.append(1.0 if cited and not answer.get("removed_ids") else 0.0)
        relevant = {str(x) for x in item.get("relevant_doc_ids") or []}
        gold_match.append(1.0 if cited & relevant else 0.0)

    def _mean(values):
        return float(np.mean(values)) if values else None

    return {"n_answered_factual": answered, "citation_in_context": _mean(in_context), "citation_matches_gold": _mean(gold_match)}


def abstention_summary(answers: list[dict]) -> dict:
    if not answers:
        return {"n": 0, "rate": None}
    flags = [1.0 if answer.get("abstained") else 0.0 for answer in answers]
    return {"n": len(flags), "rate": float(np.mean(flags))}


def reproduce_fase2(state: dict) -> dict:
    cfg = state["cfg"]
    corpus = build_f2_corpus(state["cve"], state["mitre"])
    return run_baseline(corpus, state["gold"], model_name=cfg["models"]["baseline_minilm"])


def write_experiment_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system", "split", "model", "lr", "epochs", "mrr", "recall_at_5", "mrr_without_id", "recall_at_5_without_id", "n", "seed", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def save_loss_figure(curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag, history in curves.items():
        steps, losses = [], []
        for event in history:
            if "loss" not in event:
                continue
            steps.append(event.get("step", len(steps)))
            losses.append(event["loss"])
        if losses:
            ax.plot(steps, losses, marker="o", label=tag)
    ax.set_xlabel("Paso")
    ax.set_ylabel("Pérdida InfoNCE")
    ax.set_title("Grilla de hiperparámetros — índice hardened")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison_figure(systems: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["system"] for row in systems]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, [row["mrr"] or 0 for row in systems], width, label="MRR")
    ax.bar(x + width / 2, [row["recall_at_5"] or 0 for row in systems], width, label="Recall@5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Hold-out gold Fase II — retrieval sobre hardened")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def metric_line(name: str, packed: dict, split: str, seed: int, notes: str = "", lr="", epochs="") -> dict:
    block = packed["all"]
    without = packed["without_id"]
    return {
        "system": name,
        "split": split,
        "model": name,
        "lr": lr,
        "epochs": epochs,
        "mrr": None if block["mrr"] is None else round(block["mrr"], 4),
        "recall_at_5": None if block["recall_at_k"]["5"] is None else round(block["recall_at_k"]["5"], 4),
        "mrr_without_id": None if without["mrr"] is None else round(without["mrr"], 4),
        "recall_at_5_without_id": None if without["recall_at_k"]["5"] is None else round(without["recall_at_k"]["5"], 4),
        "n": block["n"],
        "seed": seed,
        "notes": notes,
    }
