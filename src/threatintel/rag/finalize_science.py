# -*- coding: utf-8 -*-
"""Reconstruye figuras/docs/notebook desde candidates/ sin reentrenar."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from threatintel.eval.metrics import ranking_metrics
from threatintel.rag.classical import TfidfIndex, fuse_score_maps, strip_excerpt
from threatintel.rag.experiment import dev_answerable, dump_json, jsonable, load_config
from threatintel.rag.model_zoo import Doc2VecIndex, LsaIndex
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts
from threatintel.rag.run_fase3 import evaluate_dense, load_encoder, prepare
from threatintel.rag.science_expand import (
    DISTILUSE,
    MARGIN,
    REF_1EP,
    REF_2EP,
    _dense_matrix,
    _eval_dense_queries,
    _metrics,
    _rank_of,
    inject_notebook,
    save_bootstrap_bars,
    save_decision_figure,
    save_heatmap,
    save_rank_hist,
    save_zoo_figure,
)


def run(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    print("finalize: prepare…", flush=True)
    state = prepare(root)
    corpus = state["corpus"].reset_index(drop=True)
    batch = int(cfg["encode_batch_size"])
    max_seq = int(cfg["max_seq_length"])
    seed = int(cfg["seed"])

    dev_all = dev_answerable(state["splits"]["dev_items"])
    dev_wo = [item for item in dev_all if not item.get("with_id")]
    gold_items = [item for item in state["gold"] if item.get("relevant_doc_ids")]
    q_pasted = [item["question_es"] for item in dev_wo]
    q_stripped = [strip_excerpt(q) for q in q_pasted]
    gold_q = [item["question_es"] for item in gold_items]

    classical = json.loads((root / "docs/data/evaluation/fase3_classical.json").read_text(encoding="utf-8"))
    classical_rows = {r["system"]: r for r in classical["report_rows"]}
    zoo_rows: list[dict] = []

    def add_row(label, system, family, mrr_s, r5_s, mrr_p, gold_m, gold_r5):
        zoo_rows.append(
            {
                "label": label,
                "system": system,
                "family": family,
                "dev_mrr_stripped": float(mrr_s) if mrr_s is not None else None,
                "dev_r5_stripped": float(r5_s) if r5_s is not None else None,
                "dev_mrr_pasted": float(mrr_p) if mrr_p is not None else None,
                "gold_mrr": float(gold_m) if gold_m is not None else None,
                "gold_r5": float(gold_r5) if gold_r5 is not None else None,
            }
        )

    for key, label, family in [
        ("bm25_k1=1.5_b=0.9", "BM25", "lexical"),
        ("tfidf_word_df1_sub1", "TF-IDF palabras", "lexical"),
        ("tfidf_char_wb_3_5", "TF-IDF char", "lexical"),
        ("linear_w=0.75", "Fusión lineal", "lexical"),
        ("minilm", "MiniLM congelado", "dense_frozen"),
        ("multi_frozen", "MiniLM multi congelado", "dense_frozen"),
        ("biencoder_ft", "Bi-encoder 2 épocas", "dense_ft"),
        ("hybrid_rrf", "RRF publicado", "hybrid"),
    ]:
        r = classical_rows[key]
        add_row(
            label,
            key,
            family,
            r["dev_mrr_stripped"],
            r["dev_r5_stripped"],
            r["dev_mrr_pasted"],
            r["gold_mrr"],
            r["gold_r5"],
        )

    print("LSA…", flush=True)
    lsa = LsaIndex(corpus, n_components=100, random_state=seed)
    lsa_s = _metrics(dev_wo, [lsa.rank(q)[0] for q in q_stripped])
    lsa_p = _metrics(dev_wo, [lsa.rank(q)[0] for q in q_pasted])
    lsa_g = _metrics(gold_items, [lsa.rank(q)[0] for q in gold_q])
    add_row(
        "LSA (SVD 100)",
        "lsa_100",
        "unsupervised",
        lsa_s["mrr"],
        lsa_s["recall_at_k"]["5"],
        lsa_p["mrr"],
        lsa_g["mrr"],
        lsa_g["recall_at_k"]["5"],
    )

    print("Doc2Vec…", flush=True)
    d2v = Doc2VecIndex(corpus, vector_size=100, epochs=15, seed=seed)
    d2v_s = _metrics(dev_wo, [d2v.rank(q)[0] for q in q_stripped])
    d2v_p = _metrics(dev_wo, [d2v.rank(q)[0] for q in q_pasted])
    d2v_g = _metrics(gold_items, [d2v.rank(q)[0] for q in gold_q])
    add_row(
        "Doc2Vec",
        "doc2vec",
        "embedding_classic",
        d2v_s["mrr"],
        d2v_s["recall_at_k"]["5"],
        d2v_p["mrr"],
        d2v_g["mrr"],
        d2v_g["recall_at_k"]["5"],
    )

    cand_root = root / "artifacts" / "fase3" / "candidates"

    print("distiluse frozen…", flush=True)
    dist_model = load_encoder(DISTILUSE, max_seq)
    dist_matrix = _dense_matrix(dist_model, corpus, batch)
    dist_s = _metrics(dev_wo, _eval_dense_queries(dist_model, dist_matrix, corpus, q_stripped))
    dist_p = _metrics(dev_wo, _eval_dense_queries(dist_model, dist_matrix, corpus, q_pasted))
    dist_g = _metrics(gold_items, _eval_dense_queries(dist_model, dist_matrix, corpus, gold_q))
    add_row(
        "DistilUSE congelado",
        "distiluse_frozen",
        "dense_frozen",
        dist_s["mrr"],
        dist_s["recall_at_k"]["5"],
        dist_p["mrr"],
        dist_g["mrr"],
        dist_g["recall_at_k"]["5"],
    )
    del dist_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dist_ep1 = cand_root / "distiluse_ep1" / "model"
    print("distiluse ep1 desde", dist_ep1, flush=True)
    dist_ft = load_encoder(str(dist_ep1), max_seq)
    dist_ft_matrix = _dense_matrix(dist_ft, corpus, batch)
    dist_ft_dev = evaluate_dense(dist_ft, corpus, dist_ft_matrix, dev_wo, batch, "distiluse_ep1")
    dist_ft_s = _metrics(dev_wo, _eval_dense_queries(dist_ft, dist_ft_matrix, corpus, q_stripped))
    dist_ft_g = _metrics(gold_items, _eval_dense_queries(dist_ft, dist_ft_matrix, corpus, gold_q))
    add_row(
        "DistilUSE 1 época",
        "distiluse_ep1",
        "dense_ft",
        dist_ft_s["mrr"],
        dist_ft_s["recall_at_k"]["5"],
        dist_ft_dev["without_id"]["mrr"],
        dist_ft_g["mrr"],
        dist_ft_g["recall_at_k"]["5"],
    )
    del dist_ft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("rangos…", flush=True)
    bm25 = BM25Index(corpus, k1=1.5, b=0.9)
    best = load_encoder(str(root / cfg["paths"]["best_model"]), max_seq)
    best_matrix = _dense_matrix(best, corpus, batch)
    tfidf = TfidfIndex(corpus, analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    ranks = {"biencoder": [], "bm25": [], "linear": []}
    for q, item in zip(q_stripped, dev_wo):
        rel = [str(x) for x in item["relevant_doc_ids"]]
        b_ids, b_scores, _ = bm25.rank(q)
        d_ids = _eval_dense_queries(best, best_matrix, corpus, [q])[0]
        t_ids, t_scores, _ = tfidf.rank(q)
        b_map = dict(zip(b_ids, b_scores))
        t_map = dict(zip(t_ids, t_scores))
        lin = fuse_score_maps(b_map, t_map, 0.75)
        ranks["biencoder"].append(_rank_of(rel, d_ids))
        ranks["bm25"].append(_rank_of(rel, b_ids))
        ranks["linear"].append(_rank_of(rel, lin))

    print("HP desde candidates…", flush=True)
    hp_rows = []
    for lr in (2e-5, 1e-4):
        for scale in (10.0, 20.0, 40.0):
            tag = f"lr{lr:.0e}_sc{int(scale)}"
            model_dir = cand_root / tag / "model"
            print(f"  eval {tag}", flush=True)
            model = load_encoder(str(model_dir), max_seq)
            emb = _dense_matrix(model, corpus, batch)
            metrics = evaluate_dense(model, corpus, emb, dev_wo, batch, tag)
            hp_rows.append(
                {
                    "tag": tag,
                    "lr": lr,
                    "loss_scale": scale,
                    "epochs": 1,
                    "train_loss_last": None,
                    "dev_mrr_without_id": metrics["without_id"]["mrr"],
                    "dev_r5_without_id": metrics["without_id"]["recall_at_k"]["5"],
                }
            )
            print(f"    MRR={metrics['without_id']['mrr']:.3f}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hard_model = load_encoder(str(cand_root / "hardneg_bm25" / "model"), max_seq)
    hard_emb = _dense_matrix(hard_model, corpus, batch)
    hard_metrics = evaluate_dense(hard_model, corpus, hard_emb, dev_wo, batch, "hardneg_bm25")
    hardneg = {
        "tag": "hardneg_bm25",
        "n_pairs": 4000,
        "dev_mrr_without_id": hard_metrics["without_id"]["mrr"],
        "dev_r5_without_id": hard_metrics["without_id"]["recall_at_k"]["5"],
    }
    print(f"hardneg MRR={hardneg['dev_mrr_without_id']:.3f}", flush=True)
    del hard_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_1ep = max(hp_rows, key=lambda r: float(r["dev_mrr_without_id"]))
    confirm = None
    reason = (
        f"Mejor lr×scale en 1 época: {best_1ep['tag']}={best_1ep['dev_mrr_without_id']:.3f} "
        f"(referencia 1 época 0,652; publicado 2 épocas 0,688). "
        f"Hardneg BM25 llega a {hardneg['dev_mrr_without_id']:.3f} en desarrollo, "
        f"aún por debajo del publicado 0,688. DistilUSE 1 época quedó en candidates/. "
        "No se pisa biencoder_best."
    )
    if float(best_1ep["dev_mrr_without_id"]) >= REF_1EP + MARGIN:
        reason = (
            f"{best_1ep['tag']} llega a {best_1ep['dev_mrr_without_id']:.3f} "
            f"(≥ 0,652+{MARGIN}). Queda en candidates/; gold ya abierto — no pisa biencoder_best (0,688). "
            f"Hardneg={hardneg['dev_mrr_without_id']:.3f}."
        )

    reading_zoo = (
        "En desarrollo sin extracto, LSA/Doc2Vec y léxicos miden el cruce español→inglés sin pegar el documento. "
        "DistilUSE afinado 1 época mejora el protocolo con extracto pero no desplaza al bi-encoder + RRF de la rúbrica. "
        "El gold solo reporta."
    )

    figs = root / "docs" / "data" / "figures"
    fig_zoo = figs / "fase3_model_zoo.png"
    fig_hm = figs / "fase3_hp_lr_scale.png"
    fig_ranks = figs / "fase3_rank_hist.png"
    fig_boot = figs / "fase3_bootstrap_delta.png"
    fig_dec = figs / "fase3_decision_deploy.png"
    save_zoo_figure(zoo_rows, fig_zoo)
    save_heatmap(hp_rows, fig_hm)
    save_rank_hist(ranks, fig_ranks)
    save_bootstrap_bars(fig_boot, None)
    save_decision_figure(fig_dec)

    zoo_payload = {
        "report_rows": zoo_rows,
        "reading": reading_zoo,
        "n_dev": len(dev_wo),
        "n_gold": len(gold_items),
        "seed": seed,
    }
    hp_payload = {
        "rows": hp_rows,
        "hardneg": hardneg,
        "confirm": confirm,
        "decision": {
            "overwrite_published": False,
            "best_one_epoch": best_1ep,
            "reason": reason,
        },
        "ref_1ep": REF_1EP,
        "ref_2ep": REF_2EP,
        "margin": MARGIN,
    }
    dump_json(root / "docs/data/evaluation/fase3_model_zoo.json", jsonable(zoo_payload))
    dump_json(root / "artifacts/fase3/hp_interaction_rows.json", jsonable(hp_payload))

    zoo_md = root / "docs/modeling/model_zoo.md"
    zoo_md.write_text(
        "\n".join(
            [
                "# Zoo de modelos — Fase III",
                "",
                "Comparación en **desarrollo sin identificador y sin extracto pegado**. "
                "El gold se reporta una vez y no elige.",
                "",
                "```mermaid",
                "flowchart LR",
                "  lexical[TFIDF_BM25_LSA_Doc2Vec] --> decideDev[MRR_dev_sin_extracto]",
                "  frozen[MiniLM_distiluse_congelados] --> decideDev",
                "  ft[biencoder_best_distiluse_ep1] --> decideDev",
                "  decideDev --> entrega[Entrega_biencoder_RRF]",
                "  decideDev --> gold[Gold_reporta]",
                "```",
                "",
                "| Sistema | Familia | MRR sin extracto | R@5 | MRR gold |",
                "|---------|---------|-----------------:|----:|---------:|",
            ]
            + [
                f"| {r['label']} | {r['family']} | {r['dev_mrr_stripped']:.3f} | "
                f"{r['dev_r5_stripped']:.3f} | {r['gold_mrr']:.3f} |"
                for r in zoo_rows
            ]
            + [
                "",
                reading_zoo,
                "",
                "Figura: `docs/data/figures/fase3_model_zoo.png`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hp_md_path = root / "docs/modeling/hiperparametros.md"
    existing = hp_md_path.read_text(encoding="utf-8")
    section = "\n".join(
        [
            "",
            "## Interacción lr × loss_scale (1 época)",
            "",
            "Métrica: MRR de desarrollo sin identificador. Semilla 42. "
            "`loss_scale` = 1/temperatura (InfoNCE / contrastive, M5U5).",
            "",
            "```mermaid",
            "flowchart TD",
            "  grid[lr_x_scale_6_puntos] --> best1[mejor_1_epoca]",
            "  hard[hardneg_BM25] --> best1",
            "  best1 -->|bajo_publicado_0.688| keep[biencoder_best]",
            "  best1 --> cand[candidates_sin_pisar]",
            "```",
            "",
            "| Tag | lr | scale | MRR sin id | R@5 |",
            "|-----|---:|------:|-----------:|----:|",
        ]
        + [
            f"| {r['tag']} | {r['lr']} | {r['loss_scale']} | {r['dev_mrr_without_id']:.3f} | "
            f"{r['dev_r5_without_id']:.3f} |"
            for r in hp_rows
        ]
        + [
            "",
            f"Hardneg BM25: MRR {hardneg['dev_mrr_without_id']:.3f} (n={hardneg['n_pairs']}).",
            "",
            reason,
            "",
            "Figura: `docs/data/figures/fase3_hp_lr_scale.png`.",
            "",
        ]
    )
    if "## Interacción lr × loss_scale" in existing:
        existing = existing.split("## Interacción lr × loss_scale")[0].rstrip() + "\n"
    hp_md_path.write_text(existing + section, encoding="utf-8")

    payload = {"zoo": zoo_payload, "hp": hp_payload, "ranks": {k: ranks[k] for k in ranks}}
    inject_notebook(
        root,
        payload,
        {"zoo": fig_zoo, "heatmap": fig_hm, "ranks": fig_ranks, "bootstrap": fig_boot, "decision": fig_dec},
    )
    print("finalize listo", flush=True)
    return payload


if __name__ == "__main__":
    run()
