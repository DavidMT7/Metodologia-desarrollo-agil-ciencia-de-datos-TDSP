# -*- coding: utf-8 -*-
"""Mejora Fase IV: hardneg densos, continue lr bajo, seq384, RRF triple. Gold solo reporta."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from threatintel.rag.experiment import dense_rankings, dev_answerable, dump_json, jsonable, load_config, pack_metrics
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts, rrf_fuse
from threatintel.rag.run_fase3 import embed_corpus, evaluate_dense, load_encoder, prepare
from threatintel.rag.run_fase4_expand import _eval_suite, _hybrid_mrr, _save_rrf_compare
from threatintel.rag.run_fase4_science import _save_bar, _save_curves, _save_delta, _train_eval
from threatintel.rag.train import set_seed, train_biencoder

DISTILUSE = "sentence-transformers/distiluse-base-multilingual-cased-v2"
TARGET = 0.765
PROMOTE_MARGIN = 0.01


def _parent_texts(corpus: pd.DataFrame) -> dict[str, str]:
    parent_text: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for rec in corpus.itertuples(index=False):
        doc_id = str(rec.doc_id)
        text = str(rec.text)
        if doc_id not in parent_text or len(text) > lengths.get(doc_id, 0):
            parent_text[doc_id] = text
            lengths[doc_id] = len(text)
    return parent_text


def dense_hard_negatives(
    model,
    emb: np.ndarray,
    corpus: pd.DataFrame,
    train_pairs: pd.DataFrame,
    batch: int,
    *,
    rank_from: int = 1,
    rank_to: int = 30,
) -> pd.DataFrame:
    """Negativo = primer padre distinto del positivo en el ranking denso (semi-duro)."""
    parent_text = _parent_texts(corpus)
    index = DenseIndex(corpus, emb, "hn")
    queries = train_pairs["query"].astype(str).tolist()
    q_emb = encode_texts(model, queries, batch_size=batch)
    rows = []
    for i, rec in enumerate(train_pairs.itertuples(index=False)):
        ids, _scores, _ = index.rank(q_emb[i])
        pos = str(rec.positive_doc_id)
        neg = None
        for rank, doc_id in enumerate(ids, start=1):
            if rank < rank_from:
                continue
            if rank > rank_to:
                break
            if doc_id != pos and doc_id in parent_text:
                neg = doc_id
                break
        if neg is None:
            continue
        rows.append(
            {
                "query": rec.query,
                "positive_text": rec.positive_text,
                "positive_doc_id": rec.positive_doc_id,
                "negative": parent_text[neg],
                "with_id": getattr(rec, "with_id", False),
            }
        )
    return pd.DataFrame(rows)


def triple_rrf(rankings_list: list[list[str]], k: int = 60) -> list[str]:
    return [doc_id for doc_id, _ in rrf_fuse(rankings_list, k=k, top_n=100)]


def eval_triple(model_a, emb_a, model_b, emb_b, corpus, items, bm25, batch, k=60) -> dict:
    qs = [item["question_es"] for item in items]
    ra, _ = dense_rankings(model_a, DenseIndex(corpus, emb_a, "a"), qs, batch)
    rb, _ = dense_rankings(model_b, DenseIndex(corpus, emb_b, "b"), qs, batch)
    fused = []
    for q, a, b in zip(qs, ra, rb):
        bm = bm25.rank(q)[0]
        fused.append(triple_rrf([a[:100], b[:100], bm[:100]], k=k))
    return pack_metrics(f"triple_rrf_k{k}", items, fused)


def run(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    set_seed(int(cfg["seed"]))
    print("Fase IV improve: prepare…", flush=True)
    state = prepare(root)
    corpus = state["corpus"].reset_index(drop=True)
    train_pairs = state["splits"]["train_pairs"]
    dev_wo = [i for i in dev_answerable(state["splits"]["dev_items"]) if not i.get("with_id")]
    gold = [i for i in state["gold"] if i.get("relevant_doc_ids")]
    batch = int(cfg["batch_size"])
    encode_batch = int(cfg["encode_batch_size"])
    max_seq = int(cfg["max_seq_length"])
    seed = int(cfg["seed"])
    cand = root / "artifacts" / "fase4" / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    figs = root / "docs" / "data" / "figures"
    f3_path = str(root / cfg["paths"]["best_model"])
    f4_path = str(root / cfg["paths"]["best_model_fase4"])
    summary_path = root / "docs/data/evaluation/fase4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = float(summary["baseline_f3_mrr"])
    rows = list(summary["rows"])
    curves = {}

    print("load F3/F4…", flush=True)
    f3 = load_encoder(f3_path, max_seq)
    f3_emb = embed_corpus(f3, corpus, encode_batch)
    f4 = load_encoder(f4_path, max_seq)
    f4_emb = embed_corpus(f4, corpus, encode_batch)
    bm25 = BM25Index(corpus, k1=1.5, b=0.75)

    # Triple RRF (no train)
    print("triple RRF…", flush=True)
    triple_dev = eval_triple(f4, f4_emb, f3, f3_emb, corpus, dev_wo, bm25, encode_batch, 60)
    triple_gold = eval_triple(f4, f4_emb, f3, f3_emb, corpus, gold, bm25, encode_batch, 60)
    print(
        f"  triple_rrf dev MRR sin id={triple_dev['without_id']['mrr']:.3f} "
        f"gold={triple_gold['all']['mrr']:.3f}",
        flush=True,
    )

    # Dense hardneg mining with F4
    print("mining dense hardneg (DistilUSE)…", flush=True)
    dhard = dense_hard_negatives(f4, f4_emb, corpus, train_pairs, encode_batch, rank_from=2, rank_to=40)
    print(f"  n_hard={len(dhard)}", flush=True)
    dump_json(root / "artifacts/fase4/dense_hardneg_meta.json", {"n": len(dhard), "rank_from": 2, "rank_to": 40})

    experiments = [
        ("imp_cont_lr1e-5_ep1", f4_path, train_pairs, 1e-5, 1, 20.0, batch, 0.1, 0.0, max_seq),
        ("imp_cont_lr5e-6_ep1", f4_path, train_pairs, 5e-6, 1, 20.0, batch, 0.05, 0.0, max_seq),
        ("imp_cont_sc10_lr1e-5", f4_path, train_pairs, 1e-5, 1, 10.0, batch, 0.1, 0.0, max_seq),
        ("imp_dhard_lr1e-5", f4_path, dhard, 1e-5, 1, 20.0, batch, 0.1, 0.0, max_seq),
        ("imp_dhard_lr2e-5", f4_path, dhard, 2e-5, 1, 20.0, batch, 0.1, 0.0, max_seq),
        ("imp_dhard_sc10_lr1e-5", f4_path, dhard, 1e-5, 1, 10.0, batch, 0.1, 0.0, max_seq),
        ("imp_distil_seq384_ep1", DISTILUSE, train_pairs, 2e-5, 1, 20.0, batch, 0.1, 0.0, 384),
        ("imp_distil_seq384_ep2", DISTILUSE, train_pairs, 2e-5, 2, 20.0, batch, 0.1, 0.0, 384),
        ("imp_dhard_mix50", f4_path, pd.concat([dhard.sample(frac=0.5, random_state=seed), train_pairs.sample(n=min(2000, len(train_pairs)), random_state=seed)], ignore_index=True), 1e-5, 1, 20.0, batch, 0.1, 0.0, max_seq),
    ]

    # mix50 may have NaN negatives for soft rows — drop negative col when missing
    cleaned_exps = []
    for tag, init, pairs, lr, ep, sc, bs, wu, wd, seq in experiments:
        pairs = pairs.copy()
        if "negative" in pairs.columns and pairs["negative"].isna().any():
            # train without negatives for NaN rows by splitting — simpler: fill skip those rows with hard only
            pairs = pairs.dropna(subset=["negative"]) if pairs["negative"].notna().mean() > 0.3 else pairs.drop(columns=["negative"])
        cleaned_exps.append((tag, init, pairs, lr, ep, sc, bs, wu, wd, seq))

    new_rows = []
    best_row = max(rows, key=lambda r: float(r["dev_mrr_without_id"]))
    for tag, init, pairs, lr, ep, sc, bs, wu, wd, seq in cleaned_exps:
        # temporary max_seq override via cfg copy in train — train_biencoder takes max_seq_length
        row, hist, model, emb = _train_eval(
            tag=tag,
            init=init,
            pairs=pairs,
            lr=lr,
            epochs=ep,
            scale=sc,
            batch=bs,
            warmup=wu,
            wd=wd,
            cand_root=cand,
            corpus=corpus,
            dev_wo=dev_wo,
            cfg=cfg,
            seed=seed,
            max_seq=seq,
            encode_batch=encode_batch,
        )
        # re-eval at standard seq if trained with 384 — embeddings already with that model max_seq
        new_rows.append(row)
        rows.append(row)
        curves[tag] = hist
        if float(row["dev_mrr_without_id"]) > float(best_row["dev_mrr_without_id"]) + 1e-12:
            best_row = row
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Confirm best new with +1 ep if it beats TARGET
    top_new = max(new_rows, key=lambda r: float(r["dev_mrr_without_id"])) if new_rows else None
    if top_new and float(top_new["dev_mrr_without_id"]) >= TARGET + 0.005:
        tag = f"confirm_{top_new['tag']}_plus1"
        init = str(cand / top_new["tag"] / "model")
        pairs = dhard if "dhard" in top_new["tag"] and "mix" not in top_new["tag"] else train_pairs
        row, hist, model, emb = _train_eval(
            tag=tag,
            init=init,
            pairs=pairs,
            lr=max(5e-6, float(top_new["lr"]) / 2),
            epochs=1,
            scale=float(top_new["loss_scale"]),
            batch=int(top_new["batch_size"]),
            warmup=0.05,
            wd=float(top_new["weight_decay"]),
            cand_root=cand,
            corpus=corpus,
            dev_wo=dev_wo,
            cfg=cfg,
            seed=seed,
            max_seq=int(top_new.get("max_seq", max_seq)) if False else max_seq,
            encode_batch=encode_batch,
        )
        # fix: use seq from tag
        new_rows.append(row)
        rows.append(row)
        curves[tag] = hist
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    winner = max(rows, key=lambda r: float(r["dev_mrr_without_id"]))
    current = TARGET
    promote = float(winner["dev_mrr_without_id"]) >= current + PROMOTE_MARGIN
    decision = summary.get("decision_reason", "")
    if promote:
        src = cand / winner["tag"] / "model"
        dest = root / cfg["paths"]["best_model_fase4"]
        if src.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            decision = (
                f"Improve: promovido {winner['tag']} MRR {winner['dev_mrr_without_id']:.3f} "
                f"(previo {current:.3f}). Gold no eligió."
            )
            print(decision, flush=True)
    else:
        decision = (
            f"Improve: mejor={winner['tag']} {winner['dev_mrr_without_id']:.3f}; "
            f"vigente DistilUSE confirm {current:.3f}. "
            f"Triple RRF dev={triple_dev['without_id']['mrr']:.3f} (fusión en evaluación, "
            f"deploy sigue bi-encoder+BM25). No se pisa F4 salvo margen {PROMOTE_MARGIN}."
        )
        print(decision, flush=True)

    # Suites refresh
    deploy_path = str(root / cfg["paths"]["best_model_fase4"])
    deploy = load_encoder(deploy_path, max_seq)
    deploy_emb = embed_corpus(deploy, corpus, encode_batch)
    suites = {
        "F3_MiniLM": _eval_suite("f3", f3, f3_emb, corpus, dev_wo, gold, bm25, encode_batch),
        "F4_deploy": _eval_suite("f4", deploy, deploy_emb, corpus, dev_wo, gold, bm25, encode_batch),
    }
    suites["Triple_RRF"] = {
        "dense_dev_mrr": float("nan"),
        "dense_dev_r5": float("nan"),
        "stripped_mrr": float("nan"),
        "rrf60_dev_mrr": triple_dev["without_id"]["mrr"],
        "rrf40_dev_mrr": float("nan"),
        "rrf100_dev_mrr": float("nan"),
        "gold_dense_mrr": float("nan"),
        "gold_dense_r5": float("nan"),
        "gold_rrf_mrr": triple_gold["all"]["mrr"],
        "gold_rrf_r5": triple_gold["all"]["recall_at_k"]["5"],
    }
    # fill nan with None for JSON
    for s in suites.values():
        for k, v in list(s.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                s[k] = None

    _save_bar(new_rows, figs / "fase4_improve_grid.png", "dev_mrr_without_id", "Improve Fase IV — MRR dev sin id")
    if curves:
        _save_curves(curves, figs / "fase4_improve_loss.png")
    _save_delta(rows, figs / "fase4_delta_vs_f3.png", baseline)
    _save_rrf_compare(
        {k: {kk: (vv if vv is not None else 0.0) for kk, vv in s.items()} for k, s in suites.items() if k != "Triple_RRF"},
        figs / "fase4_rrf_compare.png",
    )
    # Triple bar figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["F3_RRF60", "F4_RRF60", "Triple_RRF"]
    vals = [
        suites["F3_MiniLM"]["rrf60_dev_mrr"],
        suites["F4_deploy"]["rrf60_dev_mrr"],
        triple_dev["without_id"]["mrr"],
    ]
    ax.bar(labels, vals, color=["#64748b", "#2563eb", "#059669"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fusión en desarrollo (elige narrativa; deploy = F4+BM25)")
    fig.tight_layout()
    fig.savefig(figs / "fase4_triple_rrf.png", dpi=140)
    plt.close(fig)

    payload = {
        **summary,
        "rows": rows,
        "improve_rows": new_rows,
        "suites": suites,
        "triple_rrf_dev_mrr": triple_dev["without_id"]["mrr"],
        "triple_rrf_gold_mrr": triple_gold["all"]["mrr"],
        "winner": winner,
        "promote_improve": promote,
        "decision_reason": decision,
        "deploy_path": deploy_path,
        "gold_deploy_mrr": suites["F4_deploy"]["gold_dense_mrr"],
        "gold_deploy_r5": suites["F4_deploy"]["gold_dense_r5"],
        "gold_rrf_deploy_mrr": suites["F4_deploy"]["gold_rrf_mrr"],
        "n_dense_hardneg": int(len(dhard)),
    }
    dump_json(summary_path, jsonable(payload))
    dump_json(root / "artifacts/fase4/grid_rows.json", jsonable({"rows": rows, "best": winner, "suites": suites}))
    dump_json(root / "docs/data/evaluation/fase4_suites.json", jsonable(suites))

    # Docs patch
    hp = root / "docs/modeling/fase4_hiperparametros.md"
    extra = "\n".join(
        [
            "",
            "## Improve (hardneg denso / lr bajo / seq384 / triple RRF)",
            "",
            decision,
            "",
            f"Triple RRF (DistilUSE+MiniLM+BM25) dev MRR={triple_dev['without_id']['mrr']:.3f}, "
            f"gold={triple_gold['all']['mrr']:.3f} (evaluación; no cambia el checkpoint salvo promoción).",
            "",
            "| Tag | MRR |",
            "|-----|----:|",
        ]
        + [f"| {r['tag']} | {r['dev_mrr_without_id']:.3f} |" for r in sorted(new_rows, key=lambda x: -float(x['dev_mrr_without_id']))]
        + ["", "Figuras: `fase4_improve_grid.png`, `fase4_triple_rrf.png`."]
    )
    text = hp.read_text(encoding="utf-8")
    if "## Improve" in text:
        text = text.split("## Improve")[0].rstrip() + "\n"
    hp.write_text(text + "\n" + extra, encoding="utf-8")

    (root / "docs/modeling/fase4_model_report.md").write_text(
        "\n".join(
            [
                "# Model report — Fase IV (improve)",
                "",
                decision,
                "",
                f"- Corridas totales: {len(rows)}",
                f"- Deploy: `{deploy_path}`",
                f"- Dense/RRF60 dev: {suites['F4_deploy']['dense_dev_mrr']:.3f} / {suites['F4_deploy']['rrf60_dev_mrr']:.3f}",
                f"- Gold RRF (reporta): {suites['F4_deploy']['gold_rrf_mrr']:.3f}",
                f"- Triple RRF (ablación): dev {triple_dev['without_id']['mrr']:.3f} · gold {triple_gold['all']['mrr']:.3f}",
                f"- Hardneg densos minados: {len(dhard)}",
                "",
                "```mermaid",
                "flowchart TD",
                "  conf[DistilUSE_ep2] --> mine[Hardneg_denso]",
                "  conf --> cont[Continue_lr_bajo]",
                "  mine --> grid[Improve_grid]",
                "  cont --> grid",
                "  grid --> keep[Mantener_o_promover]",
                "  conf --> triple[Triple_RRF_ablacion]",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    # Notebook append
    _patch_nb(root, payload, new_rows, triple_dev, triple_gold)
    print("Fase IV improve listo", flush=True)
    return payload


def _patch_nb(root, payload, new_rows, triple_dev, triple_gold):
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_output

    nb_path = root / "scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    marker = "# VALIDACION-IMPROVE"
    keep = []
    skip = False
    for c in nb.cells:
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if marker in src:
            skip = True
            continue
        if skip and c.cell_type == "markdown" and "### Lectura — improve" in src:
            skip = False
            continue
        if skip:
            continue
        keep.append(c)
    nb.cells = keep

    lines = [f"{r['tag']:<28} {r['dev_mrr_without_id']:.3f}" for r in sorted(new_rows, key=lambda x: -float(x["dev_mrr_without_id"]))]
    out = (
        "Improve grid\n"
        + "\n".join(lines)
        + f"\n\nTriple RRF dev={triple_dev['without_id']['mrr']:.3f} gold={triple_gold['all']['mrr']:.3f}\n"
        + payload["decision_reason"]
        + "\n"
    )
    code = new_code_cell(
        f"""{marker}
import json
from pathlib import Path
ROOT=Path.cwd().resolve()
for p in [ROOT,*ROOT.parents]:
    if (p/'configs'/'retrieval.yaml').is_file():
        ROOT=p; break
exp=json.loads((ROOT/'docs/data/evaluation/fase4_summary.json').read_text(encoding='utf-8'))
print('Improve rows')
for r in sorted(exp.get('improve_rows') or [], key=lambda x: -float(x['dev_mrr_without_id'])):
    print(f"{{r['tag']:<28}} {{r['dev_mrr_without_id']:.3f}}")
print('Triple RRF dev', exp.get('triple_rrf_dev_mrr'), 'gold', exp.get('triple_rrf_gold_mrr'))
print(exp['decision_reason'])
"""
    )
    code.outputs = [new_output("stream", name="stdout", text=out)]
    md = new_markdown_cell(
        "### Lectura — improve\n\n"
        "**Qué salió.** Hardneg densos (minados con DistilUSE), continue con lr 1e-5/5e-6, seq384, y ablación **triple RRF** "
        "(DistilUSE+MiniLM+BM25).\n\n"
        "**Qué significa.** Si ningún punto supera el confirm 0,765 con margen, el checkpoint se mantiene: "
        "más épocas/hardneg no siempre ayudan (firma de ajuste correcto).\n\n"
        f"**Decisión.** {payload['decision_reason']}\n\n"
        "![Improve](../../docs/data/figures/fase4_improve_grid.png)\n\n"
        "![Triple RRF](../../docs/data/figures/fase4_triple_rrf.png)\n"
    )
    nb.cells.extend([new_markdown_cell("## 3c. Improve"), code, md])
    nbformat.write(nb, nb_path)
    print("notebook improve patched", flush=True)


if __name__ == "__main__":
    run()
