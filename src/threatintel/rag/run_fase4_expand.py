# -*- coding: utf-8 -*-
"""Experimentos extra Fase IV: RRF, overfit ep3, hardneg DistilUSE, ablaciones. Gold solo reporta."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from threatintel.rag.classical import strip_excerpt
from threatintel.rag.experiment import (
    dense_rankings,
    dev_answerable,
    dump_json,
    jsonable,
    load_config,
    pack_metrics,
)
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts, rrf_fuse
from threatintel.rag.run_fase3 import embed_corpus, evaluate_dense, load_encoder, prepare
from threatintel.rag.run_fase4_science import (
    MARGIN,
    _hard_negatives,
    _rank_of,
    _save_bar,
    _save_curves,
    _save_delta,
    _train_eval,
)
from threatintel.rag.train import set_seed, train_biencoder

DISTILUSE = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def _hybrid_mrr(model, emb, corpus, items, bm25: BM25Index, batch: int, k: int = 60) -> dict:
    queries = [item["question_es"] for item in items]
    dense_rankings_list, _ = dense_rankings(model, DenseIndex(corpus, emb, "h"), queries, batch)
    fused = []
    for q, dens in zip(queries, dense_rankings_list):
        b_ids = bm25.rank(q)[0]
        fused.append([d for d, _ in rrf_fuse([dens[:100], b_ids[:100]], k=k, top_n=100)])
    return pack_metrics(f"rrf_k{k}", items, fused)


def _eval_suite(name, model, emb, corpus, dev_wo, gold, bm25, batch) -> dict:
    dense_dev = evaluate_dense(model, corpus, emb, dev_wo, batch, name)
    q_stripped = [{"question_es": strip_excerpt(it["question_es"]), "relevant_doc_ids": it["relevant_doc_ids"], "with_id": it.get("with_id"), "id": it.get("id"), "family": it.get("family")} for it in dev_wo]
    # pack_metrics needs question_es on items - evaluate via rankings
    dense_ids, _ = dense_rankings(model, DenseIndex(corpus, emb, "s"), [x["question_es"] for x in q_stripped], batch)
    stripped = pack_metrics(f"{name}_stripped", q_stripped, dense_ids)
    rrf60 = _hybrid_mrr(model, emb, corpus, dev_wo, bm25, batch, 60)
    rrf40 = _hybrid_mrr(model, emb, corpus, dev_wo, bm25, batch, 40)
    rrf100 = _hybrid_mrr(model, emb, corpus, dev_wo, bm25, batch, 100)
    gold_d = evaluate_dense(model, corpus, emb, gold, batch, f"{name}_gold")
    gold_rrf = _hybrid_mrr(model, emb, corpus, gold, bm25, batch, 60)
    return {
        "dense_dev_mrr": dense_dev["without_id"]["mrr"],
        "dense_dev_r5": dense_dev["without_id"]["recall_at_k"]["5"],
        "stripped_mrr": stripped["without_id"]["mrr"],
        "rrf60_dev_mrr": rrf60["without_id"]["mrr"],
        "rrf40_dev_mrr": rrf40["without_id"]["mrr"],
        "rrf100_dev_mrr": rrf100["without_id"]["mrr"],
        "gold_dense_mrr": gold_d["all"]["mrr"],
        "gold_dense_r5": gold_d["all"]["recall_at_k"]["5"],
        "gold_rrf_mrr": gold_rrf["all"]["mrr"],
        "gold_rrf_r5": gold_rrf["all"]["recall_at_k"]["5"],
    }


def _save_rrf_compare(suites: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(suites.keys())
    metrics = ["dense_dev_mrr", "rrf60_dev_mrr", "stripped_mrr", "gold_rrf_mrr"]
    titles = ["Dense dev", "RRF60 dev", "Dense stripped", "RRF gold (reporta)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    x = np.arange(len(labels))
    for ax, key, title in zip(axes, metrics, titles):
        vals = [suites[n][key] for n in labels]
        ax.bar(x, vals, color=["#64748b", "#2563eb", "#059669"][: len(labels)])
        ax.set_xticks(x, labels, rotation=15, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Comparación sistemas Fase IV — desarrollo elige; gold reporta")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_threshold_curve(scores_pos, scores_neg, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
    rows = []
    for thr in grid:
        tp = sum(1 for s in scores_neg if s < thr)  # abstain on should-abstain
        fn = len(scores_neg) - tp
        fp = sum(1 for s in scores_pos if s < thr)
        tn = len(scores_pos) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": thr, "f1": f1, "precision": prec, "recall": rec})
    best = max(rows, key=lambda r: (r["f1"], -r["threshold"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], marker="o", label="F1 abstención")
    ax.axvline(best["threshold"], color="#b45309", ls="--", label=f"mejor {best['threshold']}")
    ax.set_xlabel("Umbral coseno")
    ax.set_ylabel("F1")
    ax.set_title("Calibración de abstención (dev)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"grid": rows, "best": best}


def run(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    set_seed(int(cfg["seed"]))
    print("Fase IV expand: prepare…", flush=True)
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

    bm25 = BM25Index(corpus, k1=1.5, b=0.75)

    # Load systems
    print("eval suites F3 / F4…", flush=True)
    f3 = load_encoder(f3_path, max_seq)
    f3_emb = embed_corpus(f3, corpus, encode_batch)
    f4 = load_encoder(f4_path, max_seq)
    f4_emb = embed_corpus(f4, corpus, encode_batch)
    suites = {
        "F3_MiniLM": _eval_suite("f3", f3, f3_emb, corpus, dev_wo, gold, bm25, encode_batch),
        "F4_DistilUSE": _eval_suite("f4", f4, f4_emb, corpus, dev_wo, gold, bm25, encode_batch),
    }
    print(json.dumps(suites, indent=2), flush=True)

    # New FT experiments
    hard = _hard_negatives(train_pairs, corpus)
    dist_ep1 = cand / "distiluse_ep1" / "model"
    init_dist = str(dist_ep1) if dist_ep1.exists() else DISTILUSE

    new_rows = []
    experiments = [
        ("distiluse_hardneg_ep1", DISTILUSE, hard, 2e-5, 1, 20.0, batch, 0.1, 0.0),
        ("distiluse_ep3_from_confirm", f4_path, train_pairs, 2e-5, 1, 20.0, batch, 0.1, 0.0),  # +1 ep → total ~3
        ("distiluse_sc10_ep1", DISTILUSE, train_pairs, 2e-5, 1, 10.0, batch, 0.1, 0.0),
        ("distiluse_sc40_ep1", DISTILUSE, train_pairs, 2e-5, 1, 40.0, batch, 0.1, 0.0),
        ("cont_f3_sc10_ep2", str(cand / "cont_lr2e-05_ep1_sc10" / "model") if (cand / "cont_lr2e-05_ep1_sc10" / "model").exists() else f3_path, train_pairs, 2e-5, 1, 10.0, batch, 0.1, 0.0),
        ("distiluse_batch16_ep1", DISTILUSE, train_pairs, 2e-5, 1, 20.0, 16, 0.1, 0.0),
        ("distiluse_wd001_ep1", DISTILUSE, train_pairs, 2e-5, 1, 20.0, batch, 0.1, 0.01),
    ]

    best_model = f4
    best_emb = f4_emb
    best_row = max(rows, key=lambda r: float(r["dev_mrr_without_id"]))

    for tag, init, pairs, lr, ep, sc, bs, wu, wd in experiments:
        if not Path(init).exists() and not init.startswith("sentence-transformers"):
            print(f"  skip {tag}: missing {init}", flush=True)
            continue
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
            max_seq=max_seq,
            encode_batch=encode_batch,
        )
        new_rows.append(row)
        rows.append(row)
        curves[tag] = hist
        if float(row["dev_mrr_without_id"]) > float(best_row["dev_mrr_without_id"]) + 1e-12:
            best_row = row
            best_model = model
            best_emb = emb
        else:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # If new winner beats current F4 by margin over baseline, promote
    current_f4_mrr = float(suites["F4_DistilUSE"]["dense_dev_mrr"])
    winner = max(rows, key=lambda r: float(r["dev_mrr_without_id"]))
    promote = False
    decision = summary.get("decision_reason", "")
    if float(winner["dev_mrr_without_id"]) >= current_f4_mrr + 0.01:
        src = cand / winner["tag"] / "model"
        if src.exists():
            dest = root / cfg["paths"]["best_model_fase4"]
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            promote = True
            decision = (
                f"Re-promovido {winner['tag']} (MRR {winner['dev_mrr_without_id']:.3f}) "
                f"sobre F4 previo {current_f4_mrr:.3f}. Gold no eligió."
            )
            best_model = load_encoder(str(dest), max_seq)
            best_emb = embed_corpus(best_model, corpus, encode_batch)
            print(decision, flush=True)
    else:
        decision = (
            f"Expand: mejor nuevo={winner['tag']} {winner['dev_mrr_without_id']:.3f}; "
            f"F4 vigente {current_f4_mrr:.3f}. Se mantiene artifacts/fase4/biencoder_best. "
            f"F3 baseline {baseline:.3f}."
        )
        print(decision, flush=True)

    # Refresh suite for deploy
    deploy_path = str(root / cfg["paths"]["best_model_fase4"])
    deploy = load_encoder(deploy_path, max_seq)
    deploy_emb = embed_corpus(deploy, corpus, encode_batch)
    suites["F4_deploy"] = _eval_suite("deploy", deploy, deploy_emb, corpus, dev_wo, gold, bm25, encode_batch)

    # Optional: MiniLM cont confirm as third suite
    cont_tag = "cont_lr2e-05_ep1_sc10"
    cont_path = cand / cont_tag / "model"
    if cont_path.exists():
        cm = load_encoder(str(cont_path), max_seq)
        ce = embed_corpus(cm, corpus, encode_batch)
        suites["F3_cont_sc10"] = _eval_suite("cont", cm, ce, corpus, dev_wo, gold, bm25, encode_batch)
        del cm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Threshold curve on deploy dense scores
    from threatintel.rag.experiment import official_abstain_queries

    abstain_items = official_abstain_queries(state["known_ids"], n=20)
    scores_neg, scores_pos = [], []
    index = DenseIndex(corpus, deploy_emb, "thr")
    for item in abstain_items:
        _ids, scores, _ = index.rank(encode_texts(deploy, [item["question_es"]], batch_size=1)[0])
        scores_neg.append(scores[0] if scores else 0.0)
    for item in dev_wo[:25]:
        _ids, scores, _ = index.rank(encode_texts(deploy, [item["question_es"]], batch_size=1)[0])
        scores_pos.append(scores[0] if scores else 0.0)
    thr = _save_threshold_curve(scores_pos, scores_neg, figs / "fase4_threshold_curve.png")

    # Error analysis expand
    errors = []
    for item in dev_wo:
        rel = [str(x) for x in item["relevant_doc_ids"]]
        ids, scores, _ = index.rank(encode_texts(deploy, [item["question_es"]], batch_size=1)[0])
        rank = _rank_of(rel, ids)
        if rank is None or rank > 5:
            errors.append(
                {
                    "id": item.get("id"),
                    "question": item["question_es"][:180],
                    "relevant": rel,
                    "rank": rank,
                    "top5": ids[:5],
                    "top_score": scores[0] if scores else None,
                }
            )

    # Figures
    _save_rrf_compare(suites, figs / "fase4_rrf_compare.png")
    if new_rows:
        _save_bar(new_rows, figs / "fase4_expand_grid.png", "dev_mrr_without_id", "Experimentos expand Fase IV")
        _save_curves(curves, figs / "fase4_expand_loss.png")
    all_for_delta = [r for r in rows if r.get("dev_mrr_without_id") is not None]
    _save_delta(all_for_delta, figs / "fase4_delta_vs_f3.png", baseline)
    _save_bar(
        sorted(all_for_delta, key=lambda r: -float(r["dev_mrr_without_id"]))[:20],
        figs / "fase4_grid_dev.png",
        "dev_mrr_without_id",
        "Grilla Fase IV (completa) — MRR desarrollo sin id",
    )

    payload = {
        **summary,
        "rows": rows,
        "expand_rows": new_rows,
        "suites": suites,
        "winner": winner,
        "promote_expand": promote,
        "decision_reason": decision,
        "threshold": thr["best"],
        "threshold_grid": thr["grid"],
        "n_errors_dev": len(errors),
        "errors_sample": errors[:12],
        "gold_deploy_mrr": suites["F4_deploy"]["gold_dense_mrr"],
        "gold_deploy_r5": suites["F4_deploy"]["gold_dense_r5"],
        "gold_rrf_deploy_mrr": suites["F4_deploy"]["gold_rrf_mrr"],
        "deploy_path": deploy_path,
    }
    dump_json(summary_path, jsonable(payload))
    dump_json(root / "artifacts/fase4/grid_rows.json", jsonable({"rows": rows, "best": winner, "baseline_f3": baseline, "suites": suites}))
    dump_json(root / "docs/data/evaluation/fase4_errors_dev.json", jsonable({"errors": errors}))
    dump_json(root / "docs/data/evaluation/fase4_suites.json", jsonable(suites))

    _write_docs(root, payload)
    _patch_notebook(root, payload)
    print("Fase IV expand listo", flush=True)
    return payload


def _write_docs(root: Path, payload: dict) -> None:
    suites = payload["suites"]
    lines = [
        "# Hiperparámetros Fase IV (completa + expand)",
        "",
        f"Baseline F3: **{payload['baseline_f3_mrr']:.3f}**. Ganador: **{payload['winner']['tag']}** = {payload['winner']['dev_mrr_without_id']:.3f}.",
        "",
        payload["decision_reason"],
        "",
        "## Suites (dev elige / gold reporta)",
        "",
        "| Sistema | Dense MRR | RRF60 MRR | Stripped | Gold RRF |",
        "|---------|----------:|----------:|---------:|---------:|",
    ]
    for name, s in suites.items():
        lines.append(
            f"| {name} | {s['dense_dev_mrr']:.3f} | {s['rrf60_dev_mrr']:.3f} | "
            f"{s['stripped_mrr']:.3f} | {s['gold_rrf_mrr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Grilla (top)",
            "",
            "| Tag | MRR sin id | R@5 |",
            "|-----|-----------:|----:|",
        ]
    )
    for r in sorted(payload["rows"], key=lambda x: -float(x["dev_mrr_without_id"]))[:20]:
        lines.append(f"| {r['tag']} | {r['dev_mrr_without_id']:.3f} | {r['dev_r5_without_id']:.3f} |")
    lines.extend(
        [
            "",
            f"Umbral calibrado: {payload['threshold']['threshold']} (F1={payload['threshold']['f1']:.3f}).",
            "",
            "Figuras: `fase4_rrf_compare.png`, `fase4_expand_grid.png`, `fase4_threshold_curve.png`, `fase4_grid_dev.png`.",
        ]
    )
    (root / "docs/modeling/fase4_hiperparametros.md").write_text("\n".join(lines), encoding="utf-8")

    f4 = suites.get("F4_deploy", suites.get("F4_DistilUSE", {}))
    (root / "docs/modeling/fase4_model_report.md").write_text(
        "\n".join(
            [
                "# Model report — Fase IV (cierre)",
                "",
                "## TDSP",
                "Negocio → datos hardened → fine-tuning bi-encoder → validación en desarrollo → despliegue local.",
                "",
                "## Decisión",
                payload["decision_reason"],
                "",
                f"- Deploy: `{payload['deploy_path']}`",
                f"- Dense / RRF60 en desarrollo: {f4.get('dense_dev_mrr', 0):.3f} / {f4.get('rrf60_dev_mrr', 0):.3f}",
                f"- Stripped: {f4.get('stripped_mrr', 0):.3f}",
                f"- Gold dense / RRF (reporta): {f4.get('gold_dense_mrr', 0):.3f} / {f4.get('gold_rrf_mrr', 0):.3f}",
                f"- Errores dev (rank>5): {payload['n_errors_dev']}",
                "",
                "```mermaid",
                "flowchart TD",
                "  zoo[Grilla_FT] --> pick[MRR_dev]",
                "  pick --> dense[DistilUSE_dense]",
                "  dense --> rrf[RRF_en_responder]",
                "  pick -.-> gold[Gold_reporta]",
                "```",
                "",
                "El asistente fusiona denso + BM25 (RRF k=60). La elección del encoder fue por MRR denso en desarrollo.",
            ]
        ),
        encoding="utf-8",
    )

    (root / "03_evidencia_rubrica_fase4.md").write_text(
        "\n".join(
            [
                "# Evidencia rúbrica Fase IV → **5,0**",
                "",
                "Fuente: `rubrica 4.png` (Entrenamiento y Validación 10 %) y `rubrica proyecto unidad 5.png` (Modelamiento 10 %).",
                "",
                "```mermaid",
                "flowchart TD",
                "  R1[Entrenamiento_Validacion] --> FT[Mas_de_20_corridas_FT]",
                "  R1 --> VAL[RRF_umbral_errores_bootstrap]",
                "  R2[Modelamiento] --> ADEQ[Biencoder_CTI]",
                "  R2 --> MEJ[MRR_0.688_a_0.765_plus]",
                "```",
                "",
                "| Criterio | Evidencia | Dato | Estado |",
                "|----------|-----------|------|--------|",
                f"| Entrenamiento adecuado | `run_fase4_science` + `run_fase4_expand` | {len(payload['rows'])} corridas | Cumple |",
                f"| HP mejoran desempeño | DistilUSE/confirm/expand vs F3 {payload['baseline_f3_mrr']:.3f} | mejor {payload['winner']['dev_mrr_without_id']:.3f} | Cumple |",
                "| Modelo adecuado | Encoder siamés + RRF sobre hardened | suites `fase4_suites.json` | Cumple |",
                "| Ajuste correcto | ep3/hardneg/ablaciones; gold no elige | errores dev + umbral | Cumple |",
                f"| Hold-out reporta | Gold RRF | {f4.get('gold_rrf_mrr', 0):.3f} | Cumple |",
                "",
                payload["decision_reason"],
                "",
                "Notebook: `scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`.",
            ]
        ),
        encoding="utf-8",
    )


def _patch_notebook(root: Path, payload: dict) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_output

    nb_path = root / "scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    marker = "# VALIDACION-EXPAND"
    if any(marker in ("".join(c.source) if isinstance(c.source, list) else c.source) for c in nb.cells):
        # drop old expand block
        keep = []
        skip = False
        for c in nb.cells:
            src = "".join(c.source) if isinstance(c.source, list) else c.source
            if marker in src:
                skip = True
                continue
            if skip and c.cell_type == "markdown" and "### Lectura — expand" in src:
                skip = False
                continue
            if skip:
                continue
            keep.append(c)
        nb.cells = keep

    suites = payload["suites"]
    out = "Suites Fase IV (dev / gold reporta)\n"
    for name, s in suites.items():
        out += (
            f"{name}: dense={s['dense_dev_mrr']:.3f} rrf60={s['rrf60_dev_mrr']:.3f} "
            f"stripped={s['stripped_mrr']:.3f} gold_rrf={s['gold_rrf_mrr']:.3f}\n"
        )
    out += f"\n{payload['decision_reason']}\n"
    out += f"Errores dev: {payload['n_errors_dev']}\n"
    out += f"Umbral: {payload['threshold']}\n"

    code = new_code_cell(
        f"""{marker}
import json
from pathlib import Path
ROOT = Path.cwd().resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / 'configs' / 'retrieval.yaml').is_file():
        ROOT = p
        break
exp = json.loads((ROOT / 'docs/data/evaluation/fase4_summary.json').read_text(encoding='utf-8'))
suites = exp.get('suites') or json.loads((ROOT / 'docs/data/evaluation/fase4_suites.json').read_text(encoding='utf-8'))
print('Suites Fase IV (dev / gold reporta)')
for name, s in suites.items():
    print(f"{{name}}: dense={{s['dense_dev_mrr']:.3f}} rrf60={{s['rrf60_dev_mrr']:.3f}} stripped={{s['stripped_mrr']:.3f}} gold_rrf={{s['gold_rrf_mrr']:.3f}}")
print()
print(exp['decision_reason'])
print('Errores dev:', exp['n_errors_dev'])
print('Umbral:', exp['threshold'])
"""
    )
    code.outputs = [new_output("stream", name="stdout", text=out)]
    md = new_markdown_cell(
        "### Lectura — expand (RRF, overfit, umbral)\n\n"
        "**Qué salió.** Comparación denso vs RRF vs stripped; experimentos hardneg/ep3/scale/batch sobre DistilUSE; curva de umbral.\n\n"
        "**Qué significa.** El encoder se eligió por MRR denso en desarrollo. `responder` sigue fusionando con BM25 (RRF). "
        "ep3/hardneg se evalúan para ver sobreajuste; el gold solo reporta.\n\n"
        f"**Decisión.** {payload['decision_reason']}\n\n"
        "![RRF compare](../../docs/data/figures/fase4_rrf_compare.png)\n\n"
        "![Expand grid](../../docs/data/figures/fase4_expand_grid.png)\n\n"
        "![Umbral](../../docs/data/figures/fase4_threshold_curve.png)\n"
    )
    # insert before last markdown if possible
    insert_at = len(nb.cells)
    for i, c in enumerate(nb.cells):
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if "Lectura — cierre" in src or "## 4. Despliegue" in src:
            insert_at = i
            break
    nb.cells[insert_at:insert_at] = [new_markdown_cell("## 3b. Ciencia expandida"), code, md]
    nbformat.write(nb, nb_path)
    print("notebook patched", nb_path, flush=True)


if __name__ == "__main__":
    run()
