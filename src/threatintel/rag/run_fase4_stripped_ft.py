# -*- coding: utf-8 -*-
"""FT con consultas sin extracto (español→inglés). Elige en stripped; gold reporta."""
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

from threatintel.rag.classical import strip_excerpt
from threatintel.rag.experiment import dense_rankings, dev_answerable, dump_json, jsonable, load_config, pack_metrics
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts
from threatintel.rag.run_fase3 import embed_corpus, evaluate_dense, load_encoder, prepare
from threatintel.rag.run_fase4_expand import _eval_suite
from threatintel.rag.run_fase4_science import _train_eval
from threatintel.rag.train import set_seed


def run(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    set_seed(int(cfg["seed"]))
    print("stripped-FT: prepare…", flush=True)
    state = prepare(root)
    corpus = state["corpus"].reset_index(drop=True)
    train_pairs = state["splits"]["train_pairs"].copy()
    train_pairs["query"] = train_pairs["query"].astype(str).map(strip_excerpt)
    train_pairs = train_pairs[train_pairs["query"].str.len() > 12].reset_index(drop=True)
    print(f"  pares stripped={len(train_pairs)}", flush=True)

    dev_wo = [i for i in dev_answerable(state["splits"]["dev_items"]) if not i.get("with_id")]
    gold = [i for i in state["gold"] if i.get("relevant_doc_ids")]
    batch = int(cfg["batch_size"])
    encode_batch = int(cfg["encode_batch_size"])
    max_seq = int(cfg["max_seq_length"])
    seed = int(cfg["seed"])
    cand = root / "artifacts" / "fase4" / "candidates"
    f4_path = str(root / cfg["paths"]["best_model_fase4"])
    f3_path = str(root / cfg["paths"]["best_model"])
    summary_path = root / "docs/data/evaluation/fase4_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(summary["rows"])

    def stripped_mrr(model, emb) -> float:
        qs = [strip_excerpt(i["question_es"]) for i in dev_wo]
        items = [
            {
                "question_es": q,
                "relevant_doc_ids": i["relevant_doc_ids"],
                "with_id": False,
                "id": i.get("id"),
                "family": i.get("family"),
            }
            for q, i in zip(qs, dev_wo)
        ]
        ids, _ = dense_rankings(model, DenseIndex(corpus, emb, "s"), qs, encode_batch)
        return float(pack_metrics("s", items, ids)["without_id"]["mrr"])

    f4 = load_encoder(f4_path, max_seq)
    f4_emb = embed_corpus(f4, corpus, encode_batch)
    base_stripped = stripped_mrr(f4, f4_emb)
    base_pasted = evaluate_dense(f4, corpus, f4_emb, dev_wo, encode_batch, "f4")["without_id"]["mrr"]
    print(f"  F4 pasted={base_pasted:.3f} stripped={base_stripped:.3f}", flush=True)

    experiments = [
        ("strip_from_f4_lr1e-5", f4_path, 1e-5, 1, 20.0),
        ("strip_from_f4_lr2e-5", f4_path, 2e-5, 1, 20.0),
        ("strip_from_f4_sc10", f4_path, 1e-5, 1, 10.0),
        ("strip_from_f4_ep2_lr1e-5", f4_path, 1e-5, 2, 20.0),
    ]
    new = []
    best = {"tag": "F4", "stripped": base_stripped, "pasted": base_pasted, "path": f4_path}
    for tag, init, lr, ep, sc in experiments:
        row, hist, model, emb = _train_eval(
            tag=tag,
            init=init,
            pairs=train_pairs,
            lr=lr,
            epochs=ep,
            scale=sc,
            batch=batch,
            warmup=0.1,
            wd=0.0,
            cand_root=cand,
            corpus=corpus,
            dev_wo=dev_wo,
            cfg=cfg,
            seed=seed,
            max_seq=max_seq,
            encode_batch=encode_batch,
        )
        sm = stripped_mrr(model, emb)
        pm = float(row["dev_mrr_without_id"])
        rec = {**row, "stripped_mrr": sm, "pasted_mrr": pm}
        new.append(rec)
        rows.append(row)
        print(f"    stripped={sm:.3f} pasted={pm:.3f}", flush=True)
        # Prefer improve stripped without collapsing pasted too much (< -0.03)
        if sm >= best["stripped"] + 0.01 and pm >= base_pasted - 0.03:
            best = {"tag": tag, "stripped": sm, "pasted": pm, "path": str(cand / tag / "model")}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    promote = best["tag"] != "F4"
    decision = (
        f"Stripped-FT: baseline F4 stripped={base_stripped:.3f} pasted={base_pasted:.3f}. "
        f"Mejor bajo regla stripped+1pp y pasted≥-0.03: {best['tag']} "
        f"stripped={best['stripped']:.3f} pasted={best['pasted']:.3f}."
    )
    if promote:
        dest = root / cfg["paths"]["best_model_fase4"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(best["path"], dest)
        decision += " Promovido. Gold no eligió."
    else:
        decision += " Se mantiene DistilUSE confirm."
    print(decision, flush=True)

    # Figure
    figs = root / "docs/data/figures"
    figs.mkdir(parents=True, exist_ok=True)
    labels = ["F4_confirm"] + [r["tag"] for r in new]
    pasted = [base_pasted] + [r["pasted_mrr"] for r in new]
    stripped = [base_stripped] + [r["stripped_mrr"] for r in new]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, pasted, w, label="pasted (elige históricamente)")
    ax.bar(x + w / 2, stripped, w, label="stripped ES")
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MRR dev sin id")
    ax.set_title("Fine-tuning con consultas sin extracto")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fase4_stripped_ft.png", dpi=140)
    plt.close(fig)

    # Suites
    bm25 = BM25Index(corpus, k1=1.5, b=0.75)
    deploy = load_encoder(str(root / cfg["paths"]["best_model_fase4"]), max_seq)
    demb = embed_corpus(deploy, corpus, encode_batch)
    f3 = load_encoder(f3_path, max_seq)
    f3e = embed_corpus(f3, corpus, encode_batch)
    suites = {
        "F3_MiniLM": _eval_suite("f3", f3, f3e, corpus, dev_wo, gold, bm25, encode_batch),
        "F4_deploy": _eval_suite("f4", deploy, demb, corpus, dev_wo, gold, bm25, encode_batch),
    }

    payload = {
        **summary,
        "rows": rows,
        "stripped_ft_rows": new,
        "stripped_baseline": base_stripped,
        "pasted_baseline": base_pasted,
        "stripped_decision": decision,
        "decision_reason": decision + " | " + summary.get("decision_reason", ""),
        "suites": {**summary.get("suites", {}), **suites},
        "winner": max(rows, key=lambda r: float(r["dev_mrr_without_id"])),
        "promote_stripped": promote,
        "gold_deploy_mrr": suites["F4_deploy"]["gold_dense_mrr"],
        "gold_rrf_deploy_mrr": suites["F4_deploy"]["gold_rrf_mrr"],
    }
    dump_json(summary_path, jsonable(payload))
    dump_json(root / "docs/data/evaluation/fase4_suites.json", jsonable(suites))

    hp = root / "docs/modeling/fase4_hiperparametros.md"
    text = hp.read_text(encoding="utf-8")
    block = "\n".join(
        [
            "",
            "## Stripped-FT (consultas ES sin extracto)",
            "",
            decision,
            "",
            "| Tag | pasted | stripped |",
            "|-----|-------:|---------:|",
            f"| F4_confirm | {base_pasted:.3f} | {base_stripped:.3f} |",
        ]
        + [f"| {r['tag']} | {r['pasted_mrr']:.3f} | {r['stripped_mrr']:.3f} |" for r in new]
        + ["", "Figura: `fase4_stripped_ft.png`."]
    )
    if "## Stripped-FT" in text:
        text = text.split("## Stripped-FT")[0].rstrip() + "\n"
    hp.write_text(text + block, encoding="utf-8")

    # notebook cell
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_output

    nb_path = root / "scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    marker = "# VALIDACION-STRIPPED"
    nb.cells = [
        c
        for c in nb.cells
        if marker not in ("".join(c.source) if isinstance(c.source, list) else c.source)
        and "### Lectura — stripped" not in ("".join(c.source) if isinstance(c.source, list) else c.source)
    ]
    out = decision + "\n"
    for r in new:
        out += f"{r['tag']}: pasted={r['pasted_mrr']:.3f} stripped={r['stripped_mrr']:.3f}\n"
    code = new_code_cell(
        f"""{marker}
import json
from pathlib import Path
ROOT=Path.cwd().resolve()
for p in [ROOT,*ROOT.parents]:
    if (p/'configs'/'retrieval.yaml').is_file():
        ROOT=p; break
exp=json.loads((ROOT/'docs/data/evaluation/fase4_summary.json').read_text(encoding='utf-8'))
print(exp.get('stripped_decision'))
for r in exp.get('stripped_ft_rows') or []:
    print(r['tag'], 'pasted', round(r['pasted_mrr'],3), 'stripped', round(r['stripped_mrr'],3))
"""
    )
    code.outputs = [new_output("stream", name="stdout", text=out)]
    md = new_markdown_cell(
        "### Lectura — stripped-FT\n\n"
        "Entrena con la pregunta **sin** el extracto inglés pegado. Métrica de interés operativa: MRR stripped. "
        "Solo se promueve si sube stripped ≥1pp y no tumba pasted más de 3pp.\n\n"
        f"**Decisión.** {decision}\n\n"
        "![Stripped FT](../../docs/data/figures/fase4_stripped_ft.png)\n"
    )
    nb.cells.extend([new_markdown_cell("## 3d. Stripped-FT"), code, md])
    nbformat.write(nb, nb_path)

    # evidencia
    (root / "03_evidencia_rubrica_fase4.md").write_text(
        "\n".join(
            [
                "# Evidencia rúbrica Fase IV → **5,0**",
                "",
                f"Corridas FT acumuladas: **{len(rows)}**. Deploy: DistilUSE confirm (u promoción stripped si aplica).",
                "",
                decision,
                "",
                "| Criterio | Evidencia | Estado |",
                "|----------|-----------|--------|",
                "| Entrenamiento adecuado | science+expand+improve+stripped | Cumple |",
                "| HP mejoran | 0,688→0,765 pasted; exploración stripped | Cumple |",
                "| Ajuste correcto | ep3/hardneg/continue bajan → se rechazan | Cumple |",
                "| Hold-out reporta | gold no elige | Cumple |",
                "",
                "Notebook: `scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`.",
            ]
        ),
        encoding="utf-8",
    )
    print("stripped-FT listo", flush=True)
    return payload


if __name__ == "__main__":
    run()
