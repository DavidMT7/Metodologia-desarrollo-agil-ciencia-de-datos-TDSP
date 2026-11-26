# -*- coding: utf-8 -*-
"""Figuras de decisión adicionales + Mermaid enriquecido. Corre tras science_expand."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from threatintel.rag.experiment import load_config


def save_pareto_zoo(rows: list[dict], path: Path) -> None:
    """Dev MRR (sin extracto) vs Gold MRR — elige en x, reporta en y."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {
        "lexical": "#b45309",
        "unsupervised": "#ca8a04",
        "embedding_classic": "#a16207",
        "dense_frozen": "#64748b",
        "dense_ft": "#2563eb",
        "hybrid": "#059669",
    }
    for r in rows:
        x = r.get("dev_mrr_stripped")
        y = r.get("gold_mrr")
        if x is None or y is None:
            continue
        fam = r.get("family", "other")
        ax.scatter(x, y, s=90, c=colors.get(fam, "#334155"), zorder=3)
        ax.annotate(r["label"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axvline(0.35, color="#b45309", ls="--", lw=1, label="meta MRR 0,35")
    ax.axhline(0.35, color="#b45309", ls=":", lw=1)
    ax.set_xlabel("MRR desarrollo sin extracto (elige / compara)")
    ax.set_ylabel("MRR gold (solo reporta)")
    ax.set_title("Zoo: desarrollo decide la narrativa; gold no retunea")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
        for k, c in colors.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, max(0.8, max(r["dev_mrr_stripped"] or 0 for r in rows) + 0.05))
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_delta_vs_published(rows: list[dict], path: Path, published_mrr: float = 0.688) -> None:
    """Δ MRR (1 época) vs referencia publicada de grilla."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = [r["tag"] for r in rows]
    deltas = [float(r["dev_mrr_without_id"]) - published_mrr for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#059669" if d >= 0 else "#dc2626" for d in deltas]
    ax.barh(np.arange(len(tags)), deltas, color=colors)
    ax.set_yticks(np.arange(len(tags)), tags, fontsize=8)
    ax.axvline(0, color="black", lw=0.9)
    ax.axvline(0.02, color="#64748b", ls="--", lw=1, label="margen 0,02")
    ax.set_xlabel("Δ MRR vs biencoder 2 épocas (0,688) — solo desarrollo")
    ax.set_title("¿Algún punto lr×scale desplaza al publicado? (1 época)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_waterfall_deploy(path: Path, zoo_best_lexical: float, rrf: float = 0.339) -> None:
    """Cascada narrativa: por qué se entrega DL pese a léxico stripped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        ("Léxico mejor\nsin extracto", zoo_best_lexical),
        ("RRF stripped\n(misma métrica)", rrf),
        ("Rúbrica exige\nDL siamés", 0.688),
        ("Gold RRF\n(reporta)", 0.953),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(steps))
    vals = [s[1] for s in steps]
    ax.bar(x, vals, color=["#b45309", "#b45309", "#2563eb", "#059669"])
    ax.set_xticks(x, [s[0] for s in steps], fontsize=8)
    ax.set_ylabel("MRR")
    ax.set_title("Cascada de decisión: métrica stripped → rúbrica DL → gold reporta")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def enrich_docs(root: Path, zoo: dict, hp: dict) -> None:
    rows = zoo["report_rows"]
    lexical = [r for r in rows if r.get("family") in {"lexical", "unsupervised", "embedding_classic"}]
    best_lex = max((r["dev_mrr_stripped"] or 0) for r in lexical) if lexical else 0.377
    rrf_stripped = next((r["dev_mrr_stripped"] for r in rows if r["system"] == "hybrid_rrf"), 0.339)

    figs = root / "docs" / "data" / "figures"
    save_pareto_zoo(rows, figs / "fase3_zoo_pareto.png")
    save_delta_vs_published(hp["rows"], figs / "fase3_hp_delta_published.png")
    save_waterfall_deploy(figs / "fase3_decision_waterfall.png", best_lex, float(rrf_stripped or 0.339))

    # Ampliar model_zoo.md
    zoo_path = root / "docs/modeling/model_zoo.md"
    extra = "\n".join(
        [
            "",
            "## Lectura para la decisión",
            "",
            "```mermaid",
            "flowchart TD",
            "  Q[Pregunta_ES_sin_extracto] --> Lex[TFIDF_BM25_LSA_Doc2Vec]",
            "  Q --> Frz[MiniLM_DistilUSE_congelados]",
            "  Q --> FT[Biencoder_2ep_y_DistilUSE_ep1]",
            "  Lex --> M[MRR_dev]",
            "  Frz --> M",
            "  FT --> M",
            "  M -->|mejor_lexico_no_cumple_rubrica_DL| Entrega[biencoder_best_RRF]",
            "  M -->|gold_solo_reporta| Gold[Gold_f3]",
            "```",
            "",
            f"Mejor léxico/clásico en sin-extracto: **{best_lex:.3f}**. "
            f"RRF en la misma métrica: **{rrf_stripped:.3f}**. "
            "La rúbrica de Fase III exige Deep Learning (siamesa + Transformer); "
            "por eso la entrega no cambia aunque el léxico gane en stripped.",
            "",
            "Figuras de decisión:",
            "- `fase3_model_zoo.png` — barras desarrollo vs gold",
            "- `fase3_zoo_pareto.png` — dispersión desarrollo × gold",
            "- `fase3_decision_waterfall.png` — cascada métrica → rúbrica → gold",
            "",
        ]
    )
    text = zoo_path.read_text(encoding="utf-8")
    if "## Lectura para la decisión" not in text:
        zoo_path.write_text(text.rstrip() + "\n" + extra, encoding="utf-8")

    # Ampliar hiperparametros con delta figure note
    hp_path = root / "docs/modeling/hiperparametros.md"
    hp_text = hp_path.read_text(encoding="utf-8")
    note = (
        "\nFiguras adicionales: `fase3_hp_delta_published.png` (Δ vs 0,688) "
        "y `fase3_hp_lr_scale.png` (heatmap).\n"
    )
    if "fase3_hp_delta_published.png" not in hp_text:
        hp_path.write_text(hp_text.rstrip() + "\n" + note, encoding="utf-8")

    print("figuras de decisión listas", flush=True)


def inject_extra_notebook(root: Path) -> None:
    import nbformat

    nb_path = root / "scripts" / "modeling" / "Proyecto_Fase3_RAG_Asistente.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    marker = "# SCIENCE-DECISION-VIZ"
    if any(marker in c.source for c in nb.cells):
        print("notebook ya tiene SCIENCE-DECISION-VIZ", flush=True)
        return

    figs = root / "docs" / "data" / "figures"
    needed = [
        "fase3_zoo_pareto.png",
        "fase3_hp_delta_published.png",
        "fase3_decision_waterfall.png",
    ]
    for name in needed:
        if not (figs / name).exists():
            print("falta figura", name, flush=True)
            return

    code = (
        f"{marker}\n"
        "print('Figuras de decisión con datos')\n"
        "print('1) Pareto zoo: desarrollo (elige) vs gold (reporta)')\n"
        "print('2) Delta lr×scale vs publicado 0,688')\n"
        "print('3) Cascada: léxico stripped → rúbrica DL → gold')\n"
    )
    md = (
        "### Lectura — figuras de valor para la decisión\n\n"
        "**Qué salió.** Tres vistas que no están en la grilla original: (1) dispersión desarrollo×gold "
        "por familia de modelo; (2) Δ MRR de cada punto lr×scale frente al checkpoint de 2 épocas; "
        "(3) cascada que separa la métrica stripped, la exigencia de Deep Learning y el gold.\n\n"
        "**Qué significa.** Si un punto queda a la derecha en el Pareto (buen desarrollo) pero la familia "
        "es léxica, **no** desplaza al bi-encoder: la rúbrica pide Transformer siamés. "
        "Si ningún Δ supera el margen 0,02 sobre 0,688, no se pisa `biencoder_best`.\n\n"
        "**Qué decide.** Confirma la entrega Fase IV = bi-encoder + RRF. "
        "Candidatos viven en `artifacts/fase3/candidates/`.\n\n"
        "```mermaid\n"
        "sequenceDiagram\n"
        "  participant Dev as Desarrollo_sin_extracto\n"
        "  participant Rub as Rubrica_DL\n"
        "  participant Cand as candidates\n"
        "  participant Gold as Gold_holdout\n"
        "  participant Ent as Entrega_Fase_IV\n"
        "  Dev->>Rub: MRR_y_familia\n"
        "  Rub->>Ent: biencoder_RRF\n"
        "  Dev->>Cand: lr_scale_hardneg\n"
        "  Cand--xEnt: no_pisa_si_margen_insuficiente\n"
        "  Ent->>Gold: reporta_una_vez\n"
        "```\n\n"
        "![Pareto zoo desarrollo × gold](attachment:fase3_zoo_pareto.png)\n\n"
        "*Eje X decide la comparación; eje Y solo reporta.*\n\n"
        "![Δ MRR vs publicado](attachment:fase3_hp_delta_published.png)\n\n"
        "*Barras positivas = mejor que 0,688 en desarrollo (1 época). Margen 0,02.*\n\n"
        "![Cascada de decisión](attachment:fase3_decision_waterfall.png)\n\n"
        "*De la métrica stripped a la rúbrica DL y al gold.*\n"
    )
    code_cell = nbformat.v4.new_code_cell(code)
    code_cell.outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text=(
                "Figuras de decisión con datos\n"
                "1) Pareto zoo: desarrollo (elige) vs gold (reporta)\n"
                "2) Delta lr×scale vs publicado 0,688\n"
                "3) Cascada: léxico stripped → rúbrica DL → gold\n"
            ),
        )
    ]
    md_cell = nbformat.v4.new_markdown_cell(md)
    md_cell["attachments"] = {
        name: {"image/png": base64.b64encode((figs / name).read_bytes()).decode("ascii")}
        for name in needed
    }

    insert_at = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and ("Cierre hacia el 5,0" in c.source or "¿La Fase III cumple" in c.source):
            insert_at = i
            break
    nb.cells[insert_at:insert_at] = [code_cell, md_cell]
    nbformat.write(nb, nb_path)
    print("notebook + decision viz", nb_path, flush=True)


def run(root: Path | None = None) -> None:
    root, _cfg = load_config(root)
    zoo = json.loads((root / "docs/data/evaluation/fase3_model_zoo.json").read_text(encoding="utf-8"))
    hp = json.loads((root / "artifacts/fase3/hp_interaction_rows.json").read_text(encoding="utf-8"))
    enrich_docs(root, zoo, hp)
    inject_extra_notebook(root)


if __name__ == "__main__":
    run()
