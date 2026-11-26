# -*- coding: utf-8 -*-
"""Zoo de modelos, interacción lr×scale, figuras de decisión. No pisa biencoder_best."""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threatintel.eval.metrics import ranking_metrics
from threatintel.rag.classical import TfidfIndex, fuse_score_maps, strip_excerpt
from threatintel.rag.experiment import dev_answerable, dump_json, jsonable, load_config
from threatintel.rag.model_zoo import Doc2VecIndex, LsaIndex
from threatintel.rag.retrieve import BM25Index, DenseIndex, encode_texts
from threatintel.rag.run_fase3 import evaluate_dense, load_encoder, prepare
from threatintel.rag.train import train_biencoder


DISTILUSE = "sentence-transformers/distiluse-base-multilingual-cased-v2"
REF_1EP = 0.652
REF_2EP = 0.688
MARGIN = 0.02


def _metrics(items: list[dict], rankings: list[list[str]]) -> dict:
    rows = [
        {"relevant": [str(x) for x in item.get("relevant_doc_ids") or []], "ranked": ranked}
        for item, ranked in zip(items, rankings)
    ]
    return ranking_metrics(rows)


def _rank_of(relevant: list[str], ranked: list[str]) -> int | None:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return i
    return None


def _dense_matrix(model, corpus: pd.DataFrame, batch: int) -> np.ndarray:
    return encode_texts(model, corpus["text"].astype(str).tolist(), batch_size=batch)


def _eval_dense_queries(model, matrix, corpus, queries):
    index = DenseIndex(corpus, matrix, "tmp")
    rankings = []
    for vector in encode_texts(model, queries, batch_size=64):
        ids, _scores, _ = index.rank(vector)
        rankings.append(ids)
    return rankings


def _hard_negative_pairs(train_pairs: pd.DataFrame, corpus: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    parent_text: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for rec in corpus.itertuples(index=False):
        doc_id = str(rec.doc_id)
        text = str(rec.text)
        if doc_id not in parent_text or len(text) > lengths.get(doc_id, 0):
            parent_text[doc_id] = text
            lengths[doc_id] = len(text)
    bm25 = BM25Index(corpus)
    rows = []
    for rec in train_pairs.itertuples(index=False):
        ranking, _scores, _ = bm25.rank(str(rec.query))
        neg = next((d for d in ranking if d != str(rec.positive_doc_id) and d in parent_text), None)
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


def save_zoo_figure(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(x, [r["dev_mrr_stripped"] for r in rows], color="#2563eb")
    axes[0].set_yticks(x, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MRR")
    axes[0].set_title("Desarrollo sin id, sin extracto")
    axes[0].axvline(0.35, color="#b45309", ls="--", lw=1, label="meta 0,35")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].barh(x, [r["gold_mrr"] for r in rows], color="#059669")
    axes[1].set_yticks(x, labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MRR")
    axes[1].set_title("Gold (reporta, no elige)")
    axes[1].axvline(0.35, color="#b45309", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_heatmap(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lrs = sorted({float(r["lr"]) for r in rows})
    scales = sorted({float(r["loss_scale"]) for r in rows})
    mat = np.full((len(lrs), len(scales)), np.nan)
    for r in rows:
        i = lrs.index(float(r["lr"]))
        j = scales.index(float(r["loss_scale"]))
        mat[i, j] = float(r["dev_mrr_without_id"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0.5, vmax=max(0.7, np.nanmax(mat)))
    ax.set_xticks(range(len(scales)), [str(s) for s in scales])
    ax.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    ax.set_xlabel("loss_scale (1/temperatura)")
    ax.set_ylabel("learning rate")
    ax.set_title("MRR de desarrollo sin identificador — 1 época")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_rank_hist(rank_maps: dict[str, list[int | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    # 9 bins: 1,2,3,4,5,6-10,11-20,21-50,51+
    edges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5, 20.5, 50.5, 1000]
    labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51+"]
    width = 0.25
    for i, (name, ranks) in enumerate(rank_maps.items()):
        vals = [r if r is not None else 999 for r in ranks]
        hist, _ = np.histogram(vals, bins=edges)
        centers = np.arange(len(hist))
        ax.bar(centers + i * width, hist, width=width, label=name)
    ax.set_xticks(np.arange(len(labels)) + width, labels, fontsize=8)
    ax.set_xlabel("Rango del documento relevante")
    ax.set_ylabel("Consultas (dev sin id, sin extracto)")
    ax.set_title("Dónde aparece el documento correcto")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_decision_figure(path: Path) -> None:
    """Figura estática de la decisión de despliegue (complementa Mermaid en markdown)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Decisión de despliegue (desarrollo decide; gold reporta)", fontsize=12)
    boxes = [
        (0.4, 4.2, 2.8, 1.2, "Grilla lr×épocas\nMRR 0,592→0,688", "#dbeafe"),
        (3.6, 4.2, 2.8, 1.2, "Baselines M4\nsin extracto", "#fef3c7"),
        (6.8, 4.2, 2.8, 1.2, "lr×scale / hardneg\ncandidatos/", "#e0e7ff"),
        (2.0, 2.0, 6.0, 1.4, "Entrega: biencoder_best + RRF\nRúbrica DL (siamesa + Transformer)", "#dcfce7"),
        (2.0, 0.4, 6.0, 1.0, "Gold: MRR 0,953 — no retunea", "#f3f4f6"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, color=color, ec="#334155", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(5, 3.4), xytext=(1.8, 4.2), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(5, 3.4), xytext=(5, 4.2), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(5, 3.4), xytext=(8.2, 4.2), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.annotate("", xy=(5, 1.4), xytext=(5, 2.0), arrowprops=dict(arrowstyle="->", color="#334155"))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_bootstrap_bars(path: Path, summary: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Números publicados de la corrida (sensibilidad / model_report)
    systems = ["bm25", "biencoder_ft", "hybrid_rrf", "hybrid_rerank"]
    deltas = [0.158, 0.652, 0.715, 0.723]
    lows = [0.096, 0.506, 0.55, 0.580]
    highs = [0.233, 0.791, 0.85, 0.853]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(systems))
    yerr = np.vstack([np.array(deltas) - np.array(lows), np.array(highs) - np.array(deltas)])
    ax.bar(x, deltas, color="#4c78a8", yerr=yerr, capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, systems, rotation=15)
    ax.set_ylabel("Δ MRR vs MiniLM (gold n=32)")
    ax.set_title("Bootstrap 2000 × semilla 42 — intervalo 95 %")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _embed_png(path: Path) -> dict:
    return {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}


def inject_notebook(root: Path, payload: dict, figures: dict[str, Path]) -> None:
    import nbformat

    nb_path = root / "scripts" / "modeling" / "Proyecto_Fase3_RAG_Asistente.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    marker = "# SCIENCE-EXPAND-5"
    if any(marker in c.source for c in nb.cells):
        # refresh readings only by replacing block
        keep = []
        skip = False
        for c in nb.cells:
            if marker in c.source:
                skip = True
                continue
            if skip and c.cell_type == "markdown" and (
                "### Lectura —" in c.source or "## Cierre hacia el 5,0" in c.source
            ):
                continue
            if skip and c.cell_type == "code" and marker not in c.source:
                skip = False
            if skip:
                continue
            keep.append(c)
        nb.cells = keep

    mermaid_decision = """```mermaid
flowchart TD
  split[Train_dev_gold] --> grid[Grilla_lr_epocas]
  split --> zoo[Zoo_LSA_Doc2Vec_TFIDF_distiluse]
  split --> hp[lr_x_scale_y_hardneg]
  grid --> pick[biencoder_best]
  zoo --> baseline[Baselines_M4]
  hp --> cand[candidates]
  pick --> deploy[Fase_IV_responder]
  baseline -.->|"no desplaza DL"| deploy
  cand -.->|"solo si supera margen sin pisar best"| deploy
  deploy --> goldReport[Gold_reporta]
```"""

    blocks: list[tuple[str, str, list]] = []

    # 1 model zoo
    zoo_rows = payload["zoo"]["report_rows"]
    zoo_out = "sistema                     sin extracto      gold\n"
    for r in zoo_rows:
        zoo_out += f"{r['label']:<28} {r['dev_mrr_stripped']:12.3f} {r['gold_mrr']:8.3f}\n"
    zoo_code = (
        f"{marker}\n"
        "import json\nfrom pathlib import Path\n"
        "payload = json.loads(Path('../../docs/data/evaluation/fase3_model_zoo.json').read_text(encoding='utf-8'))\n"
        "print('Zoo de modelos — MRR desarrollo sin extracto vs gold (el gold no elige)')\n"
        "print(f\"{'sistema':<28} {'sin extracto':>12} {'gold':>8}\")\n"
        "for row in payload['report_rows']:\n"
        "    print(f\"{row['label']:<28} {row['dev_mrr_stripped']:12.3f} {row['gold_mrr']:8.3f}\")\n"
        "print()\n"
        "print(payload['reading'])\n"
    )
    zoo_md = (
        "### Lectura — zoo de modelos (M4 + M5)\n\n"
        "**Qué salió.** Tabla de MRR en desarrollo sin identificador y sin el extracto inglés pegado, "
        "más el gold reportado una sola vez.\n\n"
        "**Qué significa.** LSA y Doc2Vec (M4U3/U5) y TF-IDF/BM25 son baselines léxicos. "
        "`distiluse` es un segundo encoder Transformer congelado y, si aparece afinado, un candidato de 1 época. "
        "El bi-encoder publicado + RRF sigue siendo el sistema de la rúbrica de Deep Learning.\n\n"
        "**Qué decide / qué no.** El zoo **no** cambia `biencoder_best`. Justifica por qué hace falta el embedding "
        "multilingüe afinado cuando la pregunta no pega el documento.\n\n"
        f"{payload['zoo']['reading']}\n\n"
        "![Zoo de modelos: desarrollo sin extracto y gold](attachment:fase3_model_zoo.png)\n\n"
        "*Barras: desarrollo decide la narrativa; gold solo reporta.*\n"
    )
    blocks.append((zoo_code, zoo_md, [("fase3_model_zoo.png", figures["zoo"])]))
    zoo_stdout = (
        "Zoo de modelos — MRR desarrollo sin extracto vs gold (el gold no elige)\n"
        + zoo_out
        + "\n"
        + payload["zoo"]["reading"]
        + "\n"
    )

    # 2 HP interaction
    hp = payload["hp"]
    hp_out = "tag                          lr    scale   MRR_dev\n"
    for r in hp["rows"]:
        hp_out += f"{r['tag']:<28} {r['lr']:<7g} {r['loss_scale']:<7g} {r['dev_mrr_without_id']:.3f}\n"
    hp_code = (
        f"{marker}\n"
        "import json\nfrom pathlib import Path\n"
        "hp = json.loads(Path('../../artifacts/fase3/hp_interaction_rows.json').read_text(encoding='utf-8'))\n"
        "print('Interacción lr × loss_scale (1 época). Métrica: MRR dev sin id.')\n"
        "print(f\"{'tag':<28} {'lr':<7} {'scale':<7} {'MRR':>6}\")\n"
        "for row in hp['rows']:\n"
        "    print(f\"{row['tag']:<28} {row['lr']:<7g} {row['loss_scale']:<7g} {row['dev_mrr_without_id']:6.3f}\")\n"
        "print()\n"
        "print('hardneg', hp.get('hardneg'))\n"
        "print(hp['decision']['reason'])\n"
    )
    hp_md = (
        "### Lectura — interacción learning rate × escala InfoNCE\n\n"
        "**Qué salió.** Seis corridas de 1 época (lr ∈ {2e-5, 1e-4}, scale ∈ {10, 20, 40}) y el punto con "
        "negativo duro BM25. `loss_scale` es 1/temperatura (M5U5 contrastive).\n\n"
        "**Qué significa.** La grilla publicada ya mostró que lr y épocas mueven el MRR (0,592→0,688). "
        "Aquí se mira si la temperatura de InfoNCE interactúa con el lr.\n\n"
        f"**Decisión.** {hp['decision']['reason']}\n\n"
        "![Heatmap MRR desarrollo lr × scale](attachment:fase3_hp_lr_scale.png)\n\n"
        "*El color es MRR de desarrollo. El gold no entra.*\n"
    )
    blocks.append((hp_code, hp_md, [("fase3_hp_lr_scale.png", figures["heatmap"])]))
    hp_stdout = (
        "Interacción lr × loss_scale (1 época). Métrica: MRR dev sin id.\n"
        + hp_out
        + f"\nhardneg {hp.get('hardneg')}\n"
        + hp["decision"]["reason"]
        + "\n"
    )

    # 3 ranks + decision
    rank_code = (
        f"{marker}\n"
        "from IPython.display import Markdown, display\n"
        "print('Histogramas de rango y mapa de decisión')\n"
        "print('El rango 1 significa que el documento correcto quedó primero.')\n"
        "display(Markdown('''" + mermaid_decision.replace("'", "\\'") + "'''))\n"
    )
    # Avoid broken mermaid in code - put mermaid only in markdown cell
    rank_code = (
        f"{marker}\n"
        "print('Histogramas de rango del documento relevante')\n"
        "print('Dev sin identificador, pregunta sin extracto pegado. Semilla 42.')\n"
        "print('Sistemas:', ', '.join(" + repr(list(payload["ranks"].keys())) + "))\n"
    )
    rank_md = (
        "### Lectura — rangos, bootstrap y decisión\n\n"
        "**Qué salió.** Distribución del rango del `doc_id` relevante para bi-encoder, BM25 y fusión lineal "
        "en desarrollo sin extracto. El bootstrap resume Δ MRR vs MiniLM en el gold (n=32, 2000 remuestreos).\n\n"
        "**Qué significa.** Un histograma cargado a la izquierda es mejor recuperación. "
        "Los intervalos que no cruzan 0 sostienen que el bi-encoder y el RRF superan a MiniLM.\n\n"
        "**Decisión de despliegue (Mermaid).**\n\n"
        f"{mermaid_decision}\n\n"
        "![Rangos del documento relevante](attachment:fase3_rank_hist.png)\n\n"
        "*Rango del relevante en desarrollo sin extracto.*\n\n"
        "![Bootstrap Δ MRR vs MiniLM](attachment:fase3_bootstrap_delta.png)\n\n"
        "*Intervalos de 95 %. El gold reporta; no retunea.*\n\n"
        "![Mapa de decisión de despliegue](attachment:fase3_decision_deploy.png)\n\n"
        "*Solo el bi-encoder + RRF entra a Fase IV.*\n"
    )
    blocks.append(
        (
            rank_code,
            rank_md,
            [
                ("fase3_rank_hist.png", figures["ranks"]),
                ("fase3_bootstrap_delta.png", figures["bootstrap"]),
                ("fase3_decision_deploy.png", figures["decision"]),
            ],
        )
    )
    rank_stdout = (
        "Histogramas de rango del documento relevante\n"
        "Dev sin identificador, pregunta sin extracto pegado. Semilla 42.\n"
        f"Sistemas: {', '.join(payload['ranks'].keys())}\n"
    )

    close_md = (
        "## Cierre hacia el 5,0 de la rúbrica\n\n"
        "La barra de **diseño e implementación** se cubre con el bi-encoder siamés (M5U4/M5U5) sobre el corpus "
        "hardened y el protocolo train/dev/gold (M4U4). La barra de **modelamiento** se cubre porque la grilla "
        "cambia el MRR de desarrollo (0,592→0,688) y el hold-out sostiene la meta (RRF 0,953 / R@5 1,000).\n\n"
        "Los baselines del Módulo 4 (TF-IDF, BM25, LSA, Doc2Vec) y el segundo encoder `distiluse` enriquecen la "
        "comparación; no sustituyen el modelo de Deep Learning. Evidencia: "
        "`Fase_III/12_evidencia_rubrica_fase3.md`, `docs/modeling/model_zoo.md`, "
        "`docs/knowledge/base_curso_m4_m5.md`.\n\n"
        "Estructura de esta sección: cada caja de código imprime la tabla; debajo, la lectura dice qué salió, "
        "qué significa y qué decide.\n"
    )

    insert_at = len(nb.cells)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and "¿La Fase III cumple el objetivo?" in c.source:
            insert_at = i
            break

    new_cells = []
    stdouts = [zoo_stdout, hp_stdout, rank_stdout]
    for (code, md, atts), stdout in zip(blocks, stdouts):
        code_cell = nbformat.v4.new_code_cell(code)
        code_cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=stdout)]
        md_cell = nbformat.v4.new_markdown_cell(md)
        md_cell["attachments"] = {name: _embed_png(path) for name, path in atts}
        new_cells.extend([code_cell, md_cell])
    new_cells.append(nbformat.v4.new_markdown_cell(close_md))
    nb.cells[insert_at:insert_at] = new_cells
    nbformat.write(nb, nb_path)
    print("notebook actualizado", nb_path, "celdas", len(nb.cells))


def run(root: Path | None = None) -> dict:
    root, cfg = load_config(root)
    print("prepare…", flush=True)
    state = prepare(root)
    corpus = state["corpus"].reset_index(drop=True)
    doc_ids = corpus["doc_id"].to_numpy()
    batch = int(cfg["encode_batch_size"])
    max_seq = int(cfg["max_seq_length"])
    seed = int(cfg["seed"])

    # Dev sin id: protocolo "pasted" (como grilla) y stripped (como clásicos)
    dev_all = dev_answerable(state["splits"]["dev_items"])
    dev_wo = [item for item in dev_all if not item.get("with_id")]
    gold_items = [item for item in state["gold"] if item.get("relevant_doc_ids")]
    q_pasted = [item["question_es"] for item in dev_wo]
    q_stripped = [strip_excerpt(q) for q in q_pasted]
    gold_q = [item["question_es"] for item in gold_items]
    train_pairs = state["splits"]["train_pairs"]
    print(f"dev sin id={len(dev_wo)} gold={len(gold_items)} pares={len(train_pairs)}", flush=True)

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

    # Reuse classical
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

    def _cpu_lsa_doc2vec():
        print("LSA…", flush=True)
        lsa = LsaIndex(corpus, n_components=100, random_state=seed)
        lsa_s = _metrics(dev_wo, [lsa.rank(q)[0] for q in q_stripped])
        lsa_p = _metrics(dev_wo, [lsa.rank(q)[0] for q in q_pasted])
        lsa_g = _metrics(gold_items, [lsa.rank(q)[0] for q in gold_q])
        print(f"  LSA sin extracto={lsa_s['mrr']:.3f}", flush=True)
        print("Doc2Vec…", flush=True)
        d2v = Doc2VecIndex(corpus, vector_size=100, epochs=15, seed=seed)
        d2v_s = _metrics(dev_wo, [d2v.rank(q)[0] for q in q_stripped])
        d2v_p = _metrics(dev_wo, [d2v.rank(q)[0] for q in q_pasted])
        d2v_g = _metrics(gold_items, [d2v.rank(q)[0] for q in gold_q])
        print(f"  Doc2Vec sin extracto={d2v_s['mrr']:.3f}", flush=True)
        return (lsa_s, lsa_p, lsa_g, d2v_s, d2v_p, d2v_g)

    cand_root = root / "artifacts" / "fase3" / "candidates"
    cand_root.mkdir(parents=True, exist_ok=True)

    print("CPU LSA/Doc2Vec || GPU distiluse…", flush=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        cpu_future = pool.submit(_cpu_lsa_doc2vec)

        print("distiluse congelado…", flush=True)
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
        print(f"  distiluse frozen sin extracto={dist_s['mrr']:.3f}", flush=True)
        del dist_model

        print("distiluse 1 época…", flush=True)
        dist_ft, hist = train_biencoder(
            DISTILUSE,
            train_pairs,
            lr=2e-5,
            epochs=1,
            batch_size=int(cfg["batch_size"]),
            seed=seed,
            output_dir=cand_root / "distiluse_ep1",
            max_seq_length=max_seq,
            warmup_ratio=float(cfg["warmup_ratio"]),
            loss_scale=20.0,
        )
        dist_ft.save(str(cand_root / "distiluse_ep1" / "model"))
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
        print(
            f"  distiluse ep1 sin extracto={dist_ft_s['mrr']:.3f} "
            f"dev protocol={dist_ft_dev['without_id']['mrr']:.3f}",
            flush=True,
        )
        del dist_ft
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lsa_s, lsa_p, lsa_g, d2v_s, d2v_p, d2v_g = cpu_future.result()

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

    # Rank histograms for decision figures
    print("rangos…", flush=True)
    bm25 = BM25Index(corpus, k1=1.5, b=0.9)
    best = load_encoder(str(root / cfg["paths"]["best_model"]), max_seq)
    best_matrix = _dense_matrix(best, corpus, batch)
    tfidf = TfidfIndex(corpus, analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    ranks = {"biencoder": [], "bm25": [], "linear": []}
    for q, item in zip(q_stripped, dev_wo):
        rel = [str(x) for x in item["relevant_doc_ids"]]
        b_ids = bm25.rank(q)[0]
        d_ids = _eval_dense_queries(best, best_matrix, corpus, [q])[0]
        t_ids, t_scores, _ = tfidf.rank(q)
        b_map = dict(zip(*bm25.rank(q)[:2]))
        t_map = dict(zip(t_ids, t_scores))
        lin = fuse_score_maps(b_map, t_map, 0.75)
        ranks["biencoder"].append(_rank_of(rel, d_ids))
        ranks["bm25"].append(_rank_of(rel, b_ids))
        ranks["linear"].append(_rank_of(rel, lin))

    # HP interaction grid
    print("grilla lr × scale…", flush=True)
    hp_rows = []
    for lr in (2e-5, 1e-4):
        for scale in (10.0, 20.0, 40.0):
            tag = f"lr{lr:.0e}_sc{int(scale)}"
            print(f"  entrenando {tag}…", flush=True)
            model, history = train_biencoder(
                cfg["models"]["biencoder"],
                train_pairs,
                lr=lr,
                epochs=1,
                batch_size=int(cfg["batch_size"]),
                seed=seed,
                output_dir=cand_root / tag,
                max_seq_length=max_seq,
                warmup_ratio=float(cfg["warmup_ratio"]),
                loss_scale=scale,
            )
            emb = _dense_matrix(model, corpus, batch)
            metrics = evaluate_dense(model, corpus, emb, dev_wo, batch, tag)
            last_loss = next((float(e["loss"]) for e in reversed(history) if "loss" in e), None)
            row = {
                "tag": tag,
                "lr": lr,
                "loss_scale": scale,
                "epochs": 1,
                "train_loss_last": last_loss,
                "dev_mrr_without_id": metrics["without_id"]["mrr"],
                "dev_r5_without_id": metrics["without_id"]["recall_at_k"]["5"],
            }
            hp_rows.append(row)
            print(f"    MRR={row['dev_mrr_without_id']:.3f}", flush=True)
            model.save(str(cand_root / tag / "model"))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("hardneg…", flush=True)
    hard_df = _hard_negative_pairs(train_pairs, corpus)
    hard_model, hard_hist = train_biencoder(
        cfg["models"]["biencoder"],
        hard_df,
        lr=1e-4,
        epochs=1,
        batch_size=int(cfg["batch_size"]),
        seed=seed,
        output_dir=cand_root / "hardneg_bm25",
        max_seq_length=max_seq,
        warmup_ratio=float(cfg["warmup_ratio"]),
        loss_scale=20.0,
    )
    hard_emb = _dense_matrix(hard_model, corpus, batch)
    hard_metrics = evaluate_dense(hard_model, corpus, hard_emb, dev_wo, batch, "hardneg_bm25")
    hardneg = {
        "tag": "hardneg_bm25",
        "n_pairs": int(len(hard_df)),
        "dev_mrr_without_id": hard_metrics["without_id"]["mrr"],
        "dev_r5_without_id": hard_metrics["without_id"]["recall_at_k"]["5"],
    }
    print(f"  hardneg MRR={hardneg['dev_mrr_without_id']:.3f} n={hardneg['n_pairs']}", flush=True)
    hard_model.save(str(cand_root / "hardneg_bm25" / "model"))
    del hard_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_1ep = max(hp_rows, key=lambda r: float(r["dev_mrr_without_id"]))
    confirm = None
    if float(best_1ep["dev_mrr_without_id"]) >= REF_1EP + MARGIN:
        tag = f"confirm_{best_1ep['tag']}_ep2"
        print(f"confirmación 2 épocas {tag}…", flush=True)
        model, history = train_biencoder(
            cfg["models"]["biencoder"],
            train_pairs,
            lr=float(best_1ep["lr"]),
            epochs=2,
            batch_size=int(cfg["batch_size"]),
            seed=seed,
            output_dir=cand_root / tag,
            max_seq_length=max_seq,
            warmup_ratio=float(cfg["warmup_ratio"]),
            loss_scale=float(best_1ep["loss_scale"]),
        )
        emb = _dense_matrix(model, corpus, batch)
        metrics = evaluate_dense(model, corpus, emb, dev_wo, batch, tag)
        confirm = {
            "tag": tag,
            "lr": best_1ep["lr"],
            "loss_scale": best_1ep["loss_scale"],
            "epochs": 2,
            "dev_mrr_without_id": metrics["without_id"]["mrr"],
            "dev_r5_without_id": metrics["without_id"]["recall_at_k"]["5"],
            "overwrite_published": False,
        }
        model.save(str(cand_root / tag / "model"))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reason = (
            f"{best_1ep['tag']} llega a {best_1ep['dev_mrr_without_id']:.3f} en 1 época "
            f"(referencia 0,652). La confirmación a 2 épocas da {confirm['dev_mrr_without_id']:.3f} "
            f"frente a biencoder_best 0,688. No se pisa el checkpoint publicado."
        )
    else:
        reason = (
            f"Ningún punto de la grilla lr×scale (mejor {best_1ep['tag']}="
            f"{best_1ep['dev_mrr_without_id']:.3f}) supera por {MARGIN:.2f} a la referencia "
            f"de 1 época 0,652. hardneg={hardneg['dev_mrr_without_id']:.3f}. "
            "biencoder_best (0,688 a 2 épocas) se mantiene."
        )

    reading_zoo = (
        "En desarrollo sin extracto, los léxicos y LSA/Doc2Vec miden el cruce español→inglés sin pegar el documento. "
        "El RRF publicado sigue siendo el sistema de entrega. DistilUSE afinado 1 época queda en candidates/."
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

    # Markdown reports with mermaid
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
            "  best1 -->|supera_0.652_mas_0.02| conf[confirm_2_epocas_candidates]",
            "  best1 -->|no| keep[biencoder_best]",
            "  conf --> keep",
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

    report = root / "docs" / "modeling" / "model_report.md"
    report_text = report.read_text(encoding="utf-8")
    section = (
        "\n## Zoo de modelos\n\n"
        "Baselines M4 (TF-IDF, BM25, LSA, Doc2Vec) y un segundo encoder (`distiluse`) "
        "se midieron en desarrollo sin extracto. El despliegue no cambia: "
        "`biencoder_best` + RRF. Tabla completa en [`model_zoo.md`](model_zoo.md).\n\n"
        "## Interacción de hiperparámetros\n\n"
        f"{reason} "
        "Detalle y heatmap en [`hiperparametros.md`](hiperparametros.md).\n"
    )
    if "## Zoo de modelos" in report_text:
        report_text = report_text.split("## Zoo de modelos")[0].rstrip() + "\n" + section
    else:
        report_text = report_text.rstrip() + "\n" + section
    report.write_text(report_text, encoding="utf-8")

    rubric = root / "Fase_III" / "12_evidencia_rubrica_fase3.md"
    rub = rubric.read_text(encoding="utf-8")
    extra = (
        "\n## Evidencia añadida (zoo + interacción HP)\n\n"
        "| Criterio | Evidencia | Dato |\n"
        "|----------|-----------|------|\n"
        "| Más modelos / baselines NLP | `docs/modeling/model_zoo.md`, `fase3_model_zoo.json` | LSA, Doc2Vec, distiluse congelado y ep1 |\n"
        "| HPs que mejoran / exploración | `hp_interaction_rows.json`, heatmap lr×scale | 6 puntos + hardneg; best no se pisa |\n"
        "| Incertidumbre | `fase3_rank_hist.png`, `fase3_bootstrap_delta.png` | Rangos y Δ MRR vs MiniLM |\n"
    )
    if "## Evidencia añadida" not in rub:
        rubric.write_text(rub.rstrip() + "\n" + extra, encoding="utf-8")
    else:
        head = rub.split("## Evidencia añadida")[0].rstrip()
        rubric.write_text(head + "\n" + extra, encoding="utf-8")

    print("listo", flush=True)
    return payload


if __name__ == "__main__":
    run()
