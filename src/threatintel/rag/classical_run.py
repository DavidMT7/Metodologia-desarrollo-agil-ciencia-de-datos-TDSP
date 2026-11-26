# -*- coding: utf-8 -*-
"""Corre la comparación léxica. No entrena y no escribe en biencoder_best."""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rank_bm25 import BM25Okapi

from threatintel.eval.metrics import ranking_metrics
from threatintel.rag.classical import (
    BM25_B,
    BM25_K1,
    FUSION_WEIGHTS,
    WORD_MIN_DF,
    WORD_SUBLINEAR,
    TfidfIndex,
    choose_on_dev,
    fuse_score_maps,
    strip_excerpt,
)
from threatintel.rag.experiment import dev_answerable, dump_json, jsonable, load_config
from threatintel.rag.retrieve import best_parent_order, rrf_fuse, tokenize
from threatintel.rag.run_fase3 import load_encoder, prepare


def _metrics(items: list[dict], rankings: list[list[str]]) -> dict:
    rows = [
        {"relevant": [str(doc_id) for doc_id in item.get("relevant_doc_ids") or []], "ranked": ranked}
        for item, ranked in zip(items, rankings)
    ]
    return ranking_metrics(rows)


def _rank_bm25(doc_ids: np.ndarray, engine: BM25Okapi, query: str) -> tuple[list[str], dict[str, float]]:
    scores = np.asarray(engine.get_scores(tokenize(query)), dtype=float)
    ids, values, _chunks = best_parent_order(doc_ids, scores)
    return ids, dict(zip(ids, values))


def _dense_rankings(model, matrix: np.ndarray, doc_ids: np.ndarray, queries: list[str]) -> list[list[str]]:
    if not queries:
        return []
    vectors = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    rankings = []
    for vector in vectors:
        ids, _scores, _chunks = best_parent_order(doc_ids, matrix @ vector)
        rankings.append(ids)
    return rankings


def _rrf(dense_ids: list[str], bm25_ids: list[str]) -> list[str]:
    fused = rrf_fuse([dense_ids[:100], bm25_ids[:100]], k=60, top_n=100)
    return [doc_id for doc_id, _score in fused]


def _bar_labels(rows: list[dict]) -> list[str]:
    return [row["label"] for row in rows]


def save_figure(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = _bar_labels(rows)
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].bar(x - width / 2, [row["dev_mrr_pasted"] for row in rows], width, label="Con extracto pegado")
    axes[0].bar(x + width / 2, [row["dev_mrr_stripped"] for row in rows], width, label="Sin extracto")
    axes[0].set_title("Desarrollo sin identificador")
    axes[0].set_ylabel("MRR")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[1].bar(x, [row["gold_mrr"] for row in rows], color="#4c78a8")
    axes[1].set_title("Gold, una sola vez. No elige")
    axes[1].set_ylabel("MRR")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(path: Path, payload: dict) -> None:
    decision = payload["decision"]
    lines = [
        "# Modelos clásicos frente al bi-encoder",
        "",
        "La pregunta de desarrollo sin identificador pega, entre comillas angulares, un extracto de la descripción en inglés. "
        "Esa comparación infla a BM25 y a TF-IDF. Aquí se quita ese extracto y se vuelve a medir. "
        "El gold no entra en la elección.",
        "",
        "## Hiperparámetros elegidos en desarrollo sin extracto",
        "",
        f"- BM25: k1={payload['bm25_selected']['k1']}, b={payload['bm25_selected']['b']}, "
        f"MRR {payload['bm25_selected']['mrr_stripped']:.3f}.",
        f"- TF-IDF de palabras: min_df={payload['tfidf_word_selected']['min_df']}, "
        f"sublinear_tf={payload['tfidf_word_selected']['sublinear_tf']}, n-gramas 1–2, "
        f"MRR {payload['tfidf_word_selected']['mrr_stripped']:.3f}.",
        f"- TF-IDF de caracteres: `char_wb` 3–5, min_df=1, sublinear_tf=True, "
        f"MRR {payload['tfidf_char']['mrr_stripped']:.3f}.",
        f"- Fusión lineal: peso {payload['linear_selected']['weight']} sobre BM25 y el resto sobre "
        f"{payload['best_tfidf_name']}, MRR {payload['linear_selected']['mrr_stripped']:.3f}.",
        "",
        "El bi-encoder no se reentrenó. Se cargó `artifacts/fase3/biencoder_best` y se codificó el índice de nuevo, "
        "sin escribir en esa carpeta. La fusión RRF usa k=60 y el BM25 publicado (k1=1,5, b=0,75), no el ganador de esta grilla.",
        "",
        "## Desarrollo sin identificador",
        "",
        "| Sistema | MRR con extracto | MRR sin extracto | R@5 sin extracto |",
        "|---------|-----------------:|-----------------:|-----------------:|",
    ]
    for row in payload["report_rows"]:
        lines.append(
            f"| {row['label']} | {row['dev_mrr_pasted']:.3f} | {row['dev_mrr_stripped']:.3f} | {row['dev_r5_stripped']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Gold, reportado una vez",
            "",
            "| Sistema | MRR | R@5 |",
            "|---------|----:|----:|",
        ]
    )
    for row in payload["report_rows"]:
        lines.append(f"| {row['label']} | {row['gold_mrr']:.3f} | {row['gold_r5']:.3f} |")
    lines.extend(
        [
            "",
            "## Decisión",
            "",
            decision["reason"],
            "",
            "Figura: `docs/data/figures/fase3_clasicos_vs_dl.png`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _insert_notebook(root: Path, payload: dict, figure: Path) -> None:
    import nbformat

    notebook_path = root / "scripts" / "modeling" / "Proyecto_Fase3_RAG_Asistente.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    marker = "# CLASICOS-VS-DL"
    if any(marker in cell.source for cell in nb.cells):
        return
    decision = payload["decision"]
    code = (
        "import json\n"
        "from pathlib import Path\n\n"
        "# CLASICOS-VS-DL\n"
        "payload = json.loads(Path('../../docs/data/evaluation/fase3_classical.json').read_text(encoding='utf-8'))\n"
        "print(payload['decision']['reason'])\n"
        "print()\n"
        "print(f\"{'sistema':<22} {'con extracto':>12} {'sin extracto':>12} {'gold':>8}\")\n"
        "for row in payload['report_rows']:\n"
        "    print(\n"
        "        f\"{row['label']:<22} {row['dev_mrr_pasted']:12.3f} \"\n"
        "        f\"{row['dev_mrr_stripped']:12.3f} {row['gold_mrr']:8.3f}\"\n"
        "    )\n"
    )
    reading = (
        "### Lectura — un modelo clásico, sin el extracto pegado\n\n"
        "La pregunta de desarrollo sin identificador incluía un trozo de la descripción en inglés. "
        "Con ese trozo, el modelo léxico recupera por las palabras compartidas. "
        "Al quitarlo, la métrica mide la pregunta en español contra el documento en inglés.\n\n"
        f"{decision['reason']}\n\n"
        "La tabla de arriba es desarrollo sin identificador. La columna gold se calculó después y no cambió la elección. "
        "Los hiperparámetros de cada familia están en `docs/modeling/modelos_clasicos.md`.\n\n"
        "![MRR con el extracto pegado, sin él, y el gold que no vota](attachment:fase3_clasicos_vs_dl.png)\n\n"
        "*MRR con el extracto pegado, sin él, y el gold que no vota.*\n"
    )
    code_cell = nbformat.v4.new_code_cell(code)
    code_cell.outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stdout",
            text=_stdout(payload),
        )
    ]
    md_cell = nbformat.v4.new_markdown_cell(reading)
    md_cell["attachments"] = {
        "fase3_clasicos_vs_dl.png": {"image/png": base64.b64encode(figure.read_bytes()).decode("ascii")}
    }
    insert_at = len(nb.cells)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "¿La Fase III cumple el objetivo?" in cell.source:
            insert_at = index
            break
    nb.cells[insert_at:insert_at] = [code_cell, md_cell]
    deployed_key = decision["winner_system"] if decision["deploy"] == "classical" else "hybrid_rrf"
    deployed = next(row for row in payload["report_rows"] if row["system"] == deployed_key)
    mrr = f"{float(deployed['gold_mrr']):.3f}".replace(".", ",")
    recall = f"{float(deployed['gold_r5']):.3f}".replace(".", ",")
    old = "MRR 0,961 y Recall@5 1,000"
    new = f"MRR {mrr} y Recall@5 {recall}"
    for cell in nb.cells:
        if old in cell.source:
            cell.source = cell.source.replace(old, new)
    nbformat.write(nb, notebook_path)


def _stdout(payload: dict) -> str:
    lines = [payload["decision"]["reason"], ""]
    lines.append(f"{'sistema':<22} {'con extracto':>12} {'sin extracto':>12} {'gold':>8}")
    for row in payload["report_rows"]:
        lines.append(
            f"{row['label']:<22} {row['dev_mrr_pasted']:12.3f} "
            f"{row['dev_mrr_stripped']:12.3f} {row['gold_mrr']:8.3f}"
        )
    return "\n".join(lines) + "\n"


def _update_phase_iv(root: Path, payload: dict) -> None:
    decision = payload["decision"]
    note = (
        "\n## Comparación léxica, ya corrida\n\n"
        f"{decision['reason']}\n\n"
        "Detalle y grilla: `docs/modeling/modelos_clasicos.md`. "
        "El gold de esa corrida no eligió el sistema.\n"
    )
    entrada = root / "Fase_IV" / "01_entrada_congelada.md"
    text = entrada.read_text(encoding="utf-8")
    marker = "## Comparación léxica, ya corrida"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    entrada.write_text(text.rstrip() + "\n" + note, encoding="utf-8")
    if decision["deploy"] != "classical":
        return
    fase = root / "Fase_IV" / "00_fase_final.md"
    fase_text = fase.read_text(encoding="utf-8")
    fase_text = fase_text.replace(
        "Se despliega el bi-encoder de dos épocas, learning rate `1e-4`, InfoNCE con escala 20, fusionado con BM25 (`k1=1,5`, `b=0,75`, RRF con k=60).",
        f"Se despliega el modelo léxico `{decision['winner_system']}`, elegido en desarrollo sin extracto. "
        "El bi-encoder de dos épocas queda como control y no se borra.",
    )
    fase.write_text(fase_text, encoding="utf-8")


def run_comparison(root: Path | None = None) -> dict:
    root, _cfg = load_config(root)
    print("Compuerta de linaje y corpus hardened…")
    state = prepare(root)
    corpus = state["corpus"].reset_index(drop=True)
    doc_ids = corpus["doc_id"].to_numpy()
    dev_items = [item for item in dev_answerable(state["splits"]["dev_items"]) if not item.get("with_id")]
    gold_items = [item for item in state["gold"] if item.get("relevant_doc_ids")]
    pasted = [item["question_es"] for item in dev_items]
    stripped = [strip_excerpt(question) for question in pasted]
    gold_queries = [item["question_es"] for item in gold_items]
    print(f"Desarrollo sin identificador: {len(dev_items)}. Gold con documento: {len(gold_items)}.")

    tokenized = [tokenize(text) for text in corpus["text"].astype(str).tolist()]
    bm25_rows = []
    bm25_engines = {}
    for k1 in BM25_K1:
        for b in BM25_B:
            engine = BM25Okapi(tokenized, k1=float(k1), b=float(b))
            tag = f"bm25_k1={k1}_b={b}"
            stripped_rank = []
            pasted_rank = []
            for query_s, query_p in zip(stripped, pasted):
                stripped_rank.append(_rank_bm25(doc_ids, engine, query_s)[0])
                pasted_rank.append(_rank_bm25(doc_ids, engine, query_p)[0])
            m_s = _metrics(dev_items, stripped_rank)
            m_p = _metrics(dev_items, pasted_rank)
            row = {
                "system": tag,
                "family": "bm25",
                "k1": float(k1),
                "b": float(b),
                "mrr_stripped": m_s["mrr"],
                "r5_stripped": m_s["recall_at_k"]["5"],
                "mrr_pasted": m_p["mrr"],
            }
            bm25_rows.append(row)
            bm25_engines[tag] = engine
            print(f"  {tag} sin extracto={m_s['mrr']:.3f} con extracto={m_p['mrr']:.3f}")
    bm25_best = max(bm25_rows, key=lambda row: float(row["mrr_stripped"]))

    def _fit_tfidf(name, analyzer, ngram_range, min_df, sublinear):
        index = TfidfIndex(
            corpus,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear,
        )
        stripped_rank = [index.rank(query)[0] for query in stripped]
        pasted_rank = [index.rank(query)[0] for query in pasted]
        m_s = _metrics(dev_items, stripped_rank)
        m_p = _metrics(dev_items, pasted_rank)
        print(f"  {name} sin extracto={m_s['mrr']:.3f} con extracto={m_p['mrr']:.3f}")
        return index, {
            "system": name,
            "family": "tfidf",
            "analyzer": analyzer,
            "min_df": int(min_df),
            "sublinear_tf": bool(sublinear),
            "mrr_stripped": m_s["mrr"],
            "r5_stripped": m_s["recall_at_k"]["5"],
            "mrr_pasted": m_p["mrr"],
        }

    word_rows = []
    word_indexes = {}
    for min_df in WORD_MIN_DF:
        for sublinear in WORD_SUBLINEAR:
            name = f"tfidf_word_df{min_df}_sub{int(sublinear)}"
            index, row = _fit_tfidf(name, "word", (1, 2), min_df, sublinear)
            word_rows.append(row)
            word_indexes[name] = index
    word_best = max(word_rows, key=lambda row: float(row["mrr_stripped"]))
    char_index, char_row = _fit_tfidf("tfidf_char_wb_3_5", "char_wb", (3, 5), 1, True)
    tfidf_best = max([word_best, char_row], key=lambda row: float(row["mrr_stripped"]))
    tfidf_index = word_indexes[tfidf_best["system"]] if tfidf_best["system"] in word_indexes else char_index
    bm25_engine = bm25_engines[bm25_best["system"]]

    def _maps(rank_one, queries):
        maps = []
        for query in queries:
            _ids, scores = rank_one(query)
            maps.append(scores)
        return maps

    bm25_maps_s = _maps(lambda query: _rank_bm25(doc_ids, bm25_engine, query), stripped)
    bm25_maps_p = _maps(lambda query: _rank_bm25(doc_ids, bm25_engine, query), pasted)

    def _pair(index, query):
        ids, scores, _chunks = index.rank(query)
        return ids, dict(zip(ids, scores))

    tfidf_maps_s = [_pair(tfidf_index, query)[1] for query in stripped]
    tfidf_maps_p = [_pair(tfidf_index, query)[1] for query in pasted]

    linear_rows = []
    for weight in FUSION_WEIGHTS:
        stripped_rank = [
            fuse_score_maps(left, right, weight) for left, right in zip(bm25_maps_s, tfidf_maps_s)
        ]
        pasted_rank = [fuse_score_maps(left, right, weight) for left, right in zip(bm25_maps_p, tfidf_maps_p)]
        m_s = _metrics(dev_items, stripped_rank)
        m_p = _metrics(dev_items, pasted_rank)
        row = {
            "system": f"linear_w={weight}",
            "family": "linear",
            "weight": float(weight),
            "mrr_stripped": m_s["mrr"],
            "r5_stripped": m_s["recall_at_k"]["5"],
            "mrr_pasted": m_p["mrr"],
        }
        linear_rows.append(row)
        print(f"  {row['system']} sin extracto={m_s['mrr']:.3f}")
    linear_best = max(linear_rows, key=lambda row: float(row["mrr_stripped"]))

    print("Codificando controles densos, sin entrenar…")
    neural = {}
    specs = [
        ("minilm", state["cfg"]["models"]["baseline_minilm"]),
        ("multi_frozen", state["cfg"]["models"]["biencoder"]),
        ("biencoder_ft", str(root / state["cfg"]["paths"]["best_model"])),
    ]
    published = BM25Okapi(tokenized, k1=1.5, b=0.75)
    for name, model_name in specs:
        model = load_encoder(model_name, int(state["cfg"]["max_seq_length"]))
        matrix = np.asarray(
            model.encode(
                corpus["text"].astype(str).tolist(),
                batch_size=int(state["cfg"]["encode_batch_size"]),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        neural[name] = {"model": model, "matrix": matrix}
        print(f"  {name} listo, {matrix.shape[0]} vectores")

    def _neural_pair(name, queries):
        model = neural[name]["model"]
        matrix = neural[name]["matrix"]
        dense = _dense_rankings(model, matrix, doc_ids, queries)
        bm25 = [_rank_bm25(doc_ids, published, query)[0] for query in queries]
        rrf = [_rrf(dense_ids, bm25_ids) for dense_ids, bm25_ids in zip(dense, bm25)]
        return dense, rrf

    neural_dev = {}
    for name in ("minilm", "multi_frozen", "biencoder_ft"):
        dense_s, rrf_s = _neural_pair(name, stripped)
        dense_p, rrf_p = _neural_pair(name, pasted)
        neural_dev[name] = {
            "mrr_stripped": _metrics(dev_items, dense_s)["mrr"],
            "r5_stripped": _metrics(dev_items, dense_s)["recall_at_k"]["5"],
            "mrr_pasted": _metrics(dev_items, dense_p)["mrr"],
        }
        if name == "biencoder_ft":
            neural_dev["hybrid_rrf"] = {
                "mrr_stripped": _metrics(dev_items, rrf_s)["mrr"],
                "r5_stripped": _metrics(dev_items, rrf_s)["recall_at_k"]["5"],
                "mrr_pasted": _metrics(dev_items, rrf_p)["mrr"],
            }
        print(f"  {name} sin extracto={neural_dev[name]['mrr_stripped']:.3f}")

    selection_rows = []
    for row in bm25_rows + word_rows + [char_row] + linear_rows:
        selection_rows.append(
            {"split": "dev", "family": row["family"], "system": row["system"], "mrr": row["mrr_stripped"]}
        )
    for system, family in (("biencoder_ft", "biencoder"), ("hybrid_rrf", "rrf")):
        selection_rows.append(
            {
                "split": "dev",
                "family": family,
                "system": system,
                "mrr": neural_dev[system]["mrr_stripped"],
            }
        )
    decision = choose_on_dev(selection_rows, margin=0.02)
    print(decision["reason"])

    def _gold_bm25(engine, queries):
        return [_rank_bm25(doc_ids, engine, query)[0] for query in queries]

    gold_scores = {
        bm25_best["system"]: _metrics(gold_items, _gold_bm25(bm25_engine, gold_queries)),
        word_best["system"]: _metrics(gold_items, [word_indexes[word_best["system"]].rank(q)[0] for q in gold_queries]),
        char_row["system"]: _metrics(gold_items, [char_index.rank(q)[0] for q in gold_queries]),
    }
    gold_bmaps = [_rank_bm25(doc_ids, bm25_engine, query)[1] for query in gold_queries]
    gold_tmaps = [_pair(tfidf_index, query)[1] for query in gold_queries]
    gold_linear = [
        fuse_score_maps(left, right, linear_best["weight"]) for left, right in zip(gold_bmaps, gold_tmaps)
    ]
    gold_scores[linear_best["system"]] = _metrics(gold_items, gold_linear)
    for name in ("minilm", "multi_frozen", "biencoder_ft"):
        dense, rrf = _neural_pair(name, gold_queries)
        gold_scores[name] = _metrics(gold_items, dense)
        if name == "biencoder_ft":
            gold_scores["hybrid_rrf"] = _metrics(gold_items, rrf)

    def _report(label, key, dev_row):
        gold = gold_scores[key]
        return {
            "label": label,
            "system": key,
            "dev_mrr_pasted": float(dev_row["mrr_pasted"]),
            "dev_mrr_stripped": float(dev_row["mrr_stripped"]),
            "dev_r5_stripped": float(dev_row["r5_stripped"]),
            "gold_mrr": float(gold["mrr"]),
            "gold_r5": float(gold["recall_at_k"]["5"]),
        }

    report_rows = [
        _report(f"BM25 k1={bm25_best['k1']} b={bm25_best['b']}", bm25_best["system"], bm25_best),
        _report("TF-IDF palabras", word_best["system"], word_best),
        _report("TF-IDF caracteres", char_row["system"], char_row),
        _report(f"Fusión lineal w={linear_best['weight']}", linear_best["system"], linear_best),
        _report("MiniLM", "minilm", neural_dev["minilm"]),
        _report("Multilingüe congelado", "multi_frozen", neural_dev["multi_frozen"]),
        _report("Bi-encoder 2 épocas", "biencoder_ft", neural_dev["biencoder_ft"]),
        _report("RRF publicado", "hybrid_rrf", neural_dev["hybrid_rrf"]),
    ]
    payload = {
        "decision": decision,
        "n_dev_without_id": len(dev_items),
        "n_gold": len(gold_items),
        "bm25_selected": bm25_best,
        "bm25_grid": bm25_rows,
        "tfidf_word_selected": word_best,
        "tfidf_word_grid": word_rows,
        "tfidf_char": char_row,
        "best_tfidf_name": tfidf_best["system"],
        "linear_selected": linear_best,
        "linear_grid": linear_rows,
        "report_rows": report_rows,
    }
    payload = jsonable(payload)
    out_json = root / "docs" / "data" / "evaluation" / "fase3_classical.json"
    figure = root / "docs" / "data" / "figures" / "fase3_clasicos_vs_dl.png"
    report = root / "docs" / "modeling" / "modelos_clasicos.md"
    dump_json(out_json, payload)
    save_figure(report_rows, figure)
    write_report(report, payload)
    _insert_notebook(root, payload, figure)
    _update_phase_iv(root, payload)
    print(f"Escrito {out_json}")
    return payload


if __name__ == "__main__":
    run_comparison()
