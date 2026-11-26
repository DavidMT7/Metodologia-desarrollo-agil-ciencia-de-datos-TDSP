# -*- coding: utf-8 -*-
"""Reportes de modelado a partir de la corrida sobre parquets hardened."""
from __future__ import annotations

from pathlib import Path


def _n(value, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    return f"{float(value):.{digits}f}"


def _pct(value) -> str:
    if value is None:
        return "n/d"
    return f"{100 * float(value):.1f} %"


def _row(system: str, table: dict) -> str:
    block = table[system]
    return (
        f"| {system} | {_n(block['all']['mrr'])} | {_n(block['all']['recall_at_k']['5'])} | "
        f"{_n(block['without_id']['mrr'])} | {_n(block['without_id']['recall_at_k']['5'])} |"
    )


def write_modeling_docs(root: Path, summary: dict) -> None:
    modeling = root / "docs" / "modeling"
    modeling.mkdir(parents=True, exist_ok=True)
    (modeling / "baseline_models.md").write_text(_baseline(summary), encoding="utf-8")
    (modeling / "model_report.md").write_text(_final(summary), encoding="utf-8")


def _baseline(summary: dict) -> str:
    f2 = summary["fase2_reproduction"]
    archived = summary["fase2_archived"]
    lin = summary["lineage"]
    return f"""# Reporte del modelo baseline

## Descripción del modelo

El baseline de retrieval es `sentence-transformers/all-MiniLM-L6-v2`, congelado. No se entrena en esta fase. El índice de la reproducción usa el armado de Fase II (un documento por técnica, detección truncada) sobre los parquets **hardened**, no sobre `*_clean`.

El control de linaje de esta corrida: detection nulo en clean {_pct(lin['detection_null_clean'])} y en hardened {_pct(lin['detection_null_hardened'])}. Unknown en `likely_mitre_tactic` {_pct(lin['unknown_tactic'])}; en `likely_mitre_tactic_enriched` {_pct(lin['unknown_enriched'])}.

## Variables de entrada

- Consulta en español del gold (`question_es`).
- Texto hardened: CVE con táctica enriquecida, o técnica MITRE con `detection` STIX.

## Variable objetivo

Colocar el `doc_id` relevante lo más arriba posible. No hay una clase de severidad que predecir.

## Evaluación del modelo

### Métricas

MRR y Recall@k, misma definición que `src/threatintel/retrieval_baseline.py`. Semilla 42. Las preguntas sin documento relevante no entran al promedio.

### Resultados

| Corrida | n | MRR | R@1 | R@5 | R@10 |
|---------|--:|----:|----:|----:|-----:|
| Archivada en Fase II | {archived['n_queries_eval']} | {_n(archived['mrr'])} | {_n(archived['recall_at_k']['1'])} | {_n(archived['recall_at_k']['5'])} | {_n(archived['recall_at_k']['10'])} |
| Reproducción Fase III | {f2['n_queries_eval']} | {_n(f2['mrr'])} | {_n(f2['recall_at_k']['1'])} | {_n(f2['recall_at_k']['5'])} | {_n(f2['recall_at_k']['10'])} |

## Análisis

MiniLM está entrenado sobre todo en inglés y las preguntas están en español. MRR {_n(f2['mrr'])} y Recall@5 {_n(f2['recall_at_k']['5'])} son el piso. No se usan para generar texto libre.

## Conclusiones

El baseline sirve de control. El modelo de esta fase tiene que superarlo sin entrenar con los `doc_id` del gold y sin leer los parquets limpios.

## Referencias

- Reimers y Gurevych (2019), Sentence-BERT.
- `docs/data/evaluation/retrieval_baseline_metrics.json`.
- Registro E18/E16 en `Fase_III/09_registro_errores_imprecisiones_correcciones.md`.
"""


def _final(summary: dict) -> str:
    selected = summary["selected_checkpoint"]
    deployed = summary["deployed_encoder"]
    threshold = summary["threshold"]
    gold = summary["gold_table"]
    para = summary["paraphrase_table"]
    citations = summary["citations"]
    abstain = summary["abstention"]
    lin = summary["lineage"]
    lines = "\n".join(_row(name, gold) for name in summary["system_order"])
    para_lines = "\n".join(_row(name, para) for name in summary["system_order"])
    grid_lines = "\n".join(
        f"| {row['tag']} | {row['lr']} | {row['epochs']} | {_n(row['train_loss_last'])} | "
        f"{_n(row['dev_mrr_without_id'])} | {_n(row['dev_r5_without_id'])} |"
        for row in summary["grid"]
    )
    scenario_lines = "\n".join(
        f"| {row['id']} | {row['name']} | {row['route']} | {row['score']} | {row['note']} |"
        for row in summary["scenarios"]
    )
    best_hp = max(summary["grid"], key=lambda row: row["dev_mrr_without_id"] or -1)
    worst_hp = min(summary["grid"], key=lambda row: row["dev_mrr_without_id"] or 1)
    delta = (best_hp["dev_mrr_without_id"] or 0) - (worst_hp["dev_mrr_without_id"] or 0)
    hybrid = gold["hybrid_rerank"]["all"]
    meta = (hybrid["mrr"] or 0) >= 0.35 and (hybrid["recall_at_k"]["5"] or 0) >= 0.50
    return f"""# Reporte del modelo final

## Resumen ejecutivo

El modelo es un bi-encoder multilingüe (`{summary['biencoder']}`) afinado con MultipleNegativesRankingLoss sobre el índice **hardened** (detection STIX completo y táctica enriquecida). El checkpoint elegido en dev, sin ver el gold, es `{selected['tag']}` (MRR dev sin identificador {_n(selected['dev_mrr_without_id'])}).

En el hold-out del protocolo de Fase II el sistema desplegado ({deployed['name']} + RRF + re-ranker) obtiene MRR {_n(hybrid['mrr'])} y Recall@5 {_n(hybrid['recall_at_k']['5'])}. {"Cumple" if meta else "No cumple"} la meta MRR ≥ 0,35 y Recall@5 ≥ 0,50. En paráfrasis sin identificador el mismo sistema queda en MRR {_n(para['hybrid_rerank']['all']['mrr'])} y Recall@5 {_n(para['hybrid_rerank']['all']['recall_at_k']['5'])}.

Linaje de la corrida: detection nulo clean {_pct(lin['detection_null_clean'])} / hardened {_pct(lin['detection_null_hardened'])}. Unknown original {_pct(lin['unknown_tactic'])}; residual enriquecido {_pct(lin['unknown_enriched'])}. El índice no leyó `*_clean`.

Abstención ante identificadores ausentes: {_pct(abstain['official_rate'])} (n={abstain['official_n']}). Coincidencia de cita con el gold original: {_pct(citations['citation_matches_gold'])}. En paráfrasis: {_pct(citations.get('paraphrase_matches_gold'))}.

## Descripción del problema

El analista pregunta en español y el corpus está en inglés. Generar sin un índice útil inventa CVSS, KEV y detección. El fallo E18 de la Fase II mezcló el ETL limpio con el enriquecido; esta fase se niega a entrenar si eso vuelve a pasar.

No se optimiza la pérdida en dólares de Kaggle ni el accuracy de `severity_level`.

## Descripción del modelo

Torre siamesa, pooling del checkpoint base, coseno, InfoNCE con scale 20 (temperatura 0,05). No se entrena un transformer desde cero. El re-ranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` va congelado: es una ablación.

Hiperparámetros fijos: semilla {summary['seed']}, lote {summary['batch_size']}, longitud {summary['max_seq_length']}, warmup 10 %. Comparados: learning rate y épocas. La selección es el MRR de dev **sin** identificador. El mejor punto de la grilla supera al peor en {_n(delta)} ({best_hp['tag']} frente a {worst_hp['tag']}). El encoder desplegado es `{deployed['name']}` porque {deployed['reason']}.

La compuerta se abstiene si el identificador no está en el índice, o si el coseno máximo queda bajo {_n(threshold['threshold'], 2)} (F1 de dev {_n(threshold['f1'])}). La redacción rellena una plantilla en español solo con campos del documento recuperado.

## Evaluación del modelo

### Grilla en dev

| Configuración | lr | épocas | pérdida final | MRR dev sin id | R@5 dev sin id |
|---------------|---:|-------:|--------------:|---------------:|---------------:|
{grid_lines}

### Hold-out gold de Fase II

| Sistema | MRR | R@5 | MRR sin id | R@5 sin id |
|---------|----:|----:|-----------:|-----------:|
{lines}

### Paráfrasis sin identificador

Plantilla en español más extracto oficial. No entrenan ni eligen hiperparámetros. El extracto se solapa con el documento, así que BM25 puede verse fuerte sin que un analista haya parafraseado de memoria.

| Sistema | MRR | R@5 | MRR sin id | R@5 sin id |
|---------|----:|----:|-----------:|-----------:|
{para_lines}

### Citas, abstención y escenarios

| Control | Resultado |
|---------|-----------|
| Citas en contexto (gold respondido) | {_pct(citations['citation_in_context'])} (n={citations['n_answered_factual']}) |
| Citas del gold que coinciden con el documento | {_pct(citations['citation_matches_gold'])} |
| Citas de paráfrasis que coinciden con el documento | {_pct(citations.get('paraphrase_matches_gold'))} |
| Abstención en id ausente | {_pct(abstain['official_rate'])} (n={abstain['official_n']}) |
| Abstención en consultas sin sentido | {_pct(abstain['nonsense_rate'])} |
| Abstención falsa en gold factual | {_pct(abstain['false_abstain_rate'])} |
| Media del checklist, 5 escenarios | {_n(summary['checklist_mean_scenarios'], 2)} |
| Media del checklist, muestra factual | {_n(summary['checklist_mean_sample'], 2)} (n={summary['checklist_sample_n']}) |

| Escenario | Qué prueba | Ruta | Nota | Lectura |
|-----------|------------|------|-----:|---------|
{scenario_lines}

El checklist es operativo. No es un estudio ciego entre analistas.

## Conclusiones y recomendaciones

La grilla cambia el MRR de dev, así que no se despliega la primera configuración. El boost KEV es una fila aparte: si no mejora el MRR, no reordena el índice; la banda se escribe en la respuesta. El re-ranker se publica aunque en las paráfrasis quede por debajo de la fusión sin re-ranker.

Límites: gold factual pequeño y con el identificador en la pregunta; paráfrasis semi-sintéticas; sin telemetría real; sin QLoRA, EPSS ni CAPEC. El asistente no sustituye al SIEM y no confirma explotación en la red de quien pregunta.

## Referencias

- Karpukhin et al. (2020), Dense Passage Retrieval.
- Cormack, Clarke y Buettcher (2009), Reciprocal Rank Fusion.
- Playbook, guardrails y datasheet en `docs/data/`.
- `configs/retrieval.yaml`. Semilla {summary['seed']}.
"""
