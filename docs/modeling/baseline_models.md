# Reporte del modelo baseline

## Descripción del modelo

El baseline de retrieval es `sentence-transformers/all-MiniLM-L6-v2`, congelado. No se entrena en esta fase. El índice de la reproducción usa el armado de Fase II (un documento por técnica, detección truncada) sobre los parquets **hardened**, no sobre `*_clean`.

El control de linaje de esta corrida: detection nulo en clean 100.0 % y en hardened 0.0 %. Unknown en `likely_mitre_tactic` 39.5 %; en `likely_mitre_tactic_enriched` 16.5 %.

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
| Archivada en Fase II | 32 | 0.188 | 0.156 | 0.250 | 0.281 |
| Reproducción Fase III | 32 | 0.188 | 0.156 | 0.250 | 0.281 |

## Análisis

MiniLM está entrenado sobre todo en inglés y las preguntas están en español. MRR 0.188 y Recall@5 0.250 son el piso. No se usan para generar texto libre.

## Conclusiones

El baseline sirve de control. El modelo de esta fase tiene que superarlo sin entrenar con los `doc_id` del gold y sin leer los parquets limpios.

## Referencias

- Reimers y Gurevych (2019), Sentence-BERT.
- `docs/data/evaluation/retrieval_baseline_metrics.json`.
- Registro E18/E16 en `Fase_III/09_registro_errores_imprecisiones_correcciones.md`.
