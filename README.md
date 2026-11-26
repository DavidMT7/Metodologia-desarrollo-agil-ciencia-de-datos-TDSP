# Entrega — Fase III (Modelado RAG / bi-encoder)

**Metodología:** TDSP · modelado + evaluación

## Notebook de la fase

`scripts/modeling/Proyecto_Fase3_RAG_Asistente.ipynb`

## Estructura

```text
artifacts/fase3/biencoder_best/   metadatos del checkpoint (pesos fuera de git)
configs/
data/processed/
docs/modeling/                    model_report, hiperparámetros, zoo
docs/data/evaluation/             métricas y gold
scripts/modeling/
src/threatintel/rag/
```

## Objetivo

Entrenar y seleccionar un bi-encoder de retrieval para el asistente CTI, con grilla de hiperparámetros, validación en desarrollo y gold hold-out que **solo reporta**.

## Cómo revisar

1. Notebook Fase III  
2. `docs/modeling/`  
3. `docs/data/evaluation/`  

Los pesos grandes no están en el repositorio; ver `artifacts/fase3/biencoder_best/CHECKPOINT.md`.
