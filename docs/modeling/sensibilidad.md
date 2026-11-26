# Sensibilidad y análisis de errores

Estos números no eligen el modelo. El checkpoint oficial sigue en `artifacts/fase3/biencoder_best` (`lr1e-04_ep2`).

## Volumen, temperatura y negativos duros

Una época, learning rate 1e-4, semilla 42. La fila de 4 000 pares y scale 20 es la de la grilla, no un reentrenamiento.

| Pares | MRR dev sin id | R@5 sin id | Origen |
|------:|---------------:|-----------:|--------|
| 800 | 0.568 | 0.667 | submuestreo |
| 2000 | 0.623 | 0.700 | submuestreo |
| 4000 | 0.652 | 0.733 | grilla lr1e-04_ep1 |

| scale | MRR dev sin id | Origen |
|------:|---------------:|--------|
| 10 | 0.656 | ablación |
| 20 | 0.652 | grilla lr1e-04_ep1 |
| 40 | 0.643 | ablación |

Negativo duro de BM25, misma época: MRR dev sin id 0.679 (n=4000). La referencia en lote es 0.652.

## Decisiones fijas, medidas otra vez en desarrollo

| k de RRF | MRR sin id | R@5 sin id |
|---------:|-----------:|-----------:|
| 10 | 0.733 | 0.800 |
| 20 | 0.735 | 0.800 |
| 40 | 0.735 | 0.800 |
| 60 | 0.729 | 0.800 |
| 100 | 0.730 | 0.800 |

El k publicado es 60 (MRR 0.729).

Mismo encoder sobre el texto de Fase II, sin chunks: MRR dev sin id 0.651. Sobre el índice hardened: 0.688.

Jaccard de tokens, ya sin el identificador: gold 0.027; paráfrasis 0.288.

## Hold-out, con incertidumbre

n=32 preguntas factuales. Intervalo bootstrap de 2 000 remuestras, semilla 42.

| Sistema | MRR | IC 95 % bajo | IC 95 % alto |
|---------|----:|-------------:|-------------:|
| minilm | 0.238 | 0.110 | 0.381 |
| bm25 | 0.396 | 0.256 | 0.543 |
| biencoder_ft | 0.890 | 0.805 | 0.969 |
| hybrid_rrf | 0.953 | 0.891 | 1.000 |
| hybrid_rerank | 0.961 | 0.898 | 1.000 |

Diferencia pareada contra MiniLM. Cruzar 0 es la lectura conservadora: con 32 preguntas el intervalo puede ser ancho.

| Sistema | delta MRR | bajo | alto | cruza 0 |
|---------|----------:|-----:|-----:|:-------:|
| bm25 | 0.158 | 0.096 | 0.233 | no |
| biencoder_ft | 0.652 | 0.506 | 0.791 | no |
| hybrid_rrf | 0.715 | 0.572 | 0.843 | no |
| hybrid_rerank | 0.723 | 0.580 | 0.853 | no |

| Familia | minilm | bm25 | biencoder_ft | hybrid_rrf | hybrid_rerank |
|---------|-------:|-----:|-------------:|-----------:|--------------:|
| cve_tactic | 0.000 | 0.041 | 0.848 | 0.850 | 1.000 |
| kev_priority | 0.001 | 0.187 | 0.833 | 1.000 | 0.896 |
| mitre_detection | 0.759 | 1.000 | 1.000 | 1.000 | 1.000 |

Re-ranker en CVE que están en KEV: n=22, MRR 0.943. CVE del gold fuera de KEV: n=0, MRR n/d. En desarrollo, bi-encoder, KEV n=1 MRR 0.500; fuera de KEV n=39 MRR 0.751.

Preguntas cuyo relevante queda fuera del top-5:

| Id | Familia | Rango | Relevante | Top 3 |
|----|---------|------:|-----------|-------|
| — | — | — | — | — |

Figuras: `fase3_sensibilidad_entrenamiento.png`, `fase3_diagnostico_dev.png`, `fase3_gold_incertidumbre.png`.
