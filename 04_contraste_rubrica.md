# Contraste con la rúbrica — Fase IV

Documento de autoevaluación frente a **Entrenamiento y Validación (10 %)** y **Modelamiento (10 %)**.

## Evidencia

| Criterio [4,0–5,0] | Evidencia en esta entrega |
|--------------------|---------------------------|
| Entrenamiento adecuado | Grillas FT en `docs/modeling/fine_tuning/` y notebook |
| El modelo se ajusta | Rechazo de ep3 / hardneg / seq384 cuando bajan MRR |
| HP / comparación mejoran | F3 0,688 → DistilUSE 0,765; stripped 0,208 → 0,325 |
| Modelo adecuado al corpus | Bi-encoder + RRF sobre índice CTI hardened |
| Hold-out | Gold solo reporta (`docs/data/evaluation/`) |

## Limitaciones

Con ~30 consultas de desarrollo sin id, el intervalo bootstrap del Δ pasted DistilUSE−F3 es amplio; la mejora en RRF60 y en stripped sí se documenta con Δ pareado. El trade-off pasted/stripped del deploy queda explícito en la bitácora.

## Paquete para revisión

1. `scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`
2. `docs/modeling/fine_tuning/`
3. `03_evidencia_rubrica_fase4.md`
4. `docs/acceptance/`
