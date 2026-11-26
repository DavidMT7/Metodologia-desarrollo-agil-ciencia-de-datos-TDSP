# Informe de salida — Asistente CTI (TDSP)

## Resumen ejecutivo

Se entrega un asistente local en español que recupera CVE/MITRE del índice hardened, prioriza con KEV/CVSS y se abstiene sin evidencia. El modelo de Deep Learning es el bi-encoder MiniLM (2 épocas) + RRF. Media del checklist operativo: **5.00/5**.

## Resultados por fase

```mermaid
flowchart LR
  F1[Fase_I_datos] --> F2[Fase_II_hardening]
  F2 --> F3[Fase_III_modelado_RAG]
  F3 --> F4[Fase_IV_despliegue]
```

| Fase | Logro |
|------|-------|
| I–II | Linaje clean≠hardened; detection 0 % nulo; gold ES |
| III | Grilla 0,592→0,688; gold RRF MRR 0,953 R@5 1,000; zoo + HP |
| IV | `responder(pregunta)`; demo S1–S5; guía; acta |

## Modelo final vs baselines

- **Entrega:** bi-encoder + RRF (rúbrica DL).
- **Baselines M4:** TF-IDF/BM25/LSA/Doc2Vec documentados; no desplazan la entrega.
- **Candidatos:** DistilUSE ep1, hardneg, lr×scale en `artifacts/fase3/candidates/` — no pisan `biencoder_best`.

## Lecciones

- El extracto inglés pegado infla el léxico; la métrica stripped justifica el embedding afinado.
- Más épocas pueden subir el dev y bajar el gold denso (ep3): no se despliega.
- El gold no retunea.

## Impacto y límites

Acelera la consulta CTI del analista. No reemplaza SIEM ni hunting humano. Fuera de alcance: QLoRA, estudio ciego, EPSS/CAPEC.

## Conclusión

Fase III cumple diseño + DL + modelamiento con mejora de hiperparámetros. Fase IV congela y acepta ese sistema.
