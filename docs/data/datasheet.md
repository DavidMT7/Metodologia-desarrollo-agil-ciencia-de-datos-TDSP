# Datasheet de datos — decisiones para modelado (Fase 2→3)

## Qué SÍ usar para el asistente / priorización

| Recurso | Uso válido |
|---------|------------|
| `cve_data_hardened.parquet` | RAG, score SOC, KEV, táctica enriquecida |
| `mitre_attack_hardened.parquet` | RAG de técnicas + **detection** enriquecida STIX |
| `instructions_*_clean.parquet` | Fine-tuning / evaluación hold-out (texto EN) |
| `gold_set_es.json` | Evaluación humana / LLM-as-judge en español |
| `taxonomy_equivalence.csv` | Joins CVE↔MITRE HF |

## Qué NO usar como verdad operativa

| Campo / dataset | Por qué |
|-----------------|--------|
| Kaggle `estimated_financial_loss_usd` | Sintético; mediana ~igual en Critical…Low |
| Kaggle `response_time_hours` | No discrimina severidad |
| Kaggle `severity_level` como prueba de “buen modelo SOC” | Balance artificial ~25 %; correlaciones ~0 → accuracy engañoso |
| `mitre_attack.detection` **antes** del hardening | 100 % nulo en origen HF |
| `likely_mitre_tactic = Unknown` sin enriquecer | ~40 % del CVE; puente roto |

## Limitaciones honestas

- Corpus LLM en **inglés**; respuesta objetivo del producto en **español**.  
- Heurística CWE→táctica no sustituye mapeo ATT&CK oficial por CVE.  
- Stealth domina el catálogo MITRE HF; cobertura de Initial Access perimeter es limitada.

## Sandbox vs producción

- **Sandbox académico:** pipelines TDSP, PCA/KMeans, demos de clasificación en Kaggle.  
- **Producción CTI:** CVE + KEV + MITRE STIX + retrieval metrics + gold set ES.


## Gate visual / NLP didáctico

- Figuras de desbalance y KEV regeneradas (conteos reales; KEV n=92 visible).
- Doc2Vec/TF-IDF/W2V = **baseline didáctico**, no índice de producción.
- Kaggle plano (corr~0, loss idéntica por severidad) permanece como sandbox — no “arreglar” inventando señal.
