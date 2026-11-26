# Equivalencias de taxonomía CVE ↔ MITRE (HF)

Archivo máquina: `taxonomy_equivalence.csv` (generado por hardening).

## Problema

El campo `cve_data.likely_mitre_tactic` y `mitre_attack.tactic_primary` **no comparten el mismo vocabulario**:

- CVE trae `Defense Evasion`; el catálogo HF MITRE usa **`Defense Impairment`**.
- MITRE HF está dominado por **Stealth / Persistence**; CVE casi no etiqueta Stealth.
- ~40 % de CVE venía como **`Unknown`** antes del enriquecimiento CWE/keywords.

## Join recomendado para RAG

Usar `mitre_tactic_join_key` en CVE hardened:

- `Defense Evasion` → `Defense Impairment`
- Resto: identidad de etiqueta cuando existe en ambos lados

## Heurística Unknown → táctica

1. CWE primario (`cwe_primary`) vía tabla `CWE_TO_TACTIC`  
2. Keywords en `description` (RCE, SQLi, creds, etc.)  
3. Si falla → permanece `Unknown` (`tactic_enrichment_source=unresolved`)

Esto **no** reemplaza un grafo CWE↔CAPEC↔ATT&CK completo; es un puente operativo documentado para Fase 3.
