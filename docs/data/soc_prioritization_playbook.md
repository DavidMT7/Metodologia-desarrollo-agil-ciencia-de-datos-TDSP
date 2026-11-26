# Playbook de priorización SOC (CVE)

**Proyecto:** Asistente threat hunting CTI + LLM  
**Versión:** 1.0 (Fase 2 hardening)  
**Implementación:** `src/threatintel/soc_priority.py` → columna `soc_priority_score` / `soc_priority_band`

## Principio

> **KEV manda.** Un CVE en CISA KEV supera a un Critical CVSS sin evidencia de explotación activa catalogada.

No usar `estimated_financial_loss_usd` ni `response_time_hours` de Kaggle para priorizar parches reales (dato sintético; ver datasheet).

## Fórmula (máx. teórico ≈ 100)

| Señal | Peso | Notas |
|-------|-----:|-------|
| `in_cisa_kev = True` | **+50** | Domina el score |
| `ransomware_use = True` | **+15** | Campañas conocidas |
| `exploit_available = True` | **+15** | Público ≠ KEV |
| `cvss_score` (0–10) | **+(cvss/10)×15** | Matiz, no decisión |
| `risk_score` (0–10) | **+(risk/10)×5** | Score compuesto del dataset |

## Bandas operativas

| Banda | Score | Acción analista |
|-------|------:|-----------------|
| **P1-immediate** | ≥ 60 (o KEV+ransomware) | Parche / mitigación **hoy**; citar acción CISA |
| **P2-patch-soon** | ≥ 40 o todo KEV | Ventana corta; change management acelerado |
| **P3-plan** | ≥ 20 | Backlog priorizado del sprint |
| **P4-monitor** | &lt; 20 | Watchlist / threat intel |

## Orden de desempate

1. KEV  
2. Ransomware known  
3. Exploit público  
4. CVSS / risk_score  
5. Táctica ATT&CK enriquecida (Execution / Initial Access &gt; Collection)  
6. Exposición de asset (fuera de este dataset — Fase 3 CMDB)

## Anti-patrones

- Priorizar solo porque `cvss_severity = CRITICAL` (98 % del corpus).  
- Tratar `exploit_available` como equivalente a KEV.  
- Confiar en pérdida financiera Kaggle.
