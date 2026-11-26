# Diccionario de datos

**Proyecto:** Asistente de apoyo a *threat hunting* — CTI + LLM  
**Fuentes:** Hugging Face `Shomi28/cyber-threat-intelligence` y Kaggle `prayanshusinghgaur/cybersecurity-threat-intelligence`  
**Referencia de carga:** Fase 1 (`Proyecto_Fase1_ThreatIntel_Colab.ipynb`); tipos **post-ETL** según Fase 2 (`preprocess_and_eda.py`).

Convenciones de la tabla: **Variable** | **Descripción** | **Tipo de dato** | **Rango/Valores posibles** | **Fuente de datos**

---

## 1. Hugging Face — `cve_data` (vulnerabilidades NVD + CVSS + KEV)

**Descripción:** Registros de CVE enriquecidos con métricas CVSS v3.x, señales de explotación, nivel de riesgo compuesto, vínculo táctico MITRE y campos alineados al catálogo **CISA KEV**. Base principal para priorización de parches y contexto del asistente.

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| `cve_id` | Identificador CVE (clave de negocio) | string | Formato `CVE-AAAA-NNNNN`; 5 000 únicos | NVD / dataset HF |
| `description` | Descripción de la vulnerabilidad | string | Texto libre (NVD) | NVD |
| `published_date` | Fecha de publicación | datetime | 1999-03-01 … 2026 (según snapshot) | NVD |
| `last_modified` | Última modificación del registro | datetime | Actualizaciones NVD | NVD |
| `cwe_ids` | Debilidades CWE asociadas | string | p. ej. `CWE-276`; puede listar varios | NVD / CWE |
| `affected_products` | Productos o vendors afectados | string | CPE/vendor serializado en texto | NVD |
| `reference_urls` | Referencias y enlaces | string | URLs agregadas | NVD |
| `exploit_available` | ¿Explotación pública conocida? | boolean | `True` / `False` (parseado desde string) | Agregado CTI |
| `cvss_score` | Puntuación base CVSS | float | 0.0–10.0; media ≈ 9.73 en muestra | CVSS |
| `cvss_severity` | Etiqueta de severidad CVSS | categorical | `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` (dominante: CRITICAL ~99 %) | CVSS |
| `attack_vector` | Vector de ataque CVSS | categorical | `NETWORK`, `ADJACENT`, `LOCAL`, `PHYSICAL` | CVSS |
| `attack_complexity` | Complejidad | categorical | `LOW`, `HIGH` | CVSS |
| `privileges_required` | Privilegios requeridos | categorical | `NONE`, `LOW`, `HIGH` | CVSS |
| `user_interaction` | Interacción de usuario | categorical | `NONE`, `REQUIRED` | CVSS |
| `scope` | Alcance del impacto | categorical | `UNCHANGED`, `CHANGED` | CVSS |
| `confidentiality_impact` | Impacto confidencialidad | categorical | `NONE`, `LOW`, `HIGH` | CVSS |
| `integrity_impact` | Impacto integridad | categorical | `NONE`, `LOW`, `HIGH` | CVSS |
| `availability_impact` | Impacto disponibilidad | categorical | `NONE`, `LOW`, `HIGH` | CVSS |
| `exploitability_score` | Subscore explotabilidad | float | 0.0–10.0 | CVSS |
| `impact_score` | Subscore impacto | float | 0.0–10.0 | CVSS |
| `attack_type` | Clasificación de tipo de ataque/debilidad | string | Taxonomía del dataset | Agregado CTI |
| `likely_mitre_tactic` | Táctica MITRE ATT&CK probable | string | p. ej. `Initial Access`, `Execution` | Correlación MITRE |
| `risk_score` | Puntuación de riesgo compuesta | float | 0–100 (escala del dataset) | Agregado CTI |
| `risk_level` | Nivel de riesgo agregado | categorical | `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` | Agregado CTI |
| `primary_vendor_product` | Vendor/producto principal | string | Texto libre | NVD / CPE |
| `in_cisa_kev` | ¿Listado en CISA Known Exploited Vulnerabilities? | boolean | ~1.8 % `True` en muestra (92/5000) | CISA KEV |
| `ransomware_use` | Indicación de uso en ransomware | string | Texto / flag del dataset | CTI agregada |
| `kev_required_action` | Acción requerida según KEV | string | Texto cuando aplica | CISA KEV |

---

## 2. Hugging Face — `mitre_attack` (técnicas MITRE ATT&CK)

**Descripción:** Catálogo de técnicas y sub-técnicas para enlazar vulnerabilidades y respuestas del asistente con el marco ATT&CK (detección, plataformas, fuentes de datos).

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| `technique_id` | ID MITRE (clave) | string | `T####` o `T####.###` (697 únicos) | MITRE ATT&CK |
| `technique_name` | Nombre de la técnica | string | Texto libre | MITRE ATT&CK |
| `tactics` | Tácticas asociadas | string | Lista serializada con `\|` (p. ej. `Stealth\|Privilege Escalation`) | MITRE ATT&CK |
| `description` | Descripción de la técnica | string | Texto largo | MITRE ATT&CK |
| `detection` | Orientación de detección | string | En **`mitre_attack_clean`**: 100 % nulo tras normalizar `"nan"` del HF. En **`mitre_attack_hardened`**: **0 %** nulo (STIX). | MITRE / dataset → STIX |
| `platforms` | Plataformas relevantes | string | p. ej. `Windows`, `Linux` | MITRE ATT&CK |
| `data_sources` | Fuentes de datos para detección | string | Texto | MITRE ATT&CK |
| `is_subtechnique` | ¿Es sub-técnica? | boolean | `True` / `False` | MITRE ATT&CK |
| `url` | Enlace oficial | string | URL MITRE | MITRE ATT&CK |
| `created` | Fecha de creación del registro | datetime | Parseado desde string | MITRE ATT&CK |
| `modified` | Fecha de modificación | datetime | Parseado desde string | MITRE ATT&CK |

---

## 3. Hugging Face — `instructions_train` / `instructions_test`

**Descripción:** Pares **instrucción → contexto → respuesta** tipo analista de ciberseguridad, para supervisión, evaluación de calidad de respuestas o fine-tuning del LLM. **No mezclar** train y test en entrenamiento.

### 3.1 Split `instructions_train` (~14 189 filas)

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| `instruction` | Tarea en lenguaje natural | string | p. ej. cálculo de riesgo, remediación, explicación CVE | Sintético / curado HF |
| `input` | Contexto (CVE, métricas, técnica) | string | Texto estructurado multi-línea | Derivado de `cve_data` / MITRE |
| `output` | Respuesta esperada tipo experto | string | Markdown con severidad, pasos, prioridad | Supervisión HF |

### 3.2 Split `instructions_test` (~1 577 filas)

Mismo esquema que `instructions_train`; reservado para **métricas de evaluación** (BLEU/ROUGE, similitud semántica o revisión humana) en fases de modelado.

---

## 4. Kaggle — `Cybersecurity_Threat_Intelligence_2026.csv` (incidentes sintéticos/agregados)

**Descripción:** Tabla de incidentes de ciberseguridad con dimensiones geográficas, sectoriales y operativas (detección, respuesta, impacto). Complementa la vista CVE/MITRE con patrones agregados de amenaza para EDA y modelos tabulares (p. ej. clasificación de `severity_level`).

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| `incident_id` | Identificador sintético del incidente (clave) | int64 | 400 000 – 449 999; sin duplicados | Kaggle CTI 2026 |
| `country` | País del incidente | categorical | ~20 países balanceados (top: Canada, Australia, USA, UK, India ~8.2k c/u) | Kaggle |
| `industry` | Sector industrial | categorical | Technology, Government, E-commerce, Education, Healthcare, … | Kaggle |
| `attack_type` | Tipo de ataque | categorical | Zero-Day Exploit, Malware, Ransomware, Phishing, DDoS, SQL Injection, … (~8.3k c/u en top 5) | Kaggle |
| `attack_vector` | Vector de entrada | categorical | IoT, Web Application, Email, Credenciales, … | Kaggle |
| `incident_year` | Año del incidente | int64 | 2018 – 2026 | Kaggle |
| `incident_month` | Mes (1–12) | int64 | 1 – 12 | Kaggle |
| `severity_level` | Severidad declarada (**variable objetivo tabular principal**) | categorical | `Critical`, `High`, `Medium`, `Low` (~25 % c/u) | Kaggle |
| `affected_users_count` | Usuarios afectados | int64 | 11 – 99 992; media ≈ 50 012 | Kaggle |
| `estimated_financial_loss_usd` | Pérdida estimada (USD) | int64 | 1 218 – 4 999 963; media ≈ 2 510 816 | Kaggle |
| `vulnerability_score` | Puntuación de vulnerabilidad agregada | float64 | 1.0 – 10.0; media ≈ 5.51 | Kaggle |
| `detection_method` | Canal de detección | categorical | SIEM, Firewall, Antivirus, AI Detection, IDS, Manual Review | Kaggle |
| `response_time_hours` | Tiempo de respuesta (horas) | float64 | 0.5 – 72.0; media ≈ 36.14 | Kaggle |
| `response_action_taken` | Acción de respuesta | categorical | Monitored, Patched, Blocked, Quarantined, … | Kaggle |
| `systems_affected_count` | Sistemas afectados | int64 | 1 – 499; media ≈ 250 | Kaggle |
| `incident_date` | Fecha derivada (año-mes, día=1) | datetime | Generada en ETL Fase 2 | Derivada |

---

## Variables derivadas y objetivos de modelado (referencia Fase 3)

| Rol | Dataset | Variable(s) | Uso previsto |
|-----|---------|-------------|--------------|
| Objetivo (clasificación tabular) | Kaggle | `severity_level` | Priorización SOC / dashboards |
| Objetivo (riesgo CVE) | HF `cve_data` | `risk_level` o `cvss_severity` | Alertas y RAG |
| Objetivo (supervisión texto) | HF `instructions_*` | `output` dado `instruction` + `input` | Fine-tuning / eval (EN→ES en Fase 3) |
| Claves de enriquecimiento | HF | `likely_mitre_tactic_enriched` / `mitre_tactic_join_key` ↔ MITRE | Join semántico / grafo de amenazas |
| Priorización SOC | HF `cve_data` | `soc_priority_score`, `soc_priority_band` | Ranking KEV-first (playbook) |

> **Nota Kaggle:** `severity_level` es válido como *sandbox* de pipeline, **no** como prueba de valor SOC (ver `datasheet.md`).

---

## Columnas añadidas en hardening Fase 2b

### `cve_data_hardened.parquet`

| Variable | Descripción | Tipo / valores |
|----------|-------------|----------------|
| `cwe_primary` | Primer `CWE-###` extraído de `cwe_ids` | string / null |
| `tactic_from_cwe` | Táctica inferida por heurística CWE | string / null |
| `tactic_from_keywords` | Táctica inferida por regex en `description` | string / null |
| `likely_mitre_tactic_enriched` | Táctica final (original o inferida) | string |
| `tactic_enrichment_source` | `original` \| `cwe_heuristic` \| `keyword_heuristic` \| `unresolved` | categorical |
| `mitre_tactic_join_key` | Etiqueta alineada a taxonomía MITRE HF | string |
| `soc_priority_score` | Score 0–100 KEV-first | float |
| `soc_priority_band` | `P1-immediate` … `P4-monitor` | categorical |

### `mitre_attack_hardened.parquet`

| Variable | Descripción |
|----------|-------------|
| `official_*` | Campos STIX (nombre, descripción, detection, platforms, tactics, mitigations) |
| `detection` | Rellenado desde STIX si el HF venía nulo |
| `detection_enriched` | bool: se rellenó en hardening |
| `detection_source` | `mitre_official_stix` \| `dataset` \| `unavailable` |

---

## Calidad documentada (post-carga Fase 1 / Fase 2 / 2b)

| Conjunto | Filas | Duplicados clave | % nulos (columnas críticas) |
|--------|-------|------------------|----------------------------|
| `cve_data` hardened | 5 000 | 0 (`cve_id`) | 0 % en columnas principales; Unknown táctica residual **16,5 %** |
| `mitre_attack` hardened | 697 | 0 (`technique_id`) | `detection` **0 %** nulo tras STIX |
| `instructions_train` | 14 189 | — | Parquet exportado; sin nulos en 3 columnas |
| `instructions_test` | 1 577 | — | Parquet exportado; sin nulos en 3 columnas |
| Kaggle CSV | 50 000 | 0 (`incident_id`) | 0 % en las 15 columnas originales |

*Diccionario alineado al Project Charter, Fase 2 EDA y hardening CTI. Plan Fase 3: carpeta `Fase_III/` en la raíz del repo.*
