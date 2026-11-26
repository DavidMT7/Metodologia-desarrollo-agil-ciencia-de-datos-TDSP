# Definición de los datos

**Proyecto:** Asistente de apoyo a *threat hunting* basado en LLM y datos de CTI  
**Autor:** David Martínez — Diplomado en Machine Learning, UNAL — cohorte 2026-II  
**Fase TDSP:** Preprocesamiento y análisis exploratorio de datos (entrega parcial, 10 %)  
**Continuidad:** basado en la Fase 1 (`Proyecto_Fase1_ThreatIntel_Colab.ipynb` y `scripts/data_acquisition/`)

---

## Origen de los datos

Los datos provienen de **dos fuentes públicas de Cyber Threat Intelligence (CTI)**, alineadas al objetivo del proyecto (contextualizar CVE, MITRE ATT&CK, KEV y patrones de incidentes para un futuro asistente LLM):

| Fuente | Identificador | Formato | Obtención |
|--------|---------------|---------|-----------|
| **Hugging Face Hub** | [`Shomi28/cyber-threat-intelligence`](https://huggingface.co/datasets/Shomi28/cyber-threat-intelligence) | Cuatro archivos **Parquet** (`cve_data`, `mitre_attack`, `instructions/train`, `instructions/test`) | API `datasets` con rutas `hf://datasets/...` (no se usa `load_dataset` monolítico: el builder del Hub mezcla esquemas incompatibles) |
| **Kaggle** | [`prayanshusinghgaur/cybersecurity-threat-intelligence`](https://www.kaggle.com/datasets/prayanshusinghgaur/cybersecurity-threat-intelligence) | **CSV** `Cybersecurity_Threat_Intelligence_2026.csv` | `kagglehub` (`dataset_download` + lectura con `pandas`) |

**Licencia y uso:** conjuntos abiertos para investigación y prototipado académico. No incluyen logs ni datos internos del negocio (excluidos explícitamente en el Project Charter hasta definir gobierno de datos).

**Relación con el negocio:**

- **HF `cve_data` + `mitre_attack`:** priorización de vulnerabilidades, tácticas MITRE y señales KEV para *threat hunting* y RAG.
- **HF `instructions_*`:** pares supervisados (instrucción → respuesta de analista) para evaluación o fine-tuning del asistente.
- **Kaggle CSV:** vista tabular agregada de incidentes (sector, país, severidad, impacto) para EDA y posibles *features* de contexto operativo.

---

## Especificación de los scripts para la carga de datos

| Script / notebook | Ubicación | Propósito |
|-------------------|-----------|-----------|
| Carga Fase 1 (reproducible en Colab) | `scripts/data_acquisition/Proyecto_Fase1_ThreatIntel_Colab.ipynb` | Instalación de dependencias, carga HF por Parquet, carga Kaggle, verificación de `shape`, esquema y tabla de calidad (% nulos) |
| Preprocesamiento y EDA Fase 2 | `scripts/preprocessing_eda/Proyecto_Fase2_ThreatIntel_EDA.ipynb` (copia en raíz: `Proyecto_Fase2_ThreatIntel_EDA.ipynb`) | ETL, auditoría de calidad, EDA CTI/SOC (seaborn), exportación a `data/processed/` y figuras en `docs/data/figures/` |

**Dependencias Python (Fase 1 y 2):** `pandas>=2.0`, `datasets>=2.14`, `kagglehub[pandas-datasets]`, `matplotlib`, `seaborn`, `pyarrow`.

**Ejecución (Fase 2):** abrir el notebook en **Google Colab** o Jupyter y ejecutar todas las celdas en orden (recomendado tras clonar el repositorio o subir el `.ipynb`).

---

## Referencias a rutas o bases de datos origen y destino

### Rutas de origen de datos

| Bloque | Ubicación lógica | Archivo / split | Filas (snapshot verificado) | Columnas |
|--------|------------------|-----------------|-------------------------------|----------|
| CVE + métricas NVD/CVSS/KEV | HF Hub | `cve_data/train-00000-of-00001.parquet` | 5 000 | 28 (todas `string` en origen) |
| MITRE ATT&CK | HF Hub | `mitre_attack/train-00000-of-00001.parquet` | 697 | 11 |
| Instrucciones entrenamiento | HF Hub | `instructions/train-00000-of-00001.parquet` | 14 189 | 3 (`instruction`, `input`, `output`) |
| Instrucciones prueba | HF Hub | `instructions/test-00000-of-00001.parquet` | 1 577 | 3 |
| Incidentes agregados | Kaggle (cache local tras `dataset_download`) | `Cybersecurity_Threat_Intelligence_2026.csv` | 50 000 | 15 |

**Estructura de archivos de origen:**

- **Parquet (HF):** tablas columnares; campos numéricos y booleanos llegan como **texto** y deben convertirse en ETL.
- **CSV (Kaggle):** tipos mixtos ya tipados por `pandas` (`int64`, `float64`, `object`); sin encabezados multilínea; separador coma.

### Procedimientos de transformación y limpieza (origen → staging)

Aplicados en `Proyecto_Fase2_ThreatIntel_EDA.ipynb` (resumen):

1. **Deduplicación:** por `cve_id`, `technique_id` e `incident_id` (en la carga verificada no hay duplicados; se mantiene la regla defensiva).
2. **Tipado HF `cve_data`:** fechas (`published_date`, `last_modified` → `datetime64`); scores → `float`; `exploit_available`, `in_cisa_kev` → `boolean`; normalización de categorías (`cvss_severity`, `risk_level` en mayúsculas).
3. **Texto:** recorte de espacios; sustitución de literales `nan`/`None` por `NA` en campos clave (`description`, `cwe_ids`, `likely_mitre_tactic`).
4. **HF `mitre_attack`:** `is_subtechnique` → booleano; `detection == "nan"` → nulo; fechas `created`/`modified` parseadas.
5. **Kaggle:** normalización de `severity_level` (Title Case); validación de `incident_month` ∈ [1, 12]; construcción de `incident_date` (primer día del mes/año); estandarización de cadenas categóricas (`country`, `industry`, `attack_type`, etc.).
6. **Valores faltantes:** en Kaggle **0 %** de nulos en columnas críticas; en HF MITRE, `detection` queda **100 % nulo** en el parquet **clean** (literal `"nan"`). El parquet **hardened** restaura detection vía STIX (**0 %** nulo).

### Base de datos de destino

No se despliega un motor SQL en esta fase académica. El **destino lógico** es el directorio de datos procesados del repositorio:

| Artefacto | Ruta | Formato | Contenido |
|-----------|------|---------|-----------|
| CVE limpio | `data/processed/cve_data_clean.parquet` | Parquet | 5 000 (tipos normalizados) |
| CVE hardened | `data/processed/cve_data_hardened.parquet` | Parquet | + táctica enriquecida + `soc_priority_*` |
| MITRE limpio | `data/processed/mitre_attack_clean.parquet` | Parquet | 697 |
| MITRE hardened | `data/processed/mitre_attack_hardened.parquet` | Parquet | `detection` 0 % nulo (STIX) |
| Instructions | `data/processed/instructions_{train,test}_clean.parquet` | Parquet | 14 189 / 1 577 |
| Incidentes Kaggle | `data/processed/kaggle_cti_2026_clean.parquet` | Parquet | 50 000 (+ `incident_date`) |
| Figuras EDA | `docs/data/figures/*.png` | PNG | 29 figuras (gate visual incluido) |

**Procedimiento de carga en destino:** tras EDA + `harden_fase2_cti.py`, los Parquet (clean **y** hardened) quedan listos para Fase 3. Preferir `*_hardened` para detection / puente CVE / score SOC. Los splits `instructions_*` también están versionados en parquet (mismo particionado HF).

---

## Trazabilidad TDSP (Fase 2 ↔ rúbrica)

| Criterio de la rúbrica | Evidencia en el repositorio |
|------------------------|------------------------------|
| Código de preprocesamiento | `scripts/preprocessing_eda/Proyecto_Fase2_ThreatIntel_EDA.ipynb` |
| Código de EDA | Mismo notebook + figuras en `docs/data/figures/` |
| Definición de datos | Este documento + `data_dictionary.md` |
| Reporte resumen de datos | `data_summary.md` |

*Versión documental Fase 2 — noviembre 2026 (post-hardening CTI + gate visual).*
