# Entrega — Fase II (EDA, preprocesamiento y hardening CTI)

**Metodología:** TDSP · `docs/` · `scripts/` · `src/` · `data/`

## Notebook de la fase

`scripts/preprocessing_eda/Proyecto_Fase2_ThreatIntel_EDA.ipynb`

## Estructura

```text
data/processed/          parquets clean / hardened
docs/data/               data_summary, figures, playbooks
scripts/preprocessing_eda/
src/threatintel/         utilidades de linaje y enriquecimiento
tests/                   contratos de esquema
```

## Objetivo

Comprender el corpus, aplicar ETL/hardening CTI y dejar datos listos para modelado (Fase III), con figuras y resumen reproducibles.

## Cómo ejecutar

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests -q
```
