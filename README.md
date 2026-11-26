# Entrega — Fase IV (Fine-tuning, validación y despliegue)

**Metodología:** TDSP · entrenamiento + aceptación

## Notebook de la fase (único)

`scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`

## Índice de entrega

| Pieza | Ruta |
|-------|------|
| Entrega | `00_ENTREGA.md` |
| Evidencia rúbrica | `03_evidencia_rubrica_fase4.md` |
| Contraste rúbrica | `04_contraste_rubrica.md` |
| Fine-tuning | `docs/modeling/fine_tuning/` |
| Training CLI | `scripts/training/` |
| Aceptación | `docs/acceptance/` |
| Despliegue | `docs/deployment/` |
| Checkpoint | `artifacts/fase4/biencoder_best/CHECKPOINT.md` |

## Objetivo

Continuar el fine-tuning del retrieval, validar (incluyendo consultas en español sin extracto), calibrar abstención y demostrar el asistente (S1–S5).

## Cómo ejecutar la demo

```powershell
$env:PYTHONPATH = "src"
python scripts/training/main.py list
python scripts/evaluation/main.py
```

Requiere haber entrenado o colocado los pesos locales según `CHECKPOINT.md`.
