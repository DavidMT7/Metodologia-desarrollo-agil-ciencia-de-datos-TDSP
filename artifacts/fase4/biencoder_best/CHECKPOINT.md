# Checkpoint fase4

Los pesos (`.safetensors`) **no se versionan** en GitHub por tamaño.

Reentrenamiento reproducible desde la raíz de esta entrega:

```powershell
$env:PYTHONPATH = "src"
# Fase III
python -m threatintel.rag.run_fase3
# Fase IV
python scripts/training/main.py science
```

Métricas y decisión: `docs/data/evaluation/` y el notebook de la fase.
