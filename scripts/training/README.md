# Training / Fine-tuning — TDSP `scripts/training`

En la plantilla TDSP, **el entrenamiento se ejecuta desde aquí**.  
El notebook largo (código → lectura → figuras) está en `scripts/modeling/` porque es el entregable narrativo del evaluador.

## Entradas

```text
configs/retrieval.yaml
artifacts/fase3/biencoder_best          # baseline (no se pisa)
data/ + docs/data/evaluation/           # pares / gold
```

## Salidas

```text
artifacts/fase4/biencoder_best          # deploy Strip
artifacts/fase4/biencoder_selection_champion
artifacts/fase4/candidates/
docs/data/evaluation/fase4_*.json
docs/data/figures/fase4_*.png
docs/modeling/fine_tuning/
```

## Cómo correr (desde la raíz de Fase_IV)

```powershell
$env:PYTHONPATH = "src"
$env:CUDA_VISIBLE_DEVICES = "0"

# 1) Grilla científica inicial
python -u -m threatintel.rag.run_fase4_science

# 2) Expand / improve / stripped / TTP (según necesidad)
python -u -m threatintel.rag.run_fase4_expand
python -u -m threatintel.rag.run_fase4_improve
python -u -m threatintel.rag.run_fase4_stripped_ft
python -u -m threatintel.rag.run_fase4_ttp_pack

# 3) Orquestador documentado
python scripts/training/main.py --help
```

## Notebook

[`../modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`](../modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb)
