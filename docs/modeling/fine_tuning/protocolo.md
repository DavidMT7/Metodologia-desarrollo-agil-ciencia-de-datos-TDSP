# Protocolo de fine-tuning — Fase IV

## Datos

| Split | Uso |
|-------|-----|
| train_pairs (~4000) | InfoNCE / MNRL |
| dev answerable sin id | **Selecciona** checkpoint |
| gold hold-out | **Solo reporta** |

## Métricas

| Métrica | Rol |
|---------|-----|
| MRR dense pasted (dev sin id) | Selección histórica / campeón pasted |
| MRR stripped ES | Operación SOC (regla multi-objetivo) |
| RRF60 | Sistema en serving |
| Gold MRR | Generalización (no retunea) |

## Reglas

1. Margen pasted vs F3: **+0,02** para promover por selección.
2. Strip: stripped **+≥1 pp** y pasted **≥ −3 pp** vs DistilUSE confirm.
3. No pisar `artifacts/fase3/biencoder_best`.
4. Semilla canónica **42**; estabilidad seeds 7/13/21.
