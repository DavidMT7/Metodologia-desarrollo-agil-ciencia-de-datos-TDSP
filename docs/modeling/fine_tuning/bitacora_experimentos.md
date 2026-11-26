# Bitácora de fine-tuning — Fase IV

Fuente: `docs/data/evaluation/fase4_summary.json`. Gold **no** elige.

Corridas registradas: **37**.

| tag | lr | epochs | scale | batch | MRR_dev_sin_id | R@5 | loss |
|-----|---:|-------:|------:|------:|---------------:|----:|-----:|
| `confirm_distiluse_ep1_ep2` | 2e-05 | 2 | 20.0 | 32 | 0.7652 | 0.833 | 0.0129 |
| `imp_cont_sc10_lr1e-5` | 1e-05 | 1 | 10.0 | 32 | 0.7518 | 0.767 | 0.0406 |
| `distiluse_ep3_from_confirm` | 2e-05 | 1 | 20.0 | 32 | 0.7484 | 0.800 | 0.0072 |
| `imp_dhard_lr1e-5` | 1e-05 | 1 | 20.0 | 32 | 0.7479 | 0.833 | 0.1490 |
| `imp_distil_seq384_ep2` | 2e-05 | 2 | 20.0 | 32 | 0.7461 | 0.767 | 0.0124 |
| `imp_cont_lr1e-5_ep1` | 1e-05 | 1 | 20.0 | 32 | 0.7443 | 0.800 | 0.0021 |
| `strip_from_f4_lr1e-5` | 1e-05 | 1 | 20.0 | 32 | 0.7442 | 0.800 | 0.2933 |
| `imp_dhard_mix50` | 1e-05 | 1 | 20.0 | 32 | 0.7426 | 0.833 | 0.1404 |
| `imp_dhard_sc10_lr1e-5` | 1e-05 | 1 | 10.0 | 32 | 0.7414 | 0.800 | 0.2869 |
| `imp_cont_lr5e-6_ep1` | 5e-06 | 1 | 20.0 | 32 | 0.7399 | 0.800 | 0.0020 |
| `imp_dhard_lr2e-5` | 2e-05 | 1 | 20.0 | 32 | 0.7373 | 0.800 | 0.1395 |
| `strip_from_f4_ep2_lr1e-5` | 1e-05 | 2 | 20.0 | 32 | 0.7308 | 0.800 | 0.3243 |
| `distiluse_sc40_ep1` | 2e-05 | 1 | 40.0 | 32 | 0.7284 | 0.767 | 0.0181 |
| `strip_from_f4_sc10` | 1e-05 | 1 | 10.0 | 32 | 0.7266 | 0.767 | 0.3514 |
| `imp_distil_seq384_ep1` | 2e-05 | 1 | 20.0 | 32 | 0.7243 | 0.733 | 0.0200 |
| `distiluse_batch16_ep1` | 2e-05 | 1 | 20.0 | 16 | 0.7236 | 0.767 | 0.0246 |
| `distiluse_ep1` | 2e-05 | 1 | 20.0 | 32 | 0.7223 | 0.733 | 0.0146 |
| `distiluse_wd001_ep1` | 2e-05 | 1 | 20.0 | 32 | 0.7223 | 0.733 | 0.0146 |
| `distiluse_hardneg_ep1` | 2e-05 | 1 | 20.0 | 32 | 0.7185 | 0.767 | 0.1373 |
| `cont_lr2e-05_ep1_sc10` | 2e-05 | 1 | 10.0 | 32 | 0.7076 | 0.733 | 0.0619 |
| `strip_from_f4_lr2e-5` | 2e-05 | 1 | 20.0 | 32 | 0.7063 | 0.767 | 0.2887 |
| `cont_lr1e-04_ep1_sc10` | 0.0001 | 1 | 10.0 | 32 | 0.7042 | 0.767 | 0.0618 |
| `distiluse_sc10_ep1` | 2e-05 | 1 | 10.0 | 32 | 0.7015 | 0.700 | 0.0832 |
| `cont_f3_sc10_ep2` | 2e-05 | 1 | 10.0 | 32 | 0.6892 | 0.733 | 0.0580 |
| `hardneg_from_f3` | 2e-05 | 1 | 20.0 | 32 | 0.6800 | 0.767 | 0.1212 |
| `hardneg_from_base` | 0.0001 | 1 | 20.0 | 32 | 0.6778 | 0.733 | 0.1820 |
| `cont_lr2e-05_ep1_sc20` | 2e-05 | 1 | 20.0 | 32 | 0.6740 | 0.767 | 0.0064 |
| `cont_wd001_ep1` | 2e-05 | 1 | 20.0 | 32 | 0.6740 | 0.767 | 0.0064 |
| `base_warmup0_ep1` | 0.0001 | 1 | 20.0 | 32 | 0.6703 | 0.700 | 0.0472 |
| `base_batch16_ep1` | 0.0001 | 1 | 20.0 | 16 | 0.6698 | 0.767 | 0.0176 |
| `semihard_from_base` | 0.0001 | 1 | 20.0 | 32 | 0.6627 | 0.733 | 0.1204 |
| `base_lr1e-04_ep1_sc10` | 0.0001 | 1 | 10.0 | 32 | 0.6555 | 0.733 | 0.1455 |
| `base_wd001_ep1` | 0.0001 | 1 | 20.0 | 32 | 0.6520 | 0.733 | 0.0606 |
| `base_lr1e-04_ep1_sc20` | 0.0001 | 1 | 20.0 | 32 | 0.6518 | 0.733 | 0.0603 |
| `base_lr1e-04_ep1_sc40` | 0.0001 | 1 | 40.0 | 32 | 0.6432 | 0.733 | 0.0595 |
| `cont_lr1e-04_ep1_sc20` | 0.0001 | 1 | 20.0 | 32 | 0.6322 | 0.733 | 0.0098 |
| `base_lr2e-05_ep1_sc20` | 2e-05 | 1 | 20.0 | 32 | 0.5919 | 0.733 | 0.1024 |

## Decisiones

- Selection champion (pasted): `confirm_distiluse_ep1_ep2`
- Deploy operativo: `strip_from_f4_lr1e-5`
- Decisión: TTP: campeón de SELECCIÓN pasted = DistilUSE confirm (MRR=0.765, IC95% [0.617, 0.892]). Campeón OPERATIVO ES = Strip lr1e-5 (stripped 0.325 vs DistilUSE 0.208; Δ pareado excluye 0=True). Deploy vigente = Strip (regla multi-objetivo declarada). Seeds strip pasted σ=0.012. Gold no eligió.

## Rechazos (ajuste correcto)

| Experimento | Motivo |
|-------------|--------|
| DistilUSE ep3 | MRR < ep2 → sobreajuste |
| Hardneg BM25/denso | No supera confirm pasted |
| seq384 / continue lr bajo | Plateau ≤ 0,752 |
| Strip sc10 | stripped↑ pero pasted fuera de regla (−3 pp) |

Figuras: `docs/data/figures/fase4_*.png` · Notebook: `scripts/modeling/Proyecto_Fase4_FineTuning_Validacion.ipynb`.
