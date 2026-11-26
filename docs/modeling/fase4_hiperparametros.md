# Hiperparámetros Fase IV — consolidado

TTP: campeón de SELECCIÓN pasted = DistilUSE confirm (MRR=0.765, IC95% [0.617, 0.892]). Campeón OPERATIVO ES = Strip lr1e-5 (stripped 0.325 vs DistilUSE 0.208; Δ pareado excluye 0=True). Deploy vigente = Strip (regla multi-objetivo declarada). Seeds strip pasted σ=0.012. Gold no eligió.

## Suites bootstrap (media)

| Sistema | pasted | stripped | RRF60 |
|---------|-------:|---------:|------:|
| F3 | 0.688 | 0.198 | 0.729 |
| DistilUSE | 0.765 | 0.208 | 0.795 |
| Strip deploy | 0.744 | 0.325 | 0.776 |

---
# Hiperparámetros Fase IV (completa + expand)

Baseline F3: **0.688**. Ganador: **confirm_distiluse_ep1_ep2** = 0.765.

Expand: mejor nuevo=confirm_distiluse_ep1_ep2 0.765; F4 vigente 0.765. Se mantiene artifacts/fase4/biencoder_best. F3 baseline 0.688.

## Suites (dev elige / gold reporta)

| Sistema | Dense MRR | RRF60 MRR | Stripped | Gold RRF |
|---------|----------:|----------:|---------:|---------:|
| F3_MiniLM | 0.688 | 0.729 | 0.198 | 0.953 |
| F4_DistilUSE | 0.765 | 0.795 | 0.208 | 0.969 |
| F4_deploy | 0.765 | 0.795 | 0.208 | 0.969 |
| F3_cont_sc10 | 0.708 | 0.749 | 0.224 | 0.969 |

## Grilla (top)

| Tag | MRR sin id | R@5 |
|-----|-----------:|----:|
| confirm_distiluse_ep1_ep2 | 0.765 | 0.833 |
| distiluse_ep3_from_confirm | 0.748 | 0.800 |
| distiluse_sc40_ep1 | 0.728 | 0.767 |
| distiluse_batch16_ep1 | 0.724 | 0.767 |
| distiluse_ep1 | 0.722 | 0.733 |
| distiluse_wd001_ep1 | 0.722 | 0.733 |
| distiluse_hardneg_ep1 | 0.719 | 0.767 |
| cont_lr2e-05_ep1_sc10 | 0.708 | 0.733 |
| cont_lr1e-04_ep1_sc10 | 0.704 | 0.767 |
| distiluse_sc10_ep1 | 0.701 | 0.700 |
| cont_f3_sc10_ep2 | 0.689 | 0.733 |
| hardneg_from_f3 | 0.680 | 0.767 |
| hardneg_from_base | 0.678 | 0.733 |
| cont_lr2e-05_ep1_sc20 | 0.674 | 0.767 |
| cont_wd001_ep1 | 0.674 | 0.767 |
| base_warmup0_ep1 | 0.670 | 0.700 |
| base_batch16_ep1 | 0.670 | 0.767 |
| semihard_from_base | 0.663 | 0.733 |
| base_lr1e-04_ep1_sc10 | 0.656 | 0.733 |
| base_wd001_ep1 | 0.652 | 0.733 |

Umbral calibrado: 0.6 (F1=1.000).

Figuras: `fase4_rrf_compare.png`, `fase4_expand_grid.png`, `fase4_threshold_curve.png`, `fase4_grid_dev.png`.

## Improve (hardneg denso / lr bajo / seq384 / triple RRF)

Improve: mejor=confirm_distiluse_ep1_ep2 0.765; vigente DistilUSE confirm 0.765. Triple RRF dev=0.765 (fusión en evaluación, deploy sigue bi-encoder+BM25). No se pisa F4 salvo margen 0.01.

Triple RRF (DistilUSE+MiniLM+BM25) dev MRR=0.765, gold=1.000 (evaluación; no cambia el checkpoint salvo promoción).

| Tag | MRR |
|-----|----:|
| imp_cont_sc10_lr1e-5 | 0.752 |
| imp_dhard_lr1e-5 | 0.748 |
| imp_distil_seq384_ep2 | 0.746 |
| imp_cont_lr1e-5_ep1 | 0.744 |
| imp_dhard_mix50 | 0.743 |
| imp_dhard_sc10_lr1e-5 | 0.741 |
| imp_cont_lr5e-6_ep1 | 0.740 |
| imp_dhard_lr2e-5 | 0.737 |
| imp_distil_seq384_ep1 | 0.724 |

Figuras: `fase4_improve_grid.png`, `fase4_triple_rrf.png`.
## Stripped-FT (consultas ES sin extracto)

Stripped-FT: baseline F4 stripped=0.208 pasted=0.765. Mejor bajo regla stripped+1pp y pasted≥-0.03: strip_from_f4_lr1e-5 stripped=0.325 pasted=0.744. Promovido. Gold no eligió.

| Tag | pasted | stripped |
|-----|-------:|---------:|
| F4_confirm | 0.765 | 0.208 |
| strip_from_f4_lr1e-5 | 0.744 | 0.325 |
| strip_from_f4_lr2e-5 | 0.706 | 0.336 |
| strip_from_f4_sc10 | 0.727 | 0.382 |
| strip_from_f4_ep2_lr1e-5 | 0.731 | 0.348 |

Figura: `fase4_stripped_ft.png`.