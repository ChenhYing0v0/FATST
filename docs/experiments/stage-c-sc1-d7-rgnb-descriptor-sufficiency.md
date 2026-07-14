# SC1-D7 RGNB Descriptor Sufficiency Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate_id` | `SC1-D7` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2/3 problem/mechanism diagnostic |
| `method_training_authorized` | false |
| `forecast_test_usage` | forbidden |
| `return_if_pass` | Step 4/6 redesign；not Step 7 |
| `close_if_fail` | descriptor-generator route；RGNB remains component only |

## What We Plan To Test

Step 6已经证明PAF tensor path可projective，但generic mechanism被prior art覆盖，且B11/B14给出negative internal
evidence。D7只问：

> 在同一frozen A6 memory和rank budget下，canonical RGNB atom descriptors是否比permuted/random
> descriptors提供可复现的forecast-relevant inductive bias？

D7不测试cross-attention、history retrieval、learned horizon routing或full model performance。

## Data And Freeze Contract

- datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather；
- checkpoints: frozen natural-profile seeds 2021/2022/2023；
- source tensor: `memory [B,C,P,D]` from frozen A6 encoder；
- target: normalized future `y [B,C,720]`；
- basis: fixed RGNB `Q [720,720]`, global rank 16；
- splits: reuse accepted frozen-memory fit/inner-holdout/official-validation boundaries；
- test: never loaded；
- profile、basis rank、descriptor definition与width不得按dataset调整。

## Tensor Path

```text
memory [B,C,P,D]
  -> flatten h [B,C,R]
  -> shared branch z [B,C,256]

descriptor d_j [8]
  -> shared trunk psi_j [256]
  -> alpha_j = <z, psi_j> [B,C]

active alpha [B,C,N_H]
  x RGNB Q[:H, active] [H,N_H]
  -> prediction [B,C,H]
```

No arm receives requested $H$ as a feature. No atom accesses a different history subset.

## Arms

| Arm | Width | Descriptor | Purpose |
| --- | ---: | --- | --- |
| `free_m0` | n/a | free atom table | A6-equivalent upper/control path |
| `geo_c256` | 256 | canonical RGNB | compact geometry hypothesis |
| `perm_c256` | 256 | fixed permutation of canonical rows | compact geometry control |
| `random_c256` | 256 | fixed seeded random descriptors | compact descriptor-capacity control |
| `geo_m694` | 694 | canonical RGNB | near-A6-budget geometry hypothesis |
| `perm_m694` | 694 | same fixed permutation | matched-budget geometry control |
| `random_m694` | 694 | fixed seeded random descriptors | matched-budget capacity control |

The permutation/random seeds are global and frozen before dataset execution. They are structure seeds, not training
seeds and cannot be selected from results.

## Metrics

All reported gains use

$$
\Delta(A,B)=100\frac{\operatorname{MSE}(B)-\operatorname{MSE}(A)}
{\operatorname{MSE}(B)}.
$$

- `dense_horizon_macro_mse`: mean cumulative MSE over the frozen dense horizon set；
- `short_macro_mse`: mean over $H\le144$；
- `long_macro_mse`: mean over $H\ge336$；
- `descriptor_gain`: GEO versus median(PERM,RANDOM) within the same width；
- `free_gap`: GEO versus `free_m0`；
- `fit_holdout_gap`: fit improvement minus inner-holdout improvement，diagnoses memorization；
- `coefficient_row_reconstruction`: optional auxiliary statistic only；cannot override forecast-space gate。

## Hard Gates

D7 passes only if all hold：

1. `geo_c256` descriptor gain MSE至少`+0.5%`；
2. `geo_m694` descriptor gain MSE至少`+0.5%`；
3. 两种width均至少4/5 datasets方向为正，且每dataset至少2/3 checkpoint seeds为正；
4. 两种width的MAE descriptor gain均非负；
5. `geo_m694`相对`free_m0`不得差于`-0.5%`；
6. true geometry的fit/holdout gap不得比random control恶化超过1 percentage point；
7. completeness、finite、freeze、test-absence、prefix与basis invariants全部通过。

Gate 1/2同时要求compact与matched widths，是为了排除两种相反解释：compact gain只来自regularization，或
matched model只靠memorizing 720 descriptors。

## Failure Attribution

- numeric divergence或>100% degradation：`optimization_or_numeric_pathology`，只能否定exact probe；
- GEO与PERM/RANDOM持平：`hypothesis_false` for descriptor sufficiency；关闭descriptor route；
- compact pass、matched fail：`capacity_or_regularization_explains`；不得升method；
- matched pass、compact fail：capacity dependence；只允许重新做Step 4 claim audit，不直接升method；
- free control显著更好：`readout_or_head_design_wrong/capacity restriction`；不得用params差异掩盖；
- single-dataset pass：`dataset_specific_only`；stage gate fail。

## Decision Rule

- all gates pass：`descriptor_sufficiency_supported_return_step4_6`；
- otherwise：`descriptor_sufficiency_not_supported_close_paf`。

无论结果如何，D7都不授权future-unit retrieval、Encoder replacement、MoE或SC2-MIPR。
