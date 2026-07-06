# Phase5 StageB B3 Distance-Normalized Seasonal Residual Diagnostic

`current_step`: StageB Step 2/3 B3 diagnostic.

## Scope

[Fact] This diagnostic reuses A6-LBF-r256 `predictions_test.npz` as held-out diagnostic labels and local train split labels for train-only proxies.

[Boundary] No model code, loss code, or remote training is changed. Held-out residuals are labels for diagnostic evaluation only.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3 B3 problem-existence diagnostic |
| `problem` | Test whether train-only seasonal residual explains residual difficulty after controlling forecast distance |
| `existence_evidence` | detrending robustness, block-size robustness, bootstrap sign stability |
| `idea` | Replace raw future-unit reliability with distance-normalized structural residual reliability |
| `theory_check` | A valid proxy should remain positive after step-distance trend removal and should not be reproduced by pure step index |
| `design` | Post-hoc diagnostic over block sizes 24/48/96 and residual labels linear/rank/prefix-normalized |
| `narrative_gate` | partial_pass_needs_stronger_proxy_or_method_boundary |
| `effectiveness_gate` | not applicable; no new method trained |
| `artifacts` | this directory |
| `decision` | partial_pass_needs_stronger_proxy_or_method_boundary; do not implement B3 method unless this is upgraded by follow-up evidence |

## Seasonal Residual Alignment

| Dataset | Block | Residual Label | Spearman | Pearson | Top-Q Overlap |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh2 | 24 | `linear_step_residual` | `0.38` | `0.42` | `0.50` |
| ETTh2 | 24 | `rank_step_residual` | `-0.26` | `0.24` | `0.38` |
| ETTh2 | 24 | `prefix_normalized_residual` | `0.34` | `0.00` | `0.50` |
| ETTh2 | 48 | `linear_step_residual` | `0.35` | `0.41` | `0.50` |
| ETTh2 | 48 | `rank_step_residual` | `-0.40` | `0.21` | `0.25` |
| ETTh2 | 48 | `prefix_normalized_residual` | `0.20` | `-0.02` | `0.50` |
| ETTh2 | 96 | `linear_step_residual` | `0.31` | `0.40` | `0.50` |
| ETTh2 | 96 | `rank_step_residual` | `nan` | `nan` | `0.50` |
| ETTh2 | 96 | `prefix_normalized_residual` | `0.05` | `-0.14` | `0.50` |
| ETTm1 | 24 | `linear_step_residual` | `0.62` | `0.31` | `0.62` |
| ETTm1 | 24 | `rank_step_residual` | `0.05` | `0.17` | `0.38` |
| ETTm1 | 24 | `prefix_normalized_residual` | `0.35` | `0.00` | `0.50` |
| ETTm1 | 48 | `linear_step_residual` | `0.59` | `0.34` | `0.50` |
| ETTm1 | 48 | `rank_step_residual` | `-0.26` | `0.13` | `0.00` |
| ETTm1 | 48 | `prefix_normalized_residual` | `0.23` | `-0.02` | `0.75` |
| ETTm1 | 96 | `linear_step_residual` | `0.43` | `0.25` | `0.50` |
| ETTm1 | 96 | `rank_step_residual` | `nan` | `nan` | `0.50` |
| ETTm1 | 96 | `prefix_normalized_residual` | `0.26` | `-0.21` | `0.50` |
| Weather | 24 | `linear_step_residual` | `0.83` | `0.71` | `0.62` |
| Weather | 24 | `rank_step_residual` | `nan` | `nan` | `0.00` |
| Weather | 24 | `prefix_normalized_residual` | `-0.08` | `0.06` | `0.25` |
| Weather | 48 | `linear_step_residual` | `0.81` | `0.72` | `0.50` |
| Weather | 48 | `rank_step_residual` | `nan` | `nan` | `0.00` |
| Weather | 48 | `prefix_normalized_residual` | `0.06` | `0.09` | `0.25` |
| Weather | 96 | `linear_step_residual` | `0.81` | `0.74` | `0.50` |
| Weather | 96 | `rank_step_residual` | `nan` | `nan` | `0.00` |
| Weather | 96 | `prefix_normalized_residual` | `-0.07` | `0.10` | `0.00` |

## Block-Size Robustness

| Dataset | Block | Positive Labels | Mean Spearman | Min Spearman | Step Abs Mean | Shuffled Abs Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 24 | 2/3 | `0.15` | `-0.26` | `0.23` | `0.24` |
| ETTh2 | 48 | 2/3 | `0.05` | `-0.40` | `0.33` | `0.22` |
| ETTh2 | 96 | 2/3 | `0.18` | `0.05` | `0.18` | `0.12` |
| ETTm1 | 24 | 3/3 | `0.34` | `0.05` | `0.17` | `0.09` |
| ETTm1 | 48 | 2/3 | `0.18` | `-0.26` | `0.28` | `0.13` |
| ETTm1 | 96 | 2/3 | `0.35` | `0.26` | `0.13` | `0.60` |
| Weather | 24 | 1/3 | `0.37` | `-0.08` | `0.16` | `0.06` |
| Weather | 48 | 2/3 | `0.44` | `0.06` | `0.07` | `0.40` |
| Weather | 96 | 1/3 | `0.37` | `-0.07` | `0.08` | `0.36` |

[Note] `nan` means the residual label has no rank variance after detrending, usually because unit MSE is perfectly monotonic with step under that block size. This is treated as missing alignment evidence, not as positive support.

## Bootstrap Stability

| Dataset | Block | Residual Label | Mean Rho | P05 | P50 | P95 | Positive Fraction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 24 | `linear_step_residual` | `0.37` | `0.07` | `0.37` | `0.64` | `0.98` |
| ETTh2 | 24 | `rank_step_residual` | `-0.25` | `-0.62` | `-0.25` | `0.12` | `0.15` |
| ETTh2 | 24 | `prefix_normalized_residual` | `0.34` | `0.03` | `0.34` | `0.63` | `0.96` |
| ETTh2 | 48 | `linear_step_residual` | `0.32` | `-0.25` | `0.35` | `0.82` | `0.84` |
| ETTh2 | 48 | `rank_step_residual` | `-0.37` | `-0.80` | `-0.38` | `0.11` | `0.10` |
| ETTh2 | 48 | `prefix_normalized_residual` | `0.19` | `-0.31` | `0.22` | `0.62` | `0.75` |
| ETTh2 | 96 | `linear_step_residual` | `0.30` | `-0.32` | `0.33` | `0.92` | `0.79` |
| ETTh2 | 96 | `rank_step_residual` | `nan` | `nan` | `nan` | `nan` | `nan` |
| ETTh2 | 96 | `prefix_normalized_residual` | `0.03` | `-0.75` | `0.05` | `0.74` | `0.53` |
| ETTm1 | 24 | `linear_step_residual` | `0.60` | `0.37` | `0.61` | `0.79` | `1.00` |
| ETTm1 | 24 | `rank_step_residual` | `0.04` | `-0.31` | `0.06` | `0.36` | `0.60` |
| ETTm1 | 24 | `prefix_normalized_residual` | `0.35` | `0.03` | `0.36` | `0.62` | `0.96` |
| ETTm1 | 48 | `linear_step_residual` | `0.57` | `0.21` | `0.59` | `0.86` | `0.99` |
| ETTm1 | 48 | `rank_step_residual` | `-0.25` | `-0.69` | `-0.25` | `0.23` | `0.21` |
| ETTm1 | 48 | `prefix_normalized_residual` | `0.21` | `-0.36` | `0.23` | `0.67` | `0.75` |
| ETTm1 | 96 | `linear_step_residual` | `0.38` | `-0.23` | `0.41` | `0.92` | `0.86` |
| ETTm1 | 96 | `rank_step_residual` | `nan` | `nan` | `nan` | `nan` | `nan` |
| ETTm1 | 96 | `prefix_normalized_residual` | `0.24` | `-0.50` | `0.27` | `0.85` | `0.72` |
| Weather | 24 | `linear_step_residual` | `0.81` | `0.69` | `0.82` | `0.89` | `1.00` |
| Weather | 24 | `rank_step_residual` | `nan` | `nan` | `nan` | `nan` | `nan` |
| Weather | 24 | `prefix_normalized_residual` | `-0.09` | `-0.37` | `-0.09` | `0.21` | `0.29` |
| Weather | 48 | `linear_step_residual` | `0.78` | `0.58` | `0.80` | `0.91` | `1.00` |
| Weather | 48 | `rank_step_residual` | `nan` | `nan` | `nan` | `nan` | `nan` |
| Weather | 48 | `prefix_normalized_residual` | `0.06` | `-0.39` | `0.06` | `0.49` | `0.60` |
| Weather | 96 | `linear_step_residual` | `0.76` | `0.40` | `0.79` | `1.00` | `0.99` |
| Weather | 96 | `rank_step_residual` | `nan` | `nan` | `nan` | `nan` | `nan` |
| Weather | 96 | `prefix_normalized_residual` | `-0.10` | `-0.80` | `-0.10` | `0.54` | `0.41` |

## Gate Reasons

- seasonal residual is positive under at least two labels on >=2 block sizes for ['ETTh2', 'ETTm1']
- no block has seasonal mean rho <= -0.20
- seasonal signal does not clearly beat step control in ETTh2/b24, ETTh2/b48, ETTm1/b48
- bootstrap sign stability is weak in ETTh2/b24/rank_step_residual, ETTh2/b48/rank_step_residual, ETTh2/b96/prefix_normalized_residual, ETTm1/b48/rank_step_residual, Weather/b24/prefix_normalized_residual, Weather/b96/prefix_normalized_residual

## Decision

[Decision] B3 diagnostic decision: `partial_pass_needs_stronger_proxy_or_method_boundary`.

[Interpretation] B3 should only advance if the signal is robust across detrending forms, block sizes, and bootstrap sign checks. A positive single table is not enough for a StageB method.

[Rollback Point] If the decision is not `pass_problem_existence`, do not implement reliability-aware loss weighting. Either strengthen the train-only structural proxy or close StageB and move to a broader label-autocorrelation objective stage.

## Output Files

- `stage_b_b3_unit_residuals.csv`
- `stage_b_b3_detrending_robustness.csv`
- `stage_b_b3_blocksize_robustness.csv`
- `stage_b_b3_bootstrap_stability.csv`
- `stage_b_b3_report.md`
