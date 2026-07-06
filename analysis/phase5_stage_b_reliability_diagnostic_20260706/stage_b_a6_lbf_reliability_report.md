# Phase5 StageB B1 Reliability Diagnostic

`current_step`: StageB Step 2/3 problem-existence diagnostic with synced A6-LBF predictions.

## Scope

[Fact] This diagnostic uses synced `predictions_test.npz`, `training_log.csv`, and local train split labels.

[Boundary] Validation/test prediction errors are used only as diagnostic labels. Train-only proxies are computed from train split labels and history windows.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3 full B1 diagnostic |
| `problem` | Does A6-LBF-r256 show future-unit reliability heterogeneity that is not merely forecast distance? |
| `existence_evidence` | prediction-level unit MSE/volatility, official-last drift, train-only proxy alignment |
| `idea` | Use held-out prediction difficulty only as diagnostic labels and test whether train-only proxies can identify difficult units |
| `theory_check` | Reliability-aware supervision is plausible only if train-only proxies align with residual difficulty beyond step-index confounding |
| `design` | Post-hoc diagnostic; no model or training change |
| `narrative_gate` | partial_pass_distance_confounded |
| `effectiveness_gate` | not applicable; no new method was trained |
| `artifacts` | this directory |
| `decision` | partial_pass_distance_confounded; B2 is rejected before implementation under the current evidence |

## Prediction Unit Heterogeneity

| Dataset | Units | Unit Len | Min MSE | Max MSE | Max vs Min | Hard/Easy Ratio | Mean Error CV | Spearman(step, MSE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 15 | 48 | `0.1986` | `0.5863` | `195.16%` | `1.92` | `0.87` | `0.99` |
| ETTm1 | 15 | 48 | `0.2486` | `0.4949` | `99.09%` | `1.56` | `0.97` | `0.99` |
| Weather | 15 | 48 | `0.1116` | `0.3891` | `248.48%` | `2.09` | `0.87` | `1.00` |

[Strong Evidence] All three datasets show material future-unit MSE heterogeneity and non-trivial sample-level error volatility.

[Counter-Evidence] High `Spearman(step, MSE)` means difficulty is still strongly tied to forecast distance. A StageB method cannot be justified by this table alone.

## Train-Only Proxy Alignment

| Dataset | Proxy | Spearman(proxy, MSE) | Spearman(proxy, detrended MSE) | Raw Top-Q Overlap | Detrended Top-Q Overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| ETTh2 | `label_novelty` | `0.99` | `0.05` | `1.00` | `0.50` |
| ETTh2 | `local_variation` | `-0.67` | `0.09` | `0.00` | `0.25` |
| ETTh2 | `seasonal_residual` | `-0.74` | `0.35` | `0.00` | `0.50` |
| ETTm1 | `label_novelty` | `0.99` | `-0.17` | `0.75` | `0.00` |
| ETTm1 | `local_variation` | `0.90` | `-0.29` | `0.75` | `0.25` |
| ETTm1 | `seasonal_residual` | `-0.79` | `0.59` | `0.00` | `0.50` |
| Weather | `label_novelty` | `0.55` | `-0.02` | `0.25` | `0.50` |
| Weather | `local_variation` | `-0.56` | `0.75` | `0.00` | `0.75` |
| Weather | `seasonal_residual` | `-0.10` | `0.81` | `0.00` | `0.50` |

[Decision] The strict proxy test is the detrended column. Alignment with raw MSE can be explained by both proxy and error increasing with future distance.

## Training Trajectory

| Dataset | Best Epoch | Best Val MSE | Last Val MSE | Last vs Best | Last h720/h96 Train L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 3 | `0.4175` | `0.4587` | `9.85%` | `1.48` |
| ETTm1 | 9 | `0.6015` | `0.6019` | `0.08%` | `1.30` |
| Weather | 5 | `0.4888` | `0.4904` | `0.32%` | `1.47` |

[Moderate Evidence] ETTh2 has notable official-last drift, while ETTm1 and Weather are close to their best epochs. This suggests the stability problem is dataset-dependent, not a universal A6-LBF failure.

## Decision

[Decision] B1-RED decision: `partial_pass_distance_confounded`.

[Rollback Check] If this remains distance-confounded, StageB should return to Step 2/3 instead of implementing reliability-aware allocation. A future B2 must define a proxy that predicts residual difficulty beyond step index.

## Output Files

- `stage_b_a6_lbf_unit_reliability.csv`
- `stage_b_a6_lbf_reliability_summary.csv`
- `stage_b_a6_lbf_proxy_alignment.csv`
- `stage_b_a6_lbf_unit_proxy_detrended.csv`
- `stage_b_a6_lbf_trajectory_drift.csv`
