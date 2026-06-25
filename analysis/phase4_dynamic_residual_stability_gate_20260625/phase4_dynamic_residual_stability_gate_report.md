# Phase4 RG-B Dynamic Residual-Stability Gate Report

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10: evaluate returned artifacts and decide whether RG-B passes |
| `problem` | Fixed late routing mixed learnable/noisy conflict units and failed Weather vs R.3 |
| `existence_evidence` | RG-A gate, residual-stability diagnostic, train logs, trace buckets |
| `idea` | Route only residual-stability learnable high-novelty units to detached adapter auxiliary |
| `theory_check` | Structured residuals should benefit from adapter learning; noisy residuals should not add auxiliary pressure |
| `design` | `dynamic_residual_stability_routing` vs `full_time_mse` and R.3 on ETTh2/Weather |
| `gate` | retain full-time gain, avoid Weather 0/4 vs R.3, improve Weather late, retain ETTh2 signal, prefix zero |
| `artifacts` | `analysis/phase4_dynamic_residual_stability_gate_20260625` |
| `decision` | Fail as paper-core candidate; useful partial evidence vs full-time, but R.3 gap remains |

## Main Result

[Fact] RG-B vs `full_time_mse`: MSE wins `7/8`, mean relative MSE `-2.73%`.
[Fact] RG-B vs R.3: MSE wins `2/8`, mean relative MSE `+2.21%`.
[Fact] Weather vs R.3 remains `0/4`, mean relative MSE `+4.37%`.
[Fact] ETTh2 vs R.3 is `2/4`, mean relative MSE `+0.06%`.

## Per-Horizon MSE

| Dataset | Horizon | RG-B | Full-time | R.3 | RG-B vs full | RG-B vs R.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | 0.311002 | 0.339358 | 0.304796 | -8.36% | +2.04% |
| `ETTh2` | 192 | 0.364008 | 0.388740 | 0.369043 | -6.36% | -1.36% |
| `ETTh2` | 336 | 0.385893 | 0.394549 | 0.382910 | -2.19% | +0.78% |
| `ETTh2` | 720 | 0.405428 | 0.419201 | 0.410473 | -3.29% | -1.23% |
| `Weather` | 96 | 0.156419 | 0.154701 | 0.148026 | +1.11% | +5.67% |
| `Weather` | 192 | 0.199968 | 0.200707 | 0.192409 | -0.37% | +3.93% |
| `Weather` | 336 | 0.254646 | 0.256879 | 0.244793 | -0.87% | +4.02% |
| `Weather` | 720 | 0.333251 | 0.338482 | 0.320847 | -1.55% | +3.87% |

## Segment Gate

[Fact] Weather h720 `337-720` vs R.3: `+4.06%`.
[Fact] ETTh2 h720 `337-720` vs R.3: `-2.83%`.

| Future region | Baseline | Segments | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `early_1_96` | `D0_full_time_mse` | 8 | 4 | -2.63% |
| `late_337_720` | `D0_full_time_mse` | 2 | 2 | -3.28% |
| `middle_97_336` | `D0_full_time_mse` | 10 | 10 | -1.46% |
| `early_1_96` | `D1_r3_prefix_risk` | 8 | 0 | +3.98% |
| `late_337_720` | `D1_r3_prefix_risk` | 2 | 1 | +0.61% |
| `middle_97_336` | `D1_r3_prefix_risk` | 10 | 2 | +1.57% |

## Trace Buckets

| Dataset | Learnable blocks | Noisy blocks | Ambiguous blocks | Noisy suppression | Adapter active steps | Mean abs adapter residual | Last abs adapter residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | 1.17 | 1.44 | 1.39 | 0.36 | 56.1 | 0.030439 | 0.035329 |
| `Weather` | 1.44 | 1.87 | 0.69 | 0.47 | 69.1 | 0.044936 | 0.036922 |

## Training Dynamics

| Dataset | Strategy | Epochs ran | Best epoch | Best val MSE | Last val drift | Train loss change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `RG-B_dynamic_residual_stability` | 11 | 1 | 0.383836 | +25.30% | -26.01% |
| `Weather` | `RG-B_dynamic_residual_stability` | 13 | 3 | 0.532040 | +8.50% | -38.40% |
| `ETTh2` | `D0_full_time_mse` | 13 | 3 | 0.380270 | +14.45% | -30.94% |
| `Weather` | `D0_full_time_mse` | 14 | 4 | 0.534117 | +10.41% | -38.40% |
| `ETTh2` | `D1_r3_prefix_risk` | 11 | 1 | 0.374532 | +18.43% | -53.20% |
| `Weather` | `D1_r3_prefix_risk` | 11 | 1 | 0.527042 | +17.01% | -34.75% |

[Fact] In the current gate, all six runs reach best validation MSE by epoch `1-4`; the run length is best epoch plus patience `10`.
[Fact] Across current and adjacent Phase4 gates, `26/26` ETTh2/Weather logs reach best validation MSE by epoch `<=5`.

## Interpretation

[Strong Evidence] RG-B preserves the useful mechanism signal against `full_time_mse`: it wins 7/8 MSE settings and improves Weather long horizons versus full-time.

[Counter-Evidence] RG-B does not solve the paper-core gate. Weather remains 0/4 vs R.3, and the Weather h720 late segment is still worse than R.3. The dynamic residual-stability adapter is therefore not enough to replace R.3 as the main story.

[Inference] The trace shows the router is active rather than collapsed: Weather suppresses more noisy blocks than ETTh2 and the adapter residual becomes nonzero. The failure is more likely due to the carrier/optimization target being weak than due to a dead router.

[Training-Dynamics Analysis] The early best epoch is real and systemic. It appears in prior Phase4 controls too, while train loss keeps decreasing after validation has worsened. This points to fast overfitting or validation-objective mismatch under the current small carrier and no scheduler, not to a RG-B-specific logging bug.

[Decision] Do not continue by sweeping RG-B thresholds or aux weight. Roll back to Step 5/6 and investigate the training protocol/carrier: learning-rate schedule, shorter calibrated training, or a stronger pretraining/warmup design should be studied before adding more routing complexity.

[Fact] max prefix mismatch MSE is `1.474e-14`, so prefix consistency remains numerical-zero.
