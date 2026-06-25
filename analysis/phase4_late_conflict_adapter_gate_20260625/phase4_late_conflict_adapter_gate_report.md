# Phase4 RG-A Late-Conflict Adapter Gate Report

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10: evaluate returned artifacts and decide whether method passes |
| `problem` | Weather late/early gradient conflict suggests loss-only HSS may contaminate shared representation |
| `existence_evidence` | SRP code audit + gradient conflict diagnostic + S1/S2 failures |
| `idea` | Route late conflict auxiliary pressure into a zero-init adapter residual while dense base keeps shared path |
| `theory_check` | If conflict is mainly late/readout, adapter late residual should improve Weather late segment without damaging ETTh2 |
| `design` | `late_conflict_adapter_routing` vs `full_time_mse` and R.3 on ETTh2/Weather |
| `gate` | beat full-time, avoid Weather 0/4 vs R.3, improve Weather late segment, retain ETTh2 signal, prefix zero |
| `artifacts` | `analysis/phase4_late_conflict_adapter_gate_20260625` |
| `decision` | Fail as paper-core candidate; partial mechanism signal vs full-time, but R.3 gap and Weather late failure remain |

## Main Result

[Fact] RG-A vs `full_time_mse`: MSE wins `7/8`, mean relative MSE `-1.31%`。
[Fact] RG-A vs R.3: MSE wins `1/8`, mean relative MSE `+3.76%`。
[Fact] Weather vs R.3 remains `0/4` wins, mean relative MSE `+4.44%`。
[Fact] ETTh2 vs R.3 drops to `1/4` wins, mean relative MSE `+3.08%`。

## Overall Summary

| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE | Mean relative MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `D0_full_time_mse` | 8 | 7 | 6 | -1.31% | -0.23% |
| `D1_r3_prefix_risk` | 8 | 1 | 0 | +3.76% | +2.96% |

## Dataset Summary

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D0_full_time_mse` | 4 | 4 | -2.26% |
| `Weather` | `D0_full_time_mse` | 4 | 3 | -0.36% |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 1 | +3.08% |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | +4.44% |

## Per-Horizon Metrics

| Dataset | Horizon | RG-A MSE | Full-time MSE | R.3 MSE | RG-A vs full-time | RG-A vs R.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | 0.333902 | 0.339358 | 0.304796 | -1.61% | +9.55% |
| `ETTh2` | 192 | 0.377332 | 0.388740 | 0.369043 | -2.93% | +2.25% |
| `ETTh2` | 336 | 0.387026 | 0.394549 | 0.382910 | -1.91% | +1.07% |
| `ETTh2` | 720 | 0.408272 | 0.419201 | 0.410473 | -2.61% | -0.54% |
| `Weather` | 96 | 0.156333 | 0.154701 | 0.148026 | +1.05% | +5.61% |
| `Weather` | 192 | 0.200052 | 0.200707 | 0.192409 | -0.33% | +3.97% |
| `Weather` | 336 | 0.254317 | 0.256879 | 0.244793 | -1.00% | +3.89% |
| `Weather` | 720 | 0.334569 | 0.338482 | 0.320847 | -1.16% | +4.28% |

## Segment Gate

[Fact] Weather h720 late segment vs R.3 is `+4.74%`; gate requires improvement, so this fails.
[Fact] ETTh2 h720 late segment vs R.3 is `-1.53%`; adapter helps the intended late segment only on ETTh2.

| Future region | Baseline | Segments | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `early_1_96` | `D0_full_time_mse` | 8 | 3 | +0.12% |
| `late_337_720` | `D0_full_time_mse` | 2 | 2 | -2.33% |
| `middle_97_336` | `D0_full_time_mse` | 10 | 10 | -1.87% |
| `early_1_96` | `D1_r3_prefix_risk` | 8 | 0 | +7.00% |
| `late_337_720` | `D1_r3_prefix_risk` | 2 | 1 | +1.60% |
| `middle_97_336` | `D1_r3_prefix_risk` | 10 | 3 | +1.15% |

## Trace And Prefix

| Dataset | Unit type | Epochs | Mean adapter active steps | Mean abs adapter residual | Last abs adapter residual | Mean time loss | Mean adapter/unit loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `late_conflict_adapter` | 12 | 384 | 0.013637 | 0.016239 | 0.671773 | 0.824340 |
| `Weather` | `late_conflict_adapter` | 13 | 384 | 0.014467 | 0.015663 | 0.579310 | 0.626329 |

[Fact] max prefix mismatch MSE across all runs is `1.474e-14`, so prefix consistency remains numerical-zero.

## Interpretation

[Strong Evidence] The method validates one narrow mechanism: routing late auxiliary pressure into a zero-init adapter can still beat `full_time_mse` on average, and it improves ETTh2 h720 late segment vs R.3.

[Counter-Evidence] It fails the paper-core gate. Weather remains `0/4` vs R.3, and Weather h720 late segment is worse than R.3 by `+4.74%`. This directly falsifies the hypothesis that a fixed late adapter route is enough to repair the Weather conflict found by gradient diagnostics.

[Inference] The failure is likely not only about gradient destination. The fixed late adapter sees late residual pressure but has no state/difficulty condition and starts from a weak base that is already worse on Weather early/middle regions. It cannot decide when late signal is learnable vs noisy, so it behaves like a small late residual correction rather than a real supervision scheduler.

## Decision

[Decision] Do not enter full matrix. Do not sweep `aux_weight` or `adapter_start_step` yet. RG-A becomes a negative/partial evidence point: gradient routing is a viable axis, but fixed late routing is not enough.

[Rollback] Return to Step 5/6. Next candidate must use a dynamic conflict/predictability router, likely residual-stability-conditioned, and should route only units whose train-side evidence indicates learnable conflict rather than all late steps.
