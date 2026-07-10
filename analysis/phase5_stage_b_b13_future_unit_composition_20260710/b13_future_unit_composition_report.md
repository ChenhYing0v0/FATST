# Phase5 StageB B13-FUCO-B Prefix-Causal Composition Probe

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B13-FUCO` |
| `diagnostic_id` | `B13-FUCO-B` |
| `current_step` | Step 2/3：parameter-matched composition control |
| `scope` | frozen A6 memory；trainable diagnostic probes；not end-to-end model performance |
| `memory_source` | `coeff` |
| `decision` | `no_transition_control_explains` |

## Arms

- `parallel_no_transition`: every unit reads the same A6 coefficient memory and continuous coordinate independently;
- `prefix_causal_composed`: the same GRUCell additionally receives the previous latent unit state;
- both arms use identical modules and parameter count; no predicted values are fed back.

## Summary

| dataset | unit_size | seeds | composed_wins | mean_composed_vs_parallel_mse_pct | std_composed_vs_parallel_mse_pct | min_composed_vs_parallel_mse_pct | max_composed_vs_parallel_mse_pct | composition_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 180 | 3 | 1 | 11.3264 | 30.9635 | -31.9400 | 38.8015 | no |
| ETTh2 | 240 | 3 | 0 | 19.9064 | 8.5339 | 9.2046 | 30.0890 | no |
| ETTm1 | 180 | 3 | 0 | 4.0635 | 1.9231 | 1.6441 | 6.3491 | no |
| ETTm1 | 240 | 3 | 3 | -3.9800 | 1.7978 | -5.4391 | -1.4473 | yes |
| Weather | 180 | 3 | 2 | -3.2406 | 2.9861 | -6.9981 | 0.3074 | yes |
| Weather | 240 | 3 | 2 | -4.6688 | 3.7436 | -8.5092 | 0.4074 | yes |

## Gate Reading

[Decision] `no_transition_control_explains`.

- supported_settings=3/6
- dataset_non_degradation=ETTh2:False,ETTm1:True,Weather:True

[Fact] Maximum prefix-consistency absolute error across runs is `0.000000e+00`.
[Fact] Runs with test/validation MSE ratio above `2.0`: `0/36`.
[Fact] Trainable parameter sets by unit size: U180=[57396]; U240=[61296].

[Decision] Large-unit gradient pressure exists, but the parameter-matched no-transition probe explains the predictive value. Do not implement prefix-causal composition as a paper-core candidate.

[Rollback] Repair the intervention point once with pre-coefficient hidden memory before deciding the current GRU-based composition candidate.

## Failure Attribution Boundary

- `hypothesis_false`: not established by a frozen-coefficient probe alone;
- `intervention_point_wrong`: possible because the A6 coefficient may already discard information needed by unit composition;
- `readout_or_head_design_wrong`: possible if GRUCell/shared decoder is too weak;
- `optimization_or_numeric_pathology`: tracked by non-finite loss and val/test mismatch;
- `capacity_control_explains`: directly tested by exact parameter matching between the two arms.
