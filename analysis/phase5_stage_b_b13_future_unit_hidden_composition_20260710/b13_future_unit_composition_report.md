# Phase5 StageB B13-FUCO-B Prefix-Causal Composition Probe

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B13-FUCO` |
| `diagnostic_id` | `B13-FUCO-B2` |
| `current_step` | Step 2/3：parameter-matched composition control |
| `scope` | frozen A6 memory；trainable diagnostic probes；not end-to-end model performance |
| `memory_source` | `hidden` |
| `decision` | `no_transition_control_explains` |

## Arms

- `parallel_no_transition`: every unit reads the same A6 encoder hidden memory and continuous coordinate independently;
- `prefix_causal_composed`: the same GRUCell additionally receives the previous latent unit state;
- both arms use identical modules and parameter count; no predicted values are fed back.

## Summary

| dataset | unit_size | seeds | composed_wins | mean_composed_vs_parallel_mse_pct | std_composed_vs_parallel_mse_pct | min_composed_vs_parallel_mse_pct | max_composed_vs_parallel_mse_pct | composition_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 180 | 3 | 1 | 5.1639 | 20.0428 | -23.0849 | 21.3067 | no |
| ETTh2 | 240 | 3 | 1 | 5.3589 | 19.0149 | -21.2317 | 22.1264 | no |
| ETTm1 | 180 | 3 | 3 | -2.3454 | 0.7474 | -3.3426 | -1.5435 | yes |
| ETTm1 | 240 | 3 | 3 | -16.0953 | 2.7937 | -19.3779 | -12.5498 | yes |
| Weather | 180 | 3 | 2 | -1.8434 | 4.0261 | -5.3770 | 3.7899 | yes |
| Weather | 240 | 3 | 3 | -6.4462 | 1.5320 | -8.5504 | -4.9470 | yes |

## Gate Reading

[Decision] `no_transition_control_explains`.

- supported_settings=4/6
- dataset_non_degradation=ETTh2:False,ETTm1:True,Weather:True

[Fact] Maximum prefix-consistency absolute error across runs is `0.000000e+00`.
[Fact] Runs with test/validation MSE ratio above `2.0`: `0/36`.
[Fact] Trainable parameter sets by dataset/unit size: ETTh2-U180=[139316]; ETTh2-U240=[143216]; ETTm1-U180=[57396]; ETTm1-U240=[61296]; Weather-U180=[434228]; Weather-U240=[438128].

[Decision] Moving the intervention before the A6 coefficient bottleneck does not make the GRU-based transition beat its parameter-matched no-transition control. Close the current prefix-causal GRU composition candidate.

[Rollback] Return B13 to Step 2 and distinguish future-region-specific states from other non-recurrent future-stage generation mechanisms. Do not continue GRU/head tuning.

## Failure Attribution Boundary

- `hypothesis_false`: not established for all future-unit architectures; a failed B2 closes only the current GRU-based prefix-causal composition candidate;
- `intervention_point_wrong`: the post-coefficient bottleneck confound is removed, although a frozen hidden-memory probe is still not end-to-end adaptation;
- `readout_or_head_design_wrong`: remains possible in principle, but B2 is the pre-registered final repair and does not authorize head sweeps;
- `optimization_or_numeric_pathology`: tracked by non-finite loss and val/test mismatch;
- `capacity_control_explains`: directly tested by exact parameter matching between the two arms.
