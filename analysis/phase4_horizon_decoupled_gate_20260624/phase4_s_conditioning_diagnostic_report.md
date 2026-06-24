# Phase4-S Conditioned Scheduling 事后诊断

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 3-5 diagnostic for Phase4-S |
| `problem` | 静态 horizon-free units 有局部 wins 但整体输 R.3，是否需要 train-side condition |
| `existence_evidence` | Phase4-R segment artifacts, supervision traces, R.3 residual buckets |
| `idea` | 根据 train-side difficulty / residual proxy 调整 horizon-free unit pressure |
| `theory_check` | 如果 D2/D3 的 wins 集中在 high-residual 或 late regions，conditioned schedule 比 global static schedule 更有动机 |
| `design` | 只做 post-hoc segment diagnostic，不训练新模型 |
| `gate` | 找到局部有效区域和当前 static coverage mismatch；否则 Phase4-S 必须回退到文献调研 |
| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624` |
| `decision` | conditioned scheduling 可作为 hypothesis 继续推进，但尚未通过 implementation gate |

## 统计口径

| Quantity | Source | Computation | Meaning |
| --- | --- | --- | --- |
| `baseline_mse` | R.3 `metrics_by_segment.csv` | same dataset/horizon/segment lookup | R.3 在局部 future segment 上的误差 |
| `r3_residual_bucket` | all R.3 segment MSE rows | global tertiles over R.3 segment MSE | low/mid/high residual proxy |
| `future_region` | segment endpoint | `<=96`, `<=336`, `>336` | early/middle/late future region |
| `relative_mse_pct` | candidate vs R.3 segment MSE | `(candidate / R.3 - 1) * 100` | 局部 segment 相对变化 |
| `mean_mask_ratio` | `supervision_trace.csv` | trace average | 静态 strategy 实际监督密度 |

## Strategy-level 诊断

| strategy | segments | mse_wins | mean_relative_mse_pct | high_residual_wins | high_residual_mean_relative_mse_pct | late_wins | late_mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0_full_time_mse | 30 | 1 | +4.87% | 1 | +2.28% | 1 | +2.63% |
| D2_random_future_mask | 30 | 4 | +3.39% | 4 | +0.45% | 1 | +0.53% |
| D3_interval_supervision | 30 | 5 | +4.39% | 5 | +0.05% | 2 | -0.46% |
| D4_component_basis_top | 30 | 2 | +4.22% | 2 | +2.13% | 1 | +2.48% |
| D5_component_basis_balanced | 30 | 3 | +3.77% | 3 | +1.61% | 1 | +2.11% |
| D6_curriculum_units | 30 | 2 | +4.22% | 2 | +2.13% | 1 | +2.48% |

## R.3 Residual Bucket 汇总

| strategy | r3_residual_bucket | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | high_r3_residual | 10 | 1 | +2.28% |
| D0_full_time_mse | low_r3_residual | 10 | 0 | +5.60% |
| D0_full_time_mse | mid_r3_residual | 10 | 0 | +6.73% |
| D2_random_future_mask | high_r3_residual | 10 | 4 | +0.45% |
| D2_random_future_mask | low_r3_residual | 10 | 0 | +4.86% |
| D2_random_future_mask | mid_r3_residual | 10 | 0 | +4.86% |
| D3_interval_supervision | high_r3_residual | 10 | 5 | +0.05% |
| D3_interval_supervision | low_r3_residual | 10 | 0 | +7.90% |
| D3_interval_supervision | mid_r3_residual | 10 | 0 | +5.23% |
| D4_component_basis_top | high_r3_residual | 10 | 2 | +2.13% |
| D4_component_basis_top | low_r3_residual | 10 | 0 | +4.38% |
| D4_component_basis_top | mid_r3_residual | 10 | 0 | +6.15% |
| D5_component_basis_balanced | high_r3_residual | 10 | 3 | +1.61% |
| D5_component_basis_balanced | low_r3_residual | 10 | 0 | +4.02% |
| D5_component_basis_balanced | mid_r3_residual | 10 | 0 | +5.67% |
| D6_curriculum_units | high_r3_residual | 10 | 2 | +2.13% |
| D6_curriculum_units | low_r3_residual | 10 | 0 | +4.38% |
| D6_curriculum_units | mid_r3_residual | 10 | 0 | +6.15% |

## Future Region 汇总

| strategy | future_region | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | early_1_96 | 12 | 0 | +7.56% |
| D0_full_time_mse | late_337_720 | 3 | 1 | +2.63% |
| D0_full_time_mse | middle_97_336 | 15 | 0 | +3.17% |
| D2_random_future_mask | early_1_96 | 12 | 0 | +6.53% |
| D2_random_future_mask | late_337_720 | 3 | 1 | +0.53% |
| D2_random_future_mask | middle_97_336 | 15 | 3 | +1.45% |
| D3_interval_supervision | early_1_96 | 12 | 0 | +8.02% |
| D3_interval_supervision | late_337_720 | 3 | 2 | -0.46% |
| D3_interval_supervision | middle_97_336 | 15 | 3 | +2.46% |
| D4_component_basis_top | early_1_96 | 12 | 0 | +6.22% |
| D4_component_basis_top | late_337_720 | 3 | 1 | +2.48% |
| D4_component_basis_top | middle_97_336 | 15 | 1 | +2.96% |
| D5_component_basis_balanced | early_1_96 | 12 | 0 | +5.68% |
| D5_component_basis_balanced | late_337_720 | 3 | 1 | +2.11% |
| D5_component_basis_balanced | middle_97_336 | 15 | 2 | +2.57% |
| D6_curriculum_units | early_1_96 | 12 | 0 | +6.22% |
| D6_curriculum_units | late_337_720 | 3 | 1 | +2.48% |
| D6_curriculum_units | middle_97_336 | 15 | 1 | +2.96% |

## Trace 汇总

| strategy | unit_type | trace_rows | mean_active_steps | mean_mask_ratio | mean_loss_unit |
| --- | --- | --- | --- | --- | --- |
| D1_r3_prefix_risk | horizon_mixed | 6000 | 342.288 | 1 | 0.559692 |
| D0_full_time_mse | full_time | 4767 | 720 | 1 | 0.504166 |
| D2_random_future_mask | mask | 4826 | 384 | 0.533333 | 0.515811 |
| D3_interval_supervision | interval | 4649 | 120.79 | 0.167764 | 0.552093 |
| D4_component_basis_top | component_top | 4826 | 720 | 1 | 0.744316 |
| D5_component_basis_balanced | component_balanced | 4826 | 720 | 1 | 0.788066 |
| D6_curriculum_units | component_top | 4826 | 720 | 1 | 0.744316 |

## 解释

[Fact] `D2_random_future_mask` 在 segment-level 有 `4/30` wins，high-residual bucket 为 `4/10` wins。
[Fact] `D3_interval_supervision` 在 segment-level 有 `5/30` wins，high-residual bucket 为 `5/10` wins。

[Strong Evidence] D2/D3 的 wins 集中在 high-residual bucket：D2 为 4/10，D3 为 5/10；low/mid residual bucket 为 0 wins。

[Strong Evidence] D3 在 late region 达到 2/3 wins 且 mean relative MSE 为负；但 early region 为 0/12 wins 且显著退化。

[Inference] static strategy 的失败更像 coverage/pressure mismatch，而不是 horizon-free unit 完全无效。一个全局固定的 mask/interval 会在 early/easy regions 施加不必要 pressure，同时没有把足够 pressure 定向给 high-residual/late regions。

[Boundary] 当前 artifacts 只有 segment-level aggregate，没有 per-sample residual、label novelty 或 running-loss bucket。因此本报告不能证明具体 difficulty proxy 已成立，只能决定 Phase4-S 是否值得进入 Step 4-6 设计。

## Phase4-S 设计含义

[Decision] Phase4-S 可以作为 hypothesis 继续推进，但实现前必须先定义 train-side condition，且该 condition 不能直接使用 evaluation horizon label。

推荐优先级：

1. `S1_conditioned_future_unit_scheduling`：使用 full future dense anchor + train-side conditioned sparse unit pressure，作为独立 HSS training strategy。
2. `S2_difficulty_conditioned_interval`：用 train-label novelty 或 running loss bucket 条件化 interval sampling。
3. `S3_r3_plus_aux_control`：只作为 conflict/control，不作为 paper-core。
