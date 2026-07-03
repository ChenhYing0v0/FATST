# Phase5-A5 Unified Head Sync Gate Report

## Reader Path

[What] 本报告评估通过 narrative gate 的两个 first-principles unified head family：`A5-B_continuous_forecast_basis_operator` 与 `A5-Q_elastic_causal_target_query_decoder`。

[Why] A5 的目标是判断 direct `[B,H,C]`、architecture-level prefix-consistent head 是否能替代现有 full-720 crop / dense-head controls，并成为 Stage A 的 paper-core interface。

[How] 远程矩阵为 `ETTh2 + ETTm1 + Weather` × `a5b_r64/a5b_r128/a5q_seg48_small/a5q_seg24_wide`，共 12 runs。每个 run 使用 `metrics_by_target_horizon.csv` 和 `metrics_by_segment.csv` 生成本报告。

[Metric] `mean_relative_mse_vs_*_pct = (A5_MSE / reference_MSE - 1) * 100`；负数表示 A5 更好。`best_stage_control` 是每个 dataset/horizon 上已有 unified controls 中 MSE 最低者，包括 official unified、A2、A3C、A3D-w03、A3E-best、H1 target-set 与 H1C row-gated。fixed per-horizon 只作为非同类参考，不纳入 `best_stage_control`。

[Prefix Contract] 本轮 smoke 结果保存于 `phase5_timealign_hss_a5_smoke.json`：A5-B 的 `decode(96)` vs `decode(720)[:, :96]` mismatch 为 `0.0`，A5-Q 为约 `4.77e-07`，说明 architecture-level prefix consistency 实现成立。

## 结论

[Decision] 本轮 A5-Q/A5-B effectiveness gate 结论：`failed_as_core_candidate`。

[Fact] ALL mean MSE 最优 A5 arm 是 `a5b_r128`，对 `best_stage_control` 的平均相对 MSE 为 `+14.19%`，wins 为 `0/12`。

[Interpretation] A5-B/A5-Q 的 architecture-level prefix consistency 已由 smoke 验证，但远程 forecasting effectiveness 没有稳定超过现有 unified controls。因此，本轮结果不能把 A5-B 或 A5-Q 直接提升为 paper-core unified head。

## ALL Summary

| arm | family | mean_mse | vs_best_stage_control | wins_vs_best_stage_control | vs_h1 | vs_h1c | vs_a3d_w03 | vs_fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a5b_r64 | A5-B | 0.338574 | +19.42% | 0/12 | +18.90% | +18.29% | +18.98% | +12.72% |
| a5b_r128 | A5-B | 0.322623 | +14.19% | 0/12 | +13.70% | +13.14% | +13.77% | +8.04% |
| a5q_seg48_small | A5-Q | 0.408625 | +42.41% | 0/12 | +41.66% | +41.14% | +41.70% | +35.67% |
| a5q_seg24_wide | A5-Q | 0.449493 | +55.62% | 0/12 | +54.76% | +54.24% | +54.81% | +48.43% |

## Best A5 Per Setting

| dataset | horizon | best_a5_arm | best_a5_mse | best_stage_control | rel_vs_control | beats_control |
| --- | ---: | --- | ---: | --- | ---: | --- |
| ETTh2 | 96 | a5b_r128 | 0.294489 | a3e_best | +22.47% | False |
| ETTh2 | 192 | a5b_r128 | 0.339575 | h1_target_set | +22.49% | False |
| ETTh2 | 336 | a5q_seg48_small | 0.364704 | a3d_w03 | +20.52% | False |
| ETTh2 | 720 | a5q_seg48_small | 0.429600 | a3d_w03 | +11.79% | False |
| ETTm1 | 96 | a5b_r128 | 0.299289 | a3c_warm | +10.60% | False |
| ETTm1 | 192 | a5b_r128 | 0.328235 | a3c_warm | +6.51% | False |
| ETTm1 | 336 | a5b_r128 | 0.360417 | a3c_warm | +3.96% | False |
| ETTm1 | 720 | a5b_r128 | 0.416318 | official_unified | +2.36% | False |
| Weather | 96 | a5b_r128 | 0.157379 | a2_nested | +11.38% | False |
| Weather | 192 | a5b_r64 | 0.205526 | a3d_w03 | +12.76% | False |
| Weather | 336 | a5b_r64 | 0.259967 | a2_nested | +12.22% | False |
| Weather | 720 | a5q_seg48_small | 0.329321 | a2_nested | +8.57% | False |

## Capacity Checks

| comparison | bigger_arm | ALL relative bigger vs base | bigger wins |
| --- | --- | ---: | ---: |
| A5-B_rank_capacity | a5b_r128 | -4.07% | 10/12 |
| A5-Q_query_capacity | a5q_seg24_wide | +7.76% | 0/12 |

## Segment Notes

| dataset | arm | h720_mean_segment_mse | late_vs_early_ratio |
| --- | --- | ---: | ---: |
| ETTh2 | a5b_r64 | 0.545637 | 1.813 |
| ETTh2 | a5b_r128 | 0.499843 | 1.823 |
| ETTh2 | a5q_seg48_small | 0.438588 | 1.488 |
| ETTh2 | a5q_seg24_wide | 0.463634 | 1.556 |
| ETTm1 | a5b_r64 | 0.431372 | 1.412 |
| ETTm1 | a5b_r128 | 0.421276 | 1.443 |
| ETTm1 | a5q_seg48_small | 0.639378 | 1.074 |
| ETTm1 | a5q_seg24_wide | 0.729883 | 1.034 |
| Weather | a5b_r64 | 0.354797 | 2.118 |
| Weather | a5b_r128 | 0.354202 | 2.108 |
| Weather | a5q_seg48_small | 0.334107 | 1.767 |
| Weather | a5q_seg24_wide | 0.337905 | 1.746 |

## Gate 判断

- A5-B: `failed_as_core_candidate`。rank 128 相比 rank 64 有稳定容量收益，但仍比 best existing unified control 差 `+14.19%`，说明当前 basis/operator class 的表达上限不足。
- A5-Q: `failed_as_core_candidate`。seg24-wide 相比 seg48-small 反而 `0/12` wins，说明简单加密 target segments / FF width 未修复 query decoder 的训练与容量问题。
- A5-S/A5-I/A5-M 不应自动进入远程；若继续 A5，需要先回 Step 4/5 做新的理论诊断，而不是扩 sweep。

## Rollback

[Decision] 按 11-step loop，本轮应回退到 Step 4/5：重新评估 first-principles unified head 的 capacity 机制。A5-Q/A5-B 的 prefix-consistency contract 成立，但当前 parameterization 的 forecasting capacity 不足。
