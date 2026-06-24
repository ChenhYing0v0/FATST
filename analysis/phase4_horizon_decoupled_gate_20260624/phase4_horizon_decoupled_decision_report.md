# Phase4 Horizon-Decoupled Gate 决策报告

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 9-11 |
| `problem` | training supervision units 是否应与 evaluation horizons 解耦 |
| `existence_evidence` | 7 strategies x 3 datasets x 4 horizons remote gate |
| `idea` | horizon-decoupled future supervision units |
| `theory_check` | mask/interval/component/curriculum 应降低 task redundancy 或改善 optimization path |
| `design` | R.3 carrier, unchanged evaluation horizons, train-side unit strategies |
| `gate` | mean MSE vs R.3 < 0, wins >= 7/12, dataset degradation <= +0.5%, prefix stable, diagnostic support |
| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624` |
| `decision` | fail as paper-core; rollback to Step 4/6 and redesign supervision strategy |

## 主要结果

[Fact] Primary baseline `PatchEncoderR3PrefixRisk` 的 relative MSE 为 `0.00%`；`mse_wins` 使用严格 `<` 计算，因此 baseline 自身为 `0/12`。
[Fact] 最好的非 R.3 candidate 是 `D2_random_future_mask`，mean relative MSE +3.51%，相对 R.3 有 `1/12` 个 MSE wins。
[Fact] 相对 R.3，`full_time_mse` 为 +5.12%，random mask 为 +3.51%，interval 为 +4.12%。
[Fact] 相对 R.3，component top / component balanced / curriculum 分别为 +4.37%、+3.91%、+4.37%。

[Decision] 当前 Phase4-R 不通过 paper-core gate。没有任何 horizon-decoupled candidate 同时满足 mean relative MSE < 0、7/12 wins、dataset degradation <= +0.5%。

## 统计口径

本报告由 `scripts/analyze_phase4_horizon_decoupled_gate.py` 从 remote sync artifacts 生成。

| Quantity | Source | Computation | Meaning |
| --- | --- | --- | --- |
| `mse`, `mae` | 每个 run 的 `metrics_by_target_horizon.csv` | 直接读取 per target horizon evaluation | 固定 evaluation horizons 上的预测误差 |
| `relative_mse_pct` | candidate 与 R.3 同 dataset/horizon 的 `mse` | `(candidate / R.3 - 1) * 100` | 相对 R.3 的 MSE 变化，负数表示改进 |
| `mse_wins` | `relative_mse_pct` 对应的 raw `mse` | candidate `mse < R.3 mse` 的计数 | 12 个 dataset-horizon 设置中的严格胜出数 |
| `max_dataset_mean_degradation_pct` | dataset-level relative MSE | 每个 dataset 内取均值，再取最大 | 最坏 dataset 平均退化，用于避免单数据集被牺牲 |
| `segment_wins` | `h96/h720/metrics_by_segment.csv` | candidate segment MSE 严格小于 R.3 的计数 | H96/H720 局部误差是否改善 |
| `prefix_mismatch_mse` | `prefix_consistency.csv` | 同一 prefix 在不同 requested horizon 下的 prediction MSE mismatch | 检查 unified inference 的 prefix consistency 是否被破坏 |
| `mean_active_step_ratio` | `supervision_trace.csv` | batch trace 中 `mask_ratio` 平均 | 每步训练实际监督的 future positions 比例 |

## Strategy 汇总 vs R.3

| strategy | settings | mse_wins | mae_wins | mean_relative_mse_pct | mean_relative_mae_pct | max_dataset_mean_degradation_pct |
| --- | --- | --- | --- | --- | --- | --- |
| D1_r3_prefix_risk | 12 | 0 | 0 | +0.00% | +0.00% | +0.00% |
| D0_full_time_mse | 12 | 0 | 0 | +5.12% | +3.17% | +5.46% |
| D2_random_future_mask | 12 | 1 | 0 | +3.51% | +2.89% | +3.59% |
| D3_interval_supervision | 12 | 2 | 2 | +4.12% | +3.09% | +7.12% |
| D4_component_basis_top | 12 | 0 | 0 | +4.37% | +2.71% | +5.50% |
| D5_component_basis_balanced | 12 | 0 | 0 | +3.91% | +2.65% | +5.32% |
| D6_curriculum_units | 12 | 0 | 0 | +4.37% | +2.71% | +5.50% |

## Dataset 汇总 vs R.3

| strategy | dataset | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | ETTh2 | 0 | +5.46% | +3.54% |
| D0_full_time_mse | ETTm1 | 0 | +5.09% | +3.14% |
| D0_full_time_mse | Weather | 0 | +4.81% | +2.83% |
| D2_random_future_mask | ETTh2 | 0 | +3.59% | +2.45% |
| D2_random_future_mask | ETTm1 | 1 | +3.37% | +2.27% |
| D2_random_future_mask | Weather | 0 | +3.57% | +3.95% |
| D3_interval_supervision | ETTh2 | 2 | -0.36% | +0.56% |
| D3_interval_supervision | ETTm1 | 0 | +5.60% | +3.53% |
| D3_interval_supervision | Weather | 0 | +7.12% | +5.19% |
| D4_component_basis_top | ETTh2 | 0 | +3.71% | +2.18% |
| D4_component_basis_top | ETTm1 | 0 | +5.50% | +3.63% |
| D4_component_basis_top | Weather | 0 | +3.91% | +2.34% |
| D5_component_basis_balanced | ETTh2 | 0 | +2.79% | +1.97% |
| D5_component_basis_balanced | ETTm1 | 0 | +5.32% | +3.50% |
| D5_component_basis_balanced | Weather | 0 | +3.61% | +2.48% |
| D6_curriculum_units | ETTh2 | 0 | +3.71% | +2.18% |
| D6_curriculum_units | ETTm1 | 0 | +5.50% | +3.63% |
| D6_curriculum_units | Weather | 0 | +3.91% | +2.34% |

## Horizon 汇总 vs R.3

| strategy | horizon | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | 96 | 0 | +8.12% | +5.19% |
| D0_full_time_mse | 192 | 0 | +5.16% | +3.16% |
| D0_full_time_mse | 336 | 0 | +4.04% | +2.37% |
| D0_full_time_mse | 720 | 0 | +3.16% | +1.97% |
| D2_random_future_mask | 96 | 0 | +6.86% | +5.40% |
| D2_random_future_mask | 192 | 0 | +3.72% | +2.94% |
| D2_random_future_mask | 336 | 0 | +2.15% | +1.81% |
| D2_random_future_mask | 720 | 1 | +1.30% | +1.42% |
| D3_interval_supervision | 96 | 0 | +7.65% | +5.87% |
| D3_interval_supervision | 192 | 1 | +4.29% | +3.13% |
| D3_interval_supervision | 336 | 0 | +3.36% | +2.23% |
| D3_interval_supervision | 720 | 1 | +1.18% | +1.15% |
| D4_component_basis_top | 96 | 0 | +6.54% | +4.10% |
| D4_component_basis_top | 192 | 0 | +4.74% | +2.74% |
| D4_component_basis_top | 336 | 0 | +3.27% | +1.99% |
| D4_component_basis_top | 720 | 0 | +2.95% | +2.02% |
| D5_component_basis_balanced | 96 | 0 | +6.03% | +4.05% |
| D5_component_basis_balanced | 192 | 0 | +4.22% | +2.68% |
| D5_component_basis_balanced | 336 | 0 | +2.83% | +1.93% |
| D5_component_basis_balanced | 720 | 0 | +2.56% | +1.93% |
| D6_curriculum_units | 96 | 0 | +6.54% | +4.10% |
| D6_curriculum_units | 192 | 0 | +4.74% | +2.74% |
| D6_curriculum_units | 336 | 0 | +3.27% | +1.99% |
| D6_curriculum_units | 720 | 0 | +2.95% | +2.02% |

## Segment 汇总 vs R.3

| strategy | target_horizon | segment_wins | mean_relative_mse_pct |
| --- | --- | --- | --- |
| D0_full_time_mse | 96 | 0 | +8.12% |
| D0_full_time_mse | 720 | 1 | +3.96% |
| D2_random_future_mask | 96 | 0 | +6.86% |
| D2_random_future_mask | 720 | 2 | +2.55% |
| D3_interval_supervision | 96 | 0 | +7.65% |
| D3_interval_supervision | 720 | 3 | +3.52% |
| D4_component_basis_top | 96 | 0 | +6.54% |
| D4_component_basis_top | 720 | 1 | +3.62% |
| D5_component_basis_balanced | 96 | 0 | +6.03% |
| D5_component_basis_balanced | 720 | 2 | +3.22% |
| D6_curriculum_units | 96 | 0 | +6.54% |
| D6_curriculum_units | 720 | 1 | +3.62% |

## Prefix Consistency 诊断

| strategy | rows | max_prefix_mismatch_mse | mean_prefix_mismatch_mse |
| --- | --- | --- | --- |
| D1_r3_prefix_risk | 9 | 5.3671e-14 | 2.05406e-14 |
| D0_full_time_mse | 9 | 5.11844e-14 | 1.98958e-14 |
| D2_random_future_mask | 9 | 4.69931e-14 | 1.86752e-14 |
| D3_interval_supervision | 9 | 5.03727e-14 | 1.8317e-14 |
| D4_component_basis_top | 9 | 5.28235e-14 | 2.06733e-14 |
| D5_component_basis_balanced | 9 | 5.27741e-14 | 2.06054e-14 |
| D6_curriculum_units | 9 | 5.28235e-14 | 2.06733e-14 |

[Fact] 所有 strategy 的 prefix mismatch 都保持在 numerical-zero 量级，因此失败不是 prefix consistency 被破坏导致。

## Supervision Trace 汇总

| strategy | dataset | training_evaluation_decoupled | train_horizons_effective | epochs_ran | unit_types | curriculum_phases | mean_active_step_ratio | component_top_rank_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_r3_prefix_risk | ETTh2 | False | 96,192,336,720 | 11 | horizon_mixed:2000 | none:2000 | 1 | 0 |
| D1_r3_prefix_risk | ETTm1 | False | 96,192,336,720 | 11 | horizon_mixed:2000 | none:2000 | 1 | 0 |
| D1_r3_prefix_risk | Weather | False | 96,192,336,720 | 11 | horizon_mixed:2000 | none:2000 | 1 | 0 |
| D0_full_time_mse | ETTh2 | True | 720 | 13 | full_time:767 | none:767 | 1 | 0 |
| D0_full_time_mse | ETTm1 | True | 720 | 12 | full_time:2000 | none:2000 | 1 | 0 |
| D0_full_time_mse | Weather | True | 720 | 14 | full_time:2000 | none:2000 | 1 | 0 |
| D2_random_future_mask | ETTh2 | True | 720 | 14 | mask:826 | none:826 | 0.533333 | 0 |
| D2_random_future_mask | ETTm1 | True | 720 | 12 | mask:2000 | none:2000 | 0.533333 | 0 |
| D2_random_future_mask | Weather | True | 720 | 13 | mask:2000 | none:2000 | 0.533333 | 0 |
| D3_interval_supervision | ETTh2 | True | 720 | 11 | interval:649 | none:649 | 0.164253 | 0 |
| D3_interval_supervision | ETTm1 | True | 720 | 13 | interval:2000 | none:2000 | 0.168333 | 0 |
| D3_interval_supervision | Weather | True | 720 | 15 | interval:2000 | none:2000 | 0.168333 | 0 |
| D4_component_basis_top | ETTh2 | True | 720 | 14 | component_top:826 | none:826 | 1 | 0.873482 |
| D4_component_basis_top | ETTm1 | True | 720 | 12 | component_top:2000 | none:2000 | 1 | 0.880509 |
| D4_component_basis_top | Weather | True | 720 | 14 | component_top:2000 | none:2000 | 1 | 0.789863 |
| D5_component_basis_balanced | ETTh2 | True | 720 | 14 | component_balanced:826 | none:826 | 1 | 0.873482 |
| D5_component_basis_balanced | ETTm1 | True | 720 | 12 | component_balanced:2000 | none:2000 | 1 | 0.880509 |
| D5_component_basis_balanced | Weather | True | 720 | 14 | component_balanced:2000 | none:2000 | 1 | 0.789863 |
| D6_curriculum_units | ETTh2 | True | 720 | 14 | component_top:826 | coarse:826 | 1 | 0.873482 |
| D6_curriculum_units | ETTm1 | True | 720 | 12 | component_top:2000 | coarse:2000 | 1 | 0.880509 |
| D6_curriculum_units | Weather | True | 720 | 14 | component_top:2000 | coarse:2000 | 1 | 0.789863 |

## 结果解释

[Strong Evidence] 简单 horizon-decoupled supervision 不能取代 R.3。`full_time_mse`、mask、interval、component、curriculum 均未超过 R.3，说明 R.3 的 prefix-risk pressure 不是一个容易被 horizon-free unit sampling 替代的弱 baseline。

[Strong Evidence] 随机 mask / interval 的表现相对接近 R.3，但仍不满足 gate。这说明 stochastic future-unit scheduling 可能有 regularization 价值，但当前版本没有足够 paper-core 性能证据。

[Strong Evidence] component-based routes 系统性较差，且 component top 与 curriculum 的结果非常接近，说明第一阶段 top-component supervision 主导了 curriculum early trajectory。结合 residual projection audit，当前 component-basis route 不应作为下一步主线。

[Inference] 当前失败不是 evaluation/inference 接口问题：prefix consistency 没坏，evaluation horizons 也完整。更可能的问题是 train-side supervision unit 太粗糙：mask/interval 没有根据样本状态、future difficulty 或 error process 自适应分配 pressure；component supervision 又过度偏向 global variance basis。

## 下一步方向

[Decision] 回退到 Step 4/6，保留 training/evaluation 解耦问题，但停止当前 `D2-D6` 简单策略扩展。下一步不应继续调 mask ratio、interval length 或 component rank 的宽 sweep。

推荐进入 `Phase4-S`: State/Difficulty-Conditioned Supervision Scheduling。

核心问题：

> Horizon-free supervision units 仍然成立，但 unit pressure 不能是全局静态或随机；它应由 train-side difficulty / future-label novelty / error-process proxy 条件化。

最小下一步：

1. 用本轮 artifacts 做 post-hoc diagnostic：哪些 samples/segments 在 R.3 下高 residual，mask/interval 是否覆盖这些 regions。
2. 设计 `difficulty_conditioned_interval`：仍训练 720 future sequence，但 interval sampling probability 由 train-label novelty 或 running loss bucket 决定。
3. 设计 `r3_plus_sparse_unit_aux`：保留 R.3 base loss，只加小权重 horizon-free auxiliary unit，而不是替换 R.3 objective。
4. 先做 local diagnostic / small remote gate，再决定是否进入 full matrix。

Rollback point: Step 4/6。当前 HSS 问题保留，具体 idea 从 static horizon-decoupled replacement 改为 conditioned auxiliary scheduling。
