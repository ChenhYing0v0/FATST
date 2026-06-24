# Phase4-S CFUS Small Gate 决策报告

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 9-11 |
| `problem` | conditioned future-unit scheduling 是否能成为独立 HSS training strategy |
| `existence_evidence` | ETTh2/Weather small remote gate, full-time and R.3 controls |
| `idea` | full future dense anchor + train-side label-novelty sparse unit pressure |
| `theory_check` | 若 condition 能定向补强 hard units，应优于 full-time 且缩小 R.3 gap，不应造成 h96/Weather collapse |
| `design` | `conditioned_future_unit_scheduling` vs `full_time_mse` vs `r3_prefix_risk` on ETTh2/Weather |
| `gate` | beat full-time, close R.3 gap, no h96/Weather collapse, trace confirms conditioned units |
| `artifacts` | `analysis/phase4_s_cfus_gate_20260624` |
| `decision` | fail as paper-core; pass only as weak evidence that conditioned auxiliary improves full-time anchor |

## 主要结果

[Fact] CFUS vs `full_time_mse`: mean relative MSE -2.74%, MSE wins `6/8`。
[Fact] CFUS vs `r3_prefix_risk`: mean relative MSE +2.22%, MSE wins `3/8`。
[Fact] CFUS vs R.3 on ETTh2: mean relative MSE -0.35%, wins `3/4`。
[Fact] CFUS vs R.3 on Weather: mean relative MSE +4.78%, wins `0/4`。

[Decision] 当前 CFUS 不通过 paper-core gate。它证明 conditioned sparse auxiliary 明显优于 plain full-time dense MSE，尤其 ETTh2；但仍未形成能替代或接近 R.3 的稳定 training strategy，Weather 相对 R.3 全面退化。

## Main Metrics vs Controls

| dataset | horizon | baseline_strategy | mse | baseline_mse | relative_mse_pct | mae | baseline_mae | relative_mae_pct | mse_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | D0_full_time_mse | 0.312535 | 0.339358 | -7.90% | 0.368258 | 0.382523 | -3.73% | True |
| ETTh2 | 192 | D0_full_time_mse | 0.362985 | 0.38874 | -6.63% | 0.398901 | 0.413292 | -3.48% | True |
| ETTh2 | 336 | D0_full_time_mse | 0.38181 | 0.394549 | -3.23% | 0.414621 | 0.424757 | -2.39% | True |
| ETTh2 | 720 | D0_full_time_mse | 0.402277 | 0.419201 | -4.04% | 0.437074 | 0.448652 | -2.58% | True |
| Weather | 96 | D0_full_time_mse | 0.154855 | 0.154701 | +0.10% | 0.203708 | 0.204794 | -0.53% | False |
| Weather | 192 | D0_full_time_mse | 0.200753 | 0.200707 | +0.02% | 0.245237 | 0.246294 | -0.43% | False |
| Weather | 336 | D0_full_time_mse | 0.256679 | 0.256879 | -0.08% | 0.285803 | 0.286972 | -0.41% | True |
| Weather | 720 | D0_full_time_mse | 0.337873 | 0.338482 | -0.18% | 0.340392 | 0.341781 | -0.41% | True |
| ETTh2 | 96 | D1_r3_prefix_risk | 0.312535 | 0.304796 | +2.54% | 0.368258 | 0.358861 | +2.62% | False |
| ETTh2 | 192 | D1_r3_prefix_risk | 0.362985 | 0.369043 | -1.64% | 0.398901 | 0.400361 | -0.36% | True |
| ETTh2 | 336 | D1_r3_prefix_risk | 0.38181 | 0.38291 | -0.29% | 0.414621 | 0.415594 | -0.23% | True |
| ETTh2 | 720 | D1_r3_prefix_risk | 0.402277 | 0.410473 | -2.00% | 0.437074 | 0.439299 | -0.51% | True |
| Weather | 96 | D1_r3_prefix_risk | 0.154855 | 0.148026 | +4.61% | 0.203708 | 0.197029 | +3.39% | False |
| Weather | 192 | D1_r3_prefix_risk | 0.200753 | 0.192409 | +4.34% | 0.245237 | 0.239905 | +2.22% | False |
| Weather | 336 | D1_r3_prefix_risk | 0.256679 | 0.244793 | +4.86% | 0.285803 | 0.280433 | +1.91% | False |
| Weather | 720 | D1_r3_prefix_risk | 0.337873 | 0.320847 | +5.31% | 0.340392 | 0.333756 | +1.99% | False |

## Overall Summary

| baseline_strategy | settings | mse_wins | mae_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- |
| D0_full_time_mse | 8 | 6 | 8 | -2.74% | -1.74% |
| D1_r3_prefix_risk | 8 | 3 | 3 | +2.22% | +1.38% |

## Dataset Summary

| baseline_strategy | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | ETTh2 | 4 | 4 | -5.45% |
| D0_full_time_mse | Weather | 4 | 2 | -0.03% |
| D1_r3_prefix_risk | ETTh2 | 4 | 3 | -0.35% |
| D1_r3_prefix_risk | Weather | 4 | 0 | +4.78% |

## Horizon Summary

| baseline_strategy | horizon | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | 96 | 2 | 1 | -3.90% |
| D0_full_time_mse | 192 | 2 | 1 | -3.30% |
| D0_full_time_mse | 336 | 2 | 2 | -1.65% |
| D0_full_time_mse | 720 | 2 | 2 | -2.11% |
| D1_r3_prefix_risk | 96 | 2 | 0 | +3.58% |
| D1_r3_prefix_risk | 192 | 2 | 1 | +1.35% |
| D1_r3_prefix_risk | 336 | 2 | 1 | +2.28% |
| D1_r3_prefix_risk | 720 | 2 | 1 | +1.65% |

## Segment Future-Region Summary

| baseline_strategy | future_region | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | early_1_96 | 8 | 4 | -2.93% |
| D0_full_time_mse | late_337_720 | 2 | 2 | -2.68% |
| D0_full_time_mse | middle_97_336 | 10 | 10 | -1.52% |
| D1_r3_prefix_risk | early_1_96 | 8 | 0 | +3.67% |
| D1_r3_prefix_risk | late_337_720 | 2 | 1 | +1.26% |
| D1_r3_prefix_risk | middle_97_336 | 10 | 3 | +1.52% |

## Trace Summary

| strategy | dataset | training_evaluation_decoupled | train_horizons_effective | step_loss_weighting | unit_type | epochs_ran | mean_mask_ratio | condition_types | mean_condition_top_blocks | mean_auxiliary_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_conditioned_future_unit_scheduling | ETTh2 | True | 720 | uniform | conditioned_sparse | 11 | 0.266667 | label_novelty | 4 | 0.1 |
| S1_conditioned_future_unit_scheduling | Weather | True | 720 | uniform | conditioned_sparse | 14 | 0.266667 | label_novelty | 4 | 0.1 |
| D0_full_time_mse | ETTh2 | True | 720 | uniform | full_time | 13 | 1 | none | 0 | 0 |
| D0_full_time_mse | Weather | True | 720 | uniform | full_time | 14 | 1 | none | 0 | 0 |
| D1_r3_prefix_risk | ETTh2 | False | 96,192,336,720 | prefix_risk | horizon_mixed | 11 | 1 | none | 0 | 0 |
| D1_r3_prefix_risk | Weather | False | 96,192,336,720 | prefix_risk | horizon_mixed | 11 | 1 | none | 0 | 0 |

## Prefix Consistency

| strategy | dataset | rows | max_prefix_mismatch_mse | mean_prefix_mismatch_mse |
| --- | --- | --- | --- | --- |
| S1_conditioned_future_unit_scheduling | ETTh2 | 3 | 9.81606e-15 | 8.65452e-15 |
| S1_conditioned_future_unit_scheduling | Weather | 3 | 9.462e-15 | 3.154e-15 |
| D0_full_time_mse | ETTh2 | 3 | 1.36516e-14 | 1.19759e-14 |
| D0_full_time_mse | Weather | 3 | 9.6629e-15 | 3.22097e-15 |
| D1_r3_prefix_risk | ETTh2 | 3 | 1.47352e-14 | 1.21106e-14 |
| D1_r3_prefix_risk | Weather | 3 | 9.95966e-15 | 3.31989e-15 |

## 机制判断

[Strong Evidence] CFUS 的 train/eval 解耦实现是干净的：`train_horizons_effective=720`，`step_loss_weighting=uniform`，trace 中 `unit_type=conditioned_sparse`，并记录 `condition_type=label_novelty`。

[Strong Evidence] CFUS 相比 full-time dense MSE 有实质收益：ETTh2 四个 horizons 全赢；Weather 基本持平，h336/h720 小幅赢。这说明 conditioned sparse pressure 不是完全无效。

[Counter-Evidence] CFUS 相比 R.3 不稳定：ETTh2 在 h192/h336/h720 赢，但 h96 输；Weather 四个 horizons 全输。这直接触发 small gate 的 no Weather collapse / close R.3 gap 条件失败。

[Diagnostic Gap] 当前 trace 只记录 `condition_top_blocks=4` 和 condition score，没有记录具体 selected block indices。因此它不能证明 selected units 没有退化为固定 late blocks。下一步若继续 CFUS，必须记录 selected block ranges 或 block-index histogram。

## 下一步

[Decision] 不进入 full matrix，不继续用当前 `label_novelty + top_ratio=0.25 + aux=0.1` 宽 sweep。

建议回退到 Step 6，重新设计 condition 可观测性与 schedule：

1. 先补 trace：记录 selected block indices / block ranges / per-block condition scores。
2. 做 offline train-label condition diagnostic：判断 `label_novelty` 是否长期偏向 late blocks；若是，它只是 late weighting proxy。
3. 设计 CFUS-v2：condition 需要同时保护 early/easy regions，可考虑 `balanced condition buckets` 或 `novelty within future-region groups`。
4. 只有 CFUS-v2 local trace 证明不是固定 late weighting 后，再做 small gate。
