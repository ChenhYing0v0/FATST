# Phase4-S Predictability Gate 决策报告

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 9-11 |
| `problem` | predictability-aware downweight 是否能修复 CFUS-S1 在 Weather 上的 noisy-hard collapse |
| `existence_evidence` | train-only predictability diagnostic; ETTh2/Weather small remote gate; full-time, R.3, CFUS-S1 controls |
| `idea` | low-predictability noisy-hard blocks 降权，learnable-hard blocks 继续 auxiliary emphasis |
| `theory_check` | 若 Weather 失败来自 noisy-hard 污染 shared representation，则 downweight 应减少 Weather vs R.3 gap，同时保留 ETTh2 gain |
| `design` | `predictability_downweight` vs `full_time_mse`, `r3_prefix_risk`, and prior CFUS-S1 on ETTh2/Weather |
| `gate` | keep full-time gain; improve Weather vs R.3; no early/prefix collapse; trace confirms noisy/learnable split |
| `artifacts` | `analysis/phase4_s_predictability_gate_20260625` |
| `decision` | fail as paper-core; current predictability proxy/downweight is insufficient |

## 主要结果

[Fact] S2 vs `full_time_mse`: mean relative MSE -2.61%, MSE wins `4/8`。
[Fact] S2 vs `r3_prefix_risk`: mean relative MSE +2.35%, MSE wins `3/8`。
[Fact] S2 vs S1-CFUS: mean relative MSE +0.13%, MSE wins `2/8`。
[Fact] S2 vs R.3 on ETTh2: mean relative MSE -0.34%, wins `3/4`。
[Fact] S2 vs R.3 on Weather: mean relative MSE +5.05%, wins `0/4`。

[Decision] 当前 `predictability_downweight` 没有通过 paper-core gate。它基本保留了 ETTh2 上相对 full-time / R.3 的收益，但没有修复 Weather：相对 R.3 仍然 `0/4` wins，并且相对 full-time、相对 CFUS-S1 都略差。

## Main Metrics vs Controls

| dataset | horizon | baseline_strategy | mse | baseline_mse | relative_mse_pct | mae | baseline_mae | relative_mae_pct | mse_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | D0_full_time_mse | 0.312008 | 0.339358 | -8.06% | 0.367703 | 0.382523 | -3.87% | True |
| ETTh2 | 192 | D0_full_time_mse | 0.363125 | 0.38874 | -6.59% | 0.398805 | 0.413292 | -3.51% | True |
| ETTh2 | 336 | D0_full_time_mse | 0.382372 | 0.394549 | -3.09% | 0.41486 | 0.424757 | -2.33% | True |
| ETTh2 | 720 | D0_full_time_mse | 0.402262 | 0.419201 | -4.04% | 0.436983 | 0.448652 | -2.60% | True |
| Weather | 96 | D0_full_time_mse | 0.15519 | 0.154701 | +0.32% | 0.204754 | 0.204794 | -0.02% | False |
| Weather | 192 | D0_full_time_mse | 0.20103 | 0.200707 | +0.16% | 0.246335 | 0.246294 | +0.02% | False |
| Weather | 336 | D0_full_time_mse | 0.257228 | 0.256879 | +0.14% | 0.287003 | 0.286972 | +0.01% | False |
| Weather | 720 | D0_full_time_mse | 0.339505 | 0.338482 | +0.30% | 0.341938 | 0.341781 | +0.05% | False |
| ETTh2 | 96 | D1_r3_prefix_risk | 0.312008 | 0.304796 | +2.37% | 0.367703 | 0.358861 | +2.46% | False |
| ETTh2 | 192 | D1_r3_prefix_risk | 0.363125 | 0.369043 | -1.60% | 0.398805 | 0.400361 | -0.39% | True |
| ETTh2 | 336 | D1_r3_prefix_risk | 0.382372 | 0.38291 | -0.14% | 0.41486 | 0.415594 | -0.18% | True |
| ETTh2 | 720 | D1_r3_prefix_risk | 0.402262 | 0.410473 | -2.00% | 0.436983 | 0.439299 | -0.53% | True |
| Weather | 96 | D1_r3_prefix_risk | 0.15519 | 0.148026 | +4.84% | 0.204754 | 0.197029 | +3.92% | False |
| Weather | 192 | D1_r3_prefix_risk | 0.20103 | 0.192409 | +4.48% | 0.246335 | 0.239905 | +2.68% | False |
| Weather | 336 | D1_r3_prefix_risk | 0.257228 | 0.244793 | +5.08% | 0.287003 | 0.280433 | +2.34% | False |
| Weather | 720 | D1_r3_prefix_risk | 0.339505 | 0.320847 | +5.82% | 0.341938 | 0.333756 | +2.45% | False |

## S2 vs Prior S1-CFUS

| dataset | horizon | baseline_strategy | mse | baseline_mse | relative_mse_pct | mae | baseline_mae | relative_mae_pct | mse_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | S1_conditioned_future_unit_scheduling | 0.312008 | 0.312535 | -0.17% | 0.367703 | 0.368258 | -0.15% | True |
| ETTh2 | 192 | S1_conditioned_future_unit_scheduling | 0.363125 | 0.362985 | +0.04% | 0.398805 | 0.398901 | -0.02% | False |
| ETTh2 | 336 | S1_conditioned_future_unit_scheduling | 0.382372 | 0.38181 | +0.15% | 0.41486 | 0.414621 | +0.06% | False |
| ETTh2 | 720 | S1_conditioned_future_unit_scheduling | 0.402262 | 0.402277 | -0.00% | 0.436983 | 0.437074 | -0.02% | True |
| Weather | 96 | S1_conditioned_future_unit_scheduling | 0.15519 | 0.154855 | +0.22% | 0.204754 | 0.203708 | +0.51% | False |
| Weather | 192 | S1_conditioned_future_unit_scheduling | 0.20103 | 0.200753 | +0.14% | 0.246335 | 0.245237 | +0.45% | False |
| Weather | 336 | S1_conditioned_future_unit_scheduling | 0.257228 | 0.256679 | +0.21% | 0.287003 | 0.285803 | +0.42% | False |
| Weather | 720 | S1_conditioned_future_unit_scheduling | 0.339505 | 0.337873 | +0.48% | 0.341938 | 0.340392 | +0.45% | False |

## Overall Summary

| baseline_strategy | settings | mse_wins | mae_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- |
| D0_full_time_mse | 8 | 4 | 5 | -2.61% | -1.53% |
| D1_r3_prefix_risk | 8 | 3 | 3 | +2.35% | +1.60% |

## Dataset Summary

| baseline_strategy | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | ETTh2 | 4 | 4 | -5.44% |
| D0_full_time_mse | Weather | 4 | 0 | +0.23% |
| D1_r3_prefix_risk | ETTh2 | 4 | 3 | -0.34% |
| D1_r3_prefix_risk | Weather | 4 | 0 | +5.05% |

## Horizon Summary

| baseline_strategy | horizon | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | 96 | 2 | 1 | -3.87% |
| D0_full_time_mse | 192 | 2 | 1 | -3.21% |
| D0_full_time_mse | 336 | 2 | 1 | -1.48% |
| D0_full_time_mse | 720 | 2 | 1 | -1.87% |
| D1_r3_prefix_risk | 96 | 2 | 0 | +3.60% |
| D1_r3_prefix_risk | 192 | 2 | 1 | +1.44% |
| D1_r3_prefix_risk | 336 | 2 | 1 | +2.47% |
| D1_r3_prefix_risk | 720 | 2 | 1 | +1.91% |

## Segment Future-Region Summary

| baseline_strategy | future_region | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | early_1_96 | 8 | 4 | -2.86% |
| D0_full_time_mse | late_337_720 | 2 | 1 | -2.46% |
| D0_full_time_mse | middle_97_336 | 10 | 5 | -1.31% |
| D1_r3_prefix_risk | early_1_96 | 8 | 0 | +3.75% |
| D1_r3_prefix_risk | late_337_720 | 2 | 1 | +1.50% |
| D1_r3_prefix_risk | middle_97_336 | 10 | 3 | +1.74% |

## Trace Summary

| strategy | dataset | training_evaluation_decoupled | train_horizons_effective | unit_type | epochs_ran | mean_active_steps | mean_learnable_blocks | mean_noisy_blocks | mean_floor_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2_predictability_downweight | ETTh2 | True | 720 | predictability_downweight | 11 | 148.807 | 3.10015 | 0.906009 | 0.5 |
| S2_predictability_downweight | Weather | True | 720 | predictability_downweight | 14 | 99.912 | 2.0815 | 2.1085 | 0.5 |
| D0_full_time_mse | ETTh2 | True | 720 | full_time | 13 | 720 | 0 | 0 | 0 |
| D0_full_time_mse | Weather | True | 720 | full_time | 14 | 720 | 0 | 0 | 0 |
| D1_r3_prefix_risk | ETTh2 | False | 96,192,336,720 | horizon_mixed | 11 | 342.288 | 0 | 0 | 0 |
| D1_r3_prefix_risk | Weather | False | 96,192,336,720 | horizon_mixed | 11 | 342.288 | 0 | 0 | 0 |

## Prefix Consistency

| strategy | dataset | rows | max_prefix_mismatch_mse | mean_prefix_mismatch_mse |
| --- | --- | --- | --- | --- |
| S2_predictability_downweight | ETTh2 | 3 | 9.90429e-15 | 8.72364e-15 |
| S2_predictability_downweight | Weather | 3 | 9.4519e-15 | 3.15063e-15 |
| D0_full_time_mse | ETTh2 | 3 | 1.36516e-14 | 1.19759e-14 |
| D0_full_time_mse | Weather | 3 | 9.6629e-15 | 3.22097e-15 |
| D1_r3_prefix_risk | ETTh2 | 3 | 1.47352e-14 | 1.21106e-14 |
| D1_r3_prefix_risk | Weather | 3 | 9.95966e-15 | 3.31989e-15 |

## 机制判断

[Strong Evidence] S2 的 train/eval 解耦实现是干净的：`train_horizons_effective=720`，evaluation horizons 仍只用于测试，prefix mismatch 保持 numerical-zero。

[Strong Evidence] Trace 证明 noisy/learnable split 确实发生：ETTh2 平均约 `3.10` learnable blocks、`0.91` noisy blocks；Weather 平均约 `2.08` learnable blocks、`2.11` noisy blocks。这与 offline diagnostic 对 Weather noisy-hard 的判断一致。

[Counter-Evidence] 尽管 split 正确发生，Weather 指标没有改善。说明当前简单 proxy `top_novelty ∩ top_variation` 加 `floor_weight=0.5` 不足以解决 Weather 的 interference，或者 Weather 的主差距不是仅靠降低 noisy-hard dense weight 能解决。

[Counter-Evidence] S2 相比 S1-CFUS 没有带来实质改进：整体 mean relative MSE 为正，Weather 四个 horizons 都差于 S1。这说明当前 downweight formulation 把有效 hard-block emphasis 削弱了，但没有换来足够的 noise shielding。

## 下一步

[Decision] 不进入 full matrix，不继续 sweep 当前 `floor_weight`。当前失败的是 S2 的简单 downweight implementation，不是 predictability-conditioned scheduling 问题本身。

建议回退到 Step 5/6：

1. 重新评估 predictability proxy：仅用 local variation 过粗，需引入 train-only baseline residual / seasonal residual stability。
2. 如果继续 shielding，优先考虑 `detached/isolated auxiliary path`，而不是在 shared dense loss 中简单降权。
3. 保留 S1-CFUS 作为 evidence：hard-block emphasis 对 ETTh2 有效，但需要 dataset/state-aware gate 决定何时启用。
4. 下一轮不应再只改 scalar weights，应先做 train-side residual predictability diagnostic，确认 Weather 的 low-predictability units 是否可由更强 proxy 分离。
