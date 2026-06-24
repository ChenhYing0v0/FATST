# Phase4-S Predictability Diagnostic

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 4-6 diagnostic |
| `problem` | 当前 CFUS 把 high-novelty blocks 当作 hard units 加压，但 high novelty 可能是 low-predictability noise-hard |
| `existence_evidence` | CFUS beats full-time but collapses vs R.3 on Weather; SRP++ supports future-step interference / specialization concerns |
| `idea` | predictability-conditioned supervision scheduling: learnable-hard 加压，low-predictability hard 降权或隔离 |
| `theory_check` | 如果 hard units 含不可预测扰动，强加压会污染 shared representation；应先判断 condition 选择的 block 类型 |
| `design` | train-only block diagnostic using label novelty, seasonal naive error, local variation, and selected-block distribution |
| `gate` | 若 selected blocks 显著偏 late/high-variation/high-naive-error，则 CFUS-v2 应转向 predictability-aware downweight/isolation |
| `artifacts` | `analysis/phase4_predictability_diagnostic_20260624` |
| `decision` | current S1 condition mixes learnable-hard and noisy-hard; CFUS-v2 should use predictability-aware downweight/isolation rather than simple hard-block emphasis |

## 指标定义

- `novelty_mse`: 当前 CFUS 的 `label_novelty`，即 future block 相对最后一个 history step 的 MSE。
- `seasonal24_mse`: 24-step seasonal naive reference 的 MSE；用于粗略衡量 block 是否可由简单周期结构解释。
- `best_naive_mse`: `min(novelty_mse, seasonal24_mse)`；越高表示简单可预测性越弱。
- `local_variation`: future block 内部一阶差分能量；越高表示局部扰动越强。
- `smoothness_ratio`: `local_variation / novelty_mse`；高 novelty 且高 ratio 更像 noise-hard，低 ratio 更像 smooth shift / learnable-hard。

## Label-Novelty 选择分布

| Dataset | Group | Selected share | Balanced expectation | Mean novelty | Mean best naive | Mean variation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | early_blocks | 0.210 | 0.333 | 0.786 | 0.625 | 0.087 |
| ETTh2 | middle_blocks | 0.304 | 0.333 | 1.129 | 0.937 | 0.086 |
| ETTh2 | late_blocks | 0.485 | 0.333 | 1.342 | 1.151 | 0.085 |
| Weather | early_blocks | 0.244 | 0.333 | 1.114 | 0.915 | 0.222 |
| Weather | middle_blocks | 0.320 | 0.333 | 1.247 | 1.027 | 0.222 |
| Weather | late_blocks | 0.436 | 0.333 | 1.374 | 1.138 | 0.223 |

## Selected vs Non-Selected Blocks

| Dataset | Selection | Values | Mean novelty | Mean best naive | Mean variation | Smoothness ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | selected_by_label_novelty | 30340 | 1.901 | 1.617 | 0.098 | 0.133 |
| ETTh2 | not_selected_by_label_novelty | 83435 | 0.789 | 0.645 | 0.081 | 0.300 |
| Weather | selected_by_label_novelty | 143328 | 2.223 | 1.952 | 0.629 | 0.124 |
| Weather | not_selected_by_label_novelty | 394152 | 0.889 | 0.691 | 0.074 | 0.123 |

## 初步解释规则

[Decision Rule] 如果 selected blocks 的 `best_naive_mse` 和 `local_variation` 同时显著高于 non-selected blocks，当前 `label_novelty` 更可能混入 low-predictability / noisy-hard units；下一版不应继续简单加压。

[Decision Rule] 如果 selected share 大幅偏向 late blocks，当前 condition 可能退化为 late weighting proxy；下一版需要 region-balanced 或 predictability-aware selection。

## 当前判断

[Fact] `ETTh2` selected blocks vs non-selected: best-naive MSE ratio `2.51x`, local-variation ratio `1.20x`, late-block selected share `0.485`.
[Inference] `ETTh2` 的 high-novelty selected blocks 没有显著提高 local variation；它们更像 smooth shift / learnable-hard，因此 hard-block emphasis 可能有效。
[Fact] `Weather` selected blocks vs non-selected: best-naive MSE ratio `2.83x`, local-variation ratio `8.45x`, late-block selected share `0.436`.
[Strong Evidence] `Weather` 的 high-novelty selected blocks 同时是 high-variation blocks；这更接近 low-predictability / noisy-hard，而不只是 learnable-hard。

[Decision] 当前 `label_novelty` 不是稳定的 difficulty proxy。它在 ETTh2 上更像 learnable-hard selector，但在 Weather 上明显选中 high-variation / low-predictability blocks。下一版不应继续给所有 high-novelty blocks 加压，而应区分 learnable-hard 与 noisy-hard：前者可加压，后者应降权或隔离。
