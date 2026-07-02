# Phase5 A4S Validation Prefix Signal Export

## 诊断目标

A4S 检查 prefix-wise validation diagnostics 是否比 A4R 的 run-level logs 更能解释 `gap_to_best`。
本报告仍是 diagnostic-only，不是 routing method。

Rows analyzed: `84`.

## ALL-Level Signal Ranking

| rank | signal | valid_settings | pearson | spearman |
| --- | --- | --- | --- | --- |
| 1 | `teacher_student_mae` | 84 | 0.586 | 0.388 |
| 2 | `teacher_student_mse` | 84 | 0.558 | 0.354 |
| 3 | `full_context_prefix_mse` | 63 | -0.285 | -0.245 |
| 4 | `residual_std` | 84 | -0.267 | -0.238 |
| 5 | `validation_prefix_mse` | 84 | -0.254 | -0.236 |
| 6 | `validation_prefix_mae` | 84 | 0.082 | 0.153 |
| 7 | `residual_abs_mean` | 84 | 0.082 | 0.153 |
| 8 | `prefix_vs_full_mse` | 63 | -0.136 | -0.019 |

## Dataset-Level Top Signals

| dataset | signal | valid_settings | pearson | spearman |
| --- | --- | --- | --- | --- |
| ALL | `teacher_student_mae` | 84 | 0.586 | 0.388 |
| ALL | `teacher_student_mse` | 84 | 0.558 | 0.354 |
| ALL | `full_context_prefix_mse` | 63 | -0.285 | -0.245 |
| ETTh2 | `teacher_student_mae` | 28 | 0.567 | 0.629 |
| ETTh2 | `teacher_student_mse` | 28 | 0.424 | 0.625 |
| ETTh2 | `prefix_vs_full_mse` | 21 | -0.496 | -0.466 |
| ETTm1 | `teacher_student_mse` | 28 | -0.408 | -0.596 |
| ETTm1 | `residual_std` | 28 | -0.623 | -0.436 |
| ETTm1 | `validation_prefix_mse` | 28 | -0.605 | -0.435 |
| Weather | `full_context_prefix_mse` | 21 | -0.838 | -0.851 |
| Weather | `validation_prefix_mae` | 28 | -0.644 | -0.686 |
| Weather | `residual_abs_mean` | 28 | -0.644 | -0.686 |

## Gate Decision

- [Fact] ALL-level 最强 signal 是 `teacher_student_mae`，Spearman 为 `0.388`。
- [Decision] A4S 未通过 signal-existence gate；不能进入 learned routing，应回 Step 2/3 重审 Stage A contribution。
- [Limit] 本诊断只覆盖 unified interface paths，不覆盖 fixed specialist；fixed 仍是 problem evidence，不是 routing 候选。
