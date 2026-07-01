# Phase5 A4R Reliability Signal Diagnostic

## 诊断目标

[Step 3/4] A4 已证明 offline best path 分散，但这仍可能只是 test-oracle 现象。
A4R 使用现有 training logs 中可观测的 validation/training signals，检查它们是否能解释 path 的 `gap_to_best`。

## ALL-Level Signal Ranking

| rank | signal | pearson | spearman |
| --- | --- | --- | --- |
| 1 | `last_train_alignment_loss` | 0.355 | 0.321 |
| 2 | `last_train_prediction_l1` | 0.155 | 0.281 |
| 3 | `last_gap_to_best_val_pct` | 0.375 | 0.251 |
| 4 | `best_val_mean_mse` | -0.378 | -0.208 |
| 5 | `last_train_reconstruction_l1` | -0.115 | 0.183 |
| 6 | `last_train_teacher_l1` | -0.107 | -0.153 |
| 7 | `last_val_mean_mse` | -0.314 | -0.106 |
| 8 | `last_train_horizon_prediction_l1` | 0.049 | -0.024 |

## Dataset-Level Top Signals

| dataset | signal | pearson | spearman |
| --- | --- | --- | --- |
| ALL | `last_train_alignment_loss` | 0.355 | 0.321 |
| ALL | `last_train_prediction_l1` | 0.155 | 0.281 |
| ALL | `last_gap_to_best_val_pct` | 0.375 | 0.251 |
| ETTh2 | `last_train_teacher_l1` | -0.204 | -0.471 |
| ETTh2 | `last_train_alignment_loss` | 0.140 | 0.248 |
| ETTh2 | `last_train_prediction_l1` | -0.044 | -0.240 |
| ETTm1 | `last_train_alignment_loss` | 0.713 | 0.534 |
| ETTm1 | `last_train_horizon_prediction_l1` | -0.552 | -0.533 |
| ETTm1 | `last_train_teacher_l1` | 0.053 | 0.236 |
| Weather | `last_train_horizon_prediction_l1` | -0.356 | -0.361 |
| Weather | `last_train_alignment_loss` | -0.119 | -0.198 |
| Weather | `last_train_teacher_l1` | -0.122 | -0.154 |

## 机制判断

- [Fact] ALL-level 最强 signal 是 `last_train_alignment_loss`，Spearman 相关为 `0.321`。
- [Decision] 现有日志信号较弱，不足以支撑 learned routing；需要设计更明确的 reliability signal export。
- [Limit] 当前 signals 主要来自 run-level training log；除 `train_prediction_h{H}_l1` 外，大多不是 horizon-specific，因此不能解释所有 per-horizon path choice。
- [Next] 若继续 Stage A，应新增轻量诊断导出：teacher-student disagreement by prefix、validation prefix residual、prefix-wise validation MSE，而不是直接启动 routing head。
