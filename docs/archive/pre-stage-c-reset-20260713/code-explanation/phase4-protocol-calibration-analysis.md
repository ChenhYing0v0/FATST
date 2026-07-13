# Phase4 Protocol Calibration Analysis

本文解释 `scripts/analyze_phase4_protocol_calibration_gate.py` 的输入、输出和统计量含义。

## 输入

脚本默认读取：

- raw root: `artifacts/runs/phase4_protocol_calibration_gate`;
- main metrics:
  `lr_*/*/*/mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv`;
- training trajectory:
  `lr_*/_logs/phase4_protocol_calibration_gate/*.log`。

本次远程 artifacts 没有生成 `training_history.csv`，因此脚本从日志中的
`epoch_progress ... val_mean_mse=...` 行解析训练轨迹。

## 输出

默认输出到 `analysis/phase4_protocol_calibration_gate_20260625`：

- `phase4_protocol_main_metrics.csv`：每个 `learning_rate × strategy × dataset × horizon`
  的 test MSE/MAE；
- `phase4_protocol_hssg_delta.csv`：HSSG-A 相对 single-prefix 和 R.3 的 horizon-level
  delta；
- `phase4_protocol_lr_summary.csv`：每个 LR 的平均 test metric，以及 HSSG-A 相对 baseline
  的 wins 和 mean relative delta；
- `phase4_protocol_training_summary.csv`：每个 run 的 best epoch、epochs ran 和
  post-best validation drift；
- `phase4_protocol_training_lr_summary.csv`：按 `learning_rate × strategy` 聚合的训练轨迹；
- `phase4_protocol_calibration_gate_report.md`：面向研究决策的中文报告。

## 统计量定义

- `relative_mse_pct = (candidate_mse / baseline_mse - 1) * 100`；
- `mse_win = candidate_mse < baseline_mse`；
- `mean_mse` 在 absolute strategy 行中表示 8 个 setting 的平均 test MSE；
- `mean_mse` 在 `*_vs_*` delta 行中表示 mean relative MSE pct；
- `best_epoch` 来自 log 中最小 `val_mean_mse` 的 epoch；
- `post_best_val_drift_pct = (last_val_mean_mse / best_val_mean_mse - 1) * 100`。

## Code-Theory Consistency

[Theory] 如果 HSSG-A 的失败主要来自过快 optimization，lower LR 应同时改善 training
trajectory 和 final metrics。

[Code] 脚本同时读取 final test metrics 与 validation trajectory，并按 LR 对 HSSG-A
相对 single-prefix / R.3 的表现做 gate 判断。

[Falsification] 如果 lower LR 改善 drift 但 HSSG-A 仍无法接近 R.3，说明 protocol 不是
充分解释，下一步应回到 carrier 设计。
