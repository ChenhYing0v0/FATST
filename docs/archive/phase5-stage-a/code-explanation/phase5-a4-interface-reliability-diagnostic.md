# Phase5 A4 Interface Reliability Diagnostic 代码说明

## 目标

`scripts/analyze_phase5_timealign_hss_a4_interface_reliability.py` 是 A4 的
diagnostic-only 分析脚本。它不训练模型、不选择最终 head，只把已有 Phase5 interface
实验的 per-horizon MSE/MAE 组织成 reliability evidence。

核心问题：

> 现有 capacity-preserving / prefix-aware paths 是否存在稳定可靠性差异？这种差异是否足以支撑
> 后续 reliability-aware interface routing，而不是手工按 dataset/horizon 选路径？

## 输入

默认输入目录：

`analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/`

读取文件：

- `phase5_timealign_hss_a3e_ettm1_fixed_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_h1_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_h1c_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_a2_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_a3c_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_a3d_reference.csv`
- `phase5_timealign_hss_a3e_ettm1_metrics.csv`

分析 universe 固定为：

- datasets：`ETTh2`, `ETTm1`, `Weather`
- target horizons：`96`, `192`, `336`, `720`

## Path 定义

脚本统一生成以下 `path_id`：

| path_id | family | 含义 |
| --- | --- | --- |
| `fixed_specialist` | `fixed` | 每个 horizon 单独训练的 fixed-horizon TimeAlign |
| `h1_target_set` | `dense_target_set` | H1 target-set conditioned dense 720 projection |
| `h1c_row_gated` | `dense_target_set` | H1C row-gated dense head |
| `a2_nested` | `primary_nested` | A2 nested segment primary decoder |
| `a3c_warm_nested` | `primary_nested` | A3C H1 checkpoint warm-started nested primary decoder |
| `a3d_teacher_preserved` | `teacher_preserved_nested` | A3D teacher-preserved nested primary decoder, `w03` |
| `a3e_target_conditioned_warm` | `target_conditioned_nested` | A3E target-conditioned nested warm arm |
| `a3e_target_conditioned_scratch` | `target_conditioned_nested` | A3E target-conditioned nested scratch arm |

## 输出

默认输出目录：

`analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701/`

输出文件：

- `phase5_timealign_hss_a4_all_paths.csv`
- `phase5_timealign_hss_a4_best_path_map.csv`
- `phase5_timealign_hss_a4_path_reliability_summary.csv`
- `phase5_timealign_hss_a4_family_reliability_summary.csv`
- `phase5_timealign_hss_a4_oracle_routing_summary.csv`
- `phase5_timealign_hss_a4_interface_reliability_diagnostic.md`

## 统计量定义

`relative_vs_setting_best_pct`：

对同一 `(dataset, target_horizon)`，当前 path 的 MSE 相对该 setting 最优 path 的百分比差：

`(path_mse / best_mse - 1) * 100`

该值越小越好，`0` 表示当前 path 是该 setting 最优。

`within_0p2pct_of_best`：

当前 path 的 `relative_vs_setting_best_pct <= 0.2`。它衡量某 path 是否虽然不是最优，但已经足够接近
setting 最优。

`mean_gap_to_best_pct`：

对一个 path 或 family，在所有相关 settings 上平均 `relative_vs_setting_best_pct`。它衡量该 path
离 offline best path 的平均距离。

`wins_as_best`：

一个 path 或 family 在多少个 `(dataset, target_horizon)` 上取得最低 MSE。

`oracle_relative_vs_best_static_pct`：

每个 setting 都选择 offline best path 得到的 oracle mean MSE，相对 best static path mean MSE 的百分比差。
这是 routing 的离线上限，不是可部署方法结果。

## 代码-理论一致性评价

Intended theory：

如果没有单一 path 稳定最优，但 offline oracle 相对 best static 有增益，则 Stage A 的问题可能不是
“设计唯一 universal head”，而是“不同 capacity-preserving paths 的可靠性随 future context 变化”。

Code realization：

脚本只使用已有 per-horizon MSE/MAE，计算 best-path map、path/family gap-to-best 和 oracle upper bound。
它没有使用训练样本、validation signal 或模型内部 tensor，因此不会误把 diagnostic 写成可部署 routing。

Proxy boundary：

`best_path_map` 使用 test metrics，是离线诊断 oracle。它不能直接作为方法输入，也不能证明 learned routing
会有效。

Falsification evidence：

如果 A4R 后续发现 teacher-student disagreement、prefix residual、validation gap、segment volatility 等
可观测 signals 都不能解释 `best_path_map` 或 `mean_gap_to_best_pct`，则 reliability-aware interface routing
不应进入 paper-core。

## A4R Existing-Log Signal Diagnostic

`scripts/analyze_phase5_timealign_hss_a4r_reliability_signals.py` 是 A4 后的第一轮信号诊断。
它复用 A4 生成的 `phase5_timealign_hss_a4_all_paths.csv`，并根据每一行的 `source_path` 定位同目录下的
`training_log.csv`。

默认输入：

- `analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701/phase5_timealign_hss_a4_all_paths.csv`
- 每个 metrics 文件同目录的 `training_log.csv`

默认输出：

- `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_signal_rows.csv`
- `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_signal_correlations.csv`
- `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_best_signal_by_dataset.csv`
- `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_reliability_signal_diagnostic.md`

A4R signals：

| signal | 来源 | 含义 |
| --- | --- | --- |
| `best_val_mean_mse` | `training_log.csv` 中最小 `val_mean_mse` | 该 run 在 validation selector 上达到的最好值 |
| `last_val_mean_mse` | 最后 epoch 的 `val_mean_mse` | official-last protocol 下最终 validation 状态 |
| `last_gap_to_best_val_pct` | `(last_val / best_val - 1) * 100` | 最后 checkpoint 相对 best validation 的退化 |
| `last_train_prediction_l1` | 最后 epoch `train_prediction_l1` | run-level 训练预测误差 |
| `last_train_horizon_prediction_l1` | 最后 epoch `train_prediction_h{H}_l1`，缺失时退回 `train_prediction_l1` | 与 target horizon 对齐的训练预测误差 |
| `last_train_reconstruction_l1` | 最后 epoch `train_reconstruction_l1` | TimeAlign reconstruction pressure 的训练观测量 |
| `last_train_alignment_loss` | 最后 epoch `train_alignment_loss` | future alignment pressure 的训练观测量 |
| `last_train_teacher_l1` | 最后 epoch `train_teacher_l1`，缺失时为 0 | teacher-preservation pressure 的训练观测量 |

Correlation target：

`relative_vs_setting_best_pct`，即每个 path 在同一 `(dataset, target_horizon)` 下相对 offline best path
的 MSE gap。A4R 对每个 signal 计算 Pearson 和 Spearman correlation。

Code-theory boundary：

A4R 使用现有 logs，因此它只能判断“当前已保存的信号是否足够解释 path reliability”。它不能否定
prefix-wise validation residual、teacher-student disagreement by prefix 等尚未导出的更强 signals。
若 A4R signal 较弱，合理下一步是设计轻量 diagnostic export，而不是直接做 routing。

## A4S Validation-Prefix Signal Export

`scripts/export_timealign_validation_prefix_diagnostics.py` 是 A4S 的 checkpoint-level exporter。它不训练模型，
只加载已有远端 checkpoint，在 validation split 上导出 prefix-wise diagnostics。

默认由远程 wrapper 调用：

`scripts/remote/run_phase5_timealign_hss_a4s_validation_prefix_signal_export.sh`

同步和分析入口：

`scripts/sync_phase5_timealign_hss_a4s_results.sh`

本地分析脚本：

`scripts/analyze_phase5_timealign_hss_a4s_validation_prefix_signals.py`

A4S 覆盖的 unified path：

- `h1_target_set`
- `h1c_row_gated`
- `a2_nested`
- `a3c_warm_nested`
- `a3d_teacher_preserved`
- `a3e_target_conditioned_warm`
- `a3e_target_conditioned_scratch`

固定 horizon specialist 不进入 A4S routing candidate。它仍是 problem evidence 和 performance reference，
但不是 unified interface path routing 的候选路径。

A4S exporter 输出：

`validation_prefix_diagnostics.csv`

| column | 来源 | 含义 |
| --- | --- | --- |
| `validation_prefix_mse` | validation split, requested target_prefix | 当前 path 在 prefix horizon 上的 validation MSE |
| `validation_prefix_mae` | validation split, requested target_prefix | 当前 path 在 prefix horizon 上的 validation MAE |
| `full_context_prefix_mse` | validation split, target_prefix=720 后取前 H | 同一模型在 full-context request 下的 prefix validation MSE |
| `prefix_vs_full_mse` | requested prefix output vs full-context prefix output | 该 path 对 target_prefix 是否敏感，越大表示 prefix-conditioned behavior 越强 |
| `teacher_student_mse` | current path output vs H1 teacher output | 当前 path 相对 H1 target-set teacher 的 prefix disagreement |
| `teacher_student_mae` | current path output vs H1 teacher output | teacher-student MAE disagreement |
| `residual_abs_mean` | current prediction - validation target | validation residual 平均绝对值 |
| `residual_std` | current prediction - validation target | validation residual dispersion |

A4S analyzer 的 correlation target 仍是 A4 的 `relative_vs_setting_best_pct`。如果 prefix-wise validation
signals 能稳定解释该 target，才允许进入 `Reliability-Aware Capacity-Preserving Interface` 的 method
narrative gate。

Code-theory boundary：

A4S 只导出聚合后的 validation diagnostics，不保存 batch-level predictions，因此不会显著增加存储压力。
它也不使用 test target 作为 method input；test MSE 只在离线分析中作为被解释对象。

Gate：

- 若 ALL-level 或跨 dataset 稳定 signal 的 Spearman 绝对值达到约 `0.55`，说明 signal-existence gate
  初步通过，可进入 routing method 的 Step 4-6 narrative gate；
- 若相关性弱或方向不稳定，则 Stage A 不能继续做 routing，应回 Step 2/3 重审 interface contribution。
