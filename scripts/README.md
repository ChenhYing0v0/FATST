# Scripts

保存项目工具脚本。

脚本应尽量可重复运行，并在 destructive 或 expensive 操作前显式提示或要求确认。

## Remote Runners

- `remote/run_phase0_gate.sh`: Phase0 baseline gate。
- `remote/run_phase0_controls.sh`: Phase0 targeted controls。
- `remote/run_phase0_patch_seed_variance.sh`: selected-base seed variance。
- `remote/run_phase1_segment_decoder_gate.sh`: Phase1-A `Future-Segment Decoder Gate`。
- `remote/run_phase1_fixed_adapter_gate.sh`: Phase1-A.2 `Fixed-Head Adapter Gate`。
- `remote/run_phase1_future_aware_adapter_gate.sh`: Phase1-A.3 `Future-Aware Adapter Gate`。
- `remote/run_phase1_future_aware_repair_gate.sh`: Phase1-A.4 `Future-Aware Repair Gate`，
  对比 `AlignOnly` 与 `ScaleNorm`。该 runner 默认用 `metrics.json` 判断已完成 run，
  并删除 `checkpoint.pt` / `predictions_test.npz` 以避免 3090 quota 压力；如需保留，
  显式设置 `KEEP_HEAVY_ARTIFACTS=1`。
- `remote/run_phase1_step_specific_state_gate.sh`: Phase1-A.5
  `Step-Specific State Decoder Gate`，对比 `PatchEncoderFixedHead`、
  `PatchEncoderFixedHeadAdapter` 与 `PatchEncoderStepSpecificStateAdapter`。该 runner
  同样默认删除 `checkpoint.pt` / `predictions_test.npz`。
- `remote/run_phase1_trajectory_basis_residual_gate.sh`: Phase1-A.6
  `Output-Process Residual Gate`，对比 `PatchEncoderFixedHead`、
  `PatchEncoderFixedHeadAdapter`、`PatchEncoderStepSpecificStateAdapter` 与
  `PatchEncoderTrajectoryBasisResidual`。该 runner 默认删除 `checkpoint.pt` /
  `predictions_test.npz`。
- `remote/check_phase2_future_state_alignment_repair_progress.sh`: 在 3090 项目目录中
  检查 Phase2-R.1 repair gate 进度，输出 dataset 矩阵位置、epoch/total、ETA 和
  outer log tail。
- `sync_phase2_future_state_alignment_repair_results.sh`: 本地运行；从 3090 同步
  Phase2-R.1 repair gate artifacts 到
  `analysis/phase2_future_state_alignment_repair_gate_20260623/raw/`，排除
  `checkpoint.pt` / `predictions_test.npz`，随后运行 repair gate analyzer。
- `remote/run_phase2_error_process_decoder_gate.sh`: Phase2-B
  `PatchEncoderErrorProcessDecoder` gate。默认数据集顺序为 `ETTm1 Weather ETTh2`，
  以便两卡可用时优先并行较慢数据集。
- `remote/check_phase2_error_process_decoder_progress.sh`: 在 3090 项目目录中检查
  Phase2-B error-process gate 进度，输出 dataset 矩阵位置、epoch/total、ETA 和
  outer log tail。
- `sync_phase2_error_process_decoder_results.sh`: 本地运行；从 3090 同步 Phase2-B
  artifacts 到 `analysis/phase2_error_process_decoder_gate_20260623/raw/`，排除
  `checkpoint.pt` / `predictions_test.npz`，随后运行 error-process gate analyzer。
- `remote/run_phase2_region_balanced_gate.sh`: Phase2-C
  `PatchEncoderRegionBalanced` objective gate。默认数据集顺序为 `ETTm1 Weather ETTh2`，
  默认 `--step-loss-weighting region_balanced`，用于测试 coverage-balanced objective。
- `remote/check_phase2_region_balanced_progress.sh`: 在 3090 项目目录中检查 Phase2-C
  region-balanced gate 进度，输出 dataset 矩阵位置、epoch/total、ETA 和 outer log tail。
- `sync_phase2_region_balanced_results.sh`: 本地运行；从 3090 同步 Phase2-C artifacts 到
  `analysis/phase2_region_balanced_gate_20260623/raw/`，排除 `checkpoint.pt` /
  `predictions_test.npz`，随后运行 region-balanced gate analyzer。

## Analysis

- `analyze_phase1_future_aware_repair_gate.py`: 汇总 Phase1-A.4 repair gate 的
  metrics、alignment diagnostics、delta stats、heatmap 和中文报告。
- `analyze_phase1_step_specific_state_gate.py`: 汇总 Phase1-A.5 gate 的 metrics、
  segment comparison、state modulation diagnostics、activation similarity、heatmap 和中文报告。
- `analyze_phase1_trajectory_basis_residual_gate.py`: 汇总 Phase1-A.6 gate 的 metrics、
  segment comparison、trajectory residual diagnostics、heatmap 和中文报告。
- `analyze_phase2_future_state_alignment_repair_gate.py`: 汇总 Phase2-R.1
  confidence-weighted future alignment repair gate，对比 FixedHead 与 R.3，并输出
  leakage、prefix consistency、reconstruction confidence 诊断和 decision report。
- `analyze_phase2_alignment_conflict.py`: 使用已完成的 Phase2-A artifacts 诊断
  future-state alignment conflict，输出 MSE delta 与 teacher/student geometry、
  alignment loss、reconstruction loss 的关系。
- `analyze_output_error_process_problem.py`: 使用 H720 step-wise artifacts 诊断
  output/error-process decoder 问题，输出 segment-level relative MSE、step-wise 曲线
  和下一步 decoder pivot 依据。
- `validate_phase2_error_process_artifacts.py`: 在 Phase2-B smoke 或 remote gate 完成后
  检查 `PatchEncoderErrorProcessDecoder` 的 required artifacts、`error_process_stats.csv`
  固定列、prefix mismatch 和非 NaN 诊断值。
- `analyze_phase2_error_process_decoder_gate.py`: 汇总 Phase2-B
  `PatchEncoderErrorProcessDecoder` gate，对比 R.3 与 FixedHead，并输出
  error-process residual、H720 focus regions、prefix consistency 和 decision report。
- `analyze_phase2_objective_pressure.py`: Phase2-C objective-level diagnostic。
  复现 mixed-horizon sampler 下的 expected step pressure，比较 R.3
  `PatchEncoderPrefixRiskWeighted` 与 uniform `PatchEncoderTargetSetDecoder`，
  输出 objective pressure 分布、R.3 vs uniform 表格、相关性诊断、图片和 decision report。
- `analyze_phase2_region_balanced_gate.py`: 汇总 Phase2-C
  `PatchEncoderRegionBalanced` remote gate，对比 R.3、uniform target-set 与 FixedHead，
  输出 objective-weight stats、specialist-gap 修复情况、H720 middle/late stability、
  prefix consistency 和 decision report。
- `analyze_phase2_covariance_novelty.py`: Phase2-C.1 离线 covariance/novelty
  diagnostic。按 `ForecastDataset` 的 train split 与 scaling 计算 H720 target region
  novelty，并与 R.3、`region_balanced` 的 segment-level gain/loss 对齐，判断是否值得进入
  `step_covariance_balanced` 的 step 4-6。

远程实验前仍必须先检查 `529_Lab-3090` 的 GPU 占用；runner 中的 `nvidia-smi`
输出只作为启动时记录，不替代人工选择 GPU。
