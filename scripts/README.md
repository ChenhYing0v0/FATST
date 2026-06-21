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

## Analysis

- `analyze_phase1_future_aware_repair_gate.py`: 汇总 Phase1-A.4 repair gate 的
  metrics、alignment diagnostics、delta stats、heatmap 和中文报告。

远程实验前仍必须先检查 `529_Lab-3090` 的 GPU 占用；runner 中的 `nvidia-smi`
输出只作为启动时记录，不替代人工选择 GPU。
