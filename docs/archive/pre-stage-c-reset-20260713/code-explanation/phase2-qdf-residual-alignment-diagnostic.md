# Phase2-E2 QDF Residual Alignment Diagnostic

更新时间：2026-06-23

## 1. 目的

Phase2-E1 证明 static train-target off-diagonal block penalty 不能超过 R.3，但这没有否定 QDF
的 learned objective。Phase2-E2 的代码目标是补一个 residual-level diagnostic：

`FATST R.3 predictions -> residual vectors -> objective matrix losses -> alignment report`

该诊断不训练新模型，只回答一个问题：QDF learned `off_diag/all` matrix 是否在 FATST R.3
residual 上产生比 static proxy 更有解释力的 pressure。

## 2. Remote Artifact Runner

入口：

- `scripts/remote/run_phase2_qdf_alignment_r3_predictions.sh`
- `scripts/remote/check_phase2_qdf_alignment_r3_predictions_progress.sh`
- `scripts/sync_phase2_qdf_alignment_r3_predictions.sh`

runner 复用 `scripts/remote/run_phase1_target_set_decoder_gate.sh`，但显式设置：

- `RUN_NAME=PatchEncoderPrefixRiskWeighted`;
- `STEP_LOSS_WEIGHTING=prefix_risk`;
- `SAVE_PREDICTIONS=1`;
- `KEEP_HEAVY_ARTIFACTS=1`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions`。

`SAVE_PREDICTIONS` 是新增的 opt-in 开关。默认值是 `0`，因此不会改变既有远程 gate 的空间策略。
当它为 `1` 时，训练脚本收到 `--save-predictions`，并在每个 `h{H}` 子目录写出：

```text
predictions_test.npz
  pred: [N, H, C]
  true: [N, H, C]
```

sync wrapper 排除 `checkpoint.pt`，但保留 `predictions_test.npz`，因为 Phase2-E2 的核心输入就是
R.3 residual。

## 3. Analyzer Forward Flow

入口：

- `scripts/analyze_phase2_qdf_residual_alignment.py`

主要输入：

- R.3 predictions:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/raw/PatchEncoderPrefixRiskWeighted/{dataset}/mixed_h96_h192_h336_h720/seed2021/h{H}/predictions_test.npz`;
- QDF learned matrices:
  `analysis/phase2_qdf_upstream_gate_20260623/raw/{meta_type}/{dataset}/h{H}/seed2023/checkpoints/*/A.pth`;
- static local proxy:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/raw/PatchEncoderOffdiagBlockQuadratic/{dataset}/mixed_h96_h192_h336_h720/seed2021/offdiag_block_matrix.csv`;
- R.3 specialist gap labels:
  `analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv`。

对每个 `(dataset, horizon)`：

1. 读取 `pred` 和 `true`，形成 residual:
   `residual = pred - true`，shape `[N, H, C]`。
2. 转为 per-channel residual vectors:
   `vectors = residual.transpose(0, 2, 1).reshape(-1, H)`，shape `[N*C, H]`。
3. 计算 plain residual MSE：
   `mean(residual^2)`。
4. 计算 `prefix_risk` weighted loss，使用 R.3 相同的 analytic weights：
   `w_t = (t / 720)^(-0.5)`，再按 720-step mean normalize。
5. 读取 QDF `A.pth`，复用 `analyze_phase2_qdf_matrix_audit.load_qdf_matrix()` 还原
   covariance 和 precision。
6. 对 QDF precision 做 trace normalization，使平均 diagonal scale 接近 `1`：
   `P_norm = P * H / trace(P)`。
7. 计算 normalized quadratic loss：
   `mean(r^T P_norm r / H)`。
8. 对 static block matrix，先把 residual 压成 block means，再计算训练时同构的 projected
   penalty：
   `mean((block_values @ M^T)^2)`。

## 4. Outputs

analyzer 写出：

- `phase2_qdf_residual_alignment_losses.csv`
  - 每个 `(dataset, horizon, matrix_family)` 的 normalized loss；
  - `ratio_to_residual_mse` 表示相对同一 setting 的 plain R.3 residual MSE 的 pressure ratio；
    `static_train_target_offdiag` 也使用同一分母，而不是 block-level residual MSE。
- `phase2_qdf_residual_alignment_matrix_sources.csv`
  - QDF matrix source path、dimension、trace。
- `phase2_qdf_residual_alignment_summary.json`
  - artifact completeness gate；
  - per-matrix-family ratio statistics；
  - loss 与 residual MSE 的 Pearson/Spearman correlation。
- `phase2_qdf_residual_alignment_report.md`
  - 面向研究决策的简报。

## 5. Code-Theory Consistency

[Theory] 如果 QDF 的优势来自 learned future-step residual interaction，则 QDF `off_diag/all`
precision 应该在 FATST R.3 residual 上产生不同于 `identity/prefix_risk/static_train_target_offdiag`
的 pressure，并更好地区分 hard horizons 或 specialist gaps。

[Code Realization] analyzer 不改变模型和训练过程，只把同一批 R.3 residual 投影到不同 objective
matrix 下，比较 normalized loss 和 ratio。

[Proxy Boundary] 该诊断不是性能实验。它只能说明 learned matrix 与 FATST residual 是否对齐；
不能直接证明加入该 matrix 后训练一定提升。

[Falsification] 如果 QDF `off_diag/all` 的 ratio 与 static proxy 没有明显区别，或者不能更好地区分
R.3 的 hard settings，则 objective route 应停止，不应继续堆叠 MoE 或更复杂 loss。
