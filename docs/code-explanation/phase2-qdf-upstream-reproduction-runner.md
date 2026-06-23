# Phase2-D QDF Upstream Reproduction Runner Code Explanation

更新时间：2026-06-23 18:20 +08:00

## Purpose

本组脚本用于把 QDF 的完整 `meta_type=all/diag/off_diag` 机制作为 native upstream baseline
复现，而不是把 QDF 源码直接移植进 FATST。

涉及文件：

- `scripts/remote/run_phase2_qdf_upstream_gate.sh`
- `scripts/remote/check_phase2_qdf_upstream_progress.sh`
- `scripts/sync_phase2_qdf_upstream_results.sh`
- `scripts/analyze_phase2_qdf_upstream_gate.py`

## Remote Runner

`run_phase2_qdf_upstream_gate.sh` 在 3090 上运行。默认行为：

1. 使用 `/home/yingch/projects/QDF` 作为 QDF upstream repo；
2. 若 repo 缺失且 `CLONE_IF_MISSING=1`，从 `https://github.com/Master-PLC/QDF.git` clone；
3. 使用 `/home/yingch/dataset` 作为 dataset root；
4. 输出到 `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`；
5. 默认只跑 `META_TYPES=all`，用于最小 full-QDF gate；
6. 支持后续通过 `META_TYPES="diag off_diag"` 补 controls。

QDF `run.py` 顶层 import `cupy` 并只在当前路径中调用 `cp.random.seed`。如果远程环境缺少
`cupy`，runner 会在 `${OUTPUT_ROOT}/_shims/cupy.py` 创建一个只实现 `random.seed`
的轻量 shim，并把它 prepend 到 `PYTHONPATH`。该 shim 不改变 TQNet forward、QDF loss
或 optimizer，只解除 optional seed dependency 对 reproduction gate 的阻塞。

QDF upstream 在 `exp_long_term_forecasting_meta_ml3.py` 中用 `torch.save(self.A, ...)`
保存完整 `CovarianceMatrix` 对象。PyTorch 2.6+ 将 `torch.load` 的默认值改为
`weights_only=True`，会拒绝加载该对象。runner 会在启动前把两个 `A.pth` load 点 patch 为
`weights_only=False`。这是 trusted local upstream source 的版本兼容修复，不改变训练目标、
matrix 参数化或 forecast path。

每个 run 的输出结构：

```text
${OUTPUT_ROOT}/${meta_type}/${dataset}/h${horizon}/seed${SEED}/
  checkpoints/
  results/
  test_results/
  result_long_term_forecast.txt
  run.done
```

默认删除 `checkpoint.pth`，保留 `A.pth` 与 `cov_matrix.pdf`，因为后者是 learned covariance
诊断证据。

## Hyperparameter Mapping

runner 没有调用 QDF upstream 的 dataset shell scripts，而是把其中的 dataset/horizon
hyperparameters 显式写入 `qdf_hparams()`：

- `ETTh2`: 使用 upstream `scripts/ETTh2.sh` 中的 lr、meta lr、task 数、inner steps；
- `ETTm1`: 使用 upstream `scripts/ETTm1.sh`；
- `Weather`: 使用 upstream `scripts/Weather.sh`。

这样做的原因是：

- 可以把 output root 放到 repo-external experiment directory；
- 可以安全控制 `GPU_IDS` 和并发；
- 可以显式传入 `--meta_type`，便于 `all/diag/off_diag` ablation。

## Analyzer

`analyze_phase2_qdf_upstream_gate.py` 读取 sync 后的 `raw/` 目录。

主要输入：

- `metrics.npy`: QDF 保存的 `[mae, mse, cov_loss, rmse, mape, mspe, mre]`；
- `result_long_term_forecast.txt`: 作为 metrics fallback；
- `cov_matrix.pdf`: learned covariance heatmap artifact；
- `A.pth`: learned loss module artifact。

主要输出：

- `phase2_qdf_upstream_metrics.csv`
- `phase2_qdf_upstream_meta_type_comparison.csv`
- `phase2_qdf_upstream_summary.json`
- `phase2_qdf_upstream_decision_report.md`

## Gate Logic

QDF upstream gate 只有在 controls 完整时才可能 pass：

1. `meta_type=all` 的 12 个 runs 完成；
2. `diag` control 存在；
3. `all` 相对 `diag` 的 mean MSE 改善；
4. `all` 相对 `diag` 至少 `7/12` MSE wins；
5. FATST specialist gap 对应点至少两个改善；
6. `cov_matrix.pdf` artifacts 存在。

如果只跑了 `meta_type=all`，report 会标记 gate incomplete。这是有意设计：full QDF 本身的
绝对指标不能证明 off-diagonal 机制有效，必须有 own diagonal control。

## Code-Theory Consistency

Intended theory:

QDF 的 paper claim 是 full learned quadratic objective 能表达 future-step covariance，而
不是只改变 step weights。

Code realization:

runner 保留 upstream training/evaluation 语义，只改变输出目录、GPU 分配和 `meta_type`
ablation 控制。analyzer 首先比较 QDF 自身的 `all` 与 `diag`，避免把不同 architecture 的
FATST R.3 当作直接 control。

Remaining proxy:

QDF upstream 使用 `TQNet`，不是 FATST target-set carrier。因此通过 upstream gate 只说明
QDF 机制值得 source-informed localization，不等价于 FATST final method 通过。
