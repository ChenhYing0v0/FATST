# Phase2-D QDF Off-Diagonal Diagnostic

## 11-Step Loop Position

- `current_step`: Step 2-3 rollback check。
- `problem`: diagonal / static objective proxy 已失败，但 QDF 的完整 quadratic objective 仍可能依赖 off-diagonal future-step dependence。
- `existence_evidence`: 本诊断从 train split labels 估计 future-region correlation/covariance，不训练模型。
- `idea`: 若 future regions 存在强 off-diagonal dependence，且 diagonal proxy 已失败，则下一步不应继续调 diagonal weights，而应 native reproduce QDF full/off-diagonal baseline。

## Decision

[Decision] 进入 QDF upstream reproduction gate：完整 QDF/off-diagonal 机制值得作为下一步实验路径。

## Dataset Summary

| Dataset | Samples `[B*D]` | Mean abs offdiag corr | Max abs offdiag corr | Offdiag corr Fro share | Cov condition number |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 53095 | 0.7103 | 0.8127 | 0.6057 | 20.5236 |
| ETTm1 | 234535 | 0.8585 | 0.8897 | 0.6888 | 39.7375 |
| Weather | 752472 | 0.7342 | 0.8066 | 0.6193 | 20.9301 |

## Gate

- mean_abs_offdiag_corr: `0.7677`
- min_dataset_mean_abs_offdiag_corr: `0.7103`
- offdiag_threshold: `0.3500`
- offdiag_signal_strong: `True`
- diagonal_proxy_failed: `True`
- novelty_supported_diagonal_before_training: `True`
- supports_qdf_upstream_reproduction: `True`

## Interpretation

[Fact] QDF 的实现把误差从 `[B, P, D]` 展平成 `[B*D, P]`，再用 learned quadratic matrix 计算 loss。
本诊断按相同轴语义，把 H720 target 切成四个 future regions 后形成 `[B*D, 4]` 矩阵。

[Inference] 如果 off-diagonal correlation 很强，说明 future steps/regions 不是独立任务。
这正是 static diagonal weighting 无法表达、而 QDF full/off-diagonal matrix 能表达的部分。

[Counterargument] 该诊断只看 label-side dependence，不证明 learned QDF loss 一定提升 FATST carrier。
因此下一步应先 native reproduce upstream QDF，而不是直接把 QDF module 移植进本 repo。

## Artifacts

- `qdf_offdiag_dataset_summary.csv`
- `qdf_region_correlation_covariance.csv`
- `qdf_offdiag_summary.json`
- `qdf_region_correlation_heatmap.png`
