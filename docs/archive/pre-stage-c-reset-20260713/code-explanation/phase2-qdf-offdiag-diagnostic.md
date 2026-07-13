# Phase2-D QDF Off-Diagonal Diagnostic Code Explanation

更新时间：2026-06-23 18:02 +08:00

## Purpose

`scripts/analyze_phase2_qdf_offdiag_diagnostic.py` 用于回答一个很窄的问题：

> 在 `step_covariance_balanced` diagonal proxy 失败后，future-step / future-region 之间是否存在足够强的 off-diagonal label dependence，使得完整 QDF-style learned quadratic loss 值得作为下一步 external baseline reproduction？

该脚本不训练模型，不读取 test labels，也不修改任何训练代码。

## Data Flow

1. `ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)` 读取 train split。
2. `ForecastDataset` 内部仍使用当前 baseline 的 scaling 规则：scaler 只在 train split 上 fit。
3. `region_target_matrix(...)` 对每个 H720 target window 切四个 regions：
   `1-96`, `97-192`, `193-336`, `337-720`。
4. 每个 region 对时间维取 mean，得到 `[B, D, 4]`。
5. 脚本 reshape 为 `[B*D, 4]`。

这个 reshape 是有意的，因为 QDF 官方实现的 loss 先把误差 `E` 从 `[B, P, D]` 变成
`[B*D, P]`，再在 `P` 维上施加 quadratic matrix。

## Statistics

- `mean_abs_offdiag_corr`: 四个 future regions 的 correlation matrix 中非对角元素绝对值均值。
- `max_abs_offdiag_corr`: 非对角 correlation 的最大绝对值。
- `offdiag_corr_fro_share`: correlation matrix 的 off-diagonal Frobenius energy 占比。
- `offdiag_cov_fro_share`: covariance matrix 的 off-diagonal Frobenius energy 占比。
- `cov_condition_number`: region covariance eigenvalue ratio，用于粗略判断 label matrix 是否病态。

## Gate Logic

`supports_qdf_upstream_reproduction=True` 需要同时满足：

1. 平均 off-diagonal correlation 超过阈值；
2. 每个数据集的最低 off-diagonal signal 不低于阈值的一半；
3. Phase2-C.2 diagonal proxy 已失败；
4. Phase2-C.1 novelty diagnostic 曾支持进入 diagonal proxy 训练。

这套 gate 的含义是：

- [Fact] label-side off-diagonal dependence 真实存在；
- [Fact] diagonal approximation 已经被训练结果证伪；
- [Inference] 下一步应验证 full/off-diagonal QDF，而不是继续调 diagonal weights。

## Outputs

默认输出目录：

`analysis/phase2_qdf_offdiag_diagnostic_20260623/`

包含：

- `phase2_qdf_offdiag_diagnostic_report.md`
- `qdf_offdiag_dataset_summary.csv`
- `qdf_region_correlation_covariance.csv`
- `qdf_offdiag_summary.json`
- `qdf_region_correlation_heatmap.png`

## Code-Theory Consistency

Intended theory:

QDF 的关键不是“future steps 权重不等”这一点本身，而是 future-step residuals 之间存在 covariance structure；diagonal objective 只能表达 heterogeneous weights，不能表达 off-diagonal dependence。

Code realization:

脚本复用 QDF loss 的轴语义，将 future target regions 视为 coarse future tasks，并量化它们在 train split 上的 off-diagonal correlation/covariance。

Remaining proxy:

当前统计是 label-side proxy，不是 residual covariance，也不是 learned loss matrix。因此它只能决定是否值得进入 QDF upstream reproduction，不能直接成为模型贡献。

Falsification evidence:

如果 native QDF full/off-diagonal reproduction 不能稳定优于 own diagonal/MSE control，或者 learned matrix 接近 identity，那么 objective route 应回滚，不能继续本地化 QDF。
