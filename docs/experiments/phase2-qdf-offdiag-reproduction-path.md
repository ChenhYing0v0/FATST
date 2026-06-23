# Phase2-D: QDF Off-Diagonal Reproduction Path

更新时间：2026-06-23 18:02 +08:00

## 11-Step Loop Record

- `current_step`: Step 2-3 rollback check，准备进入 Step 6-8 的 external baseline reproduction gate。
- `problem`: Phase2-C.2 的 static diagonal / novelty-aware objective 已失败，但 QDF 论文讨论的 full quadratic objective 还没有被等价测试。
- `existence_evidence`: QDF 官方实现的 `CovarianceMatrix` 在误差维度 `[B*D, P]` 上学习 `P x P` matrix；本地 off-diagonal diagnostic 显示三个数据集的 future-region label correlation 很强。
- `idea`: 不继续调 `step_covariance_balanced` 的 diagonal weights；改为先 native reproduce QDF full/off-diagonal baseline，判断 full learned quadratic objective 是否是 objective route 的真实有效部分。
- `theory_check`: 若 future steps/regions 明显相关，identity/diagonal loss 不能表达 residual interaction；full/off-diagonal quadratic loss 至少在表达能力上覆盖该依赖。但 label-side dependence 不是 performance proof，必须通过 native reproduction 验证。
- `design`: 先保留 QDF upstream repo 的原生训练流程，复现其 `meta_type=all/diag/off_diag` 机制；本仓库只记录 audit、runner、sync/analyzer，不直接 vendor 或移植 upstream source。
- `gate`: 只在 native QDF full/off-diagonal 相对其 own MSE/diag control 有稳定收益，且结果能解释 FATST 上 diagonal proxy 失败时，才考虑本地化一个 source-informed component。
- `artifacts`: `analysis/phase2_qdf_offdiag_diagnostic_20260623/`，以及后续 upstream reproduction outputs。
- `decision`: Phase2-D diagnostic pass，下一步进入 QDF upstream reproduction gate；禁止继续在 `beta/eta` 上做宽 sweep。

## Why This Is The Next Experiment

[Fact] Phase2-C.2 `PatchEncoderStepCovarianceBalanced` failed against R.3:

- MSE wins vs R.3: `2/12`;
- mean relative MSE vs R.3: `+0.76%`;
- specialist gap wins vs R.3: `0`;
- prefix consistency remained numerically safe.

[Inference] 这说明失败不是 target-set interface 的安全性问题，而是 diagonal objective proxy 不足。

[Fact] QDF 的官方实现不是静态 diagonal weights。它使用 `CovarianceMatrix` 学习 loss matrix，并支持：

- `meta_type=all`: full Cholesky matrix；
- `meta_type=diag`: diagonal-only；
- `meta_type=off_diag`: diagonal fixed, learn off-diagonal part。

[Strong Evidence] 本地 QDF off-diagonal diagnostic 使用与 QDF loss 一致的轴语义：把 H720 target regions 构成 `[B*D, 4]` matrix。结果为：

| Dataset | Mean abs offdiag corr | Max abs offdiag corr | Offdiag corr Fro share |
| --- | ---: | ---: | ---: |
| `ETTh2` | `0.7103` | `0.8127` | `0.6057` |
| `ETTm1` | `0.8585` | `0.8897` | `0.6888` |
| `Weather` | `0.7342` | `0.8066` | `0.6193` |

[Decision] 这足以支持做 QDF upstream reproduction gate。它不支持直接把 QDF 移植到 FATST，因为 label-side off-diagonal dependence 只能证明问题真实，不证明 learned loss 能提升本地 carrier。

## Proposed Reproduction Gate

第一轮只跑最小 gate：

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- horizons: `96`, `192`, `336`, `720`;
- upstream model: QDF script 默认 `TQNet`;
- controls: `meta_type=all` 优先；若脚本可低成本切换，再补 `diag` 和 `off_diag`。

QDF upstream gate 通过需要同时满足：

1. `meta_type=all` 在至少两个数据集上相对 own MSE/diag control 改善 mean MSE；
2. 对 `ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96` 这些 FATST specialist gaps 至少有两个方向的改善证据；
3. learned covariance heatmap 不是接近 identity，也不是数值病态；
4. 结果能解释为什么 static diagonal proxy 只赢 uniform、但输给 R.3。

若 upstream QDF full/off-diagonal 也不能稳定改善，则 objective route 回滚到 Step 2：

- 不再新增 objective weighting 机制；
- 转向 base architecture 或 external baseline selection；
- MoE 继续暂停，直到有通过 gate 的 target-side carrier。

## Implementation Boundary

[Decision] 当前只新增 local diagnostic 和实验路径文档，不 vendor QDF source。

原因：

- upstream reproduction 应先在 native repo 中完成；
- 直接移植会混淆“QDF 是否有效”和“本地实现是否正确”；
- 本仓库需要保留 clean research line，只记录可审计 evidence 与 wrapper/analyzer。

## Current Artifacts

- diagnostic script: `scripts/analyze_phase2_qdf_offdiag_diagnostic.py`
- diagnostic report: `analysis/phase2_qdf_offdiag_diagnostic_20260623/phase2_qdf_offdiag_diagnostic_report.md`
- summary JSON: `analysis/phase2_qdf_offdiag_diagnostic_20260623/qdf_offdiag_summary.json`
- heatmap: `analysis/phase2_qdf_offdiag_diagnostic_20260623/qdf_region_correlation_heatmap.png`
- upstream runner: `scripts/remote/run_phase2_qdf_upstream_gate.sh`
- progress checker: `scripts/remote/check_phase2_qdf_upstream_progress.sh`
- sync wrapper: `scripts/sync_phase2_qdf_upstream_results.sh`
- upstream analyzer: `scripts/analyze_phase2_qdf_upstream_gate.py`
