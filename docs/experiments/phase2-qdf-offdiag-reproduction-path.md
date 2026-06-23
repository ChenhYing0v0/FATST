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

## Returned `meta_type=all` Result

更新时间：2026-06-23 19:40 +08:00

[Fact] QDF upstream `META_TYPES=all` gate 已完成：

- completed runs: `12/12`;
- metric files: `12/12`;
- covariance matrix PDFs: `12/12`;
- local report:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_decision_report.md`。

| Dataset | Horizon | MSE | MAE | Cov loss |
| --- | ---: | ---: | ---: | ---: |
| `ETTh2` | `96` | `0.285880` | `0.337921` | `0.077645` |
| `ETTh2` | `192` | `0.361037` | `0.388220` | `0.296859` |
| `ETTh2` | `336` | `0.407588` | `0.422399` | `0.122215` |
| `ETTh2` | `720` | `0.419218` | `0.438822` | `0.173580` |
| `ETTm1` | `96` | `0.306606` | `0.348975` | `0.121592` |
| `ETTm1` | `192` | `0.352415` | `0.376267` | `0.287310` |
| `ETTm1` | `336` | `0.382601` | `0.397518` | `0.169737` |
| `ETTm1` | `720` | `0.441164` | `0.434478` | `0.222438` |
| `Weather` | `96` | `0.159555` | `0.202416` | `0.055837` |
| `Weather` | `192` | `0.209021` | `0.246954` | `0.131674` |
| `Weather` | `336` | `0.264798` | `0.288555` | `0.127803` |
| `Weather` | `720` | `0.342472` | `0.339333` | `0.118382` |

[Decision] 该轮不判 pass。它只证明 `meta_type=all` native QDF 能完整训练、测试并产生
learned covariance artifacts；不能证明 full covariance 比 diagonal 或 off-diagonal-only
机制更有效。

## Next Control Gate

[11-Step Loop] 当前处于 Step 9-10 的 incomplete decision。下一步回到 Step 6-8，补齐
native QDF controls：

- `META_TYPES="diag off_diag"`;
- datasets: `ETTm1 Weather ETTh2`;
- horizons: `96 192 336 720`;
- output root 继续使用：
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`。

[Gate] 完整判定需要 `meta_type=all` 相对 `diag` 满足：

1. mean MSE 改善；
2. 12 个 setting 至少赢 7 个；
3. FATST specialist gaps 至少赢 2 个；
4. covariance artifacts 存在且非数值异常。

[Decision Rule] 如果 controls 返回后 `all` 不优于 `diag`，则 QDF objective route 回滚到
Step 2，不进入 FATST 本地实现；如果 `off_diag` 解释主要收益来源，再设计更小的
source-informed off-diagonal component。

## Returned Control Gate Result

更新时间：2026-06-23 20:55 +08:00

[Fact] QDF upstream controls 已完成并同步：

- completed metric rows: `36`;
- meta types: `all`, `diag`, `off_diag`;
- decision report:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_decision_report.md`;
- comparison table:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_meta_type_comparison.csv`。

[Decision] QDF upstream reproduction gate passes:

- `all` vs `diag` mean relative MSE: `-1.08%`;
- `all` vs `diag` MSE wins: `11/12`;
- `all` vs `diag` specialist gap wins: `4/4`;
- covariance artifacts present: `True`。

[Counter-Evidence] 通过 gate 不等于 full matrix 是最佳本地形态：

- `all` vs `off_diag` mean relative MSE: `+0.06%`;
- `all` vs `off_diag` MSE wins: `2/12`;
- `off_diag` 是 `10/12` 个 setting 的 best meta type。

[Interpretation] QDF 的关键借鉴点应收窄为 future-step residual interaction，而不是完整
full learned covariance。`diag` control 被稳定击败，说明 diagonal-only objective 不够；
但 `off_diag` 强于 `all`，说明固定 diagonal、只学习 off-diagonal coupling 可能更稳定。

## Next Local Experiment Direction

[11-Step Loop] Phase2-D 在 Step 10 通过 external evidence gate。下一轮 Phase2-E 进入
Step 4-6，设计本地 source-informed objective。

### Phase2-E0: Learned Matrix Audit

先不训练，审计 QDF 保存的 `A.pth`：

1. 对 `all/diag/off_diag` 提取 learned matrix 或 diagonal vector；
2. 统计 diagonal energy、off-diagonal energy、bandwidth、condition/PSD proxy；
3. 聚合到 `1-96`, `97-192`, `193-336`, `337-720` 四个 region；
4. 检查 matrix pattern 是否与 `all vs diag` 和 `off_diag vs all` 的收益一致。

该步骤的目的不是复述 QDF，而是决定本地 objective 应该是：

- 4-region off-diagonal quadratic；
- banded future-step coupling；
- low-rank residual coupling；
- 或者停止 objective route。

### Phase2-E1: Local Off-Diagonal Objective Probe

若 Phase2-E0 显示 learned matrices 有稳定 off-diagonal structure，再实现本地最小候选：

- carrier: `PatchEncoderPrefixRiskWeighted` / R.3；
- base loss: 保留 prefix-risk diagonal pressure；
- added term: 低维 future-region residual interaction；
- no full QDF bilevel loop；
- no MoE；
- no copied upstream module。

Gate:

1. mean MSE vs R.3 `< -0.3%`;
2. MSE wins vs R.3 `>=7/12`;
3. specialist gap wins `>=2/4`;
4. prefix consistency 维持数值零级别；
5. 若只赢 uniform 或只复现 QDF upstream，但输 R.3，则回滚到 Step 2。

## Phase2-E0/E1 Progress

更新时间：2026-06-23 21:20 +08:00

[Fact] Phase2-E0 learned matrix audit 已完成：

- report:
  `analysis/phase2_qdf_matrix_audit_20260623/phase2_qdf_matrix_audit_report.md`;
- metrics:
  `analysis/phase2_qdf_matrix_audit_20260623/phase2_qdf_matrix_audit_metrics.csv`;
- region blocks:
  `analysis/phase2_qdf_matrix_audit_20260623/phase2_qdf_matrix_audit_region_blocks.csv`。

Result:

- `diag` precision off-diagonal Fro share: `0.000000`;
- `all` precision off-diagonal Fro share: `0.011602`;
- `off_diag` precision off-diagonal Fro share: `0.013700`;
- `off_diag` normalized precision bandwidth: `0.202526`。

[Decision] Matrix audit supports a local off-diagonal objective probe.

[Implementation] Phase2-E1 implements `offdiag_block_quadratic`:

- preserves R.3 `prefix_risk` base loss;
- estimates a train-split block precision matrix;
- removes diagonal and normalizes by spectral norm;
- penalizes squared projected block residuals;
- default block size is `48`, giving H720 `15` blocks and H96 `2` blocks.

[Smoke] ETTh2 CPU smoke passed:

- command mode: `--step-loss-weighting offdiag_block_quadratic`;
- output:
  `artifacts/runs/smoke_phase2_offdiag_block_quadratic/SmokeOffdiagBlockQuadratic/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- max prefix mismatch MSE: `8.394639455409306e-15`.

[Next] Run the Phase2-E1 remote gate before making any paper-story claim.

## Phase2-E1 Returned Result

更新时间：2026-06-23

[Fact] Phase2-E1 `PatchEncoderOffdiagBlockQuadratic` remote gate 已完成并同步：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective`;
- local artifacts:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/`;
- interpretation:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/phase2_offdiag_block_quadratic_interpretation.md`。

[Decision] 本地 static off-diagonal block quadratic objective 不通过 gate：

- MSE wins vs R.3: `1/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+0.0464%`;
- specialist gap wins vs R.3: `1`;
- H720 stability region wins vs R.3: `0`;
- prefix consistency pass:
  max mismatch MSE `5.4052272504883855e-14`。

[Counter-Evidence] 该结果不等于 QDF 机制无效：

- vs uniform target-set:
  `10/12` MSE wins，mean relative MSE `-0.9871%`;
- vs FixedHead:
  `8/12` MSE wins，mean relative MSE `-0.3870%`;
- QDF upstream controls 仍显示 `all` vs `diag` 为 `11/12` MSE wins，
  `off_diag` 是 `10/12` 个 setting 的 best meta type。

[Interpretation] 失败点在 local proxy，而不是问题本身。Phase2-E1 使用 train-split target
covariance 构造 frozen block precision，并把它作为附加 penalty；QDF 更可能依赖 learned、
error-aware、随训练动态形成的 objective matrix。因此，下一步不应做 weight/block-size/ridge 的
宽 sweep。

## Phase2-E2 Direction: Residual/Loss Alignment Diagnostic

[11-Step Loop] 当前回滚到 Step 5-6：重新检查理论可行性与实验设计。

目的：回答 QDF learned matrix 是否真的能解释 FATST R.3 的 residual/error direction。

### Diagnostic Design

比较三类 objective matrix 在同一批 FATST R.3 residual 上的行为：

1. `identity/prefix-risk`: 当前 R.3 的 diagonal objective pressure；
2. `static train-target offdiag`: Phase2-E1 使用的 target covariance proxy；
3. `QDF learned off_diag/all`: native QDF 保存的 learned matrix。

观察量：

- per-horizon residual quadratic loss；
- matrix 对 hard horizons / specialist gaps 的 loss ranking；
- objective gradient direction 与 actual residual energy 的相关性；
- learned matrix 是否能区分 R.3 已经失败的 settings。

### Decision Rule

- 若 QDF learned matrix 在 R.3 residual 上显示稳定 alignment，则进入 learnable 或
  validation-informed local objective，不再使用 frozen target-only proxy。
- 若 alignment weak，则停止 objective route，回到 base architecture / baseline story；不进入
  MoE 或更复杂机制堆叠。

### Artifact Gap

[Fact] 当前本地 R.3 gate artifact 只确认存在 `metrics_by_target_horizon.csv`，未发现
`predictions_test.npz`。Phase2-E2 若要做 residual-level offline diagnostic，需要先补充
prediction/true artifacts，或补跑只保存预测的 R.3 diagnostic run。
