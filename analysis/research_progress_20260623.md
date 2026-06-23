# FATST Current Research Progress

更新时间：2026-06-23 18:02 +08:00

## 1. 总体目标

[Fact] 当前项目目标是围绕 time series forecasting 形成可投稿 SCI 期刊的模型创新。
候选方向包括 one-model multi-horizon forecasting、future-aware architecture 和
MoE-style conditional computation。

[Decision] 当前阶段仍处于 decoder/output strategy 的 research loop，不应进入 MoE。
原因是：尚未找到一个同时具备稳定性能收益和清晰 paper story 的 target-side state 或
decoder mechanism。MoE 需要依附于已经通过 gate 的 state/operator carrier，否则会变成
对失败机制的复杂化。

## 2. Phase0: Canonical Base

[Decision] `PatchEncoderFixedHead` 是当前 canonical internal base。

[Evidence] 它在 Phase0 中比 `DLinear` 和 `SegTSFTDenseFixedHead` 更适合作为后续机制
carrier：结构简洁、性能合理、输出端问题可诊断。

[Fact] Phase0 也证明 fixed direct head 存在可量化 prefix inconsistency 和 horizon-specific
specialist gap，但这些问题幅度中等，不足以单独支撑“variable-horizon decoder 必然显著提升”
的强 claim。

## 3. Phase1: Decoder / Target Interface

[Fact] 多个直接 decoder-side 候选未形成 paper-core：

| Candidate | Outcome |
| --- | --- |
| `PatchEncoderSegmentQueryHead` | capacity collapse, fail |
| `PatchEncoderFixedHeadAdapter` | partial signal, not paper-core |
| `PatchEncoderFutureAwareAdapter` | partial signal, unstable |
| `PatchEncoderStepSpecificStateAdapter` | partial signal, not stable |
| `PatchEncoderTrajectoryBasisResidual` | residual active but too weak |

[Inference] 这些结果说明，简单替换 fixed head 或在 output space 上加轻量 patch，不能稳定
超过 horizon-specific fixed-head specialists。

[Decision] Phase1 的有效收敛点是 `Target-Set Forecasting Decoder`：把 target horizon /
future positions 显式作为模型输入，而不是训练脚本外部参数。

## 4. Phase1-R: Target-Set Carrier

[Fact] 第一版 uniform target-set model 接近但没有通过 paper-core。

[Strong Evidence] `PatchEncoderPrefixRiskWeighted` 是目前最强的 one-model target-set carrier：

- mean relative MSE 从 uniform target-set 的 `+0.62%` 推到 `-0.43%`；
- H720-prefix h96/h192 相比 fixed H720-prefix 改善 `-2.46%`；
- prefix consistency 保持数值零级别。

[Decision] R.3 通过 compatibility carrier 判断，但不是 paper-core。它主要是 objective
weighting，不是完整 decoder/process mechanism；并且仍留下 `ETTm1/h96`、`ETTm1/h720`
和 `ETTh2/h720` 等 specialist gaps。

## 5. Phase2-A/R/B: Future-State 和 Error-Process 失败链

[Fact] `PatchEncoderFutureStateAlignment` 安全性通过，但性能 gate 失败：它在部分数据集
有正信号，却破坏 R.3 在 `ETTh2` 上的 compatibility。

[Fact] `PatchEncoderFutureStateAlignmentConfWeighted` 修复尝试仍失败，说明问题不是简单
teacher confidence 或 leakage。

[Fact] `PatchEncoderErrorProcessDecoder` 也失败：residual decomposition 安全、残差非零，
但 forecast MSE 没有稳定收益。

[Decision] 不应在这些失败 state/residual 上继续叠 MoE。当前问题更可能在 mixed-horizon
objective pressure 或更基础的 architecture choice。

## 6. Phase2-C: Objective Pressure

[Strong Evidence] objective bottleneck 真实存在。R.3 不改 architecture，却相对 uniform
target-set 获得：

- `11/12` MSE wins；
- mean relative MSE `-1.03%`；
- h96 mean relative MSE `-1.81%`；
- segment-level wins `24/30`。

[Fact] 但 R.3 的 prefix-risk weighting 将 pressure 过度集中到 early prefix：

| Region | Uniform share | Prefix-risk share |
| --- | ---: | ---: |
| `1-96` | `0.4798` | `0.7217` |
| `97-192` | `0.2298` | `0.1540` |
| `193-336` | `0.1571` | `0.0775` |
| `337-720` | `0.1333` | `0.0469` |

因此 Phase2-C 测试了 coverage-only `region_balanced`：把四个 region 的 weighted pressure
share 变为 `0.25`。

## 7. 本次返回实验结论

[Decision] `PatchEncoderRegionBalanced` 失败。

核心结果：

- MSE wins vs R.3: `2/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+1.53%`;
- dataset mean relative MSE vs R.3:
  `ETTh2 -0.29%`, `ETTm1 +3.19%`, `Weather +1.70%`;
- mean relative MSE vs uniform target-set: `+0.47%`;
- mean relative MSE vs fixed head: `+1.10%`;
- max prefix mismatch MSE: `5.2042527595328944e-14`.

[Inference] 这不是实现安全问题。Prefix consistency 仍然成立，说明 target-set interface
没有坏；失败来自 objective pressure allocation 本身。Equal-region coverage 把 early
prefix pressure 降得过多，导致 `ETTm1/h96`、`Weather/h96` 和多数 horizon 退化。

[Decision] 单纯 coverage balance 被证伪。不要手调 region multipliers，也不要在
`region_balanced` 上叠 MoE。

## 8. 当前可讲故事程度

[Strong Evidence] 目前可以讲的故事是 diagnostic story，而不是 final method story：

> one-model multi-horizon forecasting 的瓶颈不只是 decoder capacity；objective pressure
> 会显著改变 target-set decoder 的行为，但 naive pressure allocation 会产生新的 tradeoff。

[Inference] 这个故事有研究价值，但还不够成为论文主方法。论文主方法仍缺少一个通过 gate 的
机制，能够同时解释并改善 R.3 留下的 specialist gaps。

## 9. 下一步收敛路线

[Decision] 当前回到 11-step loop 的 step 2-3。

优先下一步不是直接实现 `step_covariance_balanced`，而是先做离线 diagnostic：

1. 用 training targets 计算四个 region 的 normalized covariance / novelty；
2. 检验 novelty 是否能解释 R.3 与 `region_balanced` 在 segment-level 的 gain/loss；
3. 若 novelty 与改善方向一致，再进入 `step_covariance_balanced` 的 step 4-6；
4. 若 novelty 解释力弱，停止 objective-only 主线，转向 base architecture 或 external
   baseline reproduction / selection。

[Practical Next] 当前建议新建一个 Phase2-C.1 offline diagnostic，不训练模型，只读取
dataset train split 与现有 Phase2-C artifacts。这样成本低，且能避免继续在失败 objective
上盲目训练。

## 10. Phase2-C.1: Covariance / Novelty Offline Diagnostic

[Fact] Phase2-C.1 已完成，不训练模型，只读取 `ForecastDataset` train split 与现有
Phase2-C artifacts。

Artifacts:

- report:
  `analysis/phase2_covariance_novelty_diagnostic_20260623/phase2_covariance_novelty_diagnostic_report.md`
- script:
  `scripts/analyze_phase2_covariance_novelty.py`
- code explanation:
  `docs/code-explanation/phase2-covariance-novelty-diagnostic.md`

[Strong Evidence] train-target novelty 与已有 gain/loss pattern 有一致关系：

- R.3 segment delta vs novelty share Pearson: `-0.7219`;
- R.3 segment delta vs prefix pressure share Pearson: `-0.6909`;
- `region_balanced` delta vs novelty deficit Pearson: `+0.6253`;
- aggregate R.3 delta vs novelty share Pearson: `-0.6714`;
- aggregate `region_balanced` delta vs novelty deficit Pearson: `+0.6253`。

[Fact] 三个数据集的最大 novelty region 都是 `1-96`：

| Dataset | Max novelty region | Early novelty share | Late novelty share |
| --- | --- | ---: | ---: |
| `ETTh2` | `1-96` | `0.4763` | `0.1906` |
| `ETTm1` | `1-96` | `0.6152` | `0.1299` |
| `Weather` | `1-96` | `0.4815` | `0.1623` |

[Inference] 这解释了为什么 R.3 的 early prefix pressure 提升有效，也解释了为什么
`region_balanced` 把 `1-96` share 降到 `0.25` 后会明显伤害 `ETTm1` 与 `Weather`。
因此失败的不是 objective route 本身，而是 coverage-only equal-region 假设。

[Counterargument] 该证据仍不是最终机制证明：`region_balanced` delta vs novelty deficit 的
Spearman 只有 `0.1538`，说明排序层面的单调性较弱；novelty 目前也是 dataset-level static
proxy，不是 sample-adaptive mechanism。

[Decision] 当前进入 11-step loop 的 step 4-6：允许设计 `step_covariance_balanced`，但第一版
必须保持 objective-only、固定超参、小范围 gate，并且主要比较对象仍是 R.3。若该训练候选不能
超过 R.3，应停止 objective-only 主线，转向 base architecture 或 external baseline selection。

## 11. Phase2-C.2: Step-Covariance Balanced Objective

[Fact] `step_covariance_balanced` 已实现并通过本地 smoke。

QDF 相关性：

- [Fact] QDF 指出 standard MSE 等价于 identity weighting，忽略 label autocorrelation 与
  heterogeneous task weights。
- [Inference] 当前候选只采用 QDF 的 diagonal / heterogeneous weighting 思路，不采用完整
  off-diagonal quadratic matrix，也不引入 bilevel/meta-learning。

Implementation artifacts:

- train switch:
  `--step-loss-weighting step_covariance_balanced`
- run name:
  `PatchEncoderStepCovarianceBalanced`
- code:
  `baselines/patch_encoder_target_set_decoder/train.py`
- remote runner:
  `scripts/remote/run_phase2_step_covariance_balanced_gate.sh`
- sync wrapper:
  `scripts/sync_phase2_step_covariance_balanced_results.sh`
- code explanation:
  `docs/code-explanation/phase2-step-covariance-balanced-objective.md`

[Design] 默认使用 `step_covariance_beta=0.5`, `step_covariance_eta=0.5`,
`step_covariance_eps=1e-6`。该设置不是调参 sweep，而是为了避免 `region_balanced`
把 early pressure 直接压到 `0.25` 的失败模式。

[Fact] ETTh2 local smoke 通过：

- output:
  `artifacts/runs/smoke_phase2_step_covariance_balanced/PatchEncoderStepCovarianceBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- scope: `ETTh2`, `{96,192,336,720}`, `epochs=1`, `steps_per_epoch=2`,
  `max_eval_batches=1`, CPU
- weighted pressure share:
  `1-96 = 0.4807`, `97-192 = 0.1951`, `193-336 = 0.1640`, `337-720 = 0.1603`
- prefix mismatch MSE:
  `96/720 = 8.455293790696292e-15`,
  `192/720 = 8.434740536231167e-15`,
  `336/720 = 3.5504944524786947e-15`

[Decision] 下一步是按远程实验策略在 `529_Lab-3090` 跑完整
`ETTh2/ETTm1/Weather x {96,192,336,720}` gate。Primary comparison 仍是 R.3；若
`step_covariance_balanced` 不能超过 R.3，objective-only 主线停止。

[Fact] Remote gate 已启动：

- launch note:
  `analysis/phase2_step_covariance_balanced_gate_20260623/remote_launch_note.md`
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective`
- commit:
  `f0f5d41`
- selected GPUs:
  `1`, `2`
- first launched datasets:
  `ETTm1` on GPU `1`, `Weather` on GPU `2`

## 12. Phase2-C.2 Remote Gate Result

[Decision] `PatchEncoderStepCovarianceBalanced` 未通过 Phase2-C.2 gate。

Artifacts:

- decision report:
  `analysis/phase2_step_covariance_balanced_gate_20260623/phase2_step_covariance_balanced_decision_report.md`
- interpretation:
  `analysis/phase2_step_covariance_balanced_gate_20260623/phase2_step_covariance_balanced_interpretation.md`
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective`

核心结果：

- MSE wins vs R.3: `2/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+0.76%`;
- dataset mean relative MSE vs R.3:
  `ETTh2 -0.09%`, `ETTm1 +1.35%`, `Weather +1.03%`;
- MSE wins vs uniform target-set: `12/12`;
- mean relative MSE vs uniform target-set: `-0.28%`;
- MSE wins vs FixedHead: `6/12`;
- mean relative MSE vs FixedHead: `+0.33%`;
- max prefix mismatch MSE: `5.182444710459706e-14`。

[Inference] QDF-style premise仍成立：future-step objective weights 确实重要；
static novelty-aware diagonal weighting 能稳定超过 uniform target-set。但它没有超过 R.3，
说明当前简化版 objective 不能作为 paper-core。

[Failure Pattern] 该候选没有保住 R.3 的 early-prefix 优势。`1-96` weighted pressure share 为
`ETTh2 0.4807`, `ETTm1 0.5501`, `Weather 0.4813`，明显低于 R.3 的 `0.7217`。
因此 `ETTm1/h96` 和 `Weather/h96` 仍分别比 R.3 退化 `+2.38%`、`+1.34%`。

[Decision] 不继续手调 `beta/eta`，不在该 objective carrier 上进入 MoE。当前回到 11-step
loop 的 step 2-3：要么把完整 QDF-style off-diagonal / learned quadratic objective 作为
external baseline 或 diagnostic 复现，要么停止 objective-only 主线并回到 base architecture /
external baseline selection。

## 13. Phase2-D: QDF Off-Diagonal Diagnostic

[Fact] 已完成 QDF off-diagonal diagnostic，不训练模型，只读取 `ForecastDataset` train split
与 Phase2-C.1/C.2 summary。

Artifacts:

- report:
  `analysis/phase2_qdf_offdiag_diagnostic_20260623/phase2_qdf_offdiag_diagnostic_report.md`
- script:
  `scripts/analyze_phase2_qdf_offdiag_diagnostic.py`
- experiment plan:
  `docs/experiments/phase2-qdf-offdiag-reproduction-path.md`
- code explanation:
  `docs/code-explanation/phase2-qdf-offdiag-diagnostic.md`

[Fact] QDF 官方实现的核心 loss 不是 static diagonal weights。其 `CovarianceMatrix`
将误差从 `[B, P, D]` 展平成 `[B*D, P]` 后，用 learned matrix 计算 quadratic loss，
并支持 `meta_type=all/diag/off_diag`。

[Strong Evidence] 本地诊断按相同轴语义，把 H720 future target regions 构成 `[B*D, 4]`
matrix，三个数据集均有强 off-diagonal signal：

| Dataset | Mean abs offdiag corr | Max abs offdiag corr | Offdiag corr Fro share |
| --- | ---: | ---: | ---: |
| `ETTh2` | `0.7103` | `0.8127` | `0.6057` |
| `ETTm1` | `0.8585` | `0.8897` | `0.6888` |
| `Weather` | `0.7342` | `0.8066` | `0.6193` |

[Decision] Phase2-D diagnostic pass：

- mean_abs_offdiag_corr: `0.7677`;
- min_dataset_mean_abs_offdiag_corr: `0.7103`;
- diagonal_proxy_failed: `True`;
- novelty_supported_diagonal_before_training: `True`;
- supports_qdf_upstream_reproduction: `True`。

[Inference] 当前结论不是“QDF 一定有效”，而是“完整 QDF/off-diagonal 机制值得被原生复现”。
下一步应进入 QDF upstream reproduction gate；不要继续调 `step_covariance_balanced` 的
`beta/eta`，也不要直接把 QDF module 移植进 FATST。

[Implementation Update] Phase2-D upstream reproduction tooling 已就绪：

- remote runner:
  `scripts/remote/run_phase2_qdf_upstream_gate.sh`
- progress checker:
  `scripts/remote/check_phase2_qdf_upstream_progress.sh`
- sync wrapper:
  `scripts/sync_phase2_qdf_upstream_results.sh`
- analyzer:
  `scripts/analyze_phase2_qdf_upstream_gate.py`
- code explanation:
  `docs/code-explanation/phase2-qdf-upstream-reproduction-runner.md`

[Next] commit/push 后，在 `529_Lab-3090` `git pull`，检查 GPU 与 QDF upstream 环境。
若环境满足，先启动 `META_TYPES=all` 的最小 QDF upstream gate；controls 后续用
`META_TYPES="diag off_diag"` 补齐。

[Remote Launch Update] Phase2-D QDF upstream `META_TYPES=all` gate 已在 `529_Lab-3090`
启动：

- launch note:
  `analysis/phase2_qdf_upstream_gate_20260623/remote_launch_note.md`
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`
- FATST commit:
  `9217cb8`
- QDF upstream commit:
  `eb0693a`
- selected GPUs:
  `1`, `2`
- first active jobs:
  `ETTm1/h96`, `ETTm1/h192`

[Caveat] 该轮只跑 `meta_type=all`。最终是否进入 source-informed localization，需要后续补
`diag` control 并由 `scripts/analyze_phase2_qdf_upstream_gate.py` 判定。

[Remote Repair Update] 首次 `META_TYPES=all` 启动在 `ETTm1/h96,h192` 训练到 meta-test
结束后失败，原因是 PyTorch 2.6+ 默认 `torch.load(weights_only=True)` 拒绝加载 QDF 保存的
完整 `CovarianceMatrix` 对象 `A.pth`。这不是模型机制失败，也没有产生 metrics。已将
`scripts/remote/run_phase2_qdf_upstream_gate.sh` 更新为启动前自动 patch QDF upstream 的
两个 `A.pth` load 点为 `weights_only=False`，随后需要用 `RERUN=1` 重跑 `META_TYPES=all`。

[Remote Repair Update] 后续 `ETTm1/h96` 越过 `A.pth` load 后，在 test DataLoader 报
`Too many open files`。已将 QDF runner 默认 `NUM_WORKERS=0` 并设置 `ulimit -n`，下一步先
单独重跑 `ETTm1/h96` 监控到 metrics 生成，再恢复 full matrix。

## 12. Phase2-D: QDF Upstream `meta_type=all` Gate Result

[Fact] Phase2-D QDF upstream `META_TYPES=all` gate 已完整返回：

- completed runs: `12/12`;
- metric files: `12/12`;
- covariance artifacts: `12/12`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`;
- local report:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_decision_report.md`。

`meta_type=all` 的 MSE 如下：

| Dataset | h96 | h192 | h336 | h720 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `0.285880` | `0.361037` | `0.407588` | `0.419218` | `0.368431` |
| `ETTm1` | `0.306606` | `0.352415` | `0.382601` | `0.441164` | `0.370696` |
| `Weather` | `0.159555` | `0.209021` | `0.264798` | `0.342472` | `0.243961` |

[Decision] 该结果证明 upstream QDF full covariance 路径可以稳定训练和产出完整 artifact，
但尚不能证明 QDF mechanism pass。原因是当前只有 `meta_type=all`，缺少同一 upstream
protocol 下的 `diag` 和 `off_diag` controls。

[11-Step Loop] 当前处于 Step 9-10 的未闭合状态：实验产物已返回，但 performance/story
candidate 不能判定通过。下一步回到 Step 6-8，补 `META_TYPES="diag off_diag"` control
matrix。只有当 `all` 相对 `diag` 在 mean MSE、win count 和 specialist gaps 上成立时，
才允许进入本地 source-informed localization；否则 objective route 回滚到 Step 2。

[Remote Launch Update] Phase2-D QDF upstream controls 已在 `529_Lab-3090` 启动：

- launch note:
  `analysis/phase2_qdf_upstream_gate_20260623/remote_controls_launch_note.md`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`;
- FATST commit:
  `4f5f372`;
- selected GPUs:
  `1`, `2`;
- first active jobs:
  `diag/ETTm1/h96`, `diag/ETTm1/h192`。

[Decision] 在 control gate 完成前，不进入 FATST 本地 QDF component 实现。

## 14. Phase2-D: QDF Upstream Control Gate Final Result

[Fact] Phase2-D QDF upstream controls 已完整返回：

- completed metric rows: `36`;
- meta types: `all`, `diag`, `off_diag`;
- all runs completed: `12/12`;
- comparison report:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_decision_report.md`;
- comparison table:
  `analysis/phase2_qdf_upstream_gate_20260623/phase2_qdf_upstream_meta_type_comparison.csv`。

[Strong Evidence] `all` 相对 `diag` 通过 gate：

- mean relative MSE: `-1.08%`;
- MSE wins: `11/12`;
- specialist gap wins: `4/4`;
- mean relative MAE: `-0.67%`;
- dataset mean relative MSE:
  `ETTh2 -1.89%`, `ETTm1 -1.12%`, `Weather -0.23%`。

[Counter-Evidence] `all` 并不是最优形态。相对 `off_diag`：

- mean relative MSE: `+0.06%`;
- MSE wins: `2/12`;
- specialist gap wins: `1/4`;
- `off_diag` 是 `10/12` 个 setting 的最优 meta type。

[Inference] QDF 论文关于 future steps 不应等权、且 residual interactions 重要的判断得到支持；
但本地化时不应直接追求 full learned matrix。更稳妥的解释是：diagonal-only objective 缺少
future-step interaction；而 `off_diag` 在固定 diagonal 后学习 residual coupling，反而比
`all` 更稳定。

[Decision] Phase2-D external reproduction gate passes，但通过的是 “off-diagonal interaction
值得本地化验证”，不是 “完整 QDF full matrix 可以直接成为 FATST 主方法”。

## 15. Phase2-E: Local Off-Diagonal Objective Direction

- `current_step`: Step 4-6，基于 Phase2-D pass 设计本地 source-informed candidate。
- `problem`: R.3 的 prefix-risk diagonal weighting 能改善 objective pressure，但无法表达
  future-step residual interaction；`step_covariance_balanced` 的 static diagonal proxy 已失败。
- `existence_evidence`: QDF upstream `all` vs `diag` 为 `11/12` MSE wins，specialist gaps
  `4/4` wins；本地 target-region off-diagonal correlation mean abs 为 `0.7677`。
- `idea`: 先做 local off-diagonal objective probe：保留 R.3 carrier 和 prefix-risk diagonal
  pressure，只增加低维 future-region residual interaction term。
- `theory_check`: 若收益主要来自 off-diagonal coupling，则一个 4-region 或 banded low-rank
  quadratic residual penalty 应能捕捉部分收益；若只有 QDF 的 bilevel/meta loop 有效，则该
  本地静态/轻量版本会失败。
- `design`: Phase2-E0 先审计 QDF learned `A.pth` matrix，统计 diagonal/off-diagonal energy、
  bandwidth、region aggregation、condition/PSD proxy；Phase2-E1 再实现 FATST 本地
  `offdiag_region_quadratic` objective。
- `gate`: 本地训练候选必须相对 R.3 达到 mean MSE `< -0.3%`、MSE wins `>=7/12`、
  specialist gap wins `>=2/4`，且 prefix consistency 仍为数值零级别。
- `artifacts`: 当前已有 Phase2-D upstream artifacts；Phase2-E0 应新增 matrix audit report。
- `decision`: 不进入 MoE，不复制 QDF full meta-learning；下一步先做 learned matrix audit，
  再决定是否实现本地 off-diagonal objective。

## 16. Phase2-E0/E1: Matrix Audit and Local Off-Diagonal Probe

[Fact] Phase2-E0 learned matrix audit 已完成：

- script:
  `scripts/analyze_phase2_qdf_matrix_audit.py`;
- report:
  `analysis/phase2_qdf_matrix_audit_20260623/phase2_qdf_matrix_audit_report.md`;
- code explanation:
  `docs/code-explanation/phase2-qdf-matrix-audit.md`。

[Strong Evidence] 审计支持继续本地 off-diagonal objective probe：

| Meta type | Precision offdiag fro share | Covariance offdiag fro share | Normalized precision bandwidth |
| --- | ---: | ---: | ---: |
| `all` | `0.011602` | `0.126068` | `0.201041` |
| `diag` | `0.000000` | `0.000000` | `0.000000` |
| `off_diag` | `0.013700` | `0.149504` | `0.202526` |

[Inference] QDF 的有效信号不是纯 diagonal，也不是只在局部相邻 step 上传播。`off_diag`
precision interaction 比 `all` 更强，且 normalized bandwidth 约 `0.20`，支持做
block-level future residual interaction，而不是 4-region-only 或 full 720x720 local matrix。

[Implementation Update] Phase2-E1 最小本地候选已实现并通过本地 smoke：

- train switch:
  `--step-loss-weighting offdiag_block_quadratic`;
- run name:
  `PatchEncoderOffdiagBlockQuadratic`;
- code:
  `baselines/patch_encoder_target_set_decoder/train.py`;
- remote runner:
  `scripts/remote/run_phase2_offdiag_block_quadratic_gate.sh`;
- sync wrapper:
  `scripts/sync_phase2_offdiag_block_quadratic_results.sh`;
- code explanation:
  `docs/code-explanation/phase2-offdiag-block-quadratic-objective.md`。

[Design] 该候选保留 R.3 的 `prefix_risk` base loss，并额外加入 train-split 估计的
off-diagonal block residual penalty：

- `offdiag_block_size=48`;
- H720 有 `15` 个 blocks；
- H96 有 `2` 个 blocks，因此 short-horizon specialist gap 也会接收到 off-diagonal signal；
- `offdiag_quadratic_weight=0.05`;
- `offdiag_ridge_eps=1e-3`。

[Fact] ETTh2 CPU smoke 通过：

- output:
  `artifacts/runs/smoke_phase2_offdiag_block_quadratic/SmokeOffdiagBlockQuadratic/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- scope: `ETTh2`, `{96,192,336,720}`, `epochs=1`, `steps_per_epoch=2`,
  `max_eval_batches=1`, CPU;
- `offdiag_block_count=15`;
- `offdiag_precision_fro_share_before_norm=0.0093818328`;
- max prefix mismatch MSE:
  `8.394639455409306e-15`。

[Next] commit/push 后，在 `529_Lab-3090` `git pull`，检查 GPU，启动
Phase2-E1 remote gate。Gate 仍以 R.3 为主比较对象：mean MSE `< -0.3%`、MSE wins
`>=7/12`、specialist gap wins `>=2/4`、prefix consistency 不破坏。

[Remote Launch Update] Phase2-E1 `PatchEncoderOffdiagBlockQuadratic` gate 已在
`529_Lab-3090` 启动：

- launch note:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/remote_launch_note.md`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective`;
- FATST commit:
  `b3ba31d`;
- selected GPUs:
  `1`, `2`;
- initial active jobs:
  `ETTm1`, `Weather`;
- initial progress:
  `ETTm1 epoch=3/100`, `Weather epoch=1/100`。

## 17. Phase2-E1 Returned Result: Static Off-Diagonal Proxy Fails

[Fact] Phase2-E1 `PatchEncoderOffdiagBlockQuadratic` remote gate 已完成并同步：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective`;
- local artifacts:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/`;
- completed datasets:
  `ETTh2`, `ETTm1`, `Weather`;
- completed horizons:
  `96`, `192`, `336`, `720`;
- main report:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/phase2_offdiag_block_quadratic_decision_report.md`;
- interpretation:
  `analysis/phase2_offdiag_block_quadratic_gate_20260623/phase2_offdiag_block_quadratic_interpretation.md`。

[Decision] 该候选不通过本地 paper-core gate：

- MSE wins vs R.3: `1/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+0.0464%`;
- dataset mean relative MSE vs R.3:
  `ETTh2 +0.0567%`, `ETTm1 +0.0746%`, `Weather +0.0078%`;
- specialist gap wins vs R.3: `1`;
- H720 stability region wins vs R.3: `0`;
- max prefix mismatch MSE:
  `5.4052272504883855e-14`。

[Counter-Evidence] 它不是一个数值坏机制，只是不能超过 R.3：

- MSE wins vs uniform target-set: `10/12`;
- mean relative MSE vs uniform target-set: `-0.9871%`;
- MSE wins vs FixedHead: `8/12`;
- mean relative MSE vs FixedHead: `-0.3870%`;
- all datasets satisfy no degradation over `0.3%` vs R.3。

[Inference] 该结果说明 static train-target block precision proxy 太弱。它可以作为安全正则项，
但不能解释 QDF upstream 中 `all/off_diag` 对 `diag` 的稳定收益。QDF 的可借鉴点应继续收窄为
learned/error-aware future-step residual interaction，而不是 frozen target covariance penalty。

[11-Step Loop]

- `current_step`: Step 9-10。
- `problem`: R.3 的 diagonal/prefix objective 缺少 explicit future-step residual interaction。
- `existence_evidence`: QDF upstream controls 仍支持该问题存在；Phase2-E1 只否定当前 local
  proxy。
- `idea`: static off-diagonal block quadratic objective。
- `theory_check`: target-only static covariance 不能充分近似 QDF learned objective。
- `design`: R.3 base loss + block residual projection squared penalty。
- `gate`: mean MSE vs R.3 `< -0.3%`、MSE wins `>=7/12`、specialist gap wins
  `>=2/4`、prefix consistency pass。
- `artifacts`: `phase2_offdiag_block_quadratic_*` CSV/JSON/report files。
- `decision`: fail；回滚到 Step 5-6，不进入 MoE 或其他复杂机制堆叠。

[Next] Phase2-E2 应先做 QDF-to-FATST residual/loss alignment diagnostic：

1. 在同一批 FATST R.3 residual 上比较 `identity/prefix-risk`、`static train-target offdiag`、
   `QDF learned off_diag/all` matrix 的 loss ranking 或 gradient pressure。
2. 判断 QDF learned matrix 是否真的对 FATST 的 error direction 有解释力。
3. 若 alignment positive，再实现 learnable 或 validation-informed local objective；若
   alignment weak，停止 objective route，回到 base architecture / baseline story。

[Blocker] 当前本地 R.3 artifact 只确认存在 `metrics_by_target_horizon.csv`，未发现
`predictions_test.npz`。因此 Phase2-E2 的 residual-level diagnostic 需要先补充保存
prediction/true artifacts，或补跑只保存预测的 R.3 diagnostic run。

## 18. Phase2-E2: QDF-to-FATST Residual Alignment Diagnostic Implementation

[Fact] Phase2-E2 diagnostic tooling 已完成：

- analyzer:
  `scripts/analyze_phase2_qdf_residual_alignment.py`;
- remote prediction runner:
  `scripts/remote/run_phase2_qdf_alignment_r3_predictions.sh`;
- remote progress checker:
  `scripts/remote/check_phase2_qdf_alignment_r3_predictions_progress.sh`;
- sync wrapper:
  `scripts/sync_phase2_qdf_alignment_r3_predictions.sh`;
- code explanation:
  `docs/code-explanation/phase2-qdf-residual-alignment-diagnostic.md`;
- dry report:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/phase2_qdf_residual_alignment_report.md`。

[Implementation] `scripts/remote/run_phase1_target_set_decoder_gate.sh` 新增 opt-in
`SAVE_PREDICTIONS` 开关，默认 `0`，不影响既有 gate。Phase2-E2 wrapper 显式设置：

- `RUN_NAME=PatchEncoderPrefixRiskWeighted`;
- `STEP_LOSS_WEIGHTING=prefix_risk`;
- `SAVE_PREDICTIONS=1`;
- `KEEP_HEAVY_ARTIFACTS=1`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions`。

[Verification] 本地验证已完成：

- `python -m py_compile scripts/analyze_phase2_qdf_residual_alignment.py`;
- `bash -n` for all new shell wrappers;
- dry analysis 生成 missing-artifact report；
- QDF `A.pth` load check 通过：
  `off_diag`, `L_param`, precision shape `(96, 96)`。

[Current Diagnostic Status] dry analysis 的 gate 为：

- `prediction_artifacts_complete=False`;
- `qdf_matrices_complete=True`;
- `ready_for_alignment_decision=False`;
- missing R.3 prediction artifacts: `12/12`。

[Decision] 现在还不能做 alignment conclusion。下一步是 commit/push 后在 `529_Lab-3090`
拉取代码、检查 GPU，并启动 Phase2-E2 R.3 prediction artifact run。该 run 是 artifact collection，
不是新机制训练；返回后再运行 sync/analyzer，判断 QDF learned matrices 是否比 static proxy 更能解释
FATST R.3 residual。
