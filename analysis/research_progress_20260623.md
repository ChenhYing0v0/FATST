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

## 19. Phase2-E2 Returned Result: QDF Precision Transfer Fails as R.3 Repair Evidence

[Fact] Phase2-E2 R.3 prediction artifact collection 已完成并同步：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions`;
- local analysis root:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/`;
- prediction artifacts:
  `12/12`;
- QDF matrix artifacts:
  `36/36`;
- report:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/phase2_qdf_residual_alignment_report.md`;
- interpretation:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/phase2_qdf_residual_alignment_interpretation.md`。

[Verification] analyzer 修正并重跑：

- `relative_mse_pct > 0` 用于识别 R.3 vs fixed specialist gaps；
- `static_train_target_offdiag` 的 `ratio_to_residual_mse` 已统一改为除以 plain residual MSE；
- `python -m py_compile scripts/analyze_phase2_qdf_residual_alignment.py` 通过。

[Result] Specialist gap settings 为：

- `ETTh2 / 720`;
- `ETTm1 / 96`;
- `ETTm1 / 720`;
- `Weather / 96`。

[Evidence] QDF learned matrices 没有对 specialist gaps 给出更高 pressure：

| Matrix family | Mean ratio | Specialist ratio | Non-specialist ratio | Gap / non-gap |
| --- | ---: | ---: | ---: | ---: |
| `prefix_risk` | `1.460518` | `1.528278` | `1.426638` | `1.071244` |
| `static_train_target_offdiag` | `0.156607` | `0.168521` | `0.150650` | `1.118626` |
| `qdf_off_diag_precision` | `0.531174` | `0.481565` | `0.555978` | `0.866158` |
| `qdf_all_precision` | `0.545758` | `0.502381` | `0.567446` | `0.885338` |
| `qdf_diag_precision` | `0.998441` | `0.999521` | `0.997902` | `1.001622` |

[Decision] Phase2-E2 fails as evidence for QDF precision direct transfer. QDF upstream 仍可作为
future-step interaction 的背景证据，但不应继续作为 FATST 本地 objective 的直接设计来源。

[11-Step Loop]

- `current_step`: Step 9-10。
- `problem`: R.3 仍有 short/long 两端 specialist gaps。
- `existence_evidence`: R.3 vs fixed head 有 `4/12` MSE gaps；Phase2-E2 residual artifacts
  完整确认这些 settings。
- `idea`: QDF learned `off_diag/all` precision transfer。
- `theory_check`: 若 learned precision 能解释 R.3 residual，应对 specialist gaps 有更强 pressure。
- `design`: 同一批 R.3 residual 上比较 `identity/prefix_risk/static/QDF` matrix losses。
- `gate`: QDF learned matrices 对 specialist gaps 有正向区分。
- `artifacts`: `phase2_qdf_residual_alignment_losses.csv`,
  `phase2_qdf_residual_alignment_summary.json`。
- `decision`: fail；回滚到 Step 2-3，重新定义问题为 prefix-consistency 与 specialist-performance
  tradeoff。

[Next] Phase3-A 应做 Prefix-Consistent Carrier vs Horizon Specialist Tradeoff Diagnostic：

1. 使用已回传的 R.3 `predictions_test.npz`，不先训练新模型。
2. 比较 `h96/h192/h336` 与 `h720` prefix 的 residual direction、error covariance、
   step-region energy shift。
3. 检查四个 specialist gaps 是否体现 short-prefix 或 long-tail calibration conflict。
4. 若 tradeoff 成立，再设计最小 horizon-regime residual calibration；若不成立，停止 R.3
   repair route，回到 base architecture / external baseline selection。

## 20. Phase3-A Returned Result: Prefix Identity, Regime/Segment Gaps

[Fact] Phase3-A prefix-specialist tradeoff diagnostic 已完成：

- analyzer:
  `scripts/analyze_phase3_prefix_specialist_tradeoff.py`;
- report:
  `analysis/phase3_prefix_specialist_tradeoff_20260624/phase3_prefix_specialist_report.md`;
- short-horizon table:
  `analysis/phase3_prefix_specialist_tradeoff_20260624/phase3_prefix_specialist_short_alignment.csv`;
- H720 segment table:
  `analysis/phase3_prefix_specialist_tradeoff_20260624/phase3_prefix_specialist_h720_segments.csv`;
- code explanation:
  `docs/code-explanation/phase3-prefix-specialist-tradeoff-diagnostic.md`。

[Verification]

- `python -m py_compile scripts/analyze_phase3_prefix_specialist_tradeoff.py` passed；
- input prediction artifacts: Phase2-E2 R.3 `predictions_test.npz`；
- max prediction prefix mismatch MSE:
  `5.382513303646484e-14`;
- max truth prefix alignment MSE:
  `0.0`;
- max residual prefix mismatch MSE:
  `5.382513303646484e-14`。

[Decision] 同一输入下不存在 meaningful prefix prediction/residual conflict。R.3 的剩余 gaps
不是因为 `h96/h192/h336` 与 `h720` prefix 对同一窗口输出了不同预测；它们分成两类：

1. short-horizon gap 来自 short-only extra windows；
2. long-horizon gap 来自 H720 late segment。

[Evidence] Short horizon 分解：

| Dataset | Horizon | Gap type | Full gap | H720-prefix gap | Aligned MSE | Extra MSE | Extra vs aligned |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `ETTm1` | `96` | `short_extra_window_gap` | `True` | `False` | `0.284174` | `0.549860` | `+93.49%` |
| `Weather` | `96` | `short_extra_window_gap` | `True` | `False` | `0.147463` | `0.156898` | `+6.40%` |

[Evidence] H720 segment gaps:

| Dataset | Segment | Relative MSE vs fixed |
| --- | --- | ---: |
| `ETTh2` | `193-336` | `+4.07%` |
| `ETTh2` | `337-720` | `+0.47%` |
| `ETTm1` | `337-720` | `+3.03%` |

[11-Step Loop]

- `current_step`: Step 9-10。
- `problem`: R.3 剩余 specialist gaps 的来源不清。
- `existence_evidence`: R.3 vs fixed 有 `4/12` aggregate gaps；Phase3-A 进一步定位到
  short-only extra windows 与 H720 late segments。
- `idea`: prefix-consistency vs horizon-specialist tradeoff diagnostic。
- `theory_check`: 若同输入 prefix conflict 存在，则 prefix mismatch/residual mismatch 应非零。
- `design`: 对齐 `h96/h192/h336` 与 `h720` prefix 的前 `N_720` windows，并拆出 short-only
  extra windows；H720 用 segment table 定位 late gaps。
- `gate`: prefix identity pass，short gaps 可由 extra-window regime 解释，long gaps 可由 late
  segment localization 解释。
- `artifacts`: `phase3_prefix_specialist_*`。
- `decision`: pass as diagnostic；进入 Step 4-6，设计 Phase3-B 的 regime/segment calibration
  candidate。

[Next] Phase3-B 不应回到 QDF/objective matrix，也不应直接做 full MoE。基于用户对
output residual correction 可解释性的顾虑，Phase3-B 先做 Regime/Segment Mechanism
Diagnostic：只验证困难 regime/segment 是否能被 prediction-before features 识别，不实现
output residual repair。

1. 对 short-only extra-window issue：先做 time-position/regime diagnostic，确认 extra windows 是否
   位于 test split 末端并有可识别 input-regime signal。
2. 对 H720 late-segment issue：先判断 high-error late windows 是否能由 history/window-position
   signal 识别，目标 segment 为
   `ETTh2 193-336/337-720` 和 `ETTm1 337-720`。
3. 若 diagnostic pass，下一步只允许进入 target-state / segment-operator conditioned design；
   不采用 prediction 后的 arbitrary residual correction。

## 21. Phase3-B Returned Result: Pre-Input Regime/Segment Signals Exist

[Fact] Phase3-B regime/segment mechanism diagnostic 已完成：

- analyzer:
  `scripts/analyze_phase3_regime_segment_mechanism.py`;
- report:
  `analysis/phase3_regime_segment_mechanism_20260624/phase3_regime_segment_mechanism_report.md`;
- short-regime table:
  `analysis/phase3_regime_segment_mechanism_20260624/phase3_short_regime_preinput_features.csv`;
- H720 late-segment table:
  `analysis/phase3_regime_segment_mechanism_20260624/phase3_h720_late_segment_preinput_features.csv`;
- summary:
  `analysis/phase3_regime_segment_mechanism_20260624/phase3_regime_segment_mechanism_summary.json`;
- code explanation:
  `docs/code-explanation/phase3-regime-segment-mechanism-diagnostic.md`。

[Verification]

- `python -m py_compile scripts/analyze_phase3_regime_segment_mechanism.py` passed；
- `conda run -n r2026-fsa python scripts/analyze_phase3_regime_segment_mechanism.py` completed；
- diagnostic features 只来自 history input windows 与 split 内 window position；
- R.3 predictions / truths 只用于构造 analysis labels，不作为候选模型输入。

[Gate]

- `short_regime_pre_input_signal`: `True`;
- `late_segment_pre_input_signal`: `True`;
- `no_output_residual_mechanism_used`: `True`;
- `supports_conditioned_target_operator_design`: `True`。

[Evidence] Short-regime pre-input signals:

| Dataset | Horizon | Feature | AUC | SMD | Extra MSE | Aligned MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `ETTm1` | `96` | `history_mean` | `0.997619` | `-3.221514` | `0.549860` | `0.284174` |
| `Weather` | `96` | `window_index_norm` | `1.000000` | `2.599896` | `0.156898` | `0.147463` |
| `ETTm1` | `96` | `window_index_norm` | `1.000000` | `2.586690` | `0.549860` | `0.284174` |
| `Weather` | `96` | `history_std` | `0.979425` | `2.494574` | `0.156898` | `0.147463` |

[Evidence] H720 late-segment pre-input signals:

| Dataset | Segment | Feature | AUC | SMD | High-error MSE | Other MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ETTh2` | `337-720` | `window_index_norm` | `0.845886` | `1.543815` | `0.826450` | `0.369696` |
| `ETTh2` | `337-720` | `history_slope_abs_mean` | `0.828835` | `1.262857` | `0.826450` | `0.369696` |
| `ETTm1` | `337-720` | `window_index_norm` | `0.786843` | `1.219849` | `0.814483` | `0.356796` |
| `ETTh2` | `193-336` | `window_index_norm` | `0.777454` | `1.165940` | `0.753092` | `0.241627` |

[Decision] Phase3-B pass as mechanism diagnostic. 结果说明：R.3 的剩余 gaps 并不是只能在输出后
通过 residual patch 才能观察到；这些 hard windows/segments 在 prediction 之前已经有可用信号。
因此下一步可设计 conditioned target operator，但机制应作用于 `target state`、`target segment
feature` 或 readout 前的 segment operator，而不是对最终预测做自由 residual correction。

[Caveat] `window_index_norm` 是 prediction-before time-position signal，但单独依赖它会让论文机制偏弱。
Phase3-C 需要把 window position 与 history statistics 合成 regime token，并检查 learned gates
是否真的与 history distribution shift / late-segment difficulty 对齐。

[11-Step Loop]

- `current_step`: Step 9-10 完成，进入 Step 4-6。
- `problem`: short-only extra windows 与 H720 late segments 存在可定位 performance gaps。
- `existence_evidence`: Phase3-A 定位 gaps；Phase3-B 证明这些 failure groups 有 prediction-before
  separability。
- `idea`: 用 history/window-position regime token 条件化 target-side segment operator。
- `theory_check`: 若 failure groups 在预测前可识别，则不必使用 output residual correction；可在
  latent/readout 前改变 target-state transformation。
- `design`: Phase3-C `Regime/Segment-Conditioned Target Operator`，保持 R.3 target-set carrier，
  在 output head 前引入轻量 segment-conditioned operator。
- `gate`: 修复 observed gaps，同时 non-gap degradation 受控；保留 prefix consistency；不引入自由
  output residual head。
- `artifacts`: `phase3_regime_segment_mechanism_*`。
- `decision`: pass as diagnostic；下一步实现 Phase3-C 的最小 model candidate 与 remote gate。

[Next] Phase3-C 建议：

1. 新建 `PatchEncoderRegimeSegmentTargetOperator`，只在 R.3 target state 到 output readout 之前
   做 conditioning。
2. Regime token 输入使用 history summary + `window_index_norm`，target branch 使用 segment feature
   `q_j`，避免 future target leakage。
3. 先本地 smoke + prefix consistency 检查，再远程跑最小 gate；若 observed gaps 不改善或 non-gap
   退化明显，回滚到 Step 2-3，转向 base architecture / external baseline comparison。

## 22. Phase3-C Implementation: Regime/Segment-Conditioned Target Operator

[Fact] Phase3-C 最小候选已实现并通过本地 smoke：

- model:
  `baselines/patch_encoder_target_set_decoder/model.py::PatchEncoderRegimeSegmentTargetOperator`;
- dataset support:
  `ForecastDataset(..., return_index=True)` 可选返回 `window_index_norm`;
- training registration:
  `--model-variant regime_segment_operator`;
- remote runner:
  `scripts/remote/run_phase3_regime_segment_operator_gate.sh`;
- progress checker:
  `scripts/remote/check_phase3_regime_segment_operator_progress.sh`;
- code explanation:
  `docs/code-explanation/phase3-regime-segment-target-operator.md`。

[Mechanism Boundary]

- [Fact] 新候选不在 prediction 后加 residual；
- [Fact] conditioning 发生在 `segment_output` 之前；
- [Fact] future target 不进入 prediction path；
- [Fact] `regime_segment_operator` 最后一层 zero-initialized，初始行为接近 R.3；
- [Fact] scale/shift 使用 `0.1 * tanh(...)` bounded transform，避免早期大幅破坏 readout。

[Verification]

- `python -m py_compile baselines/patch_encoder_target_set_decoder/dataset.py
  baselines/patch_encoder_target_set_decoder/model.py
  baselines/patch_encoder_target_set_decoder/train.py` passed；
- local smoke:
  `conda run -n r2026-fsa python baselines/patch_encoder_target_set_decoder/train.py ...`;
- smoke completed one train step and 1-batch eval for `ETTh2` H96/H720；
- smoke `prefix_mismatch_mse = 1.015119944076633e-14`；
- smoke wrote `regime_segment_operator_stats.csv` and `regime_feature_stats.csv`。

[11-Step Loop]

- `current_step`: Step 7 complete locally，准备 Step 8 remote training。
- `problem`: R.3 对 short extra windows 与 H720 late segments 的 operator 过于同质。
- `existence_evidence`: Phase3-A/B 定位并验证 prediction-before signals。
- `idea`: history/window-position regime token + target segment token 条件化 readout 前 hidden state。
- `theory_check`: 机制发生在 output 前，能回应 residual correction 可解释性不足的问题。
- `design`: near-identity bounded FiLM-style operator，保留 R.3 target-set carrier 和 prefix consistency。
- `gate`: remote minimal gate 先跑 `ETTm1`, `Weather`, `ETTh2` 的 H96/H720。
- `artifacts`: local smoke artifacts ignored in git；remote artifacts 将落在
  `/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator`。
- `decision`: 进入 Step 8；远程前需 commit/push，并在 `529_Lab-3090` 检查 GPU。

[Remote Launch Update]

- launch note:
  `analysis/phase3_regime_segment_operator_20260624/remote_launch_note.md`;
- remote commit:
  `bf9af01e31eb3b9ceaaa623448ee2ca5e45ae296`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator`;
- PID:
  `3736385`;
- selected GPUs:
  GPU1 and GPU2，launch 前均为 `18 MiB` used；
- avoided GPU:
  GPU0，launch 前为 `3964 MiB` used；
- command:
  `GPU_IDS="1 2" DATASETS="ETTm1 Weather ETTh2" TARGET_HORIZONS="96,720" EPOCHS=100
  bash scripts/remote/run_phase3_regime_segment_operator_gate.sh`;
- progress at 2026-06-24 10:48 CST:
  `ETTm1 3/100`, `Weather 1/100`, `ETTh2 queued`。

[Decision] Phase3-C 已进入 Step 8 remote training，且至少两个数据集正常运行。下一步等远程完成后同步
metrics、segment metrics、prefix consistency 与 regime operator diagnostics，判断是否进入 Step 9-10。

## 23. Phase3-C Returned Result: Positive but Confounded

[Fact] Phase3-C `PatchEncoderRegimeSegmentTargetOperator` minimal remote gate 已完成并同步：

- raw artifacts:
  `analysis/phase3_regime_segment_operator_20260624/raw/`;
- analyzer:
  `scripts/analyze_phase3_regime_segment_operator_gate.py`;
- report:
  `analysis/phase3_regime_segment_operator_20260624/phase3_regime_segment_operator_report.md`;
- summary:
  `analysis/phase3_regime_segment_operator_20260624/phase3_regime_segment_operator_summary.json`。

[Evidence] Numerical gate vs R.3:

| Quantity | Result |
| --- | ---: |
| MSE wins vs R.3 | `5/6` |
| mean relative MSE vs R.3 | `-0.39%` |
| observed aggregate-gap wins | `1/2` |
| observed H720 segment-gap wins | `2/3` |
| non-gap mean relative MSE vs R.3 | `-0.66%` |
| max prefix mismatch MSE | `4.925e-14` |
| mean operator abs scale | `0.079033` |
| mean operator abs shift | `0.019258` |

[Evidence] Main metrics:

| Dataset | Horizon | Candidate MSE | R.3 MSE | Rel MSE | Win |
| --- | ---: | ---: | ---: | ---: | --- |
| `ETTh2` | `96` | `0.301044` | `0.304796` | `-1.23%` | True |
| `ETTh2` | `720` | `0.410215` | `0.410473` | `-0.06%` | True |
| `ETTm1` | `96` | `0.298320` | `0.298685` | `-0.12%` | True |
| `ETTm1` | `720` | `0.414320` | `0.417293` | `-0.71%` | True |
| `Weather` | `96` | `0.148630` | `0.148026` | `+0.41%` | False |
| `Weather` | `720` | `0.318859` | `0.320847` | `-0.62%` | True |

[Evidence] Observed H720 segment gaps:

| Dataset | Segment | Candidate MSE | R.3 MSE | Rel MSE | Win |
| --- | --- | ---: | ---: | ---: | --- |
| `ETTh2` | `193-336` | `0.373090` | `0.369671` | `+0.92%` | False |
| `ETTh2` | `337-720` | `0.482149` | `0.484043` | `-0.39%` | True |
| `ETTm1` | `337-720` | `0.467490` | `0.471249` | `-0.80%` | True |

[Decision] 该结果是 positive but confounded：

1. [Strong Evidence] 数值上有收益，且 prefix consistency 仍保持数值零级别。
2. [Fact] 该 run 使用 `TARGET_HORIZONS=96,720`，而 R.3 baseline 使用
   `96,192,336,720`，因此混入了 horizon-set/objective-pressure confound。
3. [Fact] 该 run 在 prediction path 中使用 `window_index_norm`。
4. [Decision] `window_index_norm` 是 prediction-before signal，但不是稳健 causal/calendar
   variable；它按 split 内 index 归一化，可能学习 train/val/test split-position shortcut。

[11-Step Loop]

- `current_step`: Step 9-10。
- `problem`: Regime/segment-conditioned target operator 是否真实改善 R.3 hard windows/segments。
- `existence_evidence`: minimal gate 数值 positive，但存在 `window_index_norm` 与 horizon-set confounds。
- `idea`: history/window-position regime token 条件化 target-side operator。
- `theory_check`: 若收益依赖 split-position，则机制不可作为 paper-core；若 history-only 保留收益，
  机制才有可解释性。
- `design`: 当前 run 为 `window_index_norm + h96,h720`。
- `gate`: numerical pass，但 mechanism claim blocked。
- `artifacts`: `phase3_regime_segment_operator_*`。
- `decision`: 不进入 full horizon expansion claim；先做 controls，回到 Step 6-8。

[Next Controls]

1. `history_only_h96_h720`: `USE_WINDOW_POSITION=0`, `TARGET_HORIZONS=96,720`。
   目的：判断收益是否依赖 split-position shortcut。
2. 若 control 仍保留主要收益，再跑 `history_only_h96_h192_h336_h720`。
   目的：在与 R.3 相同 horizon set 下判断结构收益。
3. 可选 `window_h96_h192_h336_h720` 用于隔离 horizon-set confound。

[Implementation Update] 已修正 ablation support：`regime_segment_operator` 不再强制返回
`window_index_norm`，只有显式 `--use-window-position` 时才启用。History-only local smoke 已通过：

- `window_index_norm` feature stats: mean/std/mean_abs/max_abs all `0.0`;
- prefix mismatch MSE: `9.868e-15`。

[Remote Control Launch]

- control:
  `history_only_h96_h720`;
- launch note:
  `analysis/phase3_regime_segment_operator_20260624/history_only_control_launch_note.md`;
- remote commit:
  `1dcce5ce985647d49628e10317035b80307020a1`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator_history_only`;
- PID:
  `139925`;
- selected GPU:
  GPU1，launch 前为 `18 MiB` used；
- avoided GPUs:
  GPU0 `2649 MiB` used；GPU2 `10810 MiB` used；
- command:
  `GPU_IDS="1" RUN_NAME="PatchEncoderRegimeSegmentTargetOperatorHistoryOnly"
  USE_WINDOW_POSITION=0 TARGET_HORIZONS="96,720"
  bash scripts/remote/run_phase3_regime_segment_operator_gate.sh`;
- progress at 2026-06-24 11:15 CST:
  `ETTm1 1/100`, `Weather queued`, `ETTh2 queued`。
