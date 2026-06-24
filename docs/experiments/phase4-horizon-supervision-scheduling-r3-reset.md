# Phase4-R：从 R.3 出发的 Horizon-Decoupled Supervision Strategy

`current_step`: 11-step loop 的 Step 7 已完成；R4.2 local implementation 和 smoke 已通过，
下一步进入 Step 8 remote gate。

## 状态判定

[Decision] 当前研究目标重新锚定为：

> Horizon Supervision Scheduling for Unified Multi-Horizon Forecasting

[Decision] 这里的 `Horizon Supervision Scheduling` 不是训练时选择哪些 horizon。核心是：
evaluation horizons 与 training supervision units 解耦。

[Fact] evaluation 固定覆盖 `96,192,336,720`。这些 horizons 只用于测试、报告和诊断。

[Decision] training 可以完全不引用这些 horizons。训练策略可以基于：

- future positions；
- future intervals；
- stochastic masks；
- train-label components；
- frequency / trend-detail basis；
- residual covariance；
- coarse-to-dense curriculum；
- 其他不由 benchmark horizons 定义的 supervision units。

[Decision] `R.3` 是 target-set interface 的最小有效 carrier 和 primary baseline，不是
paper-core 终点。后续不再把问题表述为“修补 R.3 的几个坏点”，而是系统研究
horizon-decoupled supervision strategy 是否能带来性能提升和论文叙事。

[Decision] 当前不推进 future-aware、MoE、component-balanced objective 或 residual repair。
它们只有在本阶段 Step 9-10 证明主线成立后，才可能作为二级机制进入。

## Phase4 研究指南

### 11-step 原则

本阶段必须严格按 11-step loop 推进：

1. 调研分析。
2. 提出待解决问题。
3. 评估问题是否值得研究，以及问题是否真实存在。
4. 提出 idea。
5. 评估 idea 的理论可行性。
6. 设计方案。
7. 实现方案。
8. 远程训练。
9. 评估结果。
10. 判断是否通过：同时看性能收益与论文故事是否成立。
11. 若不通过，评估应回退至哪一步，然后继续循环。

本阶段执行边界：

- Step 1-3 必须先证明问题真实，不能因为已有代码方便就进入实现；
- Step 4-6 必须形成可证伪 hypothesis、数学数据流、最小实验协议和 rollback；
- Step 7-8 只能实现 Step 6 明确允许的 supervision strategy；
- Step 9-10 必须同时判断 performance evidence 和 paper narrative；
- Step 11 必须明确回退到问题定义、idea、方案设计或实现细节；
- 任何候选未通过前，不叠加 future-aware、MoE 或其他复杂 architecture。

### 阶段记录字段

每个 Phase4 子阶段必须记录：

| Field | Required Content |
| --- | --- |
| `current_step` | 当前处于 11-step loop 的哪一步 |
| `problem` | 待解决问题，以及它为什么不是已有实验已经否定的问题 |
| `existence_evidence` | 问题真实存在的 artifact、公式推导或可复查现象 |
| `idea` | 核心 idea，不超过一个主机制 |
| `theory_check` | 数据流、数学约束、可能成立的原因和反例 |
| `design` | model/training/evaluation 的最小方案 |
| `gate` | pass/fail/rollback 条件 |
| `artifacts` | 代码版本、输出路径、报告和表格 |
| `decision` | 是否通过；若不通过，回退到哪一步 |

## 第 1 步：调研分析

### 文献证据

[Strong Evidence] `TransDF` 指出 temporal MSE 忽略 label autocorrelation，且长 forecast
horizon 会产生过多 step-wise tasks，增加 multi-task optimization difficulty。对本项目的启发
不是直接复刻 TransDF，而是训练监督单位可以从 benchmark horizons 转向 transformed labels
或 components。

[Strong Evidence] `QDF` 指出 standard MSE 等价于 identity weighting，忽略 future-step
dependency 和 heterogeneous task weights。它支持 objective pressure 需要被设计，但不要求这种
设计按 horizon 划分。

[Strong Evidence] `ElasTST` 的 horizon-invariance 约束说明 unified forecaster 在请求更长
horizon 时不应任意改变已有 prefix predictions。它支持 evaluation horizon 是能力测试点，而不是
训练监督单位。

[Strong Evidence] `SRP++` 指出 future-step heterogeneity 真实存在。当前只把它作为
heterogeneity 背景，不直接采用 segment adapter 或 MoE。

### 本项目证据

[Fact] R.3 / `PatchEncoderPrefixRiskWeighted` 证明 target-set carrier 可用，并且 objective
pressure 会改变 multi-horizon 结果。

[Strong Evidence] reduced/full horizon-set control 说明 supervision composition 是 material
factor，但这不等于应该继续做 horizon subset 搜索。

[Strong Evidence] label-basis audit 显示 `pred_len=720` 的 future label sequence 有低 effective
rank 和强 step correlation，说明未来序列结构不是由 `96,192,336,720` 离散 horizon 单独定义。

[Strong Evidence] residual projection audit 显示 component structure 存在，但 known gaps 并不
集中在 dominant top components，说明不能简单把当前主线写成 top-component loss。

## 第 2 步：待解决问题

问题定义：

> Evaluation horizons define how we test a unified forecaster, not necessarily
> how we train it. Can horizon-decoupled supervision strategies improve full
> multi-horizon evaluation and produce a credible optimization narrative?

中文表述：

> `96,192,336,720` 只是评估点。训练阶段可以完全不管这些 horizons，转而设计 future
> supervision units、objective pressure、sampling mask 或 curriculum。我们要研究的是：
> training/evaluation 解耦是否能形成有叙事潜力和性能提升的 training strategy。

## 第 3 步：问题是否真实且值得研究

[Strong Evidence] 问题真实存在：

- R.3 说明 objective pressure 是 active bottleneck；
- reduced/full horizon-set control 说明 supervision composition 会显著改变结果；
- label-basis audit 说明 future labels 有 horizon-free 的结构；
- QDF/TransDF/ElasTST 提供 objective、label transform 和 horizon-invariance 三条外部证据。

[Decision] 问题值得研究，因为它具备三类价值：

1. training strategy 贡献：不改变 inference interface，只改变 supervision organization；
2. mechanism narrative 贡献：可解释为 task redundancy、label dependency、optimization path
   或 objective pressure 的改变；
3. performance 贡献：最终仍以 full evaluation horizons 上的指标判断。

[Risk] 如果策略只带来随机正则化，或只改善 aggregate MSE 而没有机制诊断支撑，就不能作为
paper-core。

## 第 4 步：核心 idea

核心 idea：

> Treat future supervision units as schedulable training signals, independent
> of evaluation horizons.

训练时显式控制：

- supervision unit：time positions、intervals、masks、components、frequency basis 等；
- objective pressure：不同 unit 或 group 的 loss pressure；
- schedule path：从 coarse 到 dense、从 low-rank 到 full time-domain、从 sparse mask 到完整监督；
- traceability：每个 batch 或 epoch 记录实际 supervision units。

推理与评估保持不变：

- one model；
- target-set / unified interface；
- evaluation horizons 固定为 `96,192,336,720`；
- prefix consistency 是硬约束。

## 第 5 步：理论可行性

设 evaluation horizon set 为：

$$
\mathcal{H}_{eval}=\{96,192,336,720\}.
$$

它只出现在 evaluation：

$$
\operatorname{Eval}(f_\theta; \mathcal{H}_{eval}).
$$

训练过程定义一组与 horizon 无关的 future supervision units：

$$
\mathcal{U}=\{u_1,u_2,\dots,u_K\}.
$$

其中 $u_k$ 可以是 future position mask、interval、component、frequency band 或 covariance
group。训练第 $t$ 步选择：

$$
S_t\subseteq\mathcal{U},\qquad a_t(u)\ge 0.
$$

训练目标：

$$
\mathcal{L}_t(\theta)
=
\sum_{u\in S_t} a_t(u)\mathcal{L}_u(\theta).
$$

[Hypothesis] 如果 future label sequence 存在相关性、低秩结构或多尺度结构，那么
horizon-free units 可能比 benchmark horizons 更适合作为训练监督单位。

[Self-Critique] 该假设可能失败。如果 horizon-free schedule 只改善某个 dataset 或只改变
regularization strength，就不能声称发现了 training strategy。必须用 trace、loss trajectory、
segment error、component residual 或 covariance diagnostic 支撑。

## 第 6 步：最小设计

### Carrier 与基线

保持 R.3 carrier：

- target-set interface；
- evaluation 固定为 `96,192,336,720`；
- same backbone and decoding path；
- no future-aware/MoE change。

主要对照：

| ID | 角色 |
| --- | --- |
| `D0_full_time_mse` | negative control |
| `D1_r3_prefix_risk` | primary baseline |

### 候选 training strategy

| ID | Training strategy | Supervision unit | Hypothesis |
| --- | --- | --- | --- |
| `D2_random_future_mask` | 随机监督 future positions 或 blocks | time mask | 降低 redundant step-wise tasks |
| `D3_interval_supervision` | 每个 batch 监督随机或结构化 intervals | future interval | interval-level units 比 horizon units 更自然 |
| `D4_component_basis_top` | 监督 train-label dominant components | component | 低秩 coarse future structure 先学更稳定 |
| `D5_component_basis_balanced` | component groups balanced pressure | component group | 避免 top-only 损伤 detail |
| `D6_curriculum_units` | coarse component/interval 到 dense time-domain | curriculum path | optimization path 比静态 objective 更关键 |

### 实现要求

- strategy choice 写入 effective config；
- training log 记录 supervision unit trace；
- evaluation protocol 不变；
- stochastic strategy 使用 explicit seed；
- prefix consistency 输出保留；
- 不引入 unrelated architecture change。

### 指标与诊断

主要指标：

- per-dataset / per-horizon MSE and MAE；
- mean relative MSE vs R.3；
- MSE wins vs R.3；
- dataset mean degradation vs R.3。

机制诊断：

- H96/H720 segment-level MSE；
- prefix mismatch / prefix consistency；
- training loss trajectory；
- supervision unit trace；
- component / interval / frequency residual attribution；
- covariance 或 task-redundancy proxy，若实现成本可控。

### 通过条件

候选策略只有同时满足以下条件，才允许进入 paper-core consideration：

1. mean relative MSE vs R.3 `< 0`，或在 `+0.2%` 内但明显修复 known gaps；
2. MSE wins vs R.3 `>= 7/12`；
3. no dataset mean degrades more than `+0.5%` vs R.3；
4. H96/H720 weak regions 不系统性恶化；
5. prefix mismatch remains near numerical zero；
6. diagnostics show a horizon-decoupled supervision mechanism, not only metric fluctuation。

### 回退规则

- 若 `D2-D6` 全部输 R.3，回退到 Step 2-3：training/evaluation 解耦可能不是当前 carrier 的
  主要瓶颈。
- 若只有 `D4/D5` 有效，回退到 Step 4：主线收窄为 transformed label / component supervision。
- 若只有 `D2/D3` 有效，回退到 Step 4：主线收窄为 stochastic future-unit scheduling。
- 若只有 `D6` 有效，停在 Step 9-10，补 curriculum path 诊断。
- 若性能通过但诊断不足，停在 Step 9-10，不进入 architecture。
- 只有性能与机制诊断同时通过，才进入扩展实验或二级机制。

## 当前任务拆分

### R4.1：Horizon-Decoupled 协议

`current_step`: Step 6 complete。

[Decision] R4.1 protocol 已完成：

- `docs/experiments/phase4-horizon-decoupled-protocol.md`

该 protocol 明确了 supervision unit API、mask / interval / component / curriculum config、
trace 格式、最小触达文件、本地 smoke 和远程 gate。下一步允许进入 R4.2。

### R4.2：本地实现

`current_step`: Step 7 complete。

[Decision] 已完成：

- `baselines/patch_encoder_target_set_decoder/train.py`
- `docs/code-explanation/phase4-horizon-decoupled-supervision.md`
- `scripts/remote/run_phase4_horizon_decoupled_gate.sh`
- `scripts/remote/check_phase4_horizon_decoupled_progress.sh`
- `scripts/sync_phase4_horizon_decoupled_results.sh`

[Verification] 已完成：

- `python -m py_compile baselines/patch_encoder_target_set_decoder/train.py`
- `bash -n scripts/remote/run_phase4_horizon_decoupled_gate.sh scripts/remote/check_phase4_horizon_decoupled_progress.sh scripts/sync_phase4_horizon_decoupled_results.sh`
- `random_future_mask` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4RandomFutureMask/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `full_time_mse` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4FullTimeMSE/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `r3_prefix_risk` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4R3PrefixRisk/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `component_basis_top` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4ComponentTop/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `component_basis_balanced` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4ComponentBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `interval_supervision` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4IntervalSupervision/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- `curriculum_units` smoke:
  `artifacts/runs/smoke_phase4_horizon_decoupled/SmokePhase4CurriculumUnits/ETTh2/mixed_h96_h192_h336_h720/seed2021`

[Fact] smoke artifacts 显示：

- `training_evaluation_decoupled=true`;
- `train_horizons_effective=[720]`;
- `evaluation_target_horizons=[96,192,336,720]`;
- `supervision_trace.csv` 存在并记录 unit type、active steps、loss；
- `curriculum_units` trace 覆盖 `coarse`、`mixed`、`dense`；
- component loss 已按 target component energy 归一化，避免 raw component scale 主导训练。

### R4.3：远程 gate

`current_step`: Step 8，下一步进入。

交付物：

- commit and push focused code state；
- inspect `nvidia-smi` on `529_Lab-3090` before launch；
- evaluate `ETTh2`, `ETTm1`, `Weather` on `96,192,336,720`；
- write outputs under `/home/yingch/exp_outputs/r-2026-fatst`；
- record GPU, command, conda env, commit hash, output path。

退出条件：

- all candidates produce comparable metrics and traces。

### R4.4：决策报告

`current_step`: Step 9-10，pending R4.3。

交付物：

- horizon-level evaluation table；
- segment-level H96/H720 analysis；
- supervision trace and loss trajectory interpretation；
- explicit pass/fail/rollback decision。

退出条件：

- decision states whether HSS remains paper-core and which step comes next。

## 当前结论

[Decision] Phase4-R 允许进入 R4.1 protocol/API definition。当前不允许启动远程训练，
也不允许加入 future-aware 或 MoE。
