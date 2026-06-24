# FATST Research Roadmap

## 当前锚点

[Fact] 本仓库是 `R_2026_FSA` 的 clean successor，但当前路线只使用本仓库内已经记录、
可复查的 evidence。旧仓库代码、配置、实验输出和未审计经验不作为默认依据。

[Decision] 当前论文 core innovation 固定为：

> Horizon Supervision Scheduling for Unified Multi-Horizon Forecasting

[Decision] 这里的 `Horizon Supervision Scheduling` 不是“训练时挑哪些 benchmark
horizon”。它的核心是 training/evaluation 解耦：

- evaluation horizons 只定义最终能力测试点，例如 `96,192,336,720`；
- training supervision 可以完全不按 horizon 组织；
- objective、schedule、curriculum 和 training strategy 可以基于 future positions、
  intervals、masks、components、frequency/basis、residual covariance 或其他
  horizon-free supervision units；
- 最终只要求训练策略能在固定 evaluation horizons 上产生更好的性能和更可信的机制叙事。

[Decision] 当前研究问题不是“继续修补 R.3 的坏点”，也不是“再做一个 decoder/operator”。
真正的问题是：

> 统一 multi-horizon forecasting 中，训练监督单位是否必须镜像 evaluation horizons；
> 如果不必须，什么样的 horizon-decoupled supervision strategy 能在完整 evaluation
> horizons 上带来性能提升，并形成有解释力的训练机制？

## 当前干净起点

[Decision] 当前从 `R.3` 重新起步，但 `R.3` 只承担三个角色：

- 证明 one-model target-set interface 可用；
- 作为 primary baseline；
- 作为 training/evaluation 解耦研究的最小 carrier。

[Decision] `R.3` 不是 paper-core 终点，也不再作为 future-aware/MoE 修补路线的 carrier。
它暴露了 mixed future supervision 的 active bottleneck，但下一步不应继续手调 horizon
weights。

[Decision] 新研究阶段命名为：

> Phase4-R: Horizon-Decoupled Supervision Strategy

主记录文件：

- `docs/experiments/phase4-horizon-supervision-scheduling-r3-reset.md`
- `docs/experiments/phase4-horizon-decoupled-protocol.md`

[Decision] 旧的 Phase4 `Component-Space Supervision` 和 `Component-Balanced Objective`
不作为当前 active route。它们保留为 candidate supervision basis 的历史证据，只有在
Phase4-R 的 11-step 判断允许时才重新激活。

## 11-Step 原则

本项目所有长研究阶段必须按下面 11-step loop 推进：

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

[Decision] 每个阶段记录必须包含：

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

## Phase4-R 当前 11-Step 状态

### 第 1 步：调研分析

[Strong Evidence] `TransDF` 和 `QDF` 都支持 objective-side 重新审视。它们的共同信号不是
“换一个 horizon 权重”，而是 future label sequence 的训练监督单位可能不应是原始 time step
或 benchmark horizon。

[Strong Evidence] `ElasTST` 的 varied-horizon / horizon-invariance 约束说明 evaluation
horizon 是能力测试点。模型请求更长 horizon 时不应任意改变已有 prefix prediction，但这不要求
训练过程按这些 evaluation horizons 组织监督。

[Strong Evidence] `SRP++` 支持 future-step heterogeneity 真实存在。当前只把它作为问题背景，
不直接启动 step-specific adapter 或 MoE。

[Strong Evidence] 本仓库 R.3、reduced horizon set、full horizon set 和 component audits
共同说明：supervision organization 会改变结果，但现有证据不足以把任何一个 horizon subset
或 component objective 直接写成 paper-core。

### 第 2 步：提出问题

[Decision] 当前问题定义为：

> Evaluation horizons define how we test a unified forecaster, but they need
> not define how we train it. Can horizon-decoupled supervision strategies
> produce better full-horizon forecasts and a credible optimization narrative?

中文表述：

> 统一 multi-horizon forecasting 中，`96,192,336,720` 只是评估点。训练时可以完全不管这些
> horizon，转而设计 future supervision units、objective pressure、sampling mask 或 curriculum。
> 我们要判断的是：这种 training/evaluation 解耦是否能带来性能提升和可讲清楚的训练机制。

### 第 3 步：评估问题是否真实且值得研究

[Strong Evidence] 问题真实存在：

- R.3 证明 objective pressure 会影响 unified multi-horizon 结果；
- reduced/full horizon-set control 说明监督组成会改变 H96/H720 的表现；
- label-basis audit 说明 future labels 有低秩和相关结构，不由 `96,192,336,720` 这些离散
  evaluation horizons 单独定义；
- residual projection audit 说明 component structure 存在，但 top-only objective 不足以直接
  解释 known gaps。

[Decision] 问题值得研究，因为它把贡献从“修修补补预测性能”提升为 training strategy：

- evaluation set 与 training supervision basis 解耦；
- 不改变 inference interface；
- 可以把 performance gain 连接到 optimization pressure、task redundancy、label dependency
  或 curriculum path；
- future-aware / MoE 只能在该主线成立后作为二级机制。

### 第 4 步：提出 idea

[Decision] 核心 idea 重新表述为：

> Horizon-decoupled supervision scheduling。

训练策略调度的不是 evaluation horizons，而是 future supervision units。候选 family：

| Family | Training unit | 当前角色 |
| --- | --- | --- |
| `full_time_mse` | full future sequence time-domain MSE | negative control |
| `R.3_prefix_risk` | historical horizon-weighted objective | primary baseline |
| `random_future_mask` | future positions / blocks 的 stochastic mask | 首轮候选 |
| `interval_supervision` | 随机或结构化 future intervals | 首轮候选 |
| `component_basis_supervision` | train-label basis / decorrelated components | 首轮候选 |
| `frequency_or_smoothness_basis` | low/high frequency basis 或 trend/detail split | 扩展候选 |
| `curriculum_over_units` | 从 coarse units 到 dense time-domain units | 首轮候选 |
| `covariance_pressure` | residual / label covariance induced objective | 扩展候选 |
| `future-aware/MoE` | 利用已证明的 heterogeneity | 暂停 |

[Decision] `96,192,336,720` 不再出现在 training schedule 的定义中。它们只用于 evaluation、
comparison、diagnostic reporting 和 paper table。

### 第 5 步：理论可行性

[Hypothesis] 如果 future label sequence 存在强相关、低秩结构或局部/全局尺度差异，那么按
evaluation horizons 同步训练可能不是最优优化路径。Horizon-decoupled supervision 可以：

1. 减少 redundant step-wise tasks；
2. 避免 benchmark horizon 切分强行定义训练任务；
3. 先学习粗粒度 future structure，再补充局部 detail；
4. 在完整 evaluation horizons 上保留 prefix consistency。

[Self-Critique] 该假设可能失败。如果训练策略只带来随机正则化，或只改善 aggregate MSE 而没有
task-redundancy、loss trajectory、segment error 或 component diagnostics 支撑，就不能作为
paper-core。

### 第 6 步：最小设计

[Decision] 第一轮设计只改变 training objective / schedule / supervision unit，不改 backbone，
不引入 future-aware 或 MoE。

最小实验组：

| ID | Training strategy | Purpose |
| --- | --- | --- |
| `D0_full_time_mse` | full future sequence time-domain MSE | negative control |
| `D1_r3_prefix_risk` | 当前 R.3 | primary baseline |
| `D2_random_future_mask` | 每个 batch 随机监督 future positions 或 blocks | 测试 task redundancy / stochastic unit |
| `D3_interval_supervision` | 每个 batch 监督随机 future intervals | 测试 interval-level training basis |
| `D4_component_basis_top` | 监督 train-label dominant components | 测试 low-rank coarse structure |
| `D5_component_basis_balanced` | component groups balanced pressure | 测试 trend/detail 平衡 |
| `D6_curriculum_units` | coarse component/interval 到 dense time-domain 的 curriculum | 测试 optimization path |

首轮诊断：

- evaluation horizons 上的 MSE/MAE；
- dataset-level mean relative MSE vs R.3；
- H96/H720 segment-level error；
- prefix consistency / prediction mismatch；
- training loss trajectory；
- supervision unit trace；
- component/interval/frequency 维度的 residual attribution。

首轮 pass gate：

1. mean relative MSE vs R.3 `< 0`，或在 `+0.2%` 内但明显修复 known gaps；
2. MSE wins vs R.3 至少 `7/12`；
3. 任一 dataset mean 不比 R.3 劣化超过 `+0.5%`；
4. H96 和 H720 weak regions 不系统性恶化；
5. prefix mismatch 保持 near numerical zero；
6. diagnostic 能说明收益来自 horizon-decoupled supervision strategy，而不是偶然调参。

回退规则：

- 若 `D2-D6` 全部输 R.3，回退到 Step 2-3：training/evaluation 解耦可能不是当前 carrier 的
  主要瓶颈。
- 若只有 `D4/D5` 有效，回退到 Step 4：主线应收窄为 transformed label / component supervision。
- 若只有 `D2/D3` 有效，回退到 Step 4：主线应收窄为 stochastic future-unit scheduling。
- 若只有 `D6` 有效，继续 Step 9-10 补 curriculum path 诊断，不直接加 architecture。
- 若性能通过但诊断不足，停在 Step 9-10，补 mechanism diagnostics。

## 当前研究任务

### R4.1：Horizon-Decoupled 协议

`current_step`: Step 6 complete。

[Decision] 已完成：

- `docs/experiments/phase4-horizon-decoupled-protocol.md`

结论：R4.1 通过，允许进入 R4.2 本地实现；仍不允许改 architecture。

### R4.2：本地实现与 smoke

`current_step`: Step 7 complete。

[Decision] 已完成 horizon-decoupled supervision 的最小本地实现：

- `random_future_mask`;
- `interval_supervision`;
- `component_basis_top`;
- `component_basis_balanced`;
- `curriculum_units`;
- `supervision_trace.csv`;
- `training_evaluation_decoupled` effective config。

[Verification] `py_compile`、remote shell syntax check、full-time smoke、R.3 smoke、
random mask smoke、component top smoke、component balanced smoke、interval smoke、
curriculum smoke 均已通过。

### R4.3：529_Lab-3090 远程 gate

`current_step`: Step 8 complete。

[Fact] 已完成 7 strategies x 3 datasets x 4 evaluation horizons 的 remote gate：

- remote output root:
  `/home/yingch/exp_outputs/r-2026-fatst`。
- local synced analysis root:
  `analysis/phase4_horizon_decoupled_gate_20260624`。
- raw sync artifacts 位于 `analysis/phase4_horizon_decoupled_gate_20260624/raw`；
  该目录按 `.gitignore` 保持不跟踪。
- 汇总表和决策报告由
  `scripts/analyze_phase4_horizon_decoupled_gate.py` 生成。

### R4.4：决策报告

`current_step`: Step 9-11 complete。

[Fact] 决策报告：

- `analysis/phase4_horizon_decoupled_gate_20260624/phase4_horizon_decoupled_decision_report.md`

[Fact] 相对 primary baseline `D1_r3_prefix_risk`：

| Strategy | MSE wins | Mean relative MSE | Worst dataset mean degradation |
| --- | ---: | ---: | ---: |
| `D0_full_time_mse` | 0/12 | +5.12% | +5.46% |
| `D2_random_future_mask` | 1/12 | +3.51% | +3.59% |
| `D3_interval_supervision` | 2/12 | +4.12% | +7.12% |
| `D4_component_basis_top` | 0/12 | +4.37% | +5.50% |
| `D5_component_basis_balanced` | 0/12 | +3.91% | +5.32% |
| `D6_curriculum_units` | 0/12 | +4.37% | +5.50% |

[Strong Evidence] 当前 `D2-D6` 静态 horizon-decoupled replacement 全部不通过
paper-core gate。最好的非 R.3 candidate 是 `D2_random_future_mask`，但仍然是
`+3.51%` mean relative MSE，只有 `1/12` MSE wins。

[Strong Evidence] `D3_interval_supervision` 在 `ETTh2` 有局部改进（dataset mean
relative MSE `-0.36%`，2/4 wins），但在 `ETTm1` 和 `Weather` 明显退化。因此它不是
可泛化 paper-core，而是提示 interval unit 可能需要 condition。

[Fact] prefix mismatch 保持在 numerical-zero 量级，说明失败不是 unified inference 或
evaluation interface 被破坏导致。

[Decision] Phase4-R 当前版本失败的是“静态、全局、替换式的 horizon-decoupled
supervision strategy”，不是 HSS 研究问题本身。回退点是 Step 4/6：保留
training/evaluation 解耦问题，重做核心 idea 和最小设计。

## 下一阶段：Phase4-S

`current_step`: Step 4-6 draft。

[Decision] 下一步主线命名为：

> Phase4-S: State/Difficulty-Conditioned Supervision Scheduling

主记录文件：

- `docs/experiments/phase4-s-conditioned-supervision-scheduling.md`

诊断产物：

- `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_diagnostic_report.md`
- `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_strategy_summary.csv`
- `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_residual_bucket_summary.csv`
- `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_future_region_summary.csv`

[Decision] 不继续做 mask ratio、interval length、component rank 的宽 sweep；也不把
future-aware 或 MoE 叠在失败策略上。

新 hypothesis：

> Horizon-free supervision units 可以成立，但 unit pressure 不能是全局静态或随机。
> 它应由 train-side difficulty、future-label novelty、residual/error process proxy 或
> running-loss state 条件化。

最小候选：

| ID | Candidate | Role |
| --- | --- | --- |
| `S2_r3_plus_sparse_unit_aux` | 保留 R.3 base loss，加小权重 horizon-free sparse auxiliary unit | 第一优先级；测试 HSS 作为辅助 schedule，而非替换 objective |
| `S1_difficulty_conditioned_interval` | 训练 `720` future sequence，但 interval sampling probability 由 label novelty / running loss bucket 决定 | 第二优先级；测试 conditioned unit pressure |
| `S3_error_process_reweighting` | 用 validation residual 或 train-side proxy 调整 future unit pressure | 测试 error-process-aware supervision |

[Strong Evidence] post-hoc diagnostic 已完成，支持 Phase4-S 作为 hypothesis 继续推进：

- `D2_random_future_mask` 的 `4/30` segment wins 全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 的 `5/30` segment wins 全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 在 late region 有 `2/3` wins，mean relative MSE 为 `-0.46%`；
- early region 上所有候选均为 `0/12` wins，说明全局静态 pressure 会损伤 easy/early regions。

[Decision] 这支持 conditioned schedule 的问题定义，但不等于 implementation gate 已通过。
下一步必须先定义不依赖 evaluation horizon identity 的 train-side condition。

实现前必须完成：

1. 为 `S2_r3_plus_sparse_unit_aux` 定义 auxiliary unit 和 $\lambda$ gate；
2. 为 `S1_difficulty_conditioned_interval` 定义 train-side difficulty proxy；
3. 确认 proxy 不直接使用 `96,192,336,720` 作为 training schedule；
4. 写出 local smoke protocol，再允许代码实现。

## 历史证据索引

[Decision] 以下历史记录保留为 evidence index，不再作为当前 active route：

| Historical Route | 当前用途 |
| --- | --- |
| Phase0 canonical base | 证明 `PatchEncoderFixedHead` 是内部 base |
| Phase1 target-set decoder / R.3 | 当前 carrier 与 primary baseline |
| Phase2 future-state / covariance / QDF variants | 证明继续修补 objective/operator 不足以形成主线 |
| Phase3 regime/segment operator | 证明 supervision composition 是 material factor |
| Phase4 component audits | 作为 supervision basis 候选，不直接作为 active route |

## 当前禁止事项

- 不把 Phase4 写成“训练时挑哪些 horizon”。
- 不把 Phase4 写成 component objective 或 residual repair。
- 不把 R.3 包装成最终贡献。
- 不在 horizon-decoupled supervision evidence 成立前启动 future-aware 或 MoE。
- 不把 reduced horizon set 的 positive signal 写成 operator success。
- 不只用 aggregate MSE/MAE 判定通过。
- 不默认使用旧 `R_2026_FSA` 证据，除非用户批准具体来源和用途。
