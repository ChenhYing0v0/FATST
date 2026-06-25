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

`current_step`: Step 9-11 complete；S1 small remote gate 已完成。当前结论是
`conditioned_future_unit_scheduling` 不通过 paper-core gate，回退到 Step 6 重做
condition 可观测性与 schedule 设计。

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
| `S1_conditioned_future_unit_scheduling` | `pred_len=720` full future dense anchor + train-side conditioned sparse unit pressure | 第一优先级；独立 HSS training strategy，不 repair R.3 |
| `S2_difficulty_conditioned_interval` | 训练 `720` future sequence，但 interval sampling probability 由 label novelty / running loss bucket 决定 | 第二优先级；测试 conditioned unit pressure |
| `S3_r3_plus_aux_control` | R.3 base loss + 小权重 sparse auxiliary unit | control only；检验 auxiliary 与 R.3 是否冲突，不作为 paper-core |
| `S4_error_process_reweighting` | 用 train-side residual proxy 调整 future unit pressure | 扩展候选；测试 error-process-aware supervision |

[Strong Evidence] post-hoc diagnostic 已完成，支持 Phase4-S 作为 hypothesis 继续推进：

- `D2_random_future_mask` 的 `4/30` segment wins 全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 的 `5/30` segment wins 全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 在 late region 有 `2/3` wins，mean relative MSE 为 `-0.46%`；
- early region 上所有候选均为 `0/12` wins，说明全局静态 pressure 会损伤 easy/early regions。

[Decision] 这支持 conditioned schedule 的问题定义。`S1_conditioned_future_unit_scheduling`
本地实现、remote small gate 与结果分析已完成，但它仍不能作为 paper-core。R.3 只作为
primary baseline、carrier sanity check 和 control 参照；修好 R.3 不是本文 core narrative。

当前 S1 实现边界：

1. `pred_len=720` full future dense anchor；
2. train-side `label_novelty` condition 选择 sparse blocks；
3. auxiliary weight 默认 `0.1`；
4. 不使用 R.3 `prefix_risk`，不采样 `96,192,336,720` 作为 training schedule；
5. local smoke 必须检查 `supervision_trace.csv` 和 prefix consistency。

[Verification] 本地 smoke 已通过：

- artifact:
  `artifacts/runs/smoke_phase4_s_conditioned/SmokePhase4SCFUS/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- `training_evaluation_decoupled=true`;
- `train_horizons_effective=[720]`;
- `step_loss_weighting=uniform`;
- `unit_type=conditioned_sparse`;
- prefix mismatch 为 numerical-zero 量级。

[Fact] S1 small remote gate 已完成：

- runner: `scripts/remote/run_phase4_s_cfus_gate.sh`;
- remote output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_s_cfus_gate`;
- local analysis root:
  `analysis/phase4_s_cfus_gate_20260624`;
- analysis script:
  `scripts/analyze_phase4_s_cfus_gate.py`;
- decision report:
  `analysis/phase4_s_cfus_gate_20260624/phase4_s_cfus_gate_decision_report.md`;
- datasets: `ETTh2`, `Weather`;
- strategies: `conditioned_future_unit_scheduling`, `full_time_mse`, `r3_prefix_risk`;
- 不进入 full matrix。

[Fact] 主要结果：

| Comparison | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| CFUS vs `D0_full_time_mse` | 8 | 6 | 8 | `-2.74%` |
| CFUS vs `D1_r3_prefix_risk` | 8 | 3 | 3 | `+2.22%` |

[Fact] dataset split：

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 3 | `-0.35%` |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | `+4.78%` |

[Decision] S1 的有效证据只到“conditioned sparse auxiliary 能改善 plain full-time dense
anchor”，不足以证明它是稳定的 HSS training strategy。它在 Weather 上相对 R.3 全面退化，
触发 no Weather collapse / close R.3 gap gate 失败。

[Diagnostic Gap] 当前 trace 没有记录 selected block indices / block ranges / per-block
condition scores，因此不能排除 `label_novelty` 退化为固定 late weighting proxy。

[Decision] 回退到 Step 6，不继续 sweep 当前 `label_novelty + top_ratio=0.25 + aux=0.1`。
下一步只做 trace-first diagnosis 和 CFUS-v2 最小重设计：

1. trace 记录 selected block indices、block ranges 和 per-block condition scores；
2. offline diagnostic 判断 `label_novelty` 是否长期偏向 late blocks；
3. 若偏向 late blocks，改为 `novelty within future-region groups` 或
   `balanced condition buckets`；
4. CFUS-v2 local trace 证明不是固定 late weighting 后，再启动新的 small gate。

### Phase4-S2：Predictability-Conditioned Scheduling

`current_step`: Step 9-11 complete；S2 small remote gate 已完成。当前结论是
`predictability_downweight` 不通过 paper-core gate，回退到 Step 5/6 重新设计
predictability proxy 和 shielding 机制。

[Question] S1 把 high-novelty / hard blocks 统一视为需要加压的对象。但当前实验提出了
更强的反问题：

> 难预测 block 是否应该被加压，还是应该被降权/隔离，避免污染模型对可预测部分的学习？

[Fact] SRP++ 不直接讨论 hard-block avoidance，但它支持 future-step/segment interference
这一前提：不同 forecast steps 需要 step-specific representations；冻结 foundation model 和
segment-specific adapters 可以避免 long-range adaptation 破坏 short-term prediction。对当前
HSS 主线的启发是：schedule 不一定只强调 hard units，也可以隔离 low-predictability units。

[Decision] 新的核心 idea 是：

> Predictability-Conditioned Supervision Scheduling

即 train-side condition 不只判断 hard/easy，而要判断 hard block 是否可预测：

| Block type | Interpretation | Training action |
| --- | --- | --- |
| `learnable-hard` | high novelty but smooth / structured | auxiliary emphasis |
| `noisy-hard` / `low-predictability` | high novelty and high local variation / weak naive predictability | downweight or isolate |
| `predictable-easy` | low novelty / stable | dense anchor / prefix protection |

[Fact] 已完成 train-only offline diagnostic：

- script: `scripts/analyze_phase4_predictability_diagnostic.py`;
- analysis root: `analysis/phase4_predictability_diagnostic_20260624`;
- report:
  `analysis/phase4_predictability_diagnostic_20260624/phase4_predictability_diagnostic_report.md`;
- datasets: `ETTh2`, `Weather`;
- split: train only；
- selected blocks: 复现当前 S1 的 `label_novelty` top-4 blocks。

[Fact] 诊断结果：

| Dataset | Selected best-naive ratio | Selected local-variation ratio | Late-block selected share | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `ETTh2` | `2.51x` | `1.20x` | `0.485` | high novelty 更像 learnable smooth shift |
| `Weather` | `2.83x` | `8.45x` | `0.436` | high novelty 明显混入 high-variation / noisy-hard |

[Strong Evidence] 这解释了 S1 small gate：ETTh2 上 selected blocks 更像 learnable-hard，因此
CFUS 能接近 R.3；Weather 上 selected blocks 更像 noisy-hard，因此 hard-block emphasis
污染 shared dense learning，导致相对 R.3 全面退化。

[Decision] 不做 `label_novelty` 参数 sweep，也不只做 region-balanced top-k。Phase4-S2
已实现并完成 small remote gate 的最小 `predictability_downweight`：

$$
\mathcal{L}
=
\sum_{u\in\mathcal{U}} w_{\text{pred}}(u)\mathcal{L}_u
+
\lambda
\sum_{u\in\mathcal{U}_{learnable-hard}} a(u)\mathcal{L}_u.
$$

其中 low-predictability blocks 保留 floor weight，避免完全丢弃；但它们不再获得额外
auxiliary pressure，也不应主导 shared representation 的梯度。

[Gate] S2 small gate 必须同时满足：

1. vs `full_time_mse` 保留收益；
2. Weather vs R.3 不再 `0/4` collapse；
3. early/predictable blocks 不被牺牲；
4. trace 记录 per-block predictability score、bucket 和实际 loss weight；
5. evaluation horizons 仍只用于测试，不进入 training schedule。

[Verification] 本地实现与 smoke 已通过：

- training strategy: `predictability_downweight`;
- code explanation:
  `docs/code-explanation/phase4-s-predictability-scheduling.md`;
- remote runner:
  `scripts/remote/run_phase4_s_predictability_gate.sh`;
- smoke artifact:
  `artifacts/runs/smoke_phase4_s_predictability/SmokePhase4SPredictabilityDownweight/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- `training_evaluation_decoupled=true`;
- `train_horizons_effective=[720]`;
- `unit_type=predictability_downweight`;
- trace 记录 `predictability_learnable_blocks`、`predictability_noisy_blocks`、
  `predictability_mean_weight` 和 `predictability_floor_weight`;
- prefix mismatch 为 numerical-zero 量级。

[Fact] S2 small remote gate 已完成：

- runner: `scripts/remote/run_phase4_s_predictability_gate.sh`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_s_predictability_gate`;
- local analysis root:
  `analysis/phase4_s_predictability_gate_20260625`;
- analysis script:
  `scripts/analyze_phase4_s_predictability_gate.py`;
- decision report:
  `analysis/phase4_s_predictability_gate_20260625/phase4_s_predictability_gate_decision_report.md`;
- datasets: `ETTh2`, `Weather`;
- strategies: `predictability_downweight`, `full_time_mse`, `r3_prefix_risk`;
- 不进入 full matrix。

[Fact] 主要结果：

| Comparison | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| S2 vs `D0_full_time_mse` | 8 | 4 | 5 | `-2.61%` |
| S2 vs `D1_r3_prefix_risk` | 8 | 3 | 3 | `+2.35%` |
| S2 vs S1-CFUS | 8 | 2 | 3 | `+0.13%` |

[Fact] dataset split：

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 3 | `-0.34%` |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | `+5.05%` |

[Strong Evidence] trace 证明 S2 的 noisy/learnable split 确实发生：

- ETTh2: mean learnable blocks `3.10`，mean noisy blocks `0.91`;
- Weather: mean learnable blocks `2.08`，mean noisy blocks `2.11`。

[Counter-Evidence] 尽管 split 发生，Weather 没有被修复。S2 相对 R.3 在 Weather 仍为
`0/4` wins，且相对 `full_time_mse` 为 `+0.23%` mean relative MSE、相对 S1-CFUS 为
全 horizon 退化。

[Decision] 当前 `predictability_downweight` 不通过 paper-core gate。不继续 sweep
`floor_weight=0.5`。失败点不是 train/eval 解耦，也不是 trace 未生效，而是简单 proxy 与
shared dense downweight formulation 不足以形成有效 shielding。

[Decision] 回退到 Step 5/6：

1. 重新评估 predictability proxy：仅用 local variation 过粗，需引入 train-only baseline
   residual、seasonal residual stability 或 running residual stability；
2. 若继续 noisy-hard shielding，优先考虑 detached/isolated auxiliary path，而不是在 shared
   dense loss 中简单降权；
3. S1-CFUS 保留为正证据：hard-block emphasis 对 ETTh2 有效，但需要 dataset/state-aware
   gate 决定何时启用；
4. 下一轮必须先做 train-side residual predictability diagnostic，再决定是否实现新 loss。

## Phase4-RG：Gradient-Routing HSS 主线升级

[Decision] Phase4 主线从 loss-only HSS 升级为 representation-aware / gradient-routing HSS：

> Horizon-agnostic supervision scheduling decides not only how much a future unit supervises,
> but also where its gradient is allowed to update.

[Current Step] 当前回到 11-step 的 Step 5/6：先验证理论可行性与诊断证据，再决定是否实现
adapter-isolated training strategy。

[Problem] S1/S2 已证明 train-side future-unit supervision 相比 `full_time_mse` 有收益，但
Weather 相对 R.3 全面失败。关键疑问不是继续调 `aux_weight` 或 `floor_weight`，而是 difficult
future units 是否污染 shared representation。

[Existence Evidence]

- [Fact] S1/S2 在 ETTh2 上能接近或超过 R.3，但 Weather 仍 `0/4` vs R.3；
- [Fact] Weather selected blocks 的 local-variation ratio 明显高于 ETTh2；
- [Fact] SRP/SRP++ 论文与本地代码 `SRP-7C55` 均支持 step/segment-specific representation
  或 tuner path 的必要性；
- [Fact] SRP finetune code 冻结 base parameters，只训练当前 tuner/group 或 MoLora shared
  tuner parameters，说明 gradient destination 是可被显式调度的设计变量。

[Idea] 把 HSS 从 “future unit loss reweighting” 扩展为：

1. `predictable/easy` 与 dense anchor 更新 shared path；
2. `learnable-hard` 可以增强 shared path 或 adapter path；
3. `noisy-hard` / conflict units 不直接污染 shared representation，而是进入 isolated adapter、
   detached auxiliary branch 或只更新 small residual path。

[Theory Check] SRP 对我们的支撑是机制级而非实现级：

- 可吸收：step/segment-specific path、base freezing、zero-init adapter、group-level metric；
- 不直接采用：two-stage SRP pretrain+finetune、按 benchmark `pred_len` 分组、直接复制 LoRA modules。

[Design] 当前先实现并运行 gradient conflict diagnostic：

- script: `scripts/analyze_phase4_gradient_conflict_diagnostic.py`;
- code explanation:
  `docs/code-explanation/phase4-gradient-conflict-diagnostic.md`;
- SRP code audit:
  `docs/experiments/phase4-srp-code-audit-for-gradient-routing-hss.md`;
- remote runner:
  `scripts/remote/run_phase4_gradient_conflict_diagnostic.sh`;
- supervision groups:
  `early_1_96`, `middle_97_192`, `middle_193_336`, `late_337_720`,
  `learnable_hard`, `noisy_hard`, `predictable_easy`, `full_1_720`;
- parameter groups:
  `encoder`, `target_path`, `readout_head`, `all_shared`;
- metric:
  pairwise gradient cosine、negative share、low-cosine share。

[Gate] 只有满足以下条件，才实现 `adapter_isolated_supervision`：

1. Weather `noisy_hard` vs `early_1_96` 或 `predictable_easy` 的 cosine 明显低于 ETTh2；
2. 冲突主要出现在 shared path，包括 `encoder`、`target_path` 或 `readout_head`；
3. block trace 中 Weather 确实存在稳定 noisy-hard bucket；
4. 诊断结论能解释 S2 scalar downweight 为什么不足。

[Rollback] 如果 gradient conflict 证据不成立，不进入 architecture-level HSS。回退到 Step 5：
重做 predictability proxy，优先 residual stability / seasonal residual stability / train-only baseline
residual，而不是继续堆 adapter 或 MoE。

[Fact] Gradient conflict diagnostic 已完成：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_gradient_conflict_diagnostic_20260625`;
- local analysis:
  `analysis/phase4_gradient_conflict_diagnostic_20260625`;
- decision report:
  `analysis/phase4_gradient_conflict_diagnostic_20260625/phase4_gradient_conflict_decision.md`;
- GPU: `1`;
- warmup steps: `40`;
- diagnostic batches: `16`;
- datasets: `ETTh2`, `Weather`。

[Fact] 关键结果：

| Pair | Parameter group | ETTh2 mean cosine | Weather mean cosine | Weather negative share |
| --- | --- | ---: | ---: | ---: |
| `noisy_hard` vs `predictable_easy` | `readout_head` | `0.6455` | `0.1190` | `0.3333` |
| `noisy_hard` vs `predictable_easy` | `all_shared` | `0.6482` | `0.1403` | `0.2667` |
| `late_337_720` vs `early_1_96` | `readout_head` | `0.2379` | `-0.0219` | `0.5625` |
| `late_337_720` vs `early_1_96` | `all_shared` | `0.1912` | `-0.0149` | `0.5625` |

[Decision] 诊断为 partial pass：支持 gradient-routing HSS 主线升级，但不支持只做
`noisy-hard isolation`。最强冲突来自 Weather 的 late vs early shared/readout path；因此下一步
method design 应调整为 `late/conflict-aware adapter routing`：

1. dense anchor 继续训练 shared base；
2. early/predictable units 保持 shared path；
3. late/conflict auxiliary pressure 不直接更新 shared readout/head；
4. adapter path zero-init，初始预测等价于 base；
5. adapter 更新范围优先限制在 readout/head 或 small residual branch，而不是全 encoder。

[Gate] 下一轮最小方法必须证明：

1. Weather vs R.3 不再 `0/4` collapse；
2. Weather `late_337_720` segment relative MSE 改善；
3. ETTh2 相对 R.3 的 `3/4` 正信号不能丢失；
4. trace 记录 routed units、adapter contribution、shared/adapter loss；
5. prefix consistency 保持 numerical-zero。

## Phase4-RG-A：Late-Conflict Adapter Routing

[Current Step] Step 6/7：基于 gradient conflict diagnostic 设计并实现最小方法。

[Idea] 不把所有 hard/noisy blocks 一刀切隔离，而是先处理诊断中最明确的冲突：
Weather `late_337_720` vs `early_1_96` 在 `readout_head/all_shared` 上 mean cosine 接近或低于
0。因此新增 zero-init late adapter residual：

$$
\hat{y}_{final}
=
\hat{y}_{base}
+
r_{\phi}(h) \cdot \mathbb{1}[t \ge 337].
$$

训练 loss：

$$
\mathcal{L}
=
\mathcal{L}_{base}(1{:}720)
+
\lambda
\mathcal{L}_{adapter}(337{:}720),
$$

其中 adapter loss 使用 `base_pred.detach() + adapter_residual`，因此 late auxiliary pressure
不通过 adapter path 更新 shared encoder/target/readout。

[Design]

- model update: `PatchEncoderTargetSetDecoder` 增加 optional `supervision_adapter_head`;
- training strategy: `late_conflict_adapter_routing`;
- code explanation:
  `docs/code-explanation/phase4-late-conflict-adapter-routing.md`;
- remote runner:
  `scripts/remote/run_phase4_late_conflict_adapter_gate.sh`;
- default adapter start: step `337`;
- default aux weight: `0.1`;
- controls: `full_time_mse`, `r3_prefix_risk`。

[Gate] Small gate 仍只跑 `ETTh2` 与 `Weather`：

1. vs `full_time_mse` 必须保留收益；
2. Weather vs R.3 不能再 `0/4` collapse；
3. Weather `late_337_720` segment 必须改善；
4. ETTh2 vs R.3 不低于 S1/S2 的正信号；
5. `adapter_mean_abs_residual` 非零但不能主导 base prediction；
6. prefix consistency numerical-zero。

[Rollback] 如果失败，不继续扩大 adapter 或上 MoE。回退 Step 5，重新定义 dynamic
conflict/predictability router。

[Fact] RG-A small remote gate 已完成：

- runner: `scripts/remote/run_phase4_late_conflict_adapter_gate.sh`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_late_conflict_adapter_gate_20260625`;
- local analysis:
  `analysis/phase4_late_conflict_adapter_gate_20260625`;
- analysis script:
  `scripts/analyze_phase4_late_conflict_adapter_gate.py`;
- decision report:
  `analysis/phase4_late_conflict_adapter_gate_20260625/phase4_late_conflict_adapter_gate_report.md`;
- strategies:
  `late_conflict_adapter_routing`, `full_time_mse`, `r3_prefix_risk`;
- datasets: `ETTh2`, `Weather`。

[Fact] 主要结果：

| Comparison | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| RG-A vs `D0_full_time_mse` | 8 | 7 | 6 | `-1.31%` |
| RG-A vs `D1_r3_prefix_risk` | 8 | 1 | 0 | `+3.76%` |

[Fact] dataset split：

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D0_full_time_mse` | 4 | 4 | `-2.26%` |
| `Weather` | `D0_full_time_mse` | 4 | 3 | `-0.36%` |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 1 | `+3.08%` |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | `+4.44%` |

[Fact] segment gate：

- Weather h720 `337-720` vs R.3: `+4.74%`，late segment 没被修复；
- ETTh2 h720 `337-720` vs R.3: `-1.53%`，adapter 只在 ETTh2 的 late segment 起效；
- prefix consistency 仍为 numerical-zero，max prefix mismatch MSE `1.474e-14`；
- adapter trace 生效：`adapter_active_steps=384`，ETTh2 mean abs residual `0.0136`，
  Weather mean abs residual `0.0145`。

[Decision] RG-A 不通过 paper-core gate，不进入 full matrix。它保留为 partial evidence：
gradient routing 是有价值的研究轴，因为它相对 `full_time_mse` 有稳定收益；但 fixed late
adapter route 不能解释或修复 Weather 相对 R.3 的失败。

[Counter-Evidence] 当前失败直接推翻“只要把 late conflict auxiliary 隔离到 adapter 就能解决
Weather”的假设。Weather 的问题不是单纯 gradient destination，而是需要判断 late signal 何时
learnable、何时 noisy。固定 late route 缺少 state/difficulty condition。

[Rollback] 回到 Step 5/6。下一步不 sweep `aux_weight` 或 `adapter_start_step`，而是设计
dynamic conflict/predictability router：

1. 用 train-side residual stability / seasonal residual stability 区分 learnable-conflict 与
   noisy-conflict；
2. 只将被判定为 learnable-conflict 的 units route 到 adapter 或 auxiliary branch；
3. noisy-conflict units 应降低或阻断 shared gradient，而不是固定 late adapter 学习；
4. gate 继续要求 Weather vs R.3 不再 `0/4` collapse，且 ETTh2 late gain 不丢失。

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
