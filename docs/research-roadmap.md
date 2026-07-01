# FATST Research Roadmap

## 论文主线总纲

[Decision] 论文级主线由 `docs/paper-mainline.md` 维护。本文档负责完整记录 11-step
研究过程、实验证据和阶段性 rollback；`docs/paper-mainline.md` 负责维护能进入论文的
核心 claim、创新点、实验总安排和转向规则。

[Decision] 阶段内部候选队列、未完成任务和“已提出但尚未执行”的 idea 由
`docs/stage-ledgers/` 维护。当前 active ledger 是
`docs/stage-ledgers/phase5-timealign-interface.md`。完整研究路径保存体系见
`docs/research-governance.md`。

[Rule] 每次重要实验节点或研究转向后，必须先在本文档完成 11-step decision，再检查
`docs/paper-mainline.md` 是否需要同步更新：

- 如果结果改变 paper claim、方法命名、核心贡献、主 baseline 或实验总安排，必须同步更新；
- 如果结果只是局部 negative evidence 或 diagnostic 细节，保留在本文档和 `docs/experiments/`，
  不写入论文总纲；
- 若 11-step rollback 到 Step 2/3，论文总纲也必须重新审视 `核心问题` 和 `目标 claim`。

[Rule] 每次用户要求“继续推进研究”“按计划继续”“设计下一步实验”或“远程实验完成请分析”
时，必须先读取 active Stage Ledger，检查 `candidate_queue` 与 `pending_tasks`。若仍存在
`proposed`、`narrative_ready`、`analysis_pending` 或 `partial_pass` 候选，不能因为当前实验失败
直接改写主线或跳到全新方向。

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
10. 判断是否通过：基于实验 artifact 评估 effectiveness；不能在此时才补做方法叙事。
11. 若不通过，评估应回退至哪一步，然后继续循环。

[Decision] 每个新方法候选的阶段记录必须包含：

| Field | Required Content |
| --- | --- |
| `current_step` | 当前处于 11-step loop 的哪一步 |
| `problem` | 待解决问题，以及它为什么不是已有实验已经否定的问题 |
| `existence_evidence` | 问题真实存在的 artifact、公式推导或可复查现象 |
| `idea` | 核心 idea，不超过一个主机制 |
| `theory_check` | 数据流、数学约束、可能成立的原因和反例 |
| `design` | model/training/evaluation 的最小方案 |
| `narrative_gate` | Step 4-6 完成；判断该方案是否具备 paper-core 叙事资格 |
| `effectiveness_gate` | Step 9-10 完成；判断实验 artifact 是否支持性能与机制有效性 |
| `artifacts` | 代码版本、输出路径、报告和表格 |
| `decision` | 是否通过；若不通过，回退到哪一步 |

[Rule] 旧记录中的单一 `gate` 保留作为历史记录，不强制回填；新的 method-candidate 记录必须拆成
`narrative_gate` 与 `effectiveness_gate`。若实验是 `diagnostic_only` 或 `control_only`，可以不写
`narrative_gate`，但必须在 `design` 或 `decision` 中明确标注其不能直接升级为 paper-core。

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

### Phase4-RG-B：Residual-Stability Dynamic Routing 诊断

`current_step`: Step 5/6 complete；RG-A 失败后重新定义 dynamic conflict proxy，并验证它是否能解释
fixed late route 的失败。

[Problem]

RG-A 相对 `full_time_mse` 有稳定收益，但相对 R.3 失败，尤其 Weather `late_337_720`
segment 没被修复。因此问题不是“late supervision 有没有冲突”，而是 fixed late route
把 learnable conflict 和 noisy conflict 混在一起，导致 adapter 也在学习不该承载的 noisy units。

[Existence Evidence]

- S1/S2 说明 high-novelty / predictability scheduling 相对 `full_time_mse` 有机制信号；
- gradient conflict diagnostic 说明 late/noisy supervision 会与 shared representation 发生冲突；
- RG-A 说明 gradient destination 是有效轴，但 fixed late destination 不够；
- residual-stability diagnostic 用 train labels 进一步验证：selected high-novelty units 内部确实存在
  learnable/noisy 分化。

[Idea]

将 HSS 主线升级为：

> Horizon-agnostic supervision scheduling decides not only how much a future unit supervises,
> but also where its gradient is allowed to update.

这里的关键不是“按 horizon 选 loss”，而是对 training 中的 future units 做
state/difficulty/residual-stability conditioned routing。evaluation 仍保持 benchmark horizons；
training strategy 可以完全 horizon-agnostic。

[Theory Check]

[Strong Evidence] 如果一个 high-novelty block 相对 seasonal/persistence baseline 有高 gain，
且 best residual 平滑，那么它更像 structured residual，适合交给 adapter 或 auxiliary branch 学习。

[Strong Evidence] 如果一个 high-novelty block 在扣除 best baseline 后 residual 仍高频震荡，
且 local variation 高，那么它更可能是 noisy conflict；继续让它更新 shared path 或 adapter
可能伤害可预测部分。

[Speculative] 这个 proxy 还没有证明能在线稳定工作。当前诊断使用 dataset-relative quantiles；
训练中需要设计 batch-level 或 warmup-calibrated threshold，并记录 trace。

[Design]

- diagnostic script:
  `scripts/analyze_phase4_residual_stability_diagnostic.py`;
- code explanation:
  `docs/code-explanation/phase4-residual-stability-diagnostic.md`;
- analysis artifacts:
  `analysis/phase4_residual_stability_diagnostic_20260625`;
- train split only，`seq_len=336`，`pred_len=720`，`block_size=48`;
- selected units: 每个 window 内 top `25%` `novelty_mse` blocks；
- baseline candidates: persistence, seasonal `24/48/96/168`;
- bucket:
  `learnable_conflict`, `noisy_conflict`, `ambiguous_conflict`。

[Fact] Selection summary：

| Dataset | Region | Selected share | Gain over persistence | Residual smoothness | Local variation |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETTh2` | `early_1_96` | `6.0%` | `2.346` | `0.453` | `0.103` |
| `ETTh2` | `middle_97_336` | `26.7%` | `2.306` | `0.349` | `0.103` |
| `ETTh2` | `late_337_720` | `67.3%` | `1.827` | `0.238` | `0.095` |
| `Weather` | `early_1_96` | `8.1%` | `2.271` | `0.294` | `0.964` |
| `Weather` | `middle_97_336` | `26.1%` | `2.411` | `0.274` | `0.783` |
| `Weather` | `late_337_720` | `65.8%` | `1.957` | `0.199` | `0.526` |

[Fact] Late bucket split：

| Dataset | Bucket | Share in late selected units | Gain | Smoothness | Variation | Baseline mode |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ETTh2` | `ambiguous_conflict` | `68.8%` | `1.527` | `0.180` | `0.094` | `persistence` |
| `ETTh2` | `learnable_conflict` | `9.5%` | `2.316` | `0.107` | `0.068` | `seasonal_168` |
| `ETTh2` | `noisy_conflict` | `21.7%` | `2.562` | `0.479` | `0.112` | `seasonal_168` |
| `Weather` | `ambiguous_conflict` | `51.0%` | `1.817` | `0.110` | `0.068` | `persistence` |
| `Weather` | `learnable_conflict` | `16.2%` | `2.839` | `0.078` | `0.040` | `seasonal_48` |
| `Weather` | `noisy_conflict` | `32.8%` | `1.738` | `0.396` | `1.478` | `seasonal_48` |

[Analysis]

[Strong Evidence] Weather 的问题不是 late units 全部不可学。Weather late 中仍有
`16.2%` learnable-conflict，且 gain `2.839`；这说明完全避开 hard units 会丢掉可学习信号。

[Strong Evidence] Weather late 同时有 `32.8%` noisy-conflict，且 local variation `1.478`，
远高于 ETTh2 late noisy 的 `0.112`。这解释了 fixed late adapter 为什么没有修复 Weather：
它把可学习 residual 与高频 noisy residual 放进同一个 adapter route。

[Strong Evidence] Weather noisy-conflict 不只出现在 late：early noisy share `48.1%`，
middle noisy share `45.3%`。因此下一步不应继续固定 late start，而应使用 bucket-conditioned
route。

[Decision]

RG-B 作为 Step 5/6 通过：它不是最终方法结果，但足以支持下一个最小方法候选
`dynamic_residual_stability_routing`。

下一步进入 Step 6/7：

1. dense base 仍使用 full 720 supervision，保留统一多 horizon carrier；
2. selected high-novelty units 内部按 residual stability 分桶；
3. `learnable_conflict` route 到 adapter auxiliary；
4. `noisy_conflict` 不进入 adapter auxiliary，并降低或阻断其对 shared path 的额外压力；
5. `ambiguous_conflict` 默认只保留 dense base 或极弱 auxiliary；
6. trace 必须记录 bucket share、routed loss、adapter residual magnitude、noisy suppression ratio。

[Implementation]

[Fact] Step 7 最小实现已完成：

- training strategy:
  `dynamic_residual_stability_routing`;
- modified training entry:
  `baselines/patch_encoder_target_set_decoder/train.py`;
- remote runner:
  `scripts/remote/run_phase4_dynamic_residual_stability_gate.sh`;
- shared runner run name:
  `PatchEncoderDynamicResidualStabilityRouting`;
- code explanation:
  `docs/code-explanation/phase4-dynamic-residual-stability-routing.md`。

[Fact] Dynamic strategy 的 adapter effective start step 被强制为 `1`，避免继承 RG-A 的
fixed late mask。否则 early/middle learnable blocks 会被 router 选中，但 adapter residual
在模型层面被置零。

[Fact] 本地 smoke 仅验证代码路径，不作为实验结果：

- dataset: `Weather`;
- strategy: `dynamic_residual_stability_routing`;
- `epochs=1`, `steps_per_epoch=1`, `max_eval_batches=1`, `batch_size=8`, `device=cpu`;
- trace example:
  `learnable_blocks=1`, `noisy_blocks=2`, `ambiguous_blocks=1`,
  `noisy_suppression_ratio=0.5`；
- 该 smoke 证明 forward/loss/trace 能运行，但不能证明性能。

[Gate]

`dynamic_residual_stability_routing` small gate 仍只跑 `ETTh2` 与 `Weather`：

1. vs `full_time_mse` 必须保留 S1/S2/RG-A 已经证明的收益；
2. Weather vs R.3 不能继续 `0/4` collapse；
3. Weather `late_337_720` segment 必须优于 RG-A；
4. ETTh2 late segment 的正信号不能消失；
5. trace 中 learnable/noisy bucket 不能塌缩为单一 bucket；
6. prefix consistency 仍必须 numerical-zero。

[Returned Result]

[Fact] RG-B small gate 已完成：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_dynamic_residual_stability_gate`;
- local raw artifacts:
  `analysis/phase4_dynamic_residual_stability_gate_20260625/raw`;
- analysis script:
  `scripts/analyze_phase4_dynamic_residual_stability_gate.py`;
- decision report:
  `analysis/phase4_dynamic_residual_stability_gate_20260625/phase4_dynamic_residual_stability_gate_report.md`。

[Fact] Overall result：

| Comparison | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| RG-B vs `full_time_mse` | 8 | 7 | 6 | `-2.73%` |
| RG-B vs R.3 | 8 | 2 | 2 | `+2.21%` |

[Fact] Dataset split vs R.3：

| Dataset | MSE wins | Mean relative MSE |
| --- | ---: | ---: |
| `ETTh2` | 2/4 | `+0.06%` |
| `Weather` | 0/4 | `+4.37%` |

[Fact] Per-horizon MSE：

| Dataset | Horizon | RG-B | Full-time | R.3 | RG-B vs full | RG-B vs R.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | `0.311002` | `0.339358` | `0.304796` | `-8.36%` | `+2.04%` |
| `ETTh2` | 192 | `0.364008` | `0.388740` | `0.369043` | `-6.36%` | `-1.36%` |
| `ETTh2` | 336 | `0.385893` | `0.394549` | `0.382910` | `-2.19%` | `+0.78%` |
| `ETTh2` | 720 | `0.405428` | `0.419201` | `0.410473` | `-3.29%` | `-1.23%` |
| `Weather` | 96 | `0.156419` | `0.154701` | `0.148026` | `+1.11%` | `+5.67%` |
| `Weather` | 192 | `0.199968` | `0.200707` | `0.192409` | `-0.37%` | `+3.93%` |
| `Weather` | 336 | `0.254646` | `0.256879` | `0.244793` | `-0.87%` | `+4.02%` |
| `Weather` | 720 | `0.333251` | `0.338482` | `0.320847` | `-1.55%` | `+3.87%` |

[Fact] Segment gate：

- Weather h720 `337-720` vs R.3: `+4.06%`，仍失败；
- ETTh2 h720 `337-720` vs R.3: `-2.83%`，dynamic adapter 在 ETTh2 late segment 有正信号；
- prefix consistency 仍为 numerical-zero，max prefix mismatch MSE `1.474e-14`。

[Fact] Trace bucket：

| Dataset | Learnable blocks | Noisy blocks | Ambiguous blocks | Noisy suppression | Adapter active steps | Mean abs adapter residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `1.17` | `1.44` | `1.39` | `0.36` | `56.1` | `0.030439` |
| `Weather` | `1.44` | `1.87` | `0.69` | `0.47` | `69.1` | `0.044936` |

[Analysis]

[Strong Evidence] RG-B 保留了相对 `full_time_mse` 的收益，说明
residual-stability routing 不是无效机制；Weather long horizons 也相对 full-time 改善。

[Counter-Evidence] RG-B 没有解决 paper-core gate。Weather vs R.3 仍为 `0/4`，且 Weather
h720 late segment 仍落后 R.3 `+4.06%`。这说明“只把 learnable residual units 送进 detached
adapter”不足以成为主线方法。

[Inference] Router 本身没有塌缩：Weather 的 noisy suppression ratio 高于 ETTh2，adapter
residual 非零。因此失败更可能来自 carrier/optimization protocol，而不是 router 完全没工作。

[Training Dynamics]

[Fact] 当前 gate 六个 run 的 best validation epoch 均在 `1-4`：

| Dataset | Strategy | Epochs ran | Best epoch | Best val MSE | Last val drift | Train loss change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | RG-B | 11 | 1 | `0.383836` | `+25.30%` | `-26.01%` |
| `Weather` | RG-B | 13 | 3 | `0.532040` | `+8.50%` | `-38.40%` |
| `ETTh2` | full-time | 13 | 3 | `0.380270` | `+14.45%` | `-30.94%` |
| `Weather` | full-time | 14 | 4 | `0.534117` | `+10.41%` | `-38.40%` |
| `ETTh2` | R.3 | 11 | 1 | `0.374532` | `+18.43%` | `-53.20%` |
| `Weather` | R.3 | 11 | 1 | `0.527042` | `+17.01%` | `-34.75%` |

[Strong Evidence] 这不是 RG-B 独有 bug。当前和相邻 Phase4 gates 中，ETTh2/Weather 共
`26/26` 条训练日志都在 epoch `<=5` 达到 best validation MSE。之后 train loss 继续下降而
validation 变差，说明是当前 carrier/protocol 的系统性 fast overfitting 或 validation-objective
mismatch。

[Decision]

RG-B 不通过 paper-core gate，不进入 full matrix，也不继续 sweep `aux_weight`、threshold 或
seasonal periods。当前应回退 Step 5/6，优先研究 training protocol / carrier：

1. 是否需要 shorter calibrated training 或 learning-rate schedule；
2. 是否需要 SRP-style pretraining / warmup，而不是从零训练时直接施加 routing；
3. 是否当前 validation objective 过早选择初始附近 checkpoint，导致后续 routing 无法显现；
4. 是否应把 routing 放在 pretrained/base-stabilized carrier 上，而不是继续在弱 carrier 上堆机制。

[Rollback]

如果 RG-B 方法 gate 失败，不继续加 MoE 或 future-aware module。回到 Step 5：
重新检验 residual-stability proxy 是否在线不稳定；必要时改成 offline-calibrated threshold
或完全转向 state-conditioned gradient blocking，而不是继续堆 adapter。

### Phase4-OP-A：Stabilized Routing Protocol

`current_step`: Step 5/6 -> Step 7。RG-B 失败后不继续调 routing threshold，而是先验证当前
training protocol 是否是瓶颈。

[Problem]

[Fact] 当前和相邻 Phase4 gates 中，ETTh2/Weather 共 `26/26` 条训练日志都在 epoch `<=5`
达到 best validation MSE；之后 train loss 继续下降而 validation 变差。

[Fact] RG-B trace 中 router 没有塌缩，Weather 的 noisy suppression ratio 高于 ETTh2，
adapter residual 非零，但 Weather vs R.3 仍 `0/4`。

[Inference] 这说明失败可能不是“routing 完全没工作”，而是 current carrier 从零训练过快进入
validation 退化区；routing adapter 在 base 尚未稳定时与 base learning 竞争，无法形成可泛化
贡献。

[Existence Evidence]

- SRP code 的 finetune path 先加载 pretrain checkpoint；
- SRP finetune 冻结 base parameters，只训练当前 tuner/group 或 MoLora shared tuner；
- 当前 Phase4 logs 系统性 early-best，说明 training protocol 本身需要被研究；
- RG-B 相对 full-time 仍有 `-2.73%` mean MSE，说明机制轴值得保留，但 carrier/protocol 不够。

[Idea]

吸收 SRP 的 protocol claim，而不是复制 SRP 架构：

> route-specific parameters should be trained on top of a stabilized base instead of competing with
> base learning from step zero.

本阶段测试 `full_time_mse pretrain -> adapter-only dynamic routing finetune`。

[Theory Check]

[Strong Evidence] 如果 adapter/routing path 只负责 residual correction，而 shared base 已被
validation-selected checkpoint 固定，那么 noisy/learnable routing 的梯度不会再干扰 base learning。

[Speculative] full-time base 可能仍弱于 R.3；因此该实验若失败，不等价于否定 pretraining，
只否定 “full-time stabilized base + current RG-B adapter” 这个组合。

[Design]

- training update:
  `baselines/patch_encoder_target_set_decoder/train.py` 增加 `--init-checkpoint` 与
  `--freeze-non-adapter`;
- code explanation:
  `docs/code-explanation/phase4-stabilized-routing-protocol.md`;
- remote runner:
  `scripts/remote/run_phase4_stabilized_routing_gate.sh`;
- datasets:
  `ETTh2`, `Weather`;
- stage 1:
  `PatchEncoderFullTimeMSE720Pretrain`，`full_time_mse`，lr `1e-4`，保留 best checkpoint；
- stage 2:
  `PatchEncoderStabilizedDynamicResidualRouting`，加载 stage-1 checkpoint，只训练
  `supervision_adapter_head`，lr `1e-3`，epochs `30`，patience `5`。

[Fact] 本地 smoke 已通过：

- dataset: `ETTh2`;
- pretrain: `full_time_mse`，`epochs=1`，`steps_per_epoch=1`；
- finetune: `dynamic_residual_stability_routing`，加载 pretrain checkpoint，
  `--freeze-non-adapter`，`epochs=1`，`steps_per_epoch=1`；
- config audit: `freeze_non_adapter_effective=true`；
- parameter audit: trainable params `12,848 / 2,025,312`；
- checkpoint audit: missing keys 仅为 `supervision_adapter_head.*`，符合 adapter 新增预期。

[Gate]

1. `effective_config.json` 必须显示 `freeze_non_adapter_effective=true`；
2. `environment.json` 的 `trainable_parameter_count` 必须远小于 `parameter_count`；
3. finetune validation best epoch 不应仍系统性卡在 epoch `1`；
4. Weather vs R.3 至少不能继续 `0/4`；
5. Weather h720 `337-720` segment 必须优于 RG-B；
6. ETTh2 late segment 正信号不能消失。

[Returned Result]

[Fact] OP-A small gate 已完成：

- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_stabilized_routing_gate`;
- local raw artifacts:
  `analysis/phase4_stabilized_routing_gate_20260625/raw`;
- analysis script:
  `scripts/analyze_phase4_stabilized_routing_gate.py`;
- decision report:
  `analysis/phase4_stabilized_routing_gate_20260625/phase4_stabilized_routing_gate_report.md`。

[Fact] Freeze / checkpoint audit 通过：

| Dataset | Freeze | Trainable params | Missing keys | Unexpected keys |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `True` | `12,848 / 2,025,312` (`0.63%`) | 4 | 0 |
| `Weather` | `True` | `12,848 / 2,025,312` (`0.63%`) | 4 | 0 |

missing keys 均为 `supervision_adapter_head.*`，符合 adapter 新增预期。

[Fact] Overall result：

| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| OP-A vs R.3 | 8 | 0 | 0 | `+8.92%` |
| OP-A vs pretrain | 8 | 0 | 0 | `+3.57%` |
| OP-A vs RG-B from scratch | 8 | 1 | 1 | `+6.64%` |

[Fact] Per-horizon MSE delta：

| Dataset | Horizon | vs pretrain | vs R.3 | vs RG-B |
| --- | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | `+5.60%` | `+17.57%` | `+15.22%` |
| `ETTh2` | 192 | `+6.39%` | `+12.06%` | `+13.61%` |
| `ETTh2` | 336 | `+3.37%` | `+6.51%` | `+5.69%` |
| `ETTh2` | 720 | `+3.79%` | `+5.99%` | `+7.31%` |
| `Weather` | 96 | `+1.08%` | `+5.64%` | `-0.03%` |
| `Weather` | 192 | `+2.01%` | `+6.41%` | `+2.39%` |
| `Weather` | 336 | `+2.86%` | `+7.94%` | `+3.76%` |
| `Weather` | 720 | `+3.51%` | `+9.20%` | `+5.14%` |

[Fact] Segment gate：

- Weather h720 `337-720` vs R.3: `+10.24%`;
- Weather h720 `337-720` vs RG-B: `+5.94%`。

[Fact] Training dynamics：

| Dataset | Stage | Epochs | Best epoch | Best val MSE | Last val drift | Train loss change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | pretrain | 13 | 3 | `0.380270` | `+14.45%` | `-30.94%` |
| `ETTh2` | adapter-only | 12 | 7 | `0.402832` | `+2.47%` | `-0.05%` |
| `Weather` | pretrain | 14 | 4 | `0.534117` | `+10.41%` | `-38.40%` |
| `Weather` | adapter-only | 11 | 6 | `0.542288` | `+0.40%` | `-0.07%` |

[Analysis]

[Strong Evidence] OP-A 成功改变了 training dynamics：adapter-only finetune 的 best epoch
从 pretrain 的 `3/4` 推迟到 `7/6`，且 train loss 几乎不再下降。这说明 base freezing 和
adapter-only optimization 确实改变了训练过程。

[Counter-Evidence] 但 OP-A 性能全面失败。它在所有 horizon 上都差于 pretrain 和 R.3；
除 Weather h96 外也差于 from-scratch RG-B。Weather late segment 也比 RG-B 更差。

[Inference] 这说明“从零训练时 base/routing 竞争”不是唯一瓶颈。当前小 adapter residual path
在 frozen full-time base 上没有足够 capacity 或合适 objective 去修正预测；full-time base
本身也不足以作为 paper-core pretrain base。

[Decision]

OP-A 不通过，不继续调 finetune lr 或 patience。回退 Step 5/6：

1. 不再推进 `full_time_mse pretrain + adapter-only RG-B`；
2. 若继续 pretraining，应测试 R.3/prefix-risk stabilized base，而不是 full-time base；
3. 若继续 gradient routing，应允许 richer target/readout subset 更新，而不只是 tiny adapter；
4. 下一步需要先诊断“R.3 为什么强”：prefix-risk 是优化 schedule、implicit regularization，
   还是更适合 current carrier 的 base objective。

[Rollback]

如果 OP-A 失败，不继续调 finetune lr。回退 Step 5/6，判断当前 adapter/routing capacity 是否过弱；
下一步应考虑 stronger base objective、R.3-style base stabilization，或更深的 carrier redesign。

### Phase4-R3D：R.3 Mechanism Diagnostic And Decomposition

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 decomposition artifacts，并决定 rollback/next direction |
| `problem` | R.3 在 Phase4 中持续压过 `full_time_mse`、RG-B 和 OP-A，但它不是单一 step-weight baseline |
| `existence_evidence` | `phase4_dynamic_residual_stability_gate`、`phase4_stabilized_routing_gate`、Phase2 objective-pressure diagnostic |
| `idea` | 把 R.3 视作 compound protocol：mixed-horizon exposure + prefix-risk pressure，而不是“简单预测段权重” |
| `theory_check` | 若不拆开 R.3，HSS 失败无法判断是输给 horizon exposure、loss weighting，还是二者交互 |
| `design` | 先生成 R.3 diagnostic；随后运行 `full_time_mse`、`horizon_mixed`、`single_720_prefix_risk`、`r3_prefix_risk` 四个 controls |
| `gate` | 若 `horizon_mixed` 接近 R.3，说明 exposure 是主因；若 `single_720_prefix_risk` 接近 R.3，说明 weighting 是主因；若两者都弱而 R.3 强，说明 interaction 是主因 |
| `artifacts` | `analysis/phase4_r3_mechanism_diagnostic_20260625/`；`analysis/phase4_r3_decomposition_gate_20260625/`；`scripts/remote/run_phase4_r3_decomposition_gate.sh` |
| `decision` | R.3 不作为 core story；prefix-risk pressure 是更干净的 HSS 起点，mixed-horizon exposure 保留为强对照而非主线 |

[Fact] 当前 R.3 vs h720-only `full_time_mse720`：`8/8` MSE wins，mean relative MSE
`-4.83%`。其中 ETTh2 为 `-5.07%`，Weather 为 `-4.59%`。

[Fact] 训练日志显示 R.3 实际使用 mixed-horizon training：ETTh2 的 h96/h192/h336/h720
training exposure share 约为 `0.25/0.23/0.25/0.27`；Weather 约为
`0.25/0.26/0.25/0.25`。`full_time_mse` 则是 h720-only supervision。

[Fact] `prefix_risk` objective pressure 把 `1-96` region 的 expected pressure share
从 `0.4798` 提高到 `0.7217`，并把 `337-720` 从 `0.1333` 降到 `0.0469`。
因此 R.3 同时改变了训练暴露分布和 future-region loss pressure。

[Counter-Evidence] Phase2 中 R.3 vs uniform target-set 的平均收益约 `-1.03%`，明显小于
当前 R.3 vs h720-only full-time 的 `-4.83%`。这支持“R.3 当前优势来自 compound protocol”
而不是“prefix-risk 单独强大”的判断。

[Returned Result]

[Fact] Decomposition gate 已完成。四个 controls 都相对 h720-only `full_time_mse720`
取得收益：

| Strategy | vs full-time mean MSE | Beats full-time | Best among candidates |
| --- | ---: | ---: | ---: |
| `mixed_horizon_uniform` | `-4.12%` | `8/8` | `1/8` |
| `single_720_prefix_risk` | `-4.39%` | `8/8` | `2/8` |
| `r3_prefix_risk` | `-4.83%` | `8/8` | `5/8` |

[Fact] `single_720_prefix_risk` 在 ETTh2 h96/h192 最优，说明 prefix-risk pressure
不依赖 training/evaluation horizon coupling 也能给出强收益。`r3_prefix_risk` 在 Weather
四个 horizons 全部最优，并在 h720 late segment 更强，说明 mixed exposure 与 prefix-risk
的 compound protocol 仍是强对照。

[Strong Evidence] `mixed_horizon_uniform` 有收益，但不是主导解释：它相对 full-time
`8/8` 改善，却只在 `1/8` setting 中最优，并且 Weather 上稳定弱于 R.3。

[Decision] 不继续 repair R.3，也不把 R.3 包装成主贡献。下一步将 HSS 重新锚定为：
在 h720-only prefix/stability pressure 下，控制 gradient 更新到哪些 future-state/readout
subspace，而不是训练时采样哪些 horizons。

[Next Direction] 进入 Phase4-HSSG：architecture-level gradient routing under
`single_720_prefix_risk` base。核心问题是：prefix-weighted supervision 可以提高前缀和
短 horizon，但 late/noisy regions 容易被牺牲；下一步应让不同 future regions 的梯度进入
不同可控参数子空间，而不是继续叠加 scalar loss reweighting。

[Rollback]

回到 Step 4/6：HSS 不再被定义为 mixed-horizon training schedule，也不继续作为 R.3 repair。
新的主线定义是 horizon-agnostic supervision pressure + gradient routing。R.3 compound
保留为 strong reference，尤其用于 Weather 与 long-horizon/late-segment gate。

### Phase4-HSSG：Horizon-Agnostic Supervision Scheduling via Gradient Routing

| Field | Content |
| --- | --- |
| `current_step` | Step 4/6：提出核心想法并设计可证伪实验 |
| `problem` | `single_720_prefix_risk` 证明 h720-only prefix pressure 有效，但 scalar loss reweighting 无法说明“梯度应该更新哪里”；R.3 compound 在 Weather/late segment 仍强，说明单纯 h720 prefix pressure 还缺少 region-aware capacity 分配 |
| `existence_evidence` | Phase4-R3D decomposition；OP-A adapter-only failure；SRP 中按 tuner/group 冻结与激活参数的训练思想 |
| `idea` | HSS 不只决定 future unit 的 loss weight，而是决定该 unit 的 gradient 被允许更新哪些 future-state/readout 参数子空间 |
| `theory_check` | 若不同 future regions 的可预测性和噪声结构不同，则将所有 region 的误差压到同一组 shared parameters 会产生 gradient conflict；restricted subspace update 可以把 prefix/stability pressure 转化为结构性学习，而非全局 loss bias |
| `design` | 在 h720-only `single_720_prefix_risk` base 下，比较 unrestricted prefix-risk、region-routed readout/update、compound R.3 strong reference |
| `gate` | 必须优于 `single_720_prefix_risk`，并至少在 Weather h720 late segment 接近或超过 R.3；若只提升 early prefix 而牺牲 late，不通过 |
| `artifacts` | `docs/code-explanation/phase4-hssg-region-routed-readout.md`；`scripts/remote/run_phase4_hssg_gradient_routing_gate.sh` |
| `decision` | HSSG-A 已进入实现与远程 gate 阶段；不立即堆 MoE/future-aware，先验证最小 architecture-level routing 是否成立 |

#### 为什么提出 HSSG

[Fact] `single_720_prefix_risk` 在不使用 mixed-horizon training 的情况下取得 `-4.39%`
mean MSE，并在 ETTh2 h96/h192 最优。这说明训练和 evaluation 解耦是可行的：
training 不必显式采样 benchmark horizons。

[Fact] R.3 compound 仍在 Weather 四个 horizons 和 h720 late segment 上最强。这说明
prefix pressure 虽然是干净起点，但单一 scalar weighting 不能充分处理 Weather/long-horizon
late region。

[Fact] OP-A 的 frozen base + tiny adapter-only 失败，说明把 gradient routing 放在
输出 residual adapter 上太晚、capacity 太弱。需要把可更新位置上移到
`target_states -> condition_head -> history_readout -> segment_output` 预测主路径附近。

[Source-Informed Evidence] SRP 的 finetune group 代码提供了一个可借鉴原则：
冻结主体参数，只激活特定 tuner/group，并对不同 tuner 单独 early stopping。我们不直接复制
SRP 的代码或训练流程，但吸收它的机制启发：supervision 可以被分配到特定参数组，而不是
全模型无差别反传。

#### 核心研究假设

[Hypothesis H1] Prefix-weighted supervision 的主要价值不是“更看重短 horizon”，而是让
可预测的 early/prefix structure 先稳定地塑造 shared representation。

[Hypothesis H2] Late/noisy regions 不应简单被忽略，也不应和 early/prefix regions 竞争同一组
readout parameters；它们需要 either protected base path or region-specific update path。

[Hypothesis H3] 如果 HSSG 成立，最小模型应在不采样 mixed horizons 的情况下达到或超过
R.3 的平均性能，并在 Weather h720 late segment 缩小当前 `single_720_prefix_risk`
相对 R.3 的缺口。

#### 方法候选

| Candidate | 机制 | 为什么先/后做 | 通过条件 |
| --- | --- | --- | --- |
| HSSG-A：Region-Routed Readout LoRA | 在 `condition_head` 或 `segment_output` 上增加 region-specific low-rank/update path；early/middle/late losses 只更新对应 path，shared base 仍由主 loss 更新 | 最小 architecture-level routing；比 tiny adapter 更靠近主预测路径 | mean MSE 优于 `single_720_prefix_risk`；Weather h720 late 不劣于 single-prefix |
| HSSG-B：Prefix-Stable Shared + Late-Protected Readout | Prefix-risk loss 更新 shared target/readout；late loss 只更新 late-specific path 或被 gradient-detach 保护 | 用来验证“保护 late/noisy regions”是否必要 | early gain 不消失，同时 late segment 接近 R.3 |
| HSSG-C：Learnability-Conditioned Gradient Mask | 用 residual stability/predictability score 决定 unit 更新 shared、region-specific、或 no-update path | 风险更高，应在 A/B 有信号后再做 | 比 A/B 更稳定，且跨 ETTh2/Weather 不发生 dataset-specific collapse |

#### 最小实验计划

第一轮只做 HSSG-A，不做 HSSG-B/C：

1. `D0_full_time_mse720`：h720-only uniform baseline。
2. `D1_single_720_prefix_risk`：当前干净起点。
3. `D2_r3_prefix_risk`：compound strong reference，不作为主线机制。
4. `HSSG-A_region_routed_readout`：h720-only prefix-risk + region-routed readout/update。

数据集先用 `ETTh2` 与 `Weather`，seed `2021`。原因是 ETTh2 能检验 short/prefix gain，
Weather 能检验 long/late robustness。若最小 gate 不通过，不扩展到 ETTm1/ETTm2。

[Implementation Note] HSSG-A 当前实现为 `hssg_region_routed_readout`：
在 `conditioned -> segment_output` 主预测路径旁增加 early/middle/late 三个 low-rank
readout residual path；训练仍为 h720-only `prefix_risk`，因此与 `single_720_prefix_risk`
的 objective 对齐，差异只来自 gradient/update subspace。

#### 分析指标

- Main horizon MSE/MAE：h96/h192/h336/h720。
- Segment MSE/MAE：尤其 h720 的 `1-96`、`97-192`、`193-336`、`337-720`。
- Region win profile：early/middle/late 哪些 region 受益。
- Gradient/update audit：每个 region-specific path 的 trainable params、mean grad norm、
  update norm、是否 collapsed。
- Prefix consistency：必须保持 numerical-zero 或接近当前 target-set decoder 水平。
- Training dynamics：best epoch、post-best drift、train loss drop；若 best epoch 仍在
  1-3 且 drift 更大，需要回到 optimization schedule，而不是继续加结构。

#### Gate

HSSG-A 通过需要同时满足：

1. vs `single_720_prefix_risk`：overall mean MSE 改善，且至少 `5/8` main settings 不劣。
2. vs R.3：Weather h720 late segment 缺口缩小到 `<= +1.0%`，或直接优于 R.3。
3. 不牺牲 ETTh2 h96/h192：相对 `single_720_prefix_risk` 不超过 `+1.0%`。
4. Audit 显示 region-routed path 非 collapse：不同 region 的 update/grad norm 有可解释差异。

#### Rollback

若 HSSG-A 失败：

- 如果 main MSE 下降但 late 改善，回 Step 6 调整 routing capacity，而不是否定 HSSG；
- 如果 early 与 late 都差，回 Step 4，说明 current target-set carrier 不适合 gradient routing；
- 如果只有 Weather 失败，回 Step 5 分析 Weather 的 late/noisy region 是否需要 explicit stability score；
- 如果 audit 显示 routed paths collapse，回 Step 7 修实现或初始化，不做理论否定。

#### HSSG-A Gate Result：Region-Routed Readout

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：远程结果评估与 gate decision |
| `problem` | HSSG-A 要验证 fixed early/middle/late low-rank readout path 是否足以把 prefix-risk supervision 转化为有效 gradient routing |
| `existence_evidence` | remote artifacts 位于 `/home/yingch/exp_outputs/r-2026-fatst/phase4_hssg_gradient_routing_gate`；本地分析位于 `analysis/phase4_hssg_gradient_routing_gate_20260625` |
| `idea` | 在 h720-only `prefix_risk` objective 下，用 region-routed readout residual path 分担 shared readout 的 future-region gradient pressure |
| `theory_check` | 若该机制成立，应保留 single-prefix 的 short/prefix gain，同时缩小 Weather h720 late 相对 R.3 的缺口 |
| `design` | ETTh2 + Weather；`full_time_mse`、`single_720_prefix_risk`、`r3_prefix_risk`、`hssg_region_routed_readout`；seed `2021` |
| `gate` | vs single overall 改善且 `>=5/8` main wins；Weather h720 late vs R.3 gap `<= +1%`；ETTh2 h96/h192 vs single 不超过 `+1%`；routed path 非 collapse |
| `artifacts` | `scripts/analyze_phase4_hssg_gradient_routing_gate.py`；`analysis/phase4_hssg_gradient_routing_gate_20260625/phase4_hssg_gradient_routing_gate_report.md` |
| `decision` | `fail_as_core_candidate`；HSSG-A 是 partial evidence，不作为 core method 继续扩展 |

[Fact] HSSG-A absolute mean MSE 相对 `single_720_prefix_risk` 为 `-0.20%`，但
per-setting relative MSE 为 `+0.23%`，且只赢 `4/8` main settings，未达到 `5/8`
gate。相对 R.3，absolute mean MSE 为 `+0.18%`，per-setting relative MSE 为
`+0.67%`，只赢 `3/8`。

[Fact] HSSG-A 的 late/long 信号不是无效：Weather h720 相对 single-prefix 为
`-1.14%`，Weather h720 late segment `337-720` 相对 single-prefix 为 `-1.74%`，
相对 R.3 仅 `+0.16%`，满足“接近 R.3”的 late gate。

[Counter-Evidence] HSSG-A 牺牲了 short/early 优势：ETTh2 h96 相对 single-prefix
为 `+1.61%`，超过 `+1%` gate；Weather early region 相对 single-prefix 和 R.3 都
不稳。这说明 fixed region-routed residual path 把一部分 late/long 修好了，但没有保住
prefix-stable shared representation。

[Audit] routed path 非 collapse，但幅度偏小。HSSG Weather h720 的
`region_readout_residual` all MAE 为 `0.036413`，late MAE 为 `0.038856`；best epoch
仍非常早，ETTh2 为 `2/12`，Weather 为 `1/11`。

[Decision] 保留 HSSG 主线，但回到 Step 6 重设计方法。下一步不是 sweep
`rank/dropout/scale`，也不是继续堆 loss weight；应进入 learnability-conditioned
gradient routing：让 late/noisy units 根据 residual stability / predictability 决定更新
shared path、region path，或 no-update path，同时保护 single-prefix 的 short-horizon gain。

#### HSSG-B/C Plan：Learnability-Conditioned Region Routing

| Field | Content |
| --- | --- |
| `current_step` | Step 6/7：重设计并实现 HSSG-A 失败后的最小候选 |
| `problem` | HSSG-A 的 fixed region path 改善 late/long，但牺牲 ETTh2 h96 和 Weather early/middle；adapter-only dynamic routing 又太弱 |
| `existence_evidence` | HSSG-A late gate partial pass；RG-B residual-stability router 非 collapse；OP-A 证明 adapter-only carrier 不足 |
| `idea` | 保留 region-routed readout 作为 richer carrier，但只让 residual-stability learnable blocks 更新 detached region path；noisy/ambiguous blocks 不给 auxiliary pressure |
| `theory_check` | 如果 early/prefix gain 来自 shared base，而 late/structured residual 需要独立 path，则 base prefix-risk + learnable-region auxiliary 应同时保住 h96/h192 并修复 h720 late |
| `design` | 新策略 `hssg_learnability_region_routing`：`base_pred` 用 h720-only `prefix_risk` 更新 shared path；`base_pred.detach() + region_residual` 只在 learnable blocks 上更新 detached-input region heads |
| `gate` | vs `single_720_prefix_risk` 至少 `5/8` main wins；ETTh2 h96/h192 不超过 `+1%`；Weather h720 late vs R.3 gap `<= +1%`；trace 中 learnable/noisy buckets 与 region grad norm 非 collapse |
| `artifacts` | `docs/code-explanation/phase4-hssg-learnability-region-routing.md`；`scripts/remote/run_phase4_hssg_learnability_routing_gate.sh` |
| `decision` | 已进入实现与 remote gate 准备；若失败，不 sweep aux/rank，先判断 failure 是 carrier interference 还是 learnability proxy 错误 |

[Implementation Note] `hssg_learnability_region_routing` 与 HSSG-A 的关键差异是
`region_routed_readout_detach_input=True` 和 masked auxiliary。shared base 不通过 region
auxiliary 更新；region path 只看 residual-stability learnable blocks。

[Rollback] 若该 gate 仍牺牲 short/early，回 Step 4/6 重新设计 carrier，不能继续在
target-set readout 旁堆 residual path；若 only Weather 失败且 trace 显示 learnability bucket
有意义，回 Step 5 改进 predictability/residual-stability proxy；若 region grad/residual 接近零，
回 Step 7 修初始化或 loss scale。

#### HSSG-B/C Gate Result：Learnability-Conditioned Region Routing

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：远程结果评估与 gate decision |
| `problem` | 验证 residual-stability learnability mask 是否能保住 single-prefix 的 early gain，同时让 structured late residual 进入 region path |
| `existence_evidence` | remote artifacts 位于 `/home/yingch/exp_outputs/r-2026-fatst/phase4_hssg_learnability_routing_gate`；本地分析位于 `analysis/phase4_hssg_learnability_routing_gate_20260625` |
| `idea` | shared base 用 h720-only prefix-risk；learnable residual blocks 只更新 detached region readout path |
| `theory_check` | 若 proxy 与 carrier 同时成立，应减少 noisy/ambiguous pressure，避免 HSSG-A 的 early damage，并保持 Weather late 接近 R.3 |
| `design` | ETTh2 + Weather；`full_time_mse`、`single_720_prefix_risk`、`r3_prefix_risk`、`hssg_region_routed_readout`、`hssg_learnability_region_routing`；seed `2021` |
| `gate` | vs single-prefix `>=5/8` main wins；ETTh2 h96/h192 不超过 `+1%`；Weather h720 late vs R.3 gap `<= +1%`；trace/grad 非 collapse |
| `artifacts` | `scripts/analyze_phase4_hssg_learnability_routing_gate.py`；`analysis/phase4_hssg_learnability_routing_gate_20260625/phase4_hssg_learnability_routing_gate_report.md` |
| `decision` | `fail_as_core_candidate`；不继续 sweep `aux_weight`、`top_ratio` 或 rank |

[Fact] HSSG-B/C absolute mean MSE 相对 `single_720_prefix_risk` 为 `+1.11%`，
相对 R.3 为 `+1.49%`，相对 HSSG-A 为 `+1.31%`。main MSE wins 分别只有
`3/8`、`3/8`、`2/8`，未达到 gate。

[Fact] early gate 仍失败：ETTh2 h96 相对 single-prefix 为 `+2.08%`，超过 `+1%`
阈值；h192 为 `-0.69%`。

[Fact] Weather late 明显退化：Weather h720 late segment `337-720` 相对 single-prefix
为 `+4.10%`，相对 R.3 为 `+6.12%`，相对 HSSG-A 为 `+5.95%`。这说明 learnability
mask 没有修复 HSSG-A 的核心问题，反而破坏了 HSSG-A 已经获得的 late partial gain。

[Audit] router 没有 collapse。Weather trace 中 mean learnable blocks 为 `1.42`，
noisy blocks 为 `1.86`，noisy suppression ratio 为 `0.465`，mean late active steps 为
`62.0`；region path residual 非零。但这些审计信号没有转化为 metric gain。

[Decision] 不继续在 target-set readout 旁增加 detached residual path。当前失败点不是
“没有 routing”，而是 routing target/carrier 错了：learnability proxy 将大量 pressure 放到
late region，但 detached low-rank region path 无法承载 Weather late structure。

[Rollback] 回 Step 5/6。下一步应停止做 `rank/dropout/aux/top_ratio` 调参，转向
training protocol / representation carrier：优先研究 prefix-risk stabilized base、
learning-rate/early-best calibration，或让 routing 更新 `condition_head/target_states` 的受控
子空间，而不是继续只更新小 residual head。

#### Phase4-PTC Plan：Protocol Calibration Gate

| Field | Content |
| --- | --- |
| `current_step` | Step 6/8：设计并运行 training protocol diagnostic |
| `problem` | Phase4 多个策略在 epoch `1-5` 达到 best validation，之后 train loss 下降但 validation drift；这会混淆 HSSG carrier 是否真的失败 |
| `existence_evidence` | HSSG-A/HSSG-B/C、RG-B、OP-A training logs 均显示 early-best pattern |
| `idea` | 不加新结构，只降低 learning rate，判断 early-best collapse 是否是 HSSG-A 牺牲 short/early 与 Weather instability 的主要原因 |
| `theory_check` | 如果 shared representation 被过快 optimization 破坏，较低 LR 应延后 best epoch、降低 drift，并给 HSSG-A region path 更稳定训练空间 |
| `design` | `single_720_prefix_risk`、`r3_prefix_risk`、`hssg_region_routed_readout`；LR `1e-4/5e-5/3e-5`；ETTh2 + Weather；seed `2021` |
| `gate` | best epoch 后移且 validation drift 下降；HSSG-A 相对 single-prefix 至少 `5/8` main wins；ETTh2 h96/h192 不超过 `+1%`；Weather h720 late 仍接近 R.3 |
| `artifacts` | `docs/code-explanation/phase4-protocol-calibration-gate.md`；`scripts/remote/run_phase4_protocol_calibration_gate.sh`；`analysis/phase4_protocol_calibration_gate_20260625/phase4_protocol_calibration_gate_report.md` |
| `decision` | Gate 已完成；lower LR 部分改善 trajectory，但不能修复 HSSG-A 对 R.3 的 Weather gap，回 Step 6 设计 richer carrier |

[Implementation Note] 该 gate 是 protocol diagnostic，不是新方法。它复用现有 HSSG runner，
只改变 `LEARNING_RATE` 和 output root；每个 LR 独立写入
`/home/yingch/exp_outputs/r-2026-fatst/phase4_protocol_calibration_gate/lr_*`。

#### Phase4-PTC Result：Protocol Helps, Carrier Still Fails

[Fact] 18 个 run 全部完成：`3 LR × 3 strategies × 2 datasets`。结果归档在
`artifacts/runs/phase4_protocol_calibration_gate`，分析报告在
`analysis/phase4_protocol_calibration_gate_20260625/phase4_protocol_calibration_gate_report.md`。

[Strong Evidence] 降低 LR 确实缓解了 early-best / drift：HSSG-A 的 mean best epoch 从
`1.5` 后移到 `6.5`，mean post-best validation drift 从 `14.86%` 降到 `6.56%`。
这说明 protocol bottleneck 是真实存在的，不能忽略。

[Strong Evidence] 但 protocol calibration 没有把 HSSG-A 修成主线候选。`3e-5` 下
HSSG-A 相对 single-prefix 达到 `6/8` MSE wins，但相对 R.3 只有 `4/8` wins；
Weather 上相对 R.3 是 `0/4` wins，Weather h720 比 R.3 差 `+3.37%`。`5e-5`
虽然是 HSSG-A 的最佳 mean MSE setting，但相对 R.3 只有 `2/8` wins。

[Decision] Phase4-PTC gate 不通过。不能继续把失败解释为“learning rate 没调好”，也不应
继续在 `hssg_region_routed_readout` 上 sweep `rank/dropout/aux/top_ratio`。下一步回
Step 6：设计能更新 `condition_head/target_states` 或 adapter 子空间的 carrier，让
supervision scheduling 决定 gradient 进入哪里，而不是只在 detached low-rank readout path
上承接难预测区域。

#### Phase4-SCC Plan：State/Condition Carrier Scheduling

| Field | Content |
| --- | --- |
| `current_step` | Step 6：重新设计 carrier，而不是继续修补 HSSG readout |
| `problem` | 当前 scheduling signal 有效但 carrier 太弱：loss / readout reweight 可以改变 exposure，却不能稳定改善 Weather/R.3 |
| `existence_evidence` | PTC 表明 lower LR 改善 drift；HSSG-A 在 ETTh2 可赢，但 Weather 相对 R.3 全输，说明瓶颈不是有没有 signal，而是 signal 更新的位置 |
| `idea` | Horizon-agnostic supervision scheduling should decide where gradient updates: shared state, condition head, or lightweight adapter, not only how much loss a future unit receives |
| `theory_check` | 若难预测 future unit 的梯度直接压 shared backbone 会损害可预测结构，则应把该梯度约束到可控 state/condition 子空间；这比纯 loss downweight 更接近 SRP-style “selective update / protect useful representation” 思路 |
| `design` | 先做最小 carrier gate：冻结 backbone 主干，只开放 `condition_head/target_states` 或小 adapter；对 high-risk future units 执行 gated gradient routing；保留 R.3/single 作为 controls |
| `gate` | Weather 相对 R.3 至少不劣于 `+0.5%` mean MSE；ETTh2 不牺牲 h96/h192；training drift 不高于 PTC `3e-5` HSSG-A；必须给出 gradient/update-path 诊断 |
| `artifacts` | `docs/code-explanation/phase4-scc-condition-carrier.md`；`scripts/remote/run_phase4_scc_condition_carrier_gate.sh`；`scripts/analyze_phase4_scc_condition_carrier_gate.py` |
| `decision` | 若 SCC gate 仍不能接近 R.3，则回 Step 2/3 重估 Phase4 主问题是否应从 scheduling 转向 architecture/pretraining |

#### Phase4-SCC-E1 Experiment Plan：Condition Carrier Gate

[Purpose] 下一步实验不再问 “哪个 horizon / region 应该加权更多”，而是问：
**同一个 horizon-agnostic scheduling signal，如果允许它更新更靠近主预测路径的
`condition_head / target_states` carrier，是否能在 official validation selection 下超过或接近 R.3？**

[Rationale] 现有证据链支持该实验，但也要求它保持克制：

1. [Strong Evidence] PTC 表明 lower LR 可以改善 trajectory，因此后续 gate 采用更稳的
   `5e-5` 为主 LR，并保留 `3e-5` 作为 sensitivity check；
2. [Strong Evidence] RG-A/RG-B/OP-A 证明 output-side tiny adapter 或 adapter-only finetune
   不是足够 carrier；
3. [Strong Evidence] HSSG-A 证明 region/readout routing 在 ETTh2 有信号，但 Weather 对 R.3
   仍失败，说明更新位置太晚或 capacity 太弱；
4. [Hypothesis] 若 Weather 的 long/late structure 需要改变 `gamma/beta` 或 target-state
   semantics，而不是只叠 residual，那么 condition/state carrier 应比 detached readout/adapter
   更有效。

**Minimal Gate Matrix**

| Group | Strategy | Purpose | LR | Dataset | Seed |
| --- | --- | --- | --- | --- | --- |
| Control | `single_720_prefix_risk` | 当前 horizon-agnostic single-prefix control | `5e-5` | ETTh2, Weather | 2021 |
| Control | `r3_prefix_risk` | 当前 strongest internal baseline | `5e-5` | ETTh2, Weather | 2021 |
| Negative carrier control | `dynamic_residual_stability_routing` | 复核同一 routing signal 在 adapter carrier 上的上限 | `5e-5` | ETTh2, Weather | 2021 |
| Candidate A | `scc_condition_delta_detached` | routing signal 只更新 zero-init condition-delta head，不更新 shared backbone | `5e-5` | ETTh2, Weather | 2021 |
| Candidate B | `scc_condition_delta_state_open` | routing signal 更新 condition-delta head + target-state path 的受控子空间 | `5e-5` | ETTh2, Weather | 2021 |

[Design Details]

- `condition_delta` 作用位置：`target_states -> condition_head -> gamma/beta -> conditioned`
  之间，优先做 zero-init low-rank 或 small MLP delta，避免初始破坏 base prediction；
- `scc_condition_delta_detached`：`target_states.detach()` 进入 condition delta，只测试 carrier
  capacity，不让 auxiliary gradient 污染 state generator；
- `scc_condition_delta_state_open`：允许 auxiliary gradient 进入 target-state / condition path，
  但冻结或弱化 encoder/history path，避免 noisy future unit 直接改 shared history representation；
- routing signal 复用 residual-stability / predictability 逻辑：`learnable_conflict` route 到
  condition carrier，`noisy_conflict` 不提供额外 positive pressure；
- main loss 仍保留 official prediction loss，condition carrier 只承担 gated auxiliary pressure；
- 初始化必须保证 `condition_delta=0` 时等价于 base，避免把收益写成参数量增加。

[Validation / Checkpoint Diagnostics]

Official model selection 继续使用当前 `val_mean_mse = mean(h96,h192,h336,h720)`，这是主结果。
但 E1 必须额外记录以下 diagnostic views：

- `best_epoch_val_mean`：官方 checkpoint；
- `best_epoch_short_mean = mean(h96,h192)`；
- `best_epoch_long_mean = mean(h336,h720)`；
- `best_epoch_h720`；
- official checkpoint 与 long/h720 oracle epoch 的 validation gap。

[Gate]

E1 只在以下条件同时满足时进入扩展实验：

1. official checkpoint 下，Candidate A 或 B 相对 R.3 的 mean MSE 不劣于 `+0.5%`；
2. Weather 相对 R.3 至少 `2/4` horizon wins，且 h720 不劣于 `+0.5%`；
3. ETTh2 h96/h192 不劣于 R.3 超过 `+1.0%`；
4. post-best validation drift 不高于 PTC `3e-5` HSSG-A 的 `6.56%` 太多，目标 `<8%`；
5. trace 必须证明 carrier 被实际使用：condition delta norm 非零但不过度主导，learnable/noisy
   route 分布非 collapse。

[Expansion If Pass]

- 加 `3e-5` LR sensitivity；
- 加 `ETTm1` 或 `ETTm2`，优先选择 Weather-like failure 是否复现；
- 加 seed `2022`；
- 若 Candidate B 赢 Candidate A，进一步做 update-path ablation：open target-state only /
  open condition-head only / open both。

[Rollback If Fail]

若 Candidate A/B 在 official checkpoint 下都不能接近 R.3，但 `best_epoch_h720` 有明显收益，
说明机制信号存在但 checkpoint selection 与主目标冲突；回 Step 5 研究 selection-sensitive
training / regularization。若 official 与 oracle views 都无收益，则回 Step 2/3，重新判断 Phase4
是否应从 supervision scheduling 转向 future-aware architecture 或 pretraining，而不是继续 carrier
局部修补。

#### Phase4-SCC-E1 Result：Carrier Signal Exists, R.3 Gate Fails

[Fact] 10 个 run 全部完成：`5 strategies × 2 datasets`，LR `5e-5`，seed `2021`。
结果归档在 `artifacts/runs/phase4_scc_condition_carrier_gate`，分析报告在
`analysis/phase4_scc_condition_carrier_gate_20260626/phase4_scc_condition_carrier_gate_report.md`。

[Strong Evidence] SCC carrier 确实被使用：`condition_delta_mean_abs_residual` 非零，
Weather 上 detached/state-open 分别达到 `0.1128/0.1202`，且
`train_condition_delta_grad_norm` 约 `0.05-0.06`。routing 也没有 collapse：
Weather mean learnable/noisy blocks 约 `1.41/1.86`，与 dynamic residual-stability control
接近。

[Strong Evidence] 但 SCC-E1 未通过 R.3 gate。相对 R.3：

- `scc_condition_delta_detached`：ETTh2 `+2.65%`，Weather `+1.96%`，均为 `0/4` wins；
- `scc_condition_delta_state_open`：ETTh2 `+2.83%`，Weather `+0.68%`，Weather 仅 h720
  `1/4` win；
- checkpoint diagnostics 显示 official vs long/h720 oracle gap 很小，Weather state-open
  h720 oracle gap 约 `+0.59%`，不足以解释整体失败。

[Inference] 更新位置上移到 condition/state carrier 比 adapter carrier 更强：state-open 在 Weather
相对 dynamic residual-stability control 是 `4/4` wins、mean MSE `-3.28%`。但它仍没有超过 R.3，
说明当前 bottleneck 不是简单的 “carrier 太晚/太弱” 可以单独解决。

[Decision] SCC-E1 fail as core route。停止继续 sweep `aux_weight/top_ratio/condition_delta_size`。
回 Step 2/3 重估 Phase4：如果仍坚持 HSS 叙事，需要转向更强的 future-aware representation /
pretraining；否则应考虑把 R.3/prefix-risk 作为强 baseline 机制分析，而不是继续局部 stacking。

### Phase4-FSA：Future-State Anchored HSS 方向重设

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3 -> Step 4/6：重新判断 Phase4 的真实问题，并设计可证伪的下一轮实验 |
| `problem` | Phase4 已证明 supervision signal 会改变训练结果，但 loss-only、readout routing、adapter routing、condition/state carrier 都没有稳定超过 R.3；核心瓶颈可能不是“如何给 future unit 加权”，而是当前 `target_states` 缺少可承接 scheduling pressure 的 future-structured representation |
| `existence_evidence` | S1/S2 相对 `full_time_mse` 有收益但 Weather vs R.3 collapse；HSSG-A 有 late/long partial gain 但牺牲 early；HSSG-B/C 与 SCC-E1 routing/carrier 非 collapse 却仍输 R.3；OP-A 改变 training dynamics 但 full-time stabilized base 全面失败；R3D 证明 `single_720_prefix_risk` 是比 `full_time_mse` 更强的 h720-only base |
| `idea` | 将 HSS 从“直接调度 raw future-unit loss / gradient”升级为 **Future-State Anchored HSS**：先用 training-only future teacher 把 `target_states` 锚定到可解释的 future latent manifold，再让 supervision scheduling 决定哪些 future units 如何塑造这个 manifold |
| `theory_check` | 若 future regions 的可预测性、噪声和长期依赖结构不同，raw loss pressure 直接压 shared state 会产生 conflict；但如果 state space 先具有 future-aware geometry，HSS 的 gradient/schedule 才有稳定落点。SRP-style pretrain 的可吸收点不是复制架构，而是“selective update 需要 stabilized representation”这一 protocol claim |
| `design` | 第一轮只做 substrate diagnostic，不叠加新的 MoE/scheduler：在 `single_720_prefix_risk` 和 `r3_prefix_risk` 上加入轻量 future-state alignment，并与各自 base 比较；`full_time_mse + future alignment` 只作为 weak control，不作为主线 |
| `gate` | `r3_prefix_risk + future alignment` 相对 R.3 mean MSE 不劣于 `+0.3%`，并在 Weather h720 或 long mean 有改善；`single_720_prefix_risk + future alignment` 相对 single-prefix 至少 `5/8` main wins；future alignment stats 不能 collapse，且 checkpoint oracle gap 不能成为唯一收益来源 |
| `artifacts` | `scripts/remote/run_phase4_future_state_anchor_gate.sh`、`scripts/remote/check_phase4_future_state_anchor_progress.sh`、`scripts/sync_phase4_future_state_anchor_results.sh`、`scripts/analyze_phase4_future_state_anchor_gate.py`、`docs/code-explanation/phase4-future-state-anchored-hss.md` |
| `decision` | 当前决策是进入 Phase4-FSA-F1 substrate diagnostic；在 F1 通过前，不继续做 SCC/HSSG 参数 sweep，也不把 R.3 包装成最终贡献 |

#### 为什么转向 Future-State Anchored HSS

[Fact] Phase4 已经排除了几个局部解释：

- 不是简单的 `full_time_mse` 弱 baseline 问题：S1/S2/RG-B 能相对 full-time 改善，但无法跨过 R.3；
- 不是 routing 没有发生：RG-B、HSSG、SCC trace 都显示 adapter/readout/condition carrier 被使用；
- 不是只由 validation checkpoint 选择造成：SCC-E1 official vs long/h720 oracle gap 很小；
- 不是只要 pretraining 就能解决：OP-A 的 full-time pretrain + adapter-only finetune 改善了 drift，却全面输给 pretrain 与 R.3；
- 不是 R.3 只有简单 step 权重：R3D 证明它是 mixed-horizon exposure + prefix-risk pressure 的 compound protocol。

[Inference] 因此下一步不应继续问“哪个 future block 应该加更大/更小 loss”，而应问：

> 当前 model state 是否有足够的 future structure，让不同 future-unit supervision 能以可泛化方式更新它？

[Hypothesis] 如果 `target_states` 本身只是 current-history representation 的弱读出，那么 HSS 的
loss/routing signal 会变成不稳定的局部 bias：ETTh2 late 可能受益，Weather early/late 可能被破坏。
如果先用 future teacher 建立 future-state geometry，再施加 prefix-risk 或 HSS pressure，
模型可能更容易把 supervision signal 转化为统一多 horizon 的有效表示。

#### Phase4-FSA-F1 最小实验设计

[Design Principle] F1 只验证 representation substrate，不同时引入新的 scheduler、MoE 或复杂
gradient mask。它回答“future-state anchor 是否让现有强 base 更强或至少不坏”。

实验矩阵：

| Arm | Strategy | Future alignment | 角色 |
| --- | --- | --- | --- |
| `F1-C0` | `single_720_prefix_risk` | off | h720-only clean HSS base |
| `F1-C1` | `r3_prefix_risk` | off | compound strong reference |
| `F1-A0` | `single_720_prefix_risk` | on | 测试 h720-only prefix pressure 是否受益于 future-state anchor |
| `F1-A1` | `r3_prefix_risk` | on | 测试最强 reference 是否仍能被 future-state anchor 改善 |
| `F1-W0` | `full_time_mse` | on | weak control；只判断 full-time base 是否仍不适合作为 anchor |

建议默认配置：

- datasets: `ETTh2`, `Weather`;
- seed: `2021`;
- LR: 先沿用 PTC/SCC 稳定设置 `5e-5`，不做 LR sweep；
- `future_teacher_layers=1`;
- `future_align_weight` 使用小权重起步；
- `future_relation_weight=0` 作为第一轮，避免 relation loss 与 prefix-risk 同时引入过强约束；
- `future_recon_weight` 使用小权重，只保证 teacher 不退化成任意 latent；
- `future_recon_normalization=target_energy`;
- `future_align_weighting=reconstruction_confidence`;
- validation checkpoint 仍使用 official `val_mean_mse`，同时保留 h720/long oracle diagnostics。

必须记录的诊断：

- main MSE/MAE vs own base、vs R.3；
- h720 segment MSE，尤其 Weather `337-720`；
- `future_alignment_stats.csv` 中的 local alignment、reconstruction loss、confidence mean/min/max；
- `checkpoint_selection_diagnostics.csv`，判断收益是否只是 checkpoint selection artifact；
- training drift 与 best epoch，判断 future anchor 是否缓解或加剧 early-best pattern。

通过门槛：

1. `F1-A1` 相对 R.3 mean MSE 不劣于 `+0.3%`，且 Weather h720 或 Weather long mean 至少一项改善；
2. `F1-A0` 相对 `single_720_prefix_risk` 至少 `5/8` main MSE wins，或 mean MSE 改善且不牺牲 ETTh2 h96/h192 超过 `+1%`；
3. future teacher diagnostics 非 collapse：confidence 不应接近全 floor，reconstruction loss 有限，alignment loss 下降不能与 MSE 明显反向；
4. 若 official checkpoint 无收益但 h720/long oracle 有一致收益，下一步先研究 validation selection，而不是继续加结构；
5. 若 `F1-W0` 仍弱于 prefix/R.3 bases，则继续否定 full-time pretrain 作为主 base。

回滚规则：

- 若 `F1-A0/A1` 均输给各自 base，且 future diagnostics 无有效 alignment，回 Step 2/3：当前 Phase4
  不应继续以 HSS 为主线，应转向 R.3 compound protocol 的机制分析或更大 architecture redesign；
- 若 `F1-A1` 接近或超过 R.3，但 `F1-A0` 不行，说明 future-state anchor 需要 mixed exposure
  作为 training support，HSS 叙事必须诚实地包含 compound exposure control；
- 若 `F1-A0` 有收益而 `F1-A1` 无收益，说明 clean h720-only HSS substrate 成立，下一步进入
  Phase4-FSA-F2：在 anchored state 上重新测试 gradient routing/scheduling；
- 若只有 oracle checkpoint 有收益，则先回 Step 6 修 validation metric，而不是修改 model。

[Implementation Status] F1 的 remote runner、progress checker、sync wrapper、analysis script
和代码说明已完成。下一步进入 Step 8：在 `529_Lab-3090` 按 F1 matrix 运行远程训练。

[Scheduling Note] 后续 FSA/Future Phase4 runner 必须避免 `Weather + ETTh2` 成对等待的
`arm_major` 调度。Weather 明显慢于 ETTh2，默认采用 `dataset_major` 或其他 workload-aware
排队方式，先把 Weather jobs 分散到可用 GPU，再填充较快数据集，减少 GPU 空等。

#### Phase4-FSA-F1 Result：Anchor Signal Exists, Not A Core Substrate Yet

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 F1 remote artifacts，并决定是否进入 FSA-F2 |
| `problem` | F1 要判断 future-state anchor 是否能作为 HSS 的稳定 representation substrate |
| `existence_evidence` | 10 个 run 完成：`5 arms × 2 datasets`；remote root `/home/yingch/exp_outputs/r-2026-fatst/phase4_future_state_anchor_gate`；local analysis `analysis/phase4_future_state_anchor_gate_20260626/` |
| `idea` | 若 substrate 成立，A0 应稳定优于 `single_720_prefix_risk` 且不伤 Weather long；A1 应接近或优于 R.3 且不伤 ETTh2 |
| `theory_check` | 有效 anchor 应产生非 collapse alignment diagnostics，同时把局部 future structure 转化为 main MSE 改善，而不是只改善某些 dataset/region |
| `design` | 比较 `F1-A0` vs `F1-C0`、`F1-A1` vs `F1-C1`，并检查 Weather h720 `337-720`、future alignment stats、checkpoint oracle gap |
| `gate` | 更严格的 ex-post gate：不能只看 mean MSE；必须同时满足 dataset split 与 Weather late no-harm。若只出现局部收益，不能直接叠加 HSSG/SCC |
| `artifacts` | `analysis/phase4_future_state_anchor_gate_20260626/phase4_future_state_anchor_gate_report.md` |
| `decision` | `partial_pass_anchor_signal_but_not_core_substrate`；回 Step 5/6 做 anchor pressure / confidence calibration，不直接进入 HSSG-on-anchor |

[Fact] `F1-A0 = single_720_prefix_risk + future anchor` 相对 `F1-C0`：

- overall: `4/8` MSE wins，mean relative MSE `-0.34%`；
- ETTh2: `3/4` wins，mean relative MSE `-1.45%`；
- Weather: `1/4` wins，mean relative MSE `+0.78%`；
- Weather h720 late `337-720`: `+2.91%`，说明 clean h720-only anchor 会伤害当前最关键的
  Weather long/late region。

[Fact] `F1-A1 = r3_prefix_risk + future anchor` 相对 `F1-C1/R.3`：

- overall: `2/8` MSE wins，mean relative MSE `+1.62%`；
- ETTh2: `0/4` wins，mean relative MSE `+3.31%`；
- Weather: `2/4` wins，mean relative MSE `-0.06%`；
- Weather h720 late `337-720`: `-1.24%`，说明 mixed exposure + future anchor 确实能改善
  Weather long/late，但代价是 ETTh2 全面退化。

[Fact] `F1-W0 = full_time_mse + future anchor` 继续失败：相对 R.3 `0/8` wins、mean MSE
`+2.53%`；相对 single-prefix mean MSE `+1.09%`。这继续否定 full-time base 作为
paper-core anchor。

[Strong Evidence] future teacher 没有 leakage，也不是完全 collapse：

- max `prediction_leakage_max_abs = 0`；
- min mean confidence `0.390767`；
- Weather 的 teacher/student cosine 高于 ETTh2；
- 但所有 future-anchor arms 的 min raw confidence 都触达 `0.05` floor，说明至少部分 segments
  仍被 floor 强制保留 alignment pressure。

[Inference] F1 证明 future-state anchor 是一个真实机制轴，但还不是稳定 substrate。它的收益依赖
base/exposure：

- h720-only prefix anchor 更适合 ETTh2 和 early/prefix；
- R.3/mixed-exposure anchor 更适合 Weather middle/late；
- 同一个全局 anchor pressure 无法同时满足 ETTh2 short/prefix 与 Weather long/late。

[Decision] 不进入 “anchored state + HSSG/SCC” stacking。下一步回 Step 5/6：
先做 anchor pressure calibration。这里的目标不是增加 diagnostic 工作量，而是用最少实验判断
F1 的副作用来自 low-confidence forced alignment、anchor pressure 过强，还是 future anchor
本身不适合当前 carrier。

#### Phase4-FSA Diagnostic 使用原则

[Decision] diagnostic artifacts 只服务于机制失效定位，不作为研究主线本身。后续实验前必须先问：
**这个中间数据是否会改变 pass/fail 判断或 rollback 点？** 如果不会，不新增保存项。

默认保留的轻量诊断：

- `metrics_by_target_horizon.csv` 与 h720 `metrics_by_segment.csv`：主判断与 Weather late gate；
- `training_log.csv`：best epoch、drift、auxiliary loss 是否异常；
- `checkpoint_selection_diagnostics.csv`：排除 validation metric artifact；
- `future_alignment_stats.csv`：仅在 future-anchor 实验中检查 leakage、confidence、teacher/student
  是否 collapse；
- `effective_config.json`：确认实验变量确实只改了预期项。

默认不扩大保存的诊断：

- 不保存 checkpoint 或 `predictions_test.npz`，除非后续需要做 residual/projection audit；
- 不把 `target_state_similarity`、`target_conditioning_stats`、`objective_weight_stats` 作为每次主报告
  的必选项。只有当 main metrics 的失败原因不清楚时才补充读取；
- 不因为 diagnostic 有可解释差异就推进方法。method gate 仍由 main/segment performance 和
  paper-story 共同决定。

#### Phase4-FSA-F2 Plan：Anchor Pressure Calibration Gate

| Field | Content |
| --- | --- |
| `current_step` | Step 5/6：基于 F1 结果重新检查 future anchor 理论可行性，并设计最小可证伪实验 |
| `problem` | F1 的 future anchor 有局部正信号，但不是稳定 substrate：A0 改善 ETTh2 却伤 Weather late；A1 改善 Weather late 却伤 ETTh2 |
| `existence_evidence` | F1 中 future alignment 非 collapse 且无 leakage；min raw confidence 触达 `0.05` floor；A0/A1 的收益随 base/exposure 改变 |
| `idea` | 不直接叠加 HSSG/SCC，而先校准 anchor pressure：判断 conflict 是来自 low-confidence units 被强制对齐，还是来自 dense/global anchor pressure 过强 |
| `theory_check` | 如果低置信 future units 主要贡献噪声，取消 confidence floor 应减少 Weather late 与 ETTh2 damage；如果全局 pressure 过强，降低 `future_align_weight` 才会缓解；若二者都无效，future anchor 不是当前 HSS substrate |
| `design` | Stage F2A 只跑 `future_confidence_floor=0.0`，复用 F1 controls；若 F2A 只部分改善，再进入 F2B 降低 `future_align_weight`，否则不 sweep |
| `gate` | F2A 需要 `single_prefix + selective anchor` 相对 F1-C0 mean MSE `<0` 且 Weather h720 late `<=+0.5%`；或 `R3 + selective anchor` 相对 F1-C1 ETTh2 damage `<=+1.0%` 且保留 Weather h720 late gain |
| `artifacts` | `scripts/remote/run_phase4_fsa_f2_anchor_pressure_gate.sh`、`scripts/remote/check_phase4_fsa_f2_anchor_pressure_progress.sh`、`scripts/sync_phase4_fsa_f2_anchor_pressure_results.sh`、`scripts/analyze_phase4_fsa_f2_anchor_pressure_gate.py` |
| `decision` | F2A 已实现，下一步进入 Step 8 remote training；F2A 通过才进入 anchored-state HSS；F2A 失败则不做 HSSG-on-anchor，回 Step 2/3 重新考虑 R.3 compound protocol 或更大 representation redesign |

F2A 最小矩阵：

| Arm | Base | Changed factor | Purpose |
| --- | --- | --- | --- |
| `F2-A0` | `single_720_prefix_risk` | `future_confidence_floor=0.0` | 检查 A0 的 Weather late damage 是否来自 low-confidence forced alignment |
| `F2-A1` | `r3_prefix_risk` | `future_confidence_floor=0.0` | 检查 A1 的 ETTh2 damage 是否来自 low-confidence forced alignment |

复用 F1 controls：

- `F1-C0` 作为 `single_720_prefix_risk` baseline；
- `F1-C1` 作为 R.3 baseline；
- `F1-A0/F1-A1` 作为 floor `0.05` 的 direct comparison。

F2B 仅在 F2A 给出接近通过但仍有轻微副作用时启动：

- candidate factor: `future_align_weight=0.003` 或 `0.005`；
- 不同时改 `future_recon_weight`、`future_relation_weight` 或 architecture；
- 如果 F2B 仍不能同时满足 Weather late 与 ETTh2 no-harm，则停止 future-anchor calibration。

#### Phase4-FSA-F2 Result：Confidence Floor Is Not The Failure Source

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 F2A remote artifacts，并决定是否继续 future-anchor calibration |
| `problem` | F2A 要判断 F1 的副作用是否来自 low-confidence future units 被 `future_confidence_floor=0.05` 强制保留 alignment pressure |
| `existence_evidence` | F2A 四个 run 完成：`2 arms × 2 datasets`；remote root `/home/yingch/exp_outputs/r-2026-fatst/phase4_fsa_f2_anchor_pressure_gate`；local analysis `analysis/phase4_fsa_f2_anchor_pressure_gate_20260626/` |
| `idea` | 若 low-confidence forced alignment 是主因，取消 floor 后应明显缓解 A0 的 Weather late damage，或缓解 A1 的 ETTh2 damage |
| `theory_check` | 代码中 `alignment_confidence` 进入 `local_alignment_loss = sum(local_alignment * confidence) / sum(confidence)`；floor 只改变相对权重，且 `future_align_weight=0.01` 较小。因此如果 floor 改动后指标几乎不变，说明主矛盾不在 floor，而在 future-anchor carrier/pressure 的整体梯度方向 |
| `design` | 只改 `future_confidence_floor=0.0`，其余 future anchor 设置保持 F1：`future_align_weight=0.01`、`future_recon_weight=0.001`、`future_relation_weight=0` |
| `gate` | A0 需相对 F1-C0 mean MSE `<0` 且 Weather h720 late `<=+0.5%`；或 A1 需相对 F1-C1 ETTh2 damage `<=+1.0%` 且保留 Weather h720 late gain |
| `artifacts` | `analysis/phase4_fsa_f2_anchor_pressure_gate_20260626/phase4_fsa_f2_anchor_pressure_gate_report.md` |
| `decision` | `fail_stop_future_anchor_stacking`；F2A 没有接近通过，不启动 F2B weight sweep；回 Step 2/3，重新设计 Phase4 主线 carrier，而不是继续在 future anchor 上叠 HSSG/SCC |

[Fact] `F2-A0 = single_720_prefix_risk + floor0 anchor` 相对 `F1-C0`：

- overall: `4/8` MSE wins，mean relative MSE `-0.34%`；
- ETTh2: `3/4` wins，mean relative MSE `-1.45%`；
- Weather: `1/4` wins，mean relative MSE `+0.77%`；
- Weather h720 late `337-720`: `+2.90%`，未解决 F1-A0 的核心副作用。

[Fact] `F2-A1 = r3_prefix_risk + floor0 anchor` 相对 `F1-C1/R.3`：

- overall: `2/8` MSE wins，mean relative MSE `+1.62%`；
- ETTh2: `0/4` wins，mean relative MSE `+3.31%`；
- Weather: `2/4` wins，mean relative MSE `-0.06%`；
- Weather h720 late `337-720`: `-1.24%`，保留了 Weather late 局部收益，但 ETTh2 damage 完全没有缓解。

[Strong Evidence] F2A 与 F1-A direct comparison 几乎完全重合：

- `F2-A0` vs `F1-A0`: mean relative MSE 约 `-0.00%`；
- `F2-A1` vs `F1-A1`: mean relative MSE 约 `+0.00%`；
- F2A config 确认 `future_confidence_floor=0.0`，不是实验配置错误；
- mean alignment confidence 只从 F1 的约 `0.391/0.623/0.465/0.704`
  变为 F2 的约 `0.389/0.620/0.463/0.701`。

[Inference] F1 的失败不主要来自 low-confidence forced alignment。更合理的解释是：当前 future
teacher anchor 作为 auxiliary representation pressure，无法稳定地产生与 main forecasting
objective 一致的梯度方向；它对 Weather late 有局部帮助，但对 ETTh2 和部分 Weather horizon
产生 conflict。继续调小 `future_align_weight` 更可能只是把机制淡化，而不是形成有叙事潜力的
Horizon-agnostic supervision scheduling。

[Decision] 停止 future-anchor stacking。下一步不做 F2B，不进入 anchored-state HSSG/SCC。Phase4
应回到 Step 2/3：重新定义真正的 HSS 主线 carrier。优先考虑从 supervision/gradient allocation
本身出发，而不是把 future-state auxiliary branch 作为核心 substrate。

### Phase5：TimeAlign Carrier Gate for HSS

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3/6/7/8：重新定义 HSS carrier，并先做最小 carrier gate |
| `problem` | Phase4 future-anchor 在当前 target-set decoder 上不稳定；但 TimeAlign 原论文证明 future reconstruction alignment 在 fixed-horizon forecasting 中有效。因此需要判断失败来自 TimeAlign 机制本身，还是来自我们把机制弱接到错误 carrier 上 |
| `existence_evidence` | TimeAlign 原文和官方代码支持 training-only future reconstruction branch、local/global alignment、stop-gradient teacher；Phase4-F1/F2 证明弱 auxiliary future anchor 不能作为 HSS substrate |
| `idea` | 不继续修补 Phase4 anchor，而是建立 TimeAlign-style carrier gate：先验证 fixed-horizon TimeAlign carrier 是否可用，再验证 unified-720 是否相对 fixed-horizon 出现 multi-horizon gap |
| `theory_check` | 若 fixed TimeAlign 强而 unified-720 退化，则 HSS 的问题支点是多个 future distributions 在共享模型中的 supervision conflict；若 fixed TimeAlign 弱，carrier 不成立；若 unified 不退化，则 HSS 缺少必要性 |
| `design` | `3 datasets × (4 fixed horizons + 1 unified-720)`；默认数据集为 `Weather/ETTm2/ETTh2`；fixed runs 分别训练 `h96/h192/h336/h720`；unified run 训练 `pred_len=720` 并评估 `96/192/336/720` prefix |
| `gate` | 先看 fixed-horizon carrier 是否合理；再看 unified-vs-fixed mean relative MSE gap 是否明显为正。只有二者同时成立，才进入 TimeAlign-based HSS schedule/gradient allocation 设计 |
| `artifacts` | `baselines/timealign_carrier/`、`scripts/remote/run_phase5_timealign_carrier_gate.sh`、`scripts/remote/check_phase5_timealign_carrier_progress.sh`、`scripts/sync_phase5_timealign_carrier_results.sh`、`scripts/analyze_phase5_timealign_carrier_gate.py`、`docs/code-explanation/phase5-timealign-carrier-gate.md` |
| `decision` | Phase5 carrier gate 已实现；下一步进入 remote training。通过后进入 Step 4/5 设计 TimeAlign-HSS；失败则回 Step 2/3 继续寻找非 TimeAlign carrier |

[Design Boundary] Phase5 不把 TimeAlign 直接包装成我们的贡献，也不声称 HSS 已经成立。它只回答：
TimeAlign 是否能提供一个比 Phase4 target-set future anchor 更合适的 HSS carrier。

最小实验矩阵：

| Run type | Run name | Training target | Evaluation target |
| --- | --- | --- | --- |
| fixed | `TimeAlignCarrierFixedH96` | `pred_len=96` | `h96` |
| fixed | `TimeAlignCarrierFixedH192` | `pred_len=192` | `h192` |
| fixed | `TimeAlignCarrierFixedH336` | `pred_len=336` | `h336` |
| fixed | `TimeAlignCarrierFixedH720` | `pred_len=720` | `h720` |
| unified | `TimeAlignCarrierUnified720` | `pred_len=720` | `h96/h192/h336/h720` |

[Dataset Choice] 第一轮默认加入 `ETTm2`，而不是同时扩到 `ETTm1+ETTm2`：

- `ETTh2` 在 TimeAlign 原论文中不一定是最强证据数据集，保留它作为 difficult counterexample；
- `Weather` 是 Phase4 中最关键的 long/late conflict 数据集，必须保留；
- `ETTm2` 提供 ETT minute-level 数据，能降低只看 `Weather+ETTh2` 导致 carrier gate 过早悲观的风险；
- `ETTm1` 暂作为可选扩展，若 `ETTm2` 与 `ETTh2` 结论冲突，再补跑 `DATASETS="ETTm1"`。

Phase5 的下一步判断：

- 若 fixed carrier 自身性能弱：停止 TimeAlign-HSS，先审计 source-faithfulness 或放弃该 carrier；
- 若 fixed carrier 可用且 unified gap 明显：进入 TimeAlign-HSS，研究 horizon-agnostic
  supervision scheduling 如何调度 future-distribution alignment pressure；
- 若 unified gap 不明显：TimeAlign 已天然适配 unified prefix evaluation，不强行做 HSS。

#### Phase5 Result：Carrier Viable, HSS Necessity Not Yet Proven

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 Phase5 carrier gate，并决定是否直接进入 TimeAlign-HSS |
| `problem` | 判断 TimeAlign 是否能作为 HSS carrier，以及 unified-720 是否相对 fixed-horizon 产生需要 HSS 修复的 gap |
| `existence_evidence` | 15 个 run 完成：`3 datasets × (4 fixed horizons + 1 unified-720)`；remote root `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_carrier_gate`；local analysis `analysis/phase5_timealign_carrier_gate_20260626/` |
| `idea` | 若 fixed TimeAlign strong 且 unified 明显退化，则进入 TimeAlign-HSS；否则先做 carrier/validation 诊断 |
| `theory_check` | 结果显示 TimeAlign unified-720 并未整体退化；ETTh2/Weather 上 unified 反而多处优于 fixed。h720 上 fixed/unified 训练目标相同，因此差异主要来自 validation/checkpoint selector |
| `design` | 比较 fixed vs unified，同时横向比较 ETTh2/Weather 上的 R.3 reference；检查 h720 segment 与 training/best epoch |
| `gate` | TimeAlign carrier 通过“可继续研究”的最低门槛；但 HSS necessity gate 未通过，因为没有稳定 unified degradation |
| `artifacts` | `analysis/phase5_timealign_carrier_gate_20260626/phase5_timealign_carrier_gate_report.md`、`analysis/phase5_timealign_carrier_gate_20260626/phase5_timealign_interpretation.md` |
| `decision` | `carrier_viable_but_hss_necessity_not_yet_proven`；不直接进入 TimeAlign-HSS，回 Step 5/6 做 Phase5-R1 validation selector 与 mechanism control |

[Fact] Unified-720 相对 fixed-horizon：

- ETTh2: `3/4` wins，mean relative MSE `-7.88%`；
- ETTm2: `0/4` wins，mean relative MSE `+1.47%`；
- Weather: `3/4` wins，mean relative MSE `-1.07%`；
- overall: `6/12` wins，mean relative MSE `-2.49%`。

[Fact] TimeAlign unified-720 相对 R.3 reference：

- ETTh2 h96/h192/h336 分别为 `-16.17%/-16.21%/-10.62%`，但 h720 为 `+5.86%`；
- Weather h96 为 `+1.60%`，h192/h336/h720 分别为 `-0.02%/-1.50%/-3.70%`。

[Inference] TimeAlign unified-720 是一个真实 carrier candidate，但它的 failure 不是
“unified 全局退化”，而是 selective failure：ETTh2 h720 与 Weather h96。HSS 若成立，
也应围绕 selective future-distribution conflict，而不是泛化地声称修复 unified multi-horizon。

[Strong Evidence] Weather h720 的 unified 优势主要是 validation/checkpoint selector：

- `TimeAlignCarrierFixedH720` 与 `TimeAlignCarrierUnified720` 在同一 dataset 上训练轨迹相同；
- Weather fixed h720 best epoch 是 `1`，unified-720 best epoch 是 `7`；
- unified h720 test MSE 相对 fixed h720 为 `-1.67%`。

[Decision] 下一步 Phase5-R1：

1. validation selector audit：分离 `val_h720`、`val_long_mean`、`val_all_mean` 对 test h720/all-horizon 的影响；
2. mechanism ablation：`w_align=0,w_recon=1`、`w_align=0.1,w_recon=0`、full TimeAlign；
3. 若 R1 证明 full TimeAlign alignment 是必要贡献，再进入 TimeAlign-HSS；否则先修 carrier 或放弃该路线。

#### Phase5-R0：Official TimeAlign Reproduction Reset

| Field | Content |
| --- | --- |
| `current_step` | Step 11 -> Step 1/6/7/8：从 local source-informed implementation 回滚到 official-source reproduction |
| `problem` | 前一版 `baselines/timealign_carrier` 的 fixed-horizon 结果与 TimeAlign 论文表现存在较大差距；如果差距来自 repo 实现、dataloader、官方 hyperparameter preset 或 checkpoint policy 错位，则后续 unified/fixed 和 HSS 判断都不可靠 |
| `existence_evidence` | 官方代码包含自己的 `data_provider`、dataset split、official scripts；并且官方 `EarlyStopping` 实际未执行 best-checkpoint 选择，`test()` 也不 reload checkpoint。这些实现细节足以改变 fixed-horizon 结果 |
| `idea` | 新建 `baselines/timealign_official/`，vendored 官方 TimeAlign 源码；只添加薄 repo adapter，不再复用本 repo 的 dataloader/model；先跑 source-faithful `official-last`，再把 `best-val` 作为 validation-selector diagnostic |
| `theory_check` | 若 official-source fixed-horizon 仍不能接近论文，问题优先是 reproduction/data/code-version audit，而不是 HSS 方法设计；若 official-source fixed-horizon 成立，再判断 unified-720 是否存在可叙事的 supervision conflict |
| `design` | `3 datasets × (4 fixed horizons + 1 unified-720)`；datasets 为 `Weather/ETTm2/ETTh2`；fixed runs 使用 dataset+horizon official preset；unified-720 使用 h720 official preset 并评估 `h96/h192/h336/h720` prefix；primary checkpoint policy 为 `official-last` |
| `gate` | Gate-1：fixed-horizon official reproduction 是否可信；Gate-2：unified-720 相对 fixed 是否存在稳定 gap；Gate-3：若 `official-last` 与 `best-val` 结论冲突，必须先解决 checkpoint protocol 再进入 HSS |
| `artifacts` | `baselines/timealign_official/`、`scripts/remote/run_phase5_timealign_official_gate.sh`、`scripts/remote/check_phase5_timealign_official_progress.sh`、`scripts/sync_phase5_timealign_official_results.sh`、`scripts/analyze_phase5_timealign_official_gate.py`、`docs/code-explanation/phase5-timealign-official-reproduction.md` |
| `decision` | Active route 切换为 official-source reproduction。暂不继续 Phase5-R1 ablation，直到 source-faithful fixed/unified 对比返回并完成分析 |

[Fact] 官方 `EarlyStopping` 的有效行为是 last-epoch checkpoint，而非 best validation checkpoint。作者在 GitHub issue #2 中确认，论文使用固定训练轮数后的 final model，理由是长时序预测中 validation/test 可能存在 distribution shift，validation-best 可能训练不足。因此该行为应视为 author-intended paper protocol，而不是 bug。

[Decision] 后续实验分成两个 protocol：

- `official-last`：primary，用于回答“当前 fixed-horizon 与论文差距是否来自我们的 repo implementation”；
- `best-val`：secondary，用于 validation-selector diagnostic，只检验 unified/fixed 结论是否依赖 checkpoint selector。

#### Phase5-R0 Result：Official Source Valid, Checkpoint Policy Blocks HSS Claim

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 official-source reproduction，并决定是否进入 HSS |
| `problem` | `official-last` 是作者确认的 paper protocol；但 unified/fixed 研究仍需检查结论是否依赖 checkpoint selector |
| `existence_evidence` | 完整矩阵 `3 datasets × (4 fixed + 1 unified)` 已完成；artifacts 在 `analysis/phase5_timealign_official_gate_20260626/` |
| `idea` | 先判定 official-source fixed carrier 是否可信，再把 checkpoint protocol 作为独立变量处理 |
| `theory_check` | official-source fixed 相对 repo-local fixed 在 11/12 个 setting 改善，说明 source/preset mismatch 确实存在；但 ETTh2 official-last 的 last-vs-best validation gap 高达 `+6.29%/+15.73%/+27.84%/+20.76%`，足以污染结论 |
| `design` | `official-last` fixed/unified 对比；同步读取 training summary 作为 checkpoint diagnostic |
| `gate` | Carrier source-faithfulness gate 通过；HSS necessity gate 暂停，必须先过 `best-val` validation-selector diagnostic |
| `artifacts` | `analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_gate_report.md`、`analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_interpretation.md` |
| `decision` | `official_source_carrier_valid_need_selector_sensitivity_check`；下一步运行 `CHECKPOINT_POLICY=best-val` 的同矩阵 validation-selector diagnostic |

[Fact] `official-last` unified-vs-fixed：

- ETTh2: `3/4` wins，mean relative MSE `-8.01%`；
- ETTm2: `0/4` wins，mean relative MSE `+3.72%`；
- Weather: `0/4` wins，mean relative MSE `+1.05%`；
- ALL: `3/12` wins，mean relative MSE `-1.08%`。

[Inference] 当前不能写成“unified 一定退化”，也不能写成“unified 没有问题”。更准确的判断是：
TimeAlign unified degradation 在 ETTm2/Weather 上存在，但 ETTh2 的反向结果需要用 validation-selector
diagnostic 验证是否只是 checkpoint artifact。

#### Phase5-R0B Result：Unified Behavior Is Dataset-Dependent, Not Selector-Driven

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 `best-val` validation-selector diagnostic |
| `problem` | `official-last` 是 paper-faithful protocol；需要判断 unified/fixed gap 是否依赖 last-epoch selector |
| `existence_evidence` | `best-val` 同矩阵 `3 datasets × (4 fixed + 1 unified)` 完成；artifacts 在 `analysis/phase5_timealign_official_bestval_20260626/` |
| `idea` | 若 `best-val` 改变 winner pattern，则 unified/fixed 结论是 selector-sensitive；若不改变，则 checkpoint policy 不是主因 |
| `theory_check` | `best-val` 与 `official-last` 的 dataset-level pattern 完全一致：ETTh2 `3/4` unified wins，ETTm2/Weather `0/4` unified wins |
| `design` | 比较 `official-last` 与 `best-val` 的 relative MSE gap、wins、dataset mean |
| `gate` | 若 ETTm2/Weather 在两种 selector 下仍稳定退化，则存在 TimeAlign unified pressure 的数据集依赖性；若 ETTh2 在两种 selector 下仍稳定受益，则不能建立“统一退化”的 HSS 叙事 |
| `artifacts` | `analysis/phase5_timealign_official_bestval_20260626/phase5_timealign_official_gate_report.md`、`analysis/phase5_timealign_official_bestval_20260626/phase5_timealign_selector_comparison.md` |
| `decision` | `dataset_dependent_unified_behavior_not_global_hss_gap`；不直接进入通用 HSS 方法设计，下一步应回 Step 2/3 重新定义 TimeAlign-HSS 的问题支点，或做 look-back horizon sweep 先对齐论文复现口径 |

[Fact] `best-val` unified-vs-fixed：

- ETTh2: `3/4` wins，mean relative MSE `-8.38%`；
- ETTm2: `0/4` wins，mean relative MSE `+3.68%`；
- Weather: `0/4` wins，mean relative MSE `+1.15%`；
- ALL: `3/12` wins，mean relative MSE `-1.18%`。

[Strong Evidence] Winner pattern 与 `official-last` 完全一致；checkpoint selector 不是造成
ETTh2 positive、ETTm2/Weather negative 分裂的主因。

[Decision] 当前 TimeAlign-HSS 不能写成“unified multi-horizon 普遍退化，需要 HSS 修复”。当前 active route 不再做论文级 look-back reproduction，而是将 TimeAlign 作为 strong carrier，回 Step 2/3 重新定义 HSS 如何进入该 carrier。

1. Active：回 Step 2/3，把研究问题改成 dataset/state-dependent future alignment scheduling，解释为什么某些数据集 unified 受益、某些数据集 unified 受损；
2. Backlog：look-back horizon sweep 只在需要论文表格级复现对齐时再做，不作为当前 HSS 主线前置条件。

#### Phase5-HSS：Integrating TimeAlign into Horizon Supervision Scheduling

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3/4/5/6：重新定义 TimeAlign-HSS 的研究问题与第一轮方法 |
| `problem` | TimeAlign 已证明是 strong carrier；`official-last` 与 `best-val` 都显示 unified behavior 是 dataset-dependent：ETTh2 unified 受益，ETTm2/Weather unified 受损。因此 HSS 不应叙述为修复“全局 unified degradation”，而应叙述为调度 future supervision 在不同 future states/units 上的作用强度与梯度路径 |
| `existence_evidence` | `official-last` 与 `best-val` winner pattern 完全一致：ETTh2 `3/4` unified wins，ETTm2/Weather `0/4` unified wins；checkpoint selector 不是主因；official-source fixed 相对 repo-local fixed 在 11/12 个 setting 改善 |
| `idea` | 把 TimeAlign 从 baseline 升级为 HSS carrier：HSS 不重新定义预测架构，而是在 TimeAlign 的 future reconstruction/alignment supervision 上做 horizon-agnostic scheduling，决定哪些 future units 提供监督、监督进入哪个 branch、何时削弱或释放 alignment gradient |
| `theory_check` | TimeAlign 的 future branch 在 training-only 读取 ground-truth future，alignment loss 将 future distribution pressure 传回 history branch；当 future units 可对齐且可预测时，这种 pressure 可提升 unified 表示；当 future units 噪声大、状态不稳定或与当前 history 表示冲突时，static full-future alignment 会产生 harmful supervision。HSS 的理论支点是 supervision reliability，而不是 benchmark horizon label |
| `design` | 第一轮不直接堆复杂结构，先做 head/interface confounder diagnostic，再做 supervision diagnostic + minimal scheduling：D0 检查 fixed `pred_len=720` head 是否造成短 prefix 监督不足；D1 诊断 future unit 的 alignability/reconstruction difficulty/residual volatility 与 unified degradation 的关系；M1 在 TimeAlign alignment/reconstruction loss 上加入 unit-level reliability schedule；M2 只改变 alignment gradient path，验证“where gradient is allowed to update”是否比 loss reweight 更有叙事与性能潜力 |
| `gate` | 必须同时满足：ETTm2/Weather unified gap 明显缩小；ETTh2 不丢失 unified benefit；机制证据显示 schedule 不是简单降低 loss，而是区分 useful vs harmful future supervision；若只改善一个 dataset 或只靠淡化 TimeAlign loss，回 Step 4/5 重设 idea |
| `artifacts` | 新建 `docs/experiments/phase5-timealign-hss-integration.md`；后续代码应在 `baselines/timealign_official/` 的 adapter 层增加最小 diagnostic/scheduling，不改官方 vendored forward 作为对照 |
| `decision` | `active_timealign_hss_integration`；下一步先运行 D0 head/interface diagnostic，再视结果进入 D1/M1/M2，不做 look-back sweep |

[Narrative Anchor] 论文主线应从 “TimeAlign 很强，所以直接改它” 变为：

> Unified multi-horizon forecasting is not only a prediction-head problem; it is also a supervision-allocation problem. A future-aware carrier such as TimeAlign exposes this issue because the same future alignment objective can be beneficial on alignable future states but harmful on unstable or noisy future units. Horizon Supervision Scheduling studies how future supervision should be scheduled, masked, or routed during training while evaluation remains multi-horizon.

[Design Constraint] HSS 仍必须 horizon-agnostic：schedule 的输入不能是 benchmark horizon id，而应来自 future unit/state 的 reliability proxy，例如 reconstruction difficulty、alignment consistency、local volatility、prediction residual structure 或 training dynamics。

[D0 Update] TimeAlign 当前没有显式 unified head 设计：official model 使用 fixed
`Linear(d_model * patch_num, pred_len)` projection；unified-720 只是训练 720 并裁剪 prefix。
因此 D1 之前必须先排除 head/interface confounder。D0 将比较 official full-horizon prediction
loss 与 `multi-prefix` prediction loss，判断 ETTm2/Weather 的 unified decrease 是否主要来自短
prefix 监督不足。

#### Phase5-HSS-D0 Result：Head / Interface Confounder Strong

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 D0 remote artifacts，并决定是否进入 D1/M1 |
| `problem` | 判断 TimeAlign unified decrease 是否主要来自 fixed `pred_len=720` head 缺少 evaluation-consistent prefix supervision |
| `existence_evidence` | D0 完成 `3 datasets x 2 prediction-loss modes` unified-only matrix；artifacts 在 `analysis/phase5_timealign_hss_d0_head_gate_20260629/` |
| `idea` | 若 `multi-prefix` prediction loss 显著优于 full-horizon loss，说明 unified head/interface 是强 confounder；HSS 不应直接从 future supervision reliability 开始 |
| `theory_check` | `multi-prefix` 不改 official TimeAlign forward、reconstruction loss 或 alignment loss，只改变 prediction loss 的 prefix aggregation；因此它隔离测试 output interface / prefix supervision 是否是主要因素 |
| `design` | 比较 `full` vs `multi-prefix`；再把两者放回 fixed-horizon reference，计算 unified gap 是否缩小 |
| `gate` | `multi-prefix` 应缩小 ETTm2/Weather unified gap，且不损伤 ETTh2 unified benefit |
| `artifacts` | `phase5_timealign_hss_d0_head_gate_report.md`、`phase5_timealign_hss_d0_interpretation.md`、`phase5_timealign_hss_d0_fixed_reference_comparison.csv` |
| `decision` | `head_interface_confounder_strong`；暂不进入 D1/M1，回 Step 4/6 先设计 TimeAlign-compatible unified head/interface carrier |

[Fact] `multi-prefix` 相对 full unified 在全部 `12/12` 个 setting 上降低 MSE；平均 MSE 相对变化：
ETTh2 `-3.36%`、ETTm2 `-1.57%`、Weather `-1.17%`、ALL `-2.03%`。

[Fact] 放回 fixed reference 后，Weather 的 unified gap 从 `+1.05%` 变成 `-0.13%`；
ETTm2 从 `+3.72%` 缩小到 `+2.06%`；ETTh2 unified benefit 从 `-8.01%` 扩大到
`-11.05%`。

[Decision] 这说明 TimeAlign-HSS 的第一层问题不是直接调度 future reconstruction/alignment
supervision，而是先让 TimeAlign 具备 evaluation-consistent unified prediction interface。
HSS 的叙事应升级为两层：

1. prefix-supervised unified prediction interface；
2. 在该 interface 稳定后，再调度 future alignment/reconstruction supervision 的 reliability
   与 gradient path。

[Next] 进入 Phase5-H0：`Prefix-Supervised TimeAlign`。第一轮先 formalize `multi-prefix`
prediction loss，并加入 `balanced-step`、`stochastic-prefix`、`continuous-prefix` 三个机制对照。
其中 `balanced-step` 判断收益是否只是 non-overlap region reweight；`stochastic-prefix` 判断
prefix supervision 能否作为 schedule；`continuous-prefix` 判断能否脱离 benchmark horizon id。
若 schedule-like variants 接近或超过 `multi-prefix`，HSS 叙事可升级为 horizon-agnostic
prediction-prefix supervision scheduling。D1 supervision reliability diagnostic 后移，只有 H0
后仍存在 residual gap 时才进入。

#### Phase5-H0 Result：Prefix Scheduling Pass

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 H0 prefix-supervision variants，并决定是否进入 D1/M1 |
| `problem` | 判断 D0 的 multi-prefix gain 是否只是 benchmark-specific full prefix loss，还是可以转化为 train-time prefix supervision schedule |
| `existence_evidence` | H0 完成 `3 datasets x 5 prediction-loss modes` unified-only matrix；artifacts 在 `analysis/phase5_timealign_hss_h0_prefix_gate_20260630/` |
| `idea` | `balanced-step` 是 non-overlap region reweight control；`stochastic-prefix` 是 benchmark-prefix schedule；`continuous-prefix` 是 horizon-agnostic schedule proxy |
| `theory_check` | 若 `balanced-step` 明显弱于 prefix modes，说明收益不只是 step/region reweight；若 `stochastic-prefix` 接近 `multi-prefix`，说明 prefix supervision 可以 schedule 化 |
| `design` | 比较 modes 相对 `full`、`multi-prefix` 和 fixed reference 的 mean MSE gap 与 per-horizon winner |
| `gate` | schedule-like mode 接近或超过 `multi-prefix`；ETTm2/Weather gap 保持缩小；ETTh2 unified benefit 不丢失 |
| `artifacts` | `phase5_timealign_hss_h0_prefix_gate_report.md`、`phase5_timealign_hss_h0_interpretation.md`、`phase5_timealign_hss_h0_summary.csv` |
| `decision` | `prefix_scheduling_pass_with_stochastic_candidate`；不进入 D1/M1，进入 H0B schedule robustness / horizon-agnostic refinement |

[Fact] 相对 full unified，所有 H0 variants 在全部 `12/12` setting 上降低 MSE：
`multi-prefix -2.03%`、`balanced-step -1.22%`、`stochastic-prefix -1.90%`、
`continuous-prefix -1.67%`。

[Strong Evidence] `stochastic-prefix` 几乎追平 `multi-prefix`：ALL mean MSE 只比
`multi-prefix` 高 `+0.13%`，wins vs fixed 同为 `7/12`，相对 fixed mean gap 为
`-2.90%`，接近 `multi-prefix` 的 `-3.04%`。

[Strong Evidence] `balanced-step` 明显弱于 `multi-prefix` 与 `stochastic-prefix`，说明收益不只是
不重叠 region reweight；prefix objective / prefix schedule 本身有价值。

[Decision] H0 通过作为 paper-story carrier，但最终主候选应优先从 `stochastic-prefix` 发展，
而不是直接把 benchmark-specific `multi-prefix` 写成方法。下一步进入 H0B：
`stochastic-prefix_k2`、`continuous-prefix_k2`、`continuous-prefix_pool96`。

#### Phase5-H0B：Schedule Robustness / Horizon-Agnostic Refinement

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 H0B schedule robustness gate，并决定 rollback/advance point |
| `problem` | H0 的 `stochastic-prefix` 接近 `multi-prefix`，但仍需判断 sample count 与 continuous pool granularity 是否能进一步提升 ETTm2 或让 schedule 更 horizon-agnostic |
| `existence_evidence` | H0 中 `stochastic-prefix` ALL mean MSE 只比 `multi-prefix` 高 `+0.13%`，但 ETTm2 仍弱于 fixed specialist |
| `idea` | 在不改 TimeAlign forward 的情况下，调整 train-time prefix schedule：增加 sampled prefixes 数量，或移除过短 continuous prefixes |
| `theory_check` | 若 `stochastic-prefix_k2` 提升，说明单 prefix sample signal 不足；若 `continuous-prefix_pool96` 提升，说明 `32-step` continuous pool 的短 prefix 噪声是主要限制；若都失败，HSS 应转向 prefix-aware / target-set readout |
| `design` | `3 datasets x 3 arms`：`stochastic_prefix_k2`、`continuous_prefix_k2`、`continuous_prefix_pool96` |
| `gate` | 至少一个 schedule arm 接近或超过 `multi-prefix`；ETTm2 residual gap 缩小；Weather 不退化；ETTh2 unified benefit 保留 |
| `artifacts` | `analysis/phase5_timealign_hss_h0b_schedule_gate_20260630/` and remote root `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h0b_schedule_gate` |
| `decision` | `prefix_scheduling_robust_but_saturated`；不继续扩大 random schedule sweep，rollback 到 Step 6 设计 prefix-aware / target-set-aware readout |

[Fact] H0B 的 `stochastic_prefix_k2` 基本追平 H0 `multi-prefix`：ALL mean MSE 相对
`full` 为 `-2.03%`，相对 `multi-prefix` 为 `+0.00%`，相对 fixed 为 `-3.04%`。

[Strong Evidence] `stochastic_prefix_k2` 是 H0B 内部最稳健 arm：`12` 个 setting 中拿到
`11` 个 best；`continuous_prefix_k2` 与 `continuous_prefix_pool96` 均弱于它。

[Limit] ETTm2 residual fixed gap 没有缩小：`stochastic_prefix_k2` 相对 fixed 为 `+2.08%`，
与 H0 `multi-prefix` 的 `+2.06%` 基本相同。

[Decision] H0B 支持 prefix supervision / stochastic schedule 作为稳健 carrier，但不支持继续
调 `prefix_samples` 或 continuous pool 作为主线。下一步进入 `Phase5-H1`：在 TimeAlign carrier
上设计 unified head / readout，而不是继续只改 loss。

#### Phase5-H1：Prefix-Aware / Target-Set-Aware TimeAlign Readout

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 H1 readout gate，并决定是否继续 readout route |
| `problem` | official TimeAlign unified 仍使用固定 `Linear(d_model * patch_num, 720)` head，再裁剪 prefix；H0/H0B 证明 loss schedule 有效但已饱和 |
| `existence_evidence` | H0B `stochastic_prefix_k2` 追平 `multi-prefix` 却没有缩小 ETTm2 residual fixed gap，说明瓶颈可能在 unified readout/interface |
| `idea` | 保留 TimeAlign backbone 与 future alignment，显式加入 prefix-conditioned 或 target-set-aware prediction path |
| `theory_check` | 如果 unified model 能知道当前请求的 target length / target set，就不必只依赖 720-head crop 来兼容多 horizon；这比继续调 loss schedule 更直接对应 unified multi-horizon forecasting |
| `design` | 最小 arms：`prefix_conditioned_stochastic_k2` 与 `target_set_decoder_multiprefix` |
| `gate` | ETTm2 相对 fixed 的 residual gap 明显缩小；ETTh2 unified benefit 与 Weather no-harm 不丢失；参数量和训练成本保持可解释 |
| `artifacts` | `analysis/phase5_timealign_hss_h1_readout_gate_20260630/` and remote root `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1_readout_gate` |
| `decision` | `readout_route_weak_pass_with_target_set_candidate`；继续 readout route，但 rollback 到 Step 6 设计 H1B variable-prefix / prefix-token readout |

[Implementation] `prefix_conditioned_stochastic_k2` 在 `proj_x` 前加入 requested-prefix condition，
并沿用 H0B 最稳健的 `stochastic-prefix, prefix_samples=2`；`target_set_decoder_multiprefix`
使用同一 readout condition，但训练时按 target set 的 `96/192/336/720` 多 prefix supervision。

[Fact] `target_set_decoder_multiprefix` 是 H1 更强 arm：ALL mean MSE 相对 H0 `full` 为
`-2.69%`，相对 H0 `multi-prefix` 为 `-0.68%`，相对 H0B `stochastic_prefix_k2` 为
`-0.69%`，并在 `12` 个 setting 中拿到 `10` 个 H1 内部 best。

[Limit] H1 没有达到 ETTm2 full pass：ETTm2 相对 fixed 仍为 `+1.81%`，相比 H0B 的
`+2.08%` 只是小幅缩小。

#### Phase5-H1B：Variable-Prefix / Prefix-Token Readout

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 H1B variable readout gate，并决定 rollback point |
| `problem` | H1 虽然证明 requested-prefix readout 有价值，但仍保留 720-step projection 后 crop，ETTm2 residual gap 未明显解决 |
| `existence_evidence` | H1 `target_set_decoder_multiprefix` 相对 H0B 改善 `-0.69%`，但 ETTm2 vs fixed 仍为 `+1.81%` |
| `idea` | 从 condition-before-720-projection 升级为真正的 variable-prefix 或 prefix-token decoder |
| `theory_check` | 如果 unified multi-horizon 的瓶颈在 readout shape，模型应按 requested target set 直接生成对应 prefix，而不是所有 request 都先生成 720 |
| `design` | 候选 arms：`target_set_prefix_head_multiprefix` 与 `prefix_token_decoder_multiprefix`；二者都使用 `multi-prefix` supervision，差别只在 head |
| `gate` | ETTm2 fixed gap 明显低于 H1 的 `+1.81%`，ETTh2 保持强收益，Weather no-harm |
| `artifacts` | `analysis/phase5_timealign_hss_h1b_variable_readout_gate_20260701/` and remote root `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1b_variable_readout_gate` |
| `decision` | `variable_readout_fail_capacity_collapse`；不继续扩展当前 variable heads，rollback 到 Step 5/6 设计 capacity-preserving prefix decoder |

[Correction] H1 的 `target_set_decoder_multiprefix` 并不是真正 decoder head，它仍然是
condition-before-720-projection。H1B 才开始测试真正的 variable-prefix / prefix-token prediction
head。

[Fact] H1B 两个真正 variable-prefix heads 均失败。`target_set_prefix_head_multiprefix` 的
ALL mean MSE 相对 H1 `target_set_decoder_multiprefix` 为 `+14.41%`，相对 fixed 为
`+10.26%`；`prefix_token_decoder_multiprefix` 更差，相对 H1 为 `+25.52%`。

[Decision] H1B 说明直接替换 dense 720 projection 会导致 readout capacity collapse。下一步若继续
decoder/head route，必须保留 TimeAlign dense projection 作为 base path，并只加入 prefix/target-set
conditioned residual、low-rank adapter 或 row-wise gate。

#### Phase5-H1C：Capacity-Preserving Prefix Decoder

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 H1C capacity-preserving decoder/head gate，并决定 rollback point |
| `problem` | H1B 证明简单 variable-prefix head 破坏 TimeAlign dense readout capacity；但 H1 证明 prefix/target-set condition 在 dense projection 上有正向信号 |
| `existence_evidence` | H1 target-set conditioned 720 projection 相对 H0B 改善 `-0.69%`；H1B variable heads 相对 H1 退化 `+14.41%` 到 `+25.52%` |
| `idea` | 保留 dense 720 projection，prefix/target-set information 只控制 residual adapter、low-rank delta 或 row-wise gate |
| `theory_check` | 如果 TimeAlign 的性能来自 dense row capacity，则 decoder 改造必须 preserve base readout，并让 HSS 只调度/调制增量路径 |
| `design` | 候选 arms：`dense_prefix_residual_adapter_multiprefix`、`row_gated_dense_head_multiprefix`、`prefix_adapter_shared_dense_multiprefix`；三者都保留 `proj_x: Linear(...,720)`，区别只在 prefix condition 进入 output residual、row-wise gate 或 hidden adapter |
| `gate` | 必须超过 H1 `target_set_decoder_multiprefix`，并让 ETTm2 fixed gap 明显低于 `+1.81%` |
| `artifacts` | `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701/`、`baselines/timealign_official/models/TimeAlign.py`、`scripts/remote/run_phase5_timealign_hss_h1c_capacity_preserving_gate.sh`、`scripts/sync_phase5_timealign_hss_h1c_results.sh`、`scripts/analyze_phase5_timealign_hss_h1c_capacity_preserving_gate.py` |
| `decision` | `capacity_preserving_readout_partial_fail_row_gate_control`；不继续扩大当前 post-hoc residual/gate/adapter sweep，rollback 到 Step 2/3/6，重设计 SCI-level unified interface；future supervision reliability diagnostic 只作为并行准备 |

[Implementation] H1C 不再尝试让 decoder 直接输出 variable prefix。它把 TimeAlign 原始 dense
720 projection 视为 base path，并测试 prefix/target-set information 应该控制哪个增量位置：

- `dense_prefix_residual_adapter_multiprefix`：`proj_x(hidden)` 先产生 `[B,C,720]`，再加一个
  zero-init、prefix-conditioned low-rank residual；
- `row_gated_dense_head_multiprefix`：`proj_x(hidden)` 先产生 `[B,C,720]`，再用
  `[step/720, target_prefix/720]` 生成 row-wise multiplicative gate；
- `prefix_adapter_shared_dense_multiprefix`：先在 `hidden` 上加入 zero-init low-rank adapter，
  再复用同一个 `proj_x` 输出 720 steps。

[Gate Clarification] 这不是回到纯 training schedule。`multi-prefix` 只作为统一控制变量，
真正被比较的是 prefix condition 在 prediction head 内部的落点：output residual、row gate、
hidden adapter。若三者均失败，说明当前 TimeAlign carrier 中“保留 dense capacity + prefix
condition”的 decoder route 也不足以支撑 HSS 主线。

[Fact] H1C 最佳 arm 是 `row_gated_dense_head_multiprefix`：ALL mean MSE 相对 H0 `full`
为 `-2.29%`，相对 H0B `stochastic_prefix_k2` 为 `-0.27%`，相对 fixed 为 `-3.29%`，
但相对 H1 `target_set_decoder_multiprefix` 仍为 `+0.43%`。它在 `12` 个 setting 中
只赢 H1 `5/12`。

[Counter-Evidence] `dense_prefix_residual_adapter_multiprefix` 相对 H1 为 `+5.04%`，
`prefix_adapter_shared_dense_multiprefix` 相对 H1 为 `+10.89%`。这说明 output residual
和 hidden low-rank adapter 都会破坏当前 TimeAlign readout balance。

[Decision] H1C 不通过 paper-core gate。`row_gated_dense_head_multiprefix` 可作为
capacity-preserving control 保留，但不能成为主线 decoder。这个失败只否定当前
post-hoc residual/gate/adapter interface 族，不能否定 unified interface 作为论文主轴。
下一步回 Step 2/3/6 设计 Stage A2：SCI-level unified interface redesign；future unit 的
reconstruction difficulty、alignment consistency、residual volatility 与 segment-level unified
gap 诊断可以并行准备，但不能替代 interface 主轴。只有 A2 再次失败后，才允许重新审稿评估
是否放弃 interface、换 carrier 或重构论文路线。

#### Phase5-A2：SCI-Level Unified Interface Redesign

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A2 最小 interface gate，并决定 rollback point |
| `problem` | H1 证明 prefix-aware interface 有价值；H1B 证明 random variable-prefix head 发生 capacity collapse；H1C 证明 post-hoc residual/gate/adapter 不能超过 H1 target-set。因此 A2 需要重新设计更有结构约束的 unified prediction interface |
| `existence_evidence` | D0/H0/H1 均说明 interface/prefix supervision 是 material factor；H1C row-gated 相对 fixed 为 `-3.29%` 但相对 H1 为 `+0.43%`，说明保守 calibration 稳定但创新不足 |
| `idea` | 不再做 full-720 crop 后的小修补，而是测试直接服务 requested prefix 的 output contract：dense-row initialized prefix decoder 与 nested segment decoder |
| `theory_check` | 如果 H1B 失败主要来自随机 variable head 丢失 dense row capacity，则 dense-row initialized prefix decoder 应避免 collapse；如果 unified interface 需要 prefix-consistent composition，则 nested segment decoder 应优于单纯 720 crop/gate |
| `design` | `dense_row_initialized_prefix_decoder_multiprefix` 直接用 `proj_x.weight[:H]` 读出 prefix，并加 zero-init low-rank delta；`nested_segment_decoder_multiprefix` 以 `[96,192,336,720]` boundaries 构造 nested segment heads |
| `gate` | 必须超过 H1 `target_set_decoder_multiprefix` 和 H1C `row_gated_dense_head_multiprefix`；ETTm2 fixed gap 明显低于 H1 的 `+1.81%`；ETTh2 benefit 与 Weather no-harm 不丢失 |
| `artifacts` | `analysis/phase5_timealign_hss_a2_interface_gate_20260701/`、`baselines/timealign_official/models/TimeAlign.py`、`scripts/remote/run_phase5_timealign_hss_a2_interface_gate.sh`、`scripts/sync_phase5_timealign_hss_a2_results.sh`、`scripts/analyze_phase5_timealign_hss_a2_interface_gate.py` |
| `decision` | `nested_interface_partial_pass_capacity_gap_remains`；不进入 reliability routing full method，不泛化 head sweep，rollback 到 Step 5/6 设计 A3：nested composition + capacity/teacher preservation |

[Implementation] A2 当前只做两个最小 arms：

- `dense_row_initialized_prefix_decoder_multiprefix`：区别于 H1B random dynamic head，它直接复用
  dense head 的前 `H` 行作为 base，因此初始 output contract 等价于 dense-row prefix readout；
- `nested_segment_decoder_multiprefix`：区别于 H1/H1C 的 720 output crop，它按 nested segments
  组合 requested prefix，显式测试 prefix-consistent composition。

[Gate Clarification] A2 不是增加更多 head sweep，而是测试 H1C 没有覆盖的两个结构假设：
capacity preservation by dense-row initialization，以及 prefix consistency by nested composition。
若二者都失败，才说明当前 TimeAlign carrier 上的 unified interface 主轴需要重新审稿评估。

[Fact] A2 中 `dense_row_initialized_prefix_decoder_multiprefix` 明确失败：ALL mean MSE 相对
H1 `target_set_decoder_multiprefix` 为 `+5.39%`，相对 H1C `row_gated_dense_head_multiprefix`
为 `+4.92%`，wins 均为 `0/12`。仅复用 dense rows 不足以形成有效 interface。

[Strong Evidence] `nested_segment_decoder_multiprefix` 是 partial pass：ALL 相对 H0 `full`
为 `-2.12%`，相对 fixed 为 `-3.13%`；在 Weather 上相对 H1 为 `-0.08%`，相对 H1C 为
`-0.09%`，且 `4/4` horizons 赢 H1C。

[Limit] `nested_segment_decoder_multiprefix` 仍未过 paper-core gate：ALL 相对 H1 为
`+0.61%`，相对 H1C 为 `+0.18%`；ETTm2 fixed gap 仍为 `+1.81%`，没有明显低于 H1。

[Decision] A2 支持 nested / prefix-composition interface 方向，但不支持当前 random nested
segment head 作为最终贡献。下一步回 Step 5/6 设计 A3：把 nested composition 与
capacity preservation 或 teacher preservation 结合。Stage B/D1 reliability diagnostic 可以并行
准备，但不能替代 A3 interface gate。

### Phase5-A3：Nested Interface Capacity Repair

| Field | Content |
| --- | --- |
| `current_step` | Step 6/7/8：设计并启动 A3-1 dense-initialized nested interface gate |
| `problem` | A2 nested segment 有 prefix-composition 正向信号，但未超过 H1/H1C；可能原因是 random segment heads 牺牲了 official dense head 的 row-level readout capacity |
| `existence_evidence` | A2 nested 相对 fixed 为 `-3.13%`，Weather 相对 H1C 为 `-0.09%` 且 `4/4` horizons 赢 H1C；但 ALL 相对 H1 为 `+0.61%`、相对 H1C 为 `+0.18%` |
| `idea` | 保留 nested composition，不改训练 objective，只把 segment heads 初始化为 `proj_x` 对应 row slices |
| `theory_check` | 若 A2 的主要漏洞是 capacity collapse，则 dense-initialized nested 应优于 A2 nested，并降低 H1/H1C gap；若仍失败，则问题不只是 initialization，而是 nested interface 或 condition signal 不足 |
| `design` | 新增 `dense-initialized-nested-segment-decoder`：`[0:96]`、`[96:192]`、`[192:336]`、`[336:720]` segment heads 分别复制 `proj_x.weight/bias` 对应 rows；forward 与 A2 nested 相同 |
| `gate` | A3-1 至少要优于 A2 nested；paper-core gate 仍要求超过 H1 target-set 和 H1C row-gated，并降低 ETTm2 fixed gap；若只优于 A2 nested，则进入 A3-2 teacher/target-conditioned repair |
| `artifacts` | `analysis/phase5_timealign_hss_a3_interface_repair_20260701/`、`baselines/timealign_official/models/TimeAlign.py`、`scripts/remote/run_phase5_timealign_hss_a3_interface_repair.sh`、`scripts/sync_phase5_timealign_hss_a3_results.sh`、`scripts/analyze_phase5_timealign_hss_a3_interface_repair.py` |
| `decision` | `shallow_dense_initialization_no_capacity_repair`；A3-1 不通过 paper-core gate，rollback 到 Step 5/6 设计真正的 teacher/target-conditioned nested preservation |

[Fact] A3-1 remote gate 已完成，`3 datasets × 1 arm × 4 horizons`。ALL mean MSE 相对
A2 nested 为 `-0.06%`，相对 fixed 为 `-3.19%`，但相对 H1 target-set 为 `+0.55%`、
相对 H1C row-gated 为 `+0.12%`。

[Critical Limit] A3-1 复制的是同一模型初始化时的 `proj_x.weight/bias` rows；`proj_x` 不是
已训练 full head。因此 A3-1 不是严格的 learned capacity preservation，只是 dense-like
initialization repair。

[Decision] A3-1 否定 shallow initialization repair，不否定 nested / prefix-composition
interface。下一步若继续 Stage A，必须做真正的 `teacher_preserved_nested_segment_decoder`、
`target_conditioned_nested_segment_decoder` 或从已训练 checkpoint 出发的
`warm_started_nested_segment_decoder`。不继续调初始化或泛化 head sweep。

### Phase5-A3B：Target-Conditioned Nested Residual Interface

| Field | Content |
| --- | --- |
| `current_step` | Step 6/7/8：从 A3-1 design error 回滚后，设计并启动 A3B nested residual gate |
| `problem` | A3-1 误把随机 `proj_x` row-copy 当作 capacity preservation；真正的问题是如何同时保留 dense readout path、target condition 和 nested prefix composition |
| `existence_evidence` | A2/A3-1 nested route 相对 fixed 仍有 `-3%` 左右收益，Weather 上接近或优于 H1C；但 A3-1 相对 H1/H1C 仍为正 gap，说明 shallow initialization 不够 |
| `idea` | 用 dense `proj_x` 作为 function-preserving base path，再叠加 zero-init、target-conditioned nested residual path |
| `theory_check` | zero-init residual 保证训练开始时等价于 dense full-head prefix；target condition 提供 H1 的 requested-prefix signal；nested residual 提供 A2 的 prefix-composition structure |
| `design` | `target-conditioned-nested-residual-decoder`：`base=proj_x(hidden)[:H]`；`residual=f(target_prefix, hidden)` 经 `[0:96]`、`[96:192]`、`[192:336]`、`[336:720]` zero-init segment heads 拼接；输出 `base + residual` |
| `narrative_gate` | 不作为 paper-core，通过条件仅为 diagnostic/control：检验 nested structure 放在 residual path 是否仍有增量 |
| `effectiveness_gate` | 必须优于 A2 nested 与 A3-1 shallow；paper-core gate 要求超过 H1 target-set 或 H1C row-gated，并降低 ETTm2 fixed gap；Weather nested gain 不能消失 |
| `artifacts` | `baselines/timealign_official/models/TimeAlign.py`、`scripts/remote/run_phase5_timealign_hss_a3b_nested_residual_gate.sh`、`scripts/sync_phase5_timealign_hss_a3b_results.sh`、`scripts/analyze_phase5_timealign_hss_a3b_nested_residual_gate.py` |
| `decision` | `nested_residual_diagnostic_failed`；A3B 不通过，降级为 diagnostic/control，不再继续 residual correction route |

[Fact] A3B remote gate 已完成：ALL 相对 A2 nested 为 `+4.42%`，相对 A3-1 shallow 为
`+4.48%`，相对 H1 target-set 为 `+5.09%`，相对 H1C row-gated 为 `+4.61%`，且
`0/12` horizon 赢 A2/H1/H1C。

[Decision] A3B 证明 nested 放在 residual path 会削弱 A2 primary nested 的正向信号。下一步回
Step 4/5/6 设计 warm-started primary nested interface。

### Phase5-A3C：Warm-Started Primary Nested Interface

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A3C warm-started primary nested gate 并决定 rollback point |
| `problem` | A2 nested 有正向信号但缺 learned capacity；A3-1 的随机 row-copy 是设计错误；A3B residual route 破坏 primary nested 叙事 |
| `existence_evidence` | A2 nested 相对 fixed 为 `-3.13%`，Weather 上赢 H1C；A3B `0/12` 赢 A2/H1/H1C，说明 nested 必须回到 primary head |
| `idea` | 从已训练 H1 target-set checkpoint warm-start shared TimeAlign carrier，并把 H1 learned `proj_x` rows 转换为 nested segment heads |
| `theory_check` | 如果 A2 失败主要来自 learned capacity 缺失，则 warm-started nested primary 应明显优于 A2/A3B；如果仍失败，则 primary nested 的瓶颈不只是 capacity |
| `design` | `checkpoint-initialized-nested-segment-decoder`：加载 H1 checkpoint 的兼容 shared weights；将 checkpoint `proj_x.weight/bias` 按 `[0:96]`、`[96:192]`、`[192:336]`、`[336:720]` 写入 nested heads；输出仍由 nested heads 直接生成 |
| `narrative_gate` | 通过：保留 primary nested interface，且 learned capacity 来自已训练 checkpoint，不再是 shallow initialization 或 residual patch |
| `effectiveness_gate` | 必须优于 A2 nested 与 A3B residual；paper-core gate 要求接近或超过 H1/H1C，并降低 ETTm2 fixed gap |
| `artifacts` | `analysis/phase5_timealign_hss_a3c_warm_started_nested_gate_20260701/`、`baselines/timealign_official/models/TimeAlign.py`、`baselines/timealign_official/train_repo.py`、`scripts/remote/run_phase5_timealign_hss_a3c_warm_started_nested_gate.sh`、`scripts/sync_phase5_timealign_hss_a3c_results.sh`、`scripts/analyze_phase5_timealign_hss_a3c_warm_started_nested_gate.py` |
| `decision` | A3C 不通过 paper-core gate；相对 A2 `+0.07%`，相对 H1 `+0.68%`，相对 H1C `+0.25%`，只相对 A3B residual 明显改善 `-4.06%`。结论是 row-slice warm-start 不足以 preserve H1 learned function；rollback 到 Step 4/5/6 后，A3D teacher-preserved nested primary 通过 narrative gate，进入实现和 remote gate |

[A3 Candidate Triage Rule] A3C 之后的候选选择不能退回到 shallow initialization 或 residual patch。
候选必须同时满足两点：第一，nested/prefix-aware structure 是 primary prediction interface 的一部分；
第二，在 Step 4-6 通过 narrative gate，能清楚解释为什么它比 dense full head 更适合作为 unified
multi-horizon interface。当前保留的 paper-core candidates 是：

- `teacher_preserved_nested_primary_decoder`：用 teacher full head 或 fixed-horizon teacher 保留原有
  dense prediction capacity，同时让 nested interface 学习 prefix-consistent decomposition；
- `target_conditioned_nested_primary_decoder`：把 target/prefix condition 显式注入 nested primary
  head，让 decoder 不再先生成 720 再 crop，而是让 requested target set 进入预测头结构；
- `teacher_preserved + target_conditioned` 最小组合：只在前两者分别具备 narrative gate 合理性后考虑，
  避免把两个未证实机制直接叠加。

### Phase5-A3D：Teacher-Preserved Nested Primary Interface

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A3D teacher-preserved nested gate 并决定下一步 |
| `problem` | A3C 证明 row-slice warm-start 不能让 nested primary 超过 H1/H1C；问题不是参数初值，而是 learned function 在切换到 nested interface 时缺少训练期 preservation constraint |
| `existence_evidence` | A3C 相对 A2 仅 `+0.07%`，相对 H1/H1C 仍为 `+0.68%` / `+0.25%`，说明 warm-start 后仍未保留 H1 target-set function |
| `idea` | 使用 H1 `target_set_decoder_multiprefix` 作为 frozen teacher；student 仍是 warm-started nested primary head；训练时同时优化 label loss 与 teacher consistency loss |
| `theory_check` | 若 A3C 的主要问题是 function preservation 而不是 nested structure 本身，则 teacher consistency 应让 nested primary 接近 H1/H1C，同时保留 prefix-consistent decomposition；若结果只复制 H1 或仍弱于 A3C，则该 preservation route 不成立 |
| `design` | 两个 teacher strength arms：`teacher_preserved_nested_w03` 与 `teacher_preserved_nested_w10`；二者均从 H1 checkpoint warm-start，并用同一 H1 checkpoint 作为 target-set teacher |
| `narrative_gate` | 通过：它直接服务 `Capacity-Preserving Prefix-Aware Interface`，且是 A3C 失败后的最小机制修复，不是 residual patch 或 shallow initialization |
| `effectiveness_gate` | 必须优于 A3C/A2；paper-core gate 要求接近或超过 H1/H1C；若只接近 H1 但无 prefix/nested 行为收益，则降级为 teacher-preservation diagnostic |
| `artifacts` | `analysis/phase5_timealign_hss_a3d_teacher_preserved_nested_gate_20260701/`、`baselines/timealign_official/train_repo.py`、`scripts/remote/run_phase5_timealign_hss_a3d_teacher_preserved_nested_gate.sh`、`scripts/sync_phase5_timealign_hss_a3d_results.sh`、`scripts/analyze_phase5_timealign_hss_a3d_teacher_preserved_nested_gate.py` |
| `decision` | A3D 为 `partial_pass`；`w03` 相对 A3C `-0.73%`、相对 H1 `-0.06%`、相对 H1C `-0.48%`，说明 teacher preservation 有效；但 ETTm2 仍负，不能作为 paper-core。下一步进入 A3E target-conditioned nested primary |

### Phase5-A3E：Target-Conditioned Nested Primary Interface

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10/11：评估 A3E target-conditioned nested gate 并决定 rollback |
| `problem` | A3D 保留 H1 function 但不能解决 minute-level failure；当前缺口更像 requested target set 没有进入 primary nested head，导致 nested decomposition 缺少 target-prefix specialization |
| `existence_evidence` | A3D teacher loss 下降且 overall 接近 H1/H1C，说明 capacity/function preservation 有效；但 ETTm2 仍弱。为避免只围绕 ETTm2 单一数据集决策，本轮用 ETTm1 替换 ETTm2 并重建 reference |
| `idea` | 将 target-prefix condition 直接注入 primary nested head 的 hidden representation，而不是在 dense residual 或 teacher loss 中间接作用 |
| `theory_check` | A3C 已证明 warm-start alone 无效；因此 A3E warm arm 只作为和 A3C 对齐的 initialization control。若 warm arm 优于 A3C/A3D，增量来自 target conditioning；scratch arm 只判断该结构是否独立于 H1 initialization |
| `design` | 数据集改为 `ETTh2 + ETTm1 + Weather`。先补跑 ETTm1 fixed、H1 target-set、H1C row-gated、A2 nested、A3C warm、A3D w03 references，再跑 A3E 双臂：`target_conditioned_nested_warm` 与 `target_conditioned_nested_scratch` |
| `narrative_gate` | 通过：它直接回答 multi-prefix evaluation 与 prediction head 不一致的问题；warm-start 不作为机制贡献 |
| `effectiveness_gate` | 未通过：warm/scratch ALL 相对 A3C 仅 `-0.25/-0.26%`，相对 A3D/H1 仍为正 gap；ETTm1 上 A3C 仍是最强候选 |
| `artifacts` | `baselines/timealign_official/models/TimeAlign.py`、`baselines/timealign_official/train_repo.py`、`scripts/remote/run_phase5_timealign_hss_a3e_ettm1_replacement_gate.sh`、`scripts/remote/run_phase5_timealign_hss_a3e_target_conditioned_nested_gate.sh`、`scripts/sync_phase5_timealign_hss_a3e_ettm1_results.sh`、`scripts/analyze_phase5_timealign_hss_a3e_ettm1_replacement_gate.py` |
| `decision` | A3E 标记为 `failed_as_core_candidate`。target conditioning 直接进入 primary nested head 的增量不足，且没有解决 ETTm1；暂不进入 A3F，rollback 到 Step 2/3/4 做 `interface reliability diagnostic`，重新判断 Stage A 是否应从 universal prefix-aware head 转向 capacity-preserving path reliability |

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
- 不把 future-aware 当作绕过 HSS 失败的装饰性模块；只能作为 Phase4-FSA 的 representation
  substrate diagnostic，并且必须保留 R.3/single-prefix controls。
- 不把 reduced horizon set 的 positive signal 写成 operator success。
- 不只用 aggregate MSE/MAE 判定通过。
- 不默认使用旧 `R_2026_FSA` 证据，除非用户批准具体来源和用途。
