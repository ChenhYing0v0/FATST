# FATST 论文主线总纲

本文档是当前项目的论文级总纲，优先级高于单个 phase 的局部实验计划。`docs/research-roadmap.md`
负责记录完整 11-step 研究过程；本文档只维护能进入论文叙事的主线、创新点、实验安排和转向规则。
阶段内部的候选队列、未完成任务和局部 decision cursor 由 `docs/stage-ledgers/` 维护；当前 active
ledger 是 `docs/stage-ledgers/phase5-timealign-interface.md`。研究路径保存体系见
`docs/research-governance.md`。

每次发生以下事件时必须回到本文档复核：

- 一个重要 gate 实验完成，例如 H1C、D1、M1/M2；
- 一个机制被判定为 `pass`、`partial pass` 或 `fail_as_core_candidate`；
- 研究方向从 head/interface 转向 future supervision routing，或从 TimeAlign carrier 转向其他 carrier；
- 新增实验族会改变论文 claim、方法命名、主 baseline 或核心贡献边界。

不属于本文档职责的内容：

- 阶段内已提出但尚未执行的并行候选；这些必须登记在 active Stage Ledger 的
  `candidate_queue`；
- 单次实验的详细 metric、CSV 解读、训练日志和 diagnostic 表；这些必须保存在 `analysis/`；
- 只影响阶段内部执行顺序、但不改变论文 claim 或贡献边界的局部判断。

主线选择原则：

- `narrative_gate` 属于 Step 4-6：在提出 core idea、theory check 和 method design 时完成，
  远程实验前必须判断该方案是否具备高水平 SCI 叙事资格；
- `effectiveness_gate` 属于 Step 9-10：实验结果返回后，再判断 MSE/MAE、segment behavior、
  stability 和 diagnostics 是否支持该方案；
- 新架构、新方法、新 training strategy 若希望成为 `paper-core`，必须先通过 `narrative_gate`，
  不能等实验结果出来后再补叙事；
- diagnostic/control 实验可以不通过 `narrative_gate`，但必须在 launch 前标注
  `diagnostic_only` 或 `control_only`，且不能因为 metric 突然变好就直接升级为 paper-core；
- 若两个 paper-core 候选性能差距不大，优先选择叙事性更强、贡献边界更清晰、能支撑高水平
  SCI 主线的方案；
- 若一个方案只是小幅修补 metric，但无法形成论文级机制叙事，不能作为 paper-core，只能作为
  control 或 diagnostic evidence。

## 当前状态

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_11_step` | Phase5-A5：Step 2/3/4，重新设计 first-principles unified prediction architecture |
| `active_carrier` | official-source TimeAlign |
| `active_stage_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |
| `active_question` | 如何在 prediction head / decoder 层面解决 unified multi-horizon 的 interface mismatch，使短 horizon 不是 naive full-720 crop |
| `current_gate` | A4S 否定 existing-path selector，但不能把 Stage A 降级为 protocol；若提出 interface mismatch，必须给出 fair enough unified prediction architecture |
| `paper_core_status` | Stage A 仍是当前 blocking core；下一步设计 A5 `Capacity-Preserving Prefix-Consistent Decoder`。Stage B routing 暂缓为 A5 成立后的第二贡献 |

## 顶级 SCI 审稿视角评判

### 总体判断

[Reviewer Assessment] 当前框架有论文潜力，但尚未达到高水平 SCI 的稳定形态。它的问题不在工作量，
而在贡献边界还容易被审稿人质疑为“基于 TimeAlign 的工程修补”或“训练 loss/schedule 的经验调参”。

若论文只写成：

> TimeAlign 在 unified multi-horizon 下有 gap，我们设计 prefix schedule/readout 修复它。

审稿风险较高。审稿人会认为 novelty 偏弱、方法依赖特定 baseline、实验逻辑不够一般化。

更强的论文必须写成：

> Unified multi-horizon forecasting exposes a supervision-allocation problem:
> evaluation horizons are discrete test points, but training supervision can
> and should be scheduled over future units and gradient paths in a
> horizon-agnostic way.

TimeAlign 应作为强 carrier 和机制放大器，而不是论文唯一理由。

[Reviewer Update after H1C] H1C 失败后，不能直接把 interface 贡献降级成诊断。顶级 SCI
审稿人会追问：如果 unified prediction interface 是核心 mismatch，为什么只尝试了少数
post-hoc modulation 就放弃？因此当前合理修订不是取消 interface 贡献，而是承认 H1C 只
否定了一个局部实现族，并把下一步回滚到 Step 2/3/6：重新设计更有结构约束和叙事力的
interface 机制。

[Reviewer Update after A4S] A2/A3C/A3D/A3E/A4/A4R/A4S 后，审稿式结论进一步收敛：
Stage A 已经证明 interface 是 material confounder，但还没有证明一个可作为 paper-core 的
unified prediction architecture。不能把这个缺口交给 Stage B routing 解决：如果论文声称
full-720 crop 存在 mismatch，就必须先提出 fair enough 的 unified head/decoder。A4S 只否定
existing-path selector，不否定 first-principles interface 方向。下一步应进入 A5，而不是直接转向
Stage B。

### 创新性评判

[Strength] 当前方向的创新性来自问题重定义，而不是某个单独 module：

- 将 `evaluation horizon` 与 `training supervision unit` 解耦；
- 把 HSS 从 loss reweighting 升级为 supervision path / gradient path scheduling；
- 在 future-aware carrier 中讨论 future supervision 何时有益、何时有害；
- 识别 unified multi-horizon 的 head/interface confounder，并用 capacity-preserving readout 处理。

[Weakness] 如果最后只有 H0/H0B 或 H1 级别结果，创新性不足：

- `multi-prefix` 容易被认为是常规 multi-task/prefix loss；
- `stochastic-prefix` 容易被认为是 sampling trick；
- `target_set_decoder_multiprefix` 仍是 condition-before-720 projection，不足以支撑 decoder-level claim；
- H1B 失败说明直接 variable-prefix head 缺少 convincing design。

[Required Revision] 论文创新点必须合并为一个更高层的机制：

> horizon-agnostic supervision scheduling decides not only which future units
> supervise training, but also which prediction/future-alignment path receives
> their gradients.

### 工作量评判

[Strength] 当前 work量已经较大：

- official-source TimeAlign reproduction；
- fixed/unified comparison；
- H0/H0B prefix scheduling；
- H1/H1B/H1C readout/interface route；
- Phase4 中多组 failed/partial evidence 支撑 rollback 逻辑。

[Weakness] 高水平 SCI 不会只看内部探索量。必须转化为论文中的最小必要证据：

- 不能展示过多失败路线；
- 不能让方法看起来是长时间试错后的 ad hoc 组合；
- 必须有清晰 main table、ablation table、diagnostic figure。

[Required Revision] 论文实验应压缩为四层证据：

1. Problem evidence：unified multi-horizon 存在 supervision/interface mismatch；
2. Method evidence：HSS module 带来稳定收益；
3. Mechanism evidence：收益来自 horizon-agnostic scheduling/path routing，而非简单 loss weakening；
4. Generality evidence：跨数据集、跨 baseline 或至少跨 carrier setting 成立。

### 逻辑性评判

[Strength] 当前 11-step loop 很强，能避免一直堆机制。

[Weakness] 论文逻辑仍有两处风险：

- 风险 1：如果只说 “unified degradation”，会被 ETTh2 unified 受益反驳；
- 风险 2：如果 HSS 先解决 prediction head/interface，再解决 future alignment routing，审稿人可能认为这是两个松散问题。

[Required Revision] 论文逻辑必须改成：

1. Unified multi-horizon 不一定整体退化，而是 dataset/state/unit-dependent；
2. 这种差异来自 training supervision allocation 与 prediction interface 的共同作用；
3. 因此方法分两层，但由同一个 HSS 原则统一：
   - evaluation-consistent prefix supervision path；
   - reliability-aware future supervision gradient path。

## 修订后的论文主线

### 核心问题

[Problem] 在 unified multi-horizon forecasting 中，`96/192/336/720` 这类 horizon 只是评估点。
如果训练仍把 full future 或固定 horizon 当作唯一监督单位，模型会面临两个 mismatch：

- `interface mismatch`：统一模型用固定 720-step dense head 生成所有 prefix，短 horizon 只是 crop；
- `supervision-path mismatch`：future-aware training branch 对所有 future units 施加相同 reconstruction/alignment pressure。

这两个 mismatch 在不同 dataset、future segment 和 state 下可能有益也可能有害，因此不能写成
“unified 一定退化”。更准确的问题是：

> Which future units should supervise a unified forecaster, and where should
> their gradients be allowed to update?

### 目标 claim

[Claim Version 0.7] Horizon-Agnostic Supervision Scheduling treats benchmark
horizons as evaluation probes rather than training units. It first requires a
capacity-preserving prefix-consistent prediction architecture so that multiple
evaluation prefixes are generated by a fair unified interface rather than
naive full-horizon cropping; on top of that architecture, future supervision can
be scheduled through reliability-aware gradient paths.

中文表述：

> HSS 不按 benchmark horizon id 组织训练，而是把 future positions / future units / gradient
> paths 作为监督调度对象。但在进入 supervision routing 前，必须先解决 unified prediction
> architecture：短 horizon 不能只是从 720 输出中被动裁剪，而应由同一套 prefix-consistent、
> capacity-preserving decoder contract 原生生成。Stage B 的 reliability-aware future supervision
> routing 是建立在该 architecture 之后的增益性贡献。

### 预期贡献

1. **Problem Formulation**
   提出 unified multi-horizon forecasting 中的 `supervision-allocation mismatch`，明确
   evaluation horizons 不等于 training supervision units。

2. **Capacity-Preserving Prefix-Consistent Unified Prediction Architecture**
   设计一个 fair enough 的 unified prediction head/decoder，使 h96/h192/h336/h720 不是
   naive full-720 crop，而是由同一套 prefix-consistent decoder contract 生成；同时通过
   teacher/full-head preservation 或 active function-preserving path 保留 TimeAlign dense/full-head
   capacity。

3. **Reliability-Aware Future Supervision Routing**
   在上述 unified prediction architecture 成立后，根据 future unit 的可重构性、alignment
   consistency、residual volatility、gradient conflict 或预测误差结构，决定 future supervision
   的强度与 gradient path。

4. **Mechanism-Level Evidence**
   用 per-horizon、per-segment、selector sensitivity、loss-only control 和 gradient-routing
   control 证明收益不是简单调 loss，也不是 validation artifact。

## 方法边界

### 必须坚持的边界

- HSS 是 horizon-agnostic：不能把 benchmark horizon id 作为最终方法的核心输入。
- TimeAlign 是 carrier，不是贡献本身。
- `multi-prefix` / `stochastic-prefix` 是控制和 carrier，不应单独包装成最终创新。
- H1 `target_set_decoder_multiprefix`、H1C `row_gated_dense_head_multiprefix` 与 A3D
  `teacher_preserved_nested` 是当前 interface controls / partial evidence，不是最终 fair head。
- A4S 只否定 existing-path selector，不否定 interface 问题本身。不能把 interface mismatch
  降级为只需控制的 confounder；必须先在 head/decoder 层面提出解决方案。
- Stage B reliability routing 必须等 A5 unified prediction architecture 至少通过 narrative gate
  并形成可运行 candidate 后再推进；它不能替代 Stage A。

### 不应写入主贡献的内容

- R.3 repair；
- 单纯 horizon loss weighting；
- 只在一个 dataset 上有效的 schedule；
- 只比 weak full-horizon baseline 好、但不超过 H1 target-set 或 row-gated control 的 interface/module；
- 不能解释机制的 diagnostic-only artifact。

## 实验总安排

### Stage A：Carrier And Interface Gate

目标：确认 TimeAlign 是否能作为 HSS carrier，以及 unified prediction interface 是否足够强。

已完成或进行中：

- official-source TimeAlign fixed/unified audit；
- H0/H0B prefix supervision / stochastic-prefix；
- H1 target-set conditioned 720 projection；
- H1B variable-prefix head failure；
- H1C capacity-preserving prefix decoder gate。
- A2 nested primary interface；
- A3C/A3D/A3E capacity-preserving nested variants；
- A4/A4R/A4S interface reliability diagnostics。

H1C pass 条件：

- ALL mean MSE 优于 H1 `target_set_decoder_multiprefix`；
- ETTm2 unified-vs-fixed gap 明显低于 `+1.81%`；
- ETTh2 unified benefit 不被明显破坏；
- Weather 至少 no-harm，最好改善 h96/h720 的薄弱点。

H1C fail 决策：

- 若 H1C 低于 H1：不继续堆 readout，回 Step 2/3 重新判断 TimeAlign 是否适合作为 HSS 主 carrier；
- 若 H1C 只在一个 dataset 改善：保留为 ablation，不作为 paper-core；
- 若 H1C 超过 H1 但 fixed gap 仍大：进入 D1，但论文 claim 需降低为 interface + routing 共同贡献。

H1C result：

- `row_gated_dense_head_multiprefix` 是唯一稳定 arm，ALL mean MSE 相对 H0 `full` 为
  `-2.29%`、相对 fixed 为 `-3.29%`；
- 但它相对 H1 `target_set_decoder_multiprefix` 仍为 `+0.43%`，只赢 H1 `5/12`；
- `dense_prefix_residual_adapter_multiprefix` 和 `prefix_adapter_shared_dense_multiprefix`
  分别相对 H1 退化 `+5.04%` 和 `+10.89%`；
- 因此 H1C 不通过 paper-core gate，但不能否定 interface 方向。历史上已进入 Stage A2/A3/A4
  继续验证结构化 interface 与 reliability signal。

Stage A decision after A4S and follow-up review：

- A2 nested 与 A3D teacher-preserved nested 证明 interface 是 material confounder 和必要控制；
- A3E 没有证明 target-conditioned nested 可作为最终 paper-core head；
- A4/A4R/A4S 证明 existing paths 的 reliability 差异存在，但没有跨 dataset 稳定可部署 signal；
- 但这并不允许跳过 Stage A。若论文声称 full-720 crop 有 interface mismatch，就必须先提出
  fair enough 的 unified prediction architecture；
- 因此 existing-path selector 暂停，Stage A 本身继续。下一步进入 A5：
  `Capacity-Preserving Prefix-Consistent Decoder`；
- Stage B `Reliability-Aware Future Supervision Routing` 暂缓为 A5 成立后的第二贡献，不能用来补
  Stage A 的逻辑缺口。

### Stage A5：Capacity-Preserving Prefix-Consistent Decoder

目标：基于 A2 nested 与 A3D teacher-preserved 的正向信号，重新设计一个 first-principles
unified prediction architecture。它必须在 head/decoder 层面解决 interface mismatch，而不是
继续做 shallow initialization、residual patch 或 existing-path selector。

A5 design requirements：

1. **Direct multi-prefix generation**
   h96/h192/h336/h720 由 decoder 根据 requested prefix 原生生成，不是先生成 720 再 crop。

2. **Prefix consistency**
   短 horizon 与长 horizon 的同一 future prefix 应共享一致的生成 contract，避免不同 horizon
   的输出语义互相冲突。

3. **Capacity preservation**
   新 head 不能因为 nested/query/basis 结构而丢掉 TimeAlign dense/full-head capacity。capacity
   preservation 必须来自 trained teacher、active full-head path、function-preserving constraint
   或明确的 distillation 机制；不能再把随机 row-copy 当作证据。

4. **Target-prefix awareness**
   requested prefix / target set 必须进入 decoder/head 本身，而不是只在 720 projection 前后做
   post-hoc modulation。

A5 narrative gate：

- 必须能解释为什么它比 A2 nested、A3D teacher-preserved、A3E target-conditioned 更根本；
- 必须说明 tensor/gradient path：prefix request 如何进入 decoder，capacity 如何保留，prefix
  consistency 如何实现；
- 若只是把 A2/A3D/A3E 机械叠加，不能作为 paper-core candidate，只能作为 diagnostic/control。

A5 effectiveness gate：

- 至少要超过 H1 `target_set_decoder_multiprefix` 与 A3D `teacher_preserved_nested` controls；
- 不能只在单个 dataset 或单个 horizon 上有效；
- 当前 gate universe 先维持 `ETTh2 + ETTm1 + Weather`，通过后再扩展到论文主表数据集。

### Stage A2：SCI-Level Unified Interface Redesign Historical Route

目标：不再尝试简单 post-hoc modulation，而是重新设计能回答审稿人质疑的 unified prediction
interface。该 interface 必须说明：一个模型如何原生服务多个 prefix horizons，而不是先生成
720 再裁剪。

H1C 已否定的实现族：

- output residual adapter；
- row-wise multiplicative gate 作为主贡献；
- hidden low-rank adapter before shared dense projection。

H1C 没有否定的 interface 方向：

1. **Prefix-Consistent Factorized Decoder**
   将 720-step output 分解为 shared temporal basis + prefix-conditioned coefficient /
   composition。核心不是给每个 horizon 一个 head，而是让任意 prefix 的输出由同一组可解释
   basis 组合得到，并保持 prefix consistency。

2. **Nested Segment / Cumulative Decoder**
   将未来预测拆成若干 nested segments 或 cumulative increments。短 horizon 预测是长 horizon
   预测的前缀组成部分，而不是从 720 输出中被动裁剪。该路线可直接连接 unified evaluation。

3. **Teacher-Preserved Unified Interface**
   用 fixed-horizon specialist 或 H1 target-set 作为 teacher，约束 unified interface 不丢失
   dense 720 capacity。该路线把 “preserve capacity” 从 initialization/architecture 变成
   explicit training constraint。

4. **Target-Set Query Decoder With Dense-Head Initialization**
   重新设计 H1B 的 query decoder，但必须从 dense head rows 或 fixed specialists 初始化 /
   distill，而不是随机训练 variable-prefix head。H1B 失败主要说明 random variable head
   capacity collapse，不说明 query decoder 思路不可行。

Stage A2 gate：

- 必须超过 H1 `target_set_decoder_multiprefix` 和 H1C `row_gated_dense_head_multiprefix`；
- 至少在 ETTm2 上明显降低 fixed gap，而不是只在 ETTh2 保留已有 unified benefit；
- 必须有结构解释：prefix consistency、capacity preservation 或 teacher preservation 至少成立一个；
- 若 A2 仍失败，才允许从论文逻辑上放弃 interface 贡献，并必须重新构造 SCI 级新路线。

A2 result：

- `dense_row_initialized_prefix_decoder_multiprefix` 失败，ALL 相对 H1 `+5.39%`，相对 H1C
  row-gated `+4.92%`，wins 均为 `0/12`；
- `nested_segment_decoder_multiprefix` 是 partial pass，ALL 相对 H0 `full` 为 `-2.12%`，
  相对 fixed 为 `-3.13%`，Weather 上 `4/4` horizon 赢 H1C；
- 但 nested segment 仍未过 gate：ALL 相对 H1 `+0.61%`，相对 H1C `+0.18%`，ETTm2
  fixed gap 仍约 `+1.81%`；
- 因此 A2 不进入 final method，但保留 nested composition 作为 A3 substrate。

### Stage A3：Nested Interface With Capacity / Teacher Preservation

目标：保留 A2 nested composition 的正向信号，同时修复它相对 H1/H1C 仍不足的问题。A3 不是
继续扩大 head sweep，而是围绕一个 substrate 做机制增强。

优先候选：

1. **Dense-Initialized Nested Segment Decoder**
   用 `proj_x.weight` 与 `proj_x.bias` 的对应 row slices 初始化 nested segment heads，测试
   nested composition + dense capacity preservation 是否能超过 H1/H1C。

2. **Teacher-Preserved Nested Segment Decoder**
   用 H1 target-set 或 H1C row-gated 作为 teacher，增加 prediction consistency /
   distillation loss，防止 nested interface 丢失当前 best carrier 的输出能力。

3. **Target-Conditioned Nested Segment Decoder**
   在 nested segment heads 中加入 target-set condition，测试 H1 的 requested-prefix signal
   能否与 nested composition 结合。

A3 gate：

- 必须超过 H1 `target_set_decoder_multiprefix` 和 H1C `row_gated_dense_head_multiprefix`；
- ETTm2 fixed gap 必须明显低于 H1/A2 的约 `+1.81%`；
- ETTh2 strong unified benefit 不能明显下降；
- Weather 的 A2 nested gain 不能消失。

A3-1 implementation：

- `readout-mode=dense-initialized-nested-segment-decoder`；
- forward 与 A2 `nested-segment-decoder` 一致；
- 每个 segment head 从 `proj_x` 对应 row slice 初始化；
- 只跑一个 arm：`dense_initialized_nested_segment_decoder_multiprefix`；
- comparison 同时包含 A2 nested、H1 target-set、H1C row-gated 和 fixed。

A3-1 result：

- ALL 相对 A2 nested 为 `-0.06%`，说明 shallow dense initialization 只有极弱修复；
- ALL 相对 H1 target-set 为 `+0.55%`，相对 H1C row-gated 为 `+0.12%`，未过 paper-core gate；
- ETTh2 相对 fixed 仍强 `-11.05%`，Weather 相对 H1C 为 `-0.06%`，说明 nested route 仍有
  partial evidence；
- 但当前 A3-1 复制的是同一模型中未训练的 `proj_x` rows，不是 learned full-head capacity。

A3-1 decision：

- 不能写成 capacity-preserving contribution；
- 不能继续调 initialization 或扩展 shallow head sweep；
- 下一步如果保留 Stage A，必须进入真正的 `teacher_preserved_nested_segment_decoder`、
  `target_conditioned_nested_segment_decoder` 或从已训练 checkpoint 初始化的
  `warm_started_nested_segment_decoder`。

A3B implementation：

- `readout-mode=target-conditioned-nested-residual-decoder`；
- base path：`proj_x(hidden)[:, :, :H]`，保留 dense full-head prefix 能力；
- residual path：`target_prefix` condition 后进入 zero-init nested segment residual heads；
- 初始函数严格等价于 dense full-head prefix，不再把随机 row-copy 误写成 learned capacity；
- comparison 同时包含 A2 nested、A3-1 shallow、H1 target-set、H1C row-gated 和 fixed。

A3B gate：

- primary：ALL 必须优于 A2 nested 与 A3-1 shallow；
- paper-core：ALL 或至少多数 dataset/horizon 必须超过 H1 target-set 或 H1C row-gated；
- ETTm2 fixed gap 必须低于 A2/A3-1；
- Weather 的 nested partial gain 不能消失。

A3B result：

- ALL 相对 A2 nested 为 `+4.42%`，相对 A3-1 shallow 为 `+4.48%`；
- ALL 相对 H1 target-set 为 `+5.09%`，相对 H1C row-gated 为 `+4.61%`；
- `0/12` horizon 赢 A2 nested、H1 或 H1C；
- 因此 A3B 不通过 effectiveness gate，并且只作为 diagnostic/control：nested 放在 residual
  path 上会削弱 primary nested 的正向信号。

### Stage A3C：Warm-Started Primary Nested Interface

目标：回到 A2 中有正向信号的 primary nested output contract，但用真正的 learned capacity
preservation 修复 A2/A3-1 的 capacity 问题。

Design：

- `readout-mode=checkpoint-initialized-nested-segment-decoder`；
- 从 H1 `target_set_decoder_multiprefix` checkpoint warm-start shared TimeAlign carrier；
- 将 checkpoint 中已训练的 `proj_x.weight/bias` row slices 复制到 nested segment heads；
- output 仍由 nested segment heads 直接生成，是 primary prediction interface，不是 residual correction。

Narrative gate：

- 通过。它直接服务于论文第一贡献 `Capacity-Preserving Prefix-Aware Interface`；
- 它保留 A2 的 primary nested interface 叙事，同时修复 A3-1 的 shallow initialization 错误；
- 它比 A3B 更符合 SCI 主线，因为 nested 仍是 prediction head 的主体，而不是 dense head 的附属补丁。

Effectiveness gate：

- 必须优于 A2 nested 与 A3B residual；
- paper-core gate 要求接近或超过 H1 target-set / H1C row-gated；
- 若 warm-start 后仍不能超过 H1/H1C，只能说明 `learned capacity preservation` 这一分支不足以
  修复 primary nested interface，不能直接判定 Stage A interface 主线失败；
- A3C 失败后的 rollback point 是 Step 4/5/6：对 remaining primary-interface candidates
  做 narrative-gate triage，优先评估 `teacher_preserved_nested_primary_decoder`、
  `target_conditioned_nested_primary_decoder`，以及二者的最小组合；
- 只有这些候选在 narrative gate 或 effectiveness gate 上连续失败，才进入 Stage A interface
  主线的顶级 SCI 审稿式重评估。

A3C result：

- ALL mean MSE 相对 A2 nested 为 `+0.07%`，相对 A3B residual 为 `-4.06%`；
- 相对 H1 target-set 为 `+0.68%`，相对 H1C row-gated 为 `+0.25%`；
- 因此 A3C 不通过 paper-core gate。它说明 primary nested 比 residual path 更合理，但
  row-slice warm-start 不足以 preserve H1 learned function。

### Stage A3D：Teacher-Preserved Primary Nested Interface

目标：在 A3C 的 primary nested output contract 上加入 H1 teacher consistency，使 nested primary
head 学习 prefix-consistent decomposition 的同时保留 H1 target-set function。

Design：

- student：`checkpoint-initialized-nested-segment-decoder`；
- teacher：frozen H1 `target_set_decoder_multiprefix` checkpoint；
- loss：label prediction loss + teacher consistency loss；
- gate arms：`teacher_preserved_nested_w03` 与 `teacher_preserved_nested_w10`。

Narrative gate：

- 通过。它是 A3C 失败后的最小机制修复，直接服务 `Capacity-Preserving Prefix-Aware Interface`；
- 与 A3B 不同，nested 仍然是 primary prediction interface；
- 与 A3C 不同，它不是只复制 weights，而是在训练期约束 student 保留 teacher learned function。

Effectiveness gate：

- 必须优于 A3C/A2；
- paper-core gate 要求接近或超过 H1/H1C；
- 若只复制 teacher 而不能改善 prefix/nested behavior，则降级为 teacher-preservation diagnostic。

A3D result：

- `teacher_preserved_nested_w03` 相对 A3C 为 `-0.73%`，相对 H1 为 `-0.06%`，相对 H1C 为
  `-0.48%`；
- 但 ETTm2 仍弱于 H1C，说明 teacher preservation 是有效修复，但还不能作为最终 paper-core；
- 因此 A3D 标记为 `partial_pass`，下一步进入 A3E。

### Stage A3E：Target-Conditioned Primary Nested Interface

目标：让 requested target set / prefix condition 直接进入 primary nested head，而不是只在 loss、
dense projection 前后或 teacher preservation 中间接体现。

Design：

- `target_conditioned_nested_warm`：与 A3C 使用同样 H1 warm-start，只用于隔离 target conditioning
  相对 A3C 的增量；
- `target_conditioned_nested_scratch`：不使用 H1 warm-start，只作为 diagnostic/control，判断该结构是否
  完全依赖 H1 initialization；
- target condition 通过 zero-init modulation 进入 nested primary hidden，使 warm arm 初始等价于 A3C。

Narrative gate：

- 通过。它直接回答 unified multi-prefix evaluation 和 prediction head 不一致的问题；
- warm-start 不是机制贡献，只是 initialization control；
- scratch arm 不作为 paper-core，只用于判断 target-conditioned nested 是否有独立结构能力。

Effectiveness gate：

- warm arm 必须优于 A3C，并至少接近 A3D/H1/H1C；
- paper-core gate 使用 `ETTh2 + ETTm1 + Weather`，其中 ETTm1 替换 ETTm2；目标不是只扩大 ETTh2
  收益，而是验证 target-conditioned primary nested 是否能在新的 minute-level dataset 上减少
  interface gap；
- 若 warm arm 改善而 scratch arm 失败，说明 target conditioning 有增量但依赖 capacity initialization；
- 若两者都失败，则优先回 Step 2/3/4 做 interface reliability diagnostic，而不是直接叠加 A3F。

A3E result：

- `target_conditioned_nested_warm` ALL 相对 A3C 为 `-0.25%`，相对 A3D 为 `+0.25%`，相对 H1 为
  `+0.20%`；
- `target_conditioned_nested_scratch` ALL 相对 A3C 为 `-0.26%`，相对 A3D 为 `+0.24%`，相对 H1
  为 `+0.18%`；
- ETTm1 上 A3C warm-started nested 仍是当前最强候选，A3E warm/scratch 均未超过 A3C；
- 因此 A3E 标记为 `failed_as_core_candidate`。A3F 暂不启动，因为 teacher preservation 与
  target conditioning 在 ETTm1 上没有同时正向，直接叠加不通过 narrative gate。

### Stage A4：Interface Reliability Diagnostic

目标：在不新增 head 的前提下，判断 Stage A 失败是否来自“没有一个 universal prefix-aware head”，
还是来自不同 capacity-preserving paths 在不同 future context 下可靠性不同。

A4 result：

- best path map 分散：ETTh2 偏向 H1/A3D/A3E，ETTm1 偏向 A3C，Weather 的短 horizon 甚至由
  fixed specialist 最优；
- ALL best static path 是 `a3d_teacher_preserved`，但 per-setting oracle 相对它只有
  `-0.431%` MSE 改善上限；
- 因此该结果只支持 reliability 差异真实存在，不支持把 dataset/horizon 手工路径选择写成
  paper-core 方法。

A4 decision：

- Stage A 不再表述为寻找单一 universal head；
- Stage A 也不能转成 manual per-dataset/per-horizon routing；
- 下一步进入 A4R：验证是否存在训练期或 validation-time 可观测的 reliability signals，例如
  teacher-student disagreement、prefix residual、validation gap 或 segment volatility；
- 若 A4R 找不到可解释 signal，必须回 Step 2/3 重审 interface contribution，而不是直接实现 routing。

A4R result：

- 使用现有 `training_log.csv` 中的 best/last validation MSE、train prediction L1、horizon-level train
  prediction L1、alignment loss、teacher L1 作为 signals；
- ALL-level 最强 signal 是 `last_train_alignment_loss`，Spearman 相关仅 `0.321`；
- ETTm1 内部有较高相关 signal，但 ETTh2/Weather 方向不稳定，且多数 signal 不是 horizon-specific。

A4R decision：

- 不进入 learned routing；
- 不把现有日志指标包装成 reliability selector；
- 下一步只做 A4S：新增轻量 prefix-wise validation diagnostic export，重点看 validation prefix MSE、
  teacher-student disagreement by prefix、prefix residual 是否能解释 A4 best-path map。

A4S design：

- 不训练新模型，只加载已有 checkpoints；
- 在 validation split 上导出 `validation_prefix_mse`、`prefix_vs_full_mse`、
  `teacher_student_mse`、`residual_abs_mean` 和 `residual_std`；
- 只覆盖 unified interface paths：H1、H1C、A2、A3C、A3D、A3E warm/scratch；
- fixed specialist 仍作为 problem evidence 和 performance reference，不作为 unified interface
  routing candidate。

A4S result：

- ALL-level 最强 signal 是 `teacher_student_mae`，Spearman 仅 `0.388`，未达到 signal-existence gate；
- ETTh2 的 teacher-student disagreement 有正相关，ETTm1 上 teacher-student MSE 变成负相关，
  Weather 上 full-context prefix MSE / residual 类 signal 更强但方向也不同；
- 这说明现有 unified paths 的 reliability 不是一个稳定、可部署的 validation-prefix selector 能解释的现象。

A4S decision：

- 不进入 learned routing；
- 不继续 existing-path mixture / selector；
- A4S 只否定 existing-path selector。Stage A 不能降级为 protocol/control；下一步进入 A5
  first-principles unified prediction architecture。

### Stage B：Future Supervision Reliability Diagnostic Deferred

目标：在 A5 unified prediction architecture 成立后，证明 future supervision 的 useful/harmful
差异真实存在，并判断这种差异是否应通过 gradient-path routing 处理。Stage B 当前暂缓，不是 active
next route。

建议诊断：

- `unit_reconstruction_error`：future unit 是否容易被 autoencoder 重构；
- `alignment_consistency`：history/future representation alignment 的稳定性；
- `residual_volatility`：该 future unit 的 local volatility 或 residual variance；
- `segment_unified_gap`：unified 相对 fixed 的 segment-level gap；
- `gradient_conflict_proxy`：不同 future units 对 shared representation 的梯度方向是否冲突。

Gate：

- A5 至少通过 narrative gate 并形成可运行 candidate；
- 至少一个 reliability proxy 与 future supervision 的 harmful/useful behavior 存在跨 dataset 可解释关系；
- proxy 不能只解释一个 isolated horizon，也不能依赖 test horizon id；
- 必须区分 `loss-only weighting` 与 `gradient-path routing`，否则不进入 method gate；
- 若 proxy 与 future branch harm 无关，不进入 M1/M2，回 Step 2/3 重新定义 HSS problem。

### Stage C：HSS Method Gate

目标：实现真正的 HSS 方法，而不是 diagnostic。

最小方法族：

1. `loss-only reliability weighting`：只作为 control；
2. `reliability-aware alignment scheduling`：调 future reconstruction/alignment supervision 强度；
3. `gradient-routed future supervision`：决定 future unit gradient 更新 shared path、adapter path 或被 detach；
4. `combined interface + routing`：最终候选。

Pass 条件：

- 比 A5 unified prediction architecture 稳定更好，并同时保留 H1/A3D/H1C controls；
- 比 loss-only control 更好；
- 不靠简单降低 TimeAlign future loss 获益；
- 至少在 `ETTh2 + ETTm1 + Weather` 的当前 gate universe 中稳定成立；后续论文主表再扩展到
  ETTm2/ETTh1 等 datasets。

Fail 决策：

- 若 loss-only 有效但 routing 无效：论文主线降级为 supervision reliability weighting，创新性降低；
- 若 routing 改善 segment 但 aggregate 不改善：回 Step 6 调 carrier capacity，不直接否定机制；
- 若所有 routing 无效：放弃 TimeAlign-HSS 主 carrier，转向更一般的 unified interface paper 或新 carrier。

### Stage D：论文级主实验

主数据集：

- ETTm1、ETTm2；
- ETTh1、ETTh2；
- Weather；
- ECL；
- 资源允许时加入 Traffic、Exchange。

主 baselines：

- TimeAlign fixed specialist；
- TimeAlign unified official；
- H0/H0B/H1/H1C internal controls；
- PatchTST、DLinear、TimesNet、iTransformer；
- 至少一个近期 strong baseline，具体依据后续 Zotero/官方代码审计确定。

主表：

- unified single-model multi-horizon MSE/MAE；
- fixed specialist reference gap；
- per-dataset average rank；
- parameter/training cost。

Ablation：

- without capacity-preserving prefix interface；
- direct variable-prefix head；
- loss-only reliability weighting；
- gradient routing without reliability；
- reliability routing without prefix-aware interface；
- official-last vs best-val selector sensitivity。

Mechanism figures：

- per-segment unified gap；
- reliability proxy vs improvement；
- gradient path usage / gate values；
- short-prefix vs long-prefix tradeoff。

## 论文结构草案

1. Introduction
   - unified multi-horizon forecasting 的实际需求；
   - evaluation horizon 与 training supervision unit 不应混同；
   - future-aware supervision 的双刃剑效应；
   - HSS 的核心贡献。

2. Related Work
   - multi-horizon forecasting；
   - unified / flexible horizon forecasting；
   - future-aware representation learning；
   - curriculum / dynamic supervision / gradient routing。

3. Problem Formulation
   - unified multi-horizon setting；
   - evaluation horizons as probes；
   - future supervision units；
   - supervision path scheduling。

4. Method
   - unified prediction interface；
   - future supervision reliability estimation；
   - reliability-aware loss/path routing；
   - training and inference complexity。

5. Experiments
   - datasets and baselines；
   - main results；
   - unified-vs-fixed gap；
   - ablation；
   - diagnostics。

6. Discussion
   - when HSS helps；
   - limitations；
   - relation to TimeAlign and other carriers。

## 转向规则

### 进入下一部分工作的条件

- H1C 已不通过 paper-core gate；A2 nested partial pass；A3-1 shallow initialization repair 不通过；
- Stage A3 candidate triage 产生超过 H1/H1C controls 的 teacher-preserved 或
  target-conditioned nested interface 后，进入 Stage B/D1 和 Stage C/M1/M2；
- D1 可以并行准备诊断，但只有在 A3 或当前最强 interface carrier 上验证后，才能支撑方法主线；
- M2 超过 loss-only control 后，进入 Stage D 主实验；
- Stage D 稳定后，开始 paper writing 与 figure/table freeze。

### 暂停或转向条件

- H1C 失败：停止当前 post-hoc readout sweep，但不停止 interface 主轴；回 Step 2/3/6 设计 A2；
- A3C 失败：只否定 warm-started nested primary 这一候选；回 Step 4/5/6 对
  teacher-preserved、target-conditioned 等 remaining primary-interface candidates 做 triage；
- A3 candidate triage 失败：必须重新进行顶级 SCI 审稿评估，再决定是放弃 interface、换
  carrier，还是重构论文主线；
- D1 失败：不做 routing 方法，回 Step 2/3 重定义 supervision reliability；
- M1/M2 只弱于 Stage A2/H1/H1C controls：不作为 paper-core，保留为 negative evidence；
- 方法只在一个 dataset 上有效：不扩主表，先做 failure analysis；
- 任何新机制无法同时解释 performance 和 paper story：不进入 full matrix。

## 与 11-Step 的同步方式

每次更新 `docs/research-roadmap.md` 的重要阶段时，必须同步检查本文档：

| 11-Step Node | 本文档同步动作 |
| --- | --- |
| Step 2/3 | 检查 `核心问题` 和 `目标 claim` 是否需要变化 |
| Step 4/5 | 检查 `预期贡献` 和 `方法边界` 是否仍成立 |
| Step 6 | 更新 `实验总安排` 中对应 stage 的 design/gate |
| Step 8 | 记录当前实验是否服务于 paper-core gate |
| Step 9/10 | 根据结果更新 `当前状态`、`转向规则` 和候选贡献 |
| Step 11 | 如果 rollback，明确是回到 interface、reliability、routing 还是 carrier 选择 |

本文档不追求记录所有实验细节。详细实验结果仍写入 `docs/research-roadmap.md` 和
`docs/experiments/`；本文档只保留能影响论文主线的结论。
