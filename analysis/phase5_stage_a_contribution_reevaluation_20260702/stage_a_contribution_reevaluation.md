# Phase5 Stage A Contribution Re-evaluation

## 2026-07-02 共识修正

[Decision Update] 本报告最初倾向于把 Stage A 从 standalone method route 降级为
`interface-controlled evaluation`，并让主方法转入 Stage B。经过进一步审稿式讨论后，该判断被修正。

最新共识是：

1. 如果论文明确提出 `interface mismatch`，就必须给出一个能实质解决或缓解 mismatch 的
   unified prediction architecture；
2. `interface-controlled evaluation protocol` 只能作为实验规范，不能替代方法贡献；
3. Stage B `Reliability-Aware Future Supervision Routing` 是建立在 unified prediction architecture
   之后的增益性工作，不能用来补 Stage A 的逻辑缺口；
4. 当前应回到 Stage A，启动 A5 `Capacity-Preserving Prefix-Consistent Decoder` 的 Step 2/3/4
   设计，而不是新建 Stage B method ledger。

因此，本报告中“直接转入 Stage B”的内容仅作为被否定的中间判断保留。后续研究以
`docs/paper-mainline.md` 和 `docs/stage-ledgers/phase5-timealign-interface.md` 中的 A5 路线为准。

## 评估目标

本次重评估对应 11-step 的 Step 2/3：在 A4S 未通过 signal-existence gate 后，重新判断
`Capacity-Preserving Prefix-Aware Interface` 是否还能作为论文贡献 1 继续推进。

这不是一次新方法设计，也不是把 A4S 失败扩大为 HSS 失败。它只回答三个问题：

1. Stage A 已完成证据是否足以支撑一个 standalone interface method contribution；
2. 如果不足，interface 问题是否仍应保留在论文叙事中；
3. 下一步应回到新的 interface mechanism，还是转入 reliability-aware future supervision routing。

## 审稿人视角判断

### 结论 1：当前 Stage A 不能作为 standalone method contribution

[Strong Evidence] Stage A 的核心 method route 已经连续遇到三个边界：

- `A2_nested_segment_primary` 有局部正向信号，但相对 H1/H1C 不够强；
- `A3D_teacher_preserved_nested_primary` 是 partial pass，说明 function preservation 有效，但仍不能形成稳定 paper-core；
- `A3E_target_conditioned_nested_primary` 没有带来清晰增量，ALL 相对 A3C 只有约 `-0.25%/-0.26%`，且相对 A3D/H1 仍弱。

如果把这些结果包装成“我们提出一个更好的 unified head”，审稿人会认为证据不足。当前最强证据只能说明：

> interface design is a material confounder and a necessary control, not yet a validated standalone method.

### 结论 2：不能继续 existing-path routing

[Strong Evidence] A4/A4R/A4S 共同否定了 existing-path routing route：

- A4 证明 best path 分散，但 ALL oracle 相对 best static 只有 `-0.431%`；
- A4R 使用现有 training logs，ALL 最强 Spearman 只有 `0.321`；
- A4S 使用 prefix-wise validation signals，ALL 最强 `teacher_student_mae` Spearman 只有 `0.388`，且 dataset-level signal 方向不一致。

这说明不能把 H1/H1C/A2/A3C/A3D/A3E 之间的选择写成可部署方法。否则方法会退化为 test-oracle
或 dataset/horizon-specific rule，叙事上弱于高水平 SCI 要求。

### 结论 3：interface 问题仍然必须保留

[Strong Evidence] Stage A 不应被简单删除。已有结果仍然证明了 interface 是论文必须控制的因素：

- H1/H1C/A2/A3D/A3E 的相对表现不同，说明 head/interface 会实质影响 unified multi-horizon 结果；
- A2 nested 和 A3D teacher-preserved nested 都有正向信号，说明 prefix-aware / capacity-preserving 不是伪问题；
- 如果后续直接转向 future supervision routing，而不控制 interface，审稿人会质疑收益来自 head capacity 或 crop protocol。

因此，正确处理不是把 interface 降级为无关诊断。更严格地说，Stage A 必须继续承担论文核心结构问题：

> 在 prediction head / decoder 层面提出一个 fair enough 的 unified prediction architecture，
> 否则后续 HSS routing 的收益归因不成立。

## 贡献边界重构

### 被拒绝的写法

不应继续使用以下论文叙事：

> Contribution 1: We propose a new prefix-aware prediction head that solves unified multi-horizon forecasting.

当前没有足够证据支撑这个 claim。A3D 只能 partial pass，A3E/A4S 均未通过。

也不应写成：

> Contribution 1: We route among existing heads according to validation reliability.

A4S 已经否定了跨 dataset 稳定的可观测 signal。

### 保留并重构的写法

更稳健的贡献边界应改为：

1. **Problem Formulation**
   形式化 unified multi-horizon 中的 `supervision-interface mismatch`：benchmark horizons 是
   evaluation probes，不应直接等同于 training units；同时证明 naive full-720 crop 会引入
   interface confounder。

2. **Capacity-Preserving Prefix-Consistent Unified Prediction Architecture**
   在 head/decoder 层面解决 mismatch：短 horizon 不是从 720 被动 crop，而是由同一套
   prefix-consistent decoder contract 原生生成；同时必须保留 TimeAlign dense/full-head 的有效 capacity。

3. **Reliability-Aware Future Supervision Routing**
   在上述 architecture 成立后，进一步控制 future supervision 的 gradient path：哪些 future
   units 监督模型、监督强度多大、梯度允许进入哪些模块。

这个重构不会把未通过 gate 的旧 head 强行写成贡献，但也不会跳过 prediction architecture 这个核心问题。

## 为什么必须继续做新 interface head

[Hypothesis] 仍可能存在更强的 first-principles prefix interface，例如 shared cumulative basis、
monotonic nested operator 或 function-preserving conditional decoder。最新共识认为当前阶段必须继续 A5，
但不能做零散 head sweep。

原因如下：

1. 若论文声称 naive full-720 crop 有 mismatch，就必须提出 fair unified head；
2. Stage B 是增益性 supervision routing，不能替代 architecture；
3. A2 nested 与 A3D teacher-preserved 已经给出正向信号，说明 prefix-consistent / capacity-preserving
   不是伪方向；
4. 失败的是 shallow initialization、residual patch、existing-path selector，而不是 first-principles
   unified prediction architecture。

[Self-Critique] A5 的风险是继续陷入 head variant search。因此 A5 必须先过 narrative gate：
它要有清楚的 decoder contract、capacity-preservation path 和 prefix-consistency mechanism，不能只是
把 A2/A3D 组件机械叠加。

## 下一步研究计划

### Step 2/3：新问题定义

从现在开始，A5 的问题应定义为：

> 如何设计一个 fair enough 的 unified prediction architecture，使 h96/h192/h336/h720
> 不是 naive 720 crop，而是由同一套 prefix-consistent、capacity-preserving decoder contract 生成？

这里的核心不是哪个 existing head 最好，而是 prediction architecture 本身是否能同时满足
direct multi-prefix generation、prefix consistency、capacity preservation 和 target-prefix awareness。

### Step 4/5：候选核心 idea

下一步优先设计 `Capacity-Preserving Prefix-Consistent Decoder`：

- nested / cumulative decoder 提供 prefix-consistent structure；
- teacher/full-head preservation 提供 function/capacity preservation；
- target prefix 直接进入 decoder，而不是先生成完整 720 再裁剪；
- 训练和评估围绕 direct prefix output，而不是 post-hoc crop。

### Step 6：实验前 gate

进入实现前必须先完成一个 A5 design plan：

- 明确 decoder forward contract：输入、输出、prefix request 如何进入 head；
- 明确 capacity preservation 来自 trained teacher、active function-preserving path 或 explicit consistency loss；
- 明确为什么它比 A2/A3D/A3E 更根本，而不是旧组件堆叠；
- 明确 effectiveness gate：必须至少超过 H1 和 A3D controls，且不能只靠一个 dataset。

## 决策

[Decision] existing-head sweep 与 existing-path selector 暂停，但 Stage A 不能暂停。

[Decision] Stage A 必须回到 Step 2/3/4，设计 first-principles `Capacity-Preserving Prefix-Consistent
Unified Prediction Architecture`。

[Decision] Stage B `Reliability-Aware Future Supervision Routing` 暂缓，只有在 A5 architecture
通过 narrative gate 并形成可运行 candidate 后，才作为第二贡献继续推进。
