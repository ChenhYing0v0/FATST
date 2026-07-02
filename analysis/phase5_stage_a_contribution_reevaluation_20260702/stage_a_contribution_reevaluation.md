# Phase5 Stage A Contribution Re-evaluation

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

因此，正确处理不是把 interface 降级为无关诊断，而是把它从“已验证 method contribution”重构为：

> a required capacity-preserving prefix-aware carrier constraint and confounder-control protocol for HSS.

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

1. **Problem Formulation and Interface-Controlled Evaluation**
   形式化 unified multi-horizon 中的 `supervision-interface mismatch`：benchmark horizons 是
   evaluation probes，不应直接等同于 training units；同时证明 naive full-720 crop 会引入
   interface confounder。因此后续 HSS 必须在 capacity-preserving prefix-aware carrier 上评估。

2. **Reliability-Aware Future Supervision Routing**
   论文主方法不再是在 existing prediction heads 之间选择，而是控制 future supervision 的
   gradient path：哪些 future units 监督模型、监督强度多大、梯度允许进入哪些模块。

3. **Mechanism-Level Evidence Under Interface Control**
   用 H1/A3D 等 interface controls 证明方法收益不是 head choice、validation artifact 或简单
   loss weakening。

这个重构不会取消 interface 的价值，但会避免把一个尚未通过 effectiveness gate 的 head 方案强行写成主贡献。

## 为什么不继续做新 interface head

[Hypothesis] 仍可能存在更强的 first-principles prefix interface，例如 shared cumulative basis、
monotonic nested operator 或 function-preserving conditional decoder。但当前阶段不建议立即继续 A5 head sweep。

原因如下：

1. 已执行的 A2/A3C/A3D/A3E 覆盖了 primary nested、warm-start、teacher preservation、target conditioning 四类主要修复；
2. A4S 说明 existing path 的 validation reliability 不足，继续增加 path 很可能只扩大 search space；
3. 论文核心题目是 `Horizon-Agnostic Supervision Scheduling`，不是 prediction head architecture；
4. 高水平 SCI 的主线应把 interface 作为 HSS 的必要控制条件，而不是把研究资源继续耗在 head 微结构上。

[Self-Critique] 这个判断不等于“interface 方向失败”。它只是说：在当前证据下，继续设计 head 的边际论文价值低于进入 future-supervision gradient routing。若 Stage B 后续发现方法收益高度依赖某个 head，必须回到 interface mechanism 重新设计。

## 下一步研究计划

### Step 2/3：新问题定义

从现在开始，Stage B 的问题应定义为：

> 在 capacity-preserving prefix-aware carrier 已被控制的前提下，future-aware branch 的监督是否应该按 future-unit reliability 决定梯度路径？

这里的 reliability 不再是“哪个 existing head 最好”，而是“某个 future unit 的监督信号是否值得更新 shared encoder、future branch、alignment module 或 prediction head”。

### Step 4/5：候选核心 idea

下一步优先设计 `Reliability-Aware Future Supervision Routing`：

- reliable future units 可以更新 shared representation 和 prediction path；
- unreliable future units 只更新 future-specific / alignment auxiliary path，或降低进入 shared path 的梯度；
- routing signal 应来自训练期可观测的 future-unit behavior，例如 residual volatility、alignment consistency、reconstruction predictability 或 gradient conflict，而不是 test horizon id。

### Step 6：实验前 gate

进入实现前必须先完成一个 Stage B diagnostic plan：

- 选择固定 carrier：至少包含 H1 target-set 和 A3D teacher-preserved nested 作为 interface controls；
- 设计最小诊断：检查 future branch supervision 是否与 prediction loss 存在梯度冲突或 state-dependent harm；
- 明确 narrative gate：方法必须回答 gradient path scheduling，而不是简单 loss reweighting；
- 明确 effectiveness gate：必须在 `ETTh2 + ETTm1 + Weather` 上超过 H1/A3D controls，且不能只靠一个 dataset。

## 决策

[Decision] Stage A 作为 standalone `Capacity-Preserving Prefix-Aware Interface` method contribution 暂停，不再继续
existing-head sweep 或 existing-path selector。

[Decision] interface 问题保留为 paper-level `Problem Formulation and Interface-Controlled Evaluation`，并作为
Stage B 的必要 carrier constraint。

[Decision] 论文主方法进入 Stage B：`Reliability-Aware Future Supervision Routing`。下一次研究推进应先建立
Stage B ledger 和 diagnostic plan，再实现远程实验。
