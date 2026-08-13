# ISCF-BSCA 论文结构与叙事共识稿

## 文档状态

| Field | Content |
| --- | --- |
| `document_role` | ISCF-BSCA 论文全文结构、术语、claim 与实验布局的权威讨论稿 |
| `version` | `v0.69` |
| `last_updated` | `2026-08-13` |
| `paper_candidate` | architecture family frozen；`ISCF-BSCA-v1`=ablation anchor；`ISCF-BSCA-MAIN-v1`=tuned main candidate |
| `current_review_cursor` | writing=Sections 5--7 v0.2 author-fixed structure temporarily frozen；experiments=H5B fallback frozen，H5C 54/54 training complete、manifest frozen、formal test authorized |
| `restart_handoff` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md` |
| `experiment_handoff` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-experiments-restart-handoff-20260731.md` |
| `experiment_protocol` | `configs/iscf_bsca_paper_experiment_protocol.json` |
| `paper_table_registry` | `docs/iscf-bsca-paper-table-registry.md`；machine contract=`configs/iscf_bsca_paper_table_registry.json` |
| `frozen_consensus` | 论文七章结构并保留standalone Discussion；varied-horizon主问题；CHPC为basic property；ISCF decoder-side scope framework；BSCA train-only contribution boundary |
| `temporarily_frozen_content` | Introduction P1--P6 v0.9正文 + approved Figure 1；Section 2 v0.2正文、subsection structure、citations与claim boundaries；Section 3 v0.7正文 + approved Figures 2--3；Section 4 v0.7正文、公式与Figure 4 integration/caption；Method Figure 4 visual design；Sections 5--7 v0.2 structural design |
| `provisional_content` | Method Figure 4 stable vector-asset synchronization；remaining manuscript prose and pending experiment evidence |
| `authorization_source` | 2026-08-13用户进一步授权将ETTh1 HPO matrix扩大约50%并继续优化；H5C限54个seed2021 profiles、三GPU training及complete-manifest后的完整formal test，extra seeds/architecture redesign/automatic table mutation未授权 |

本文档用于逐段讨论论文，而不是宣告全文已经定稿。标记为
`frozen_consensus` 的内容在出现新证据或明确讨论结论前保持不变；
`temporarily_frozen_content` 只有在后续章节或证据产生明确矛盾且用户同意后才解冻；
`provisional_content` 只表示当前最佳结构，后续按章节继续修订。

H5B result amendment：ETTh1 expanded HPO已完成36/36 checkpoints与144/144 formal-test
rows，选择`h5b_seq640_p20`。Main II ETTh1 best cells由2/8提高到4/8，four-H mean
MSE/MAE均改善，但H336两项metric轻微退化。该结果只更新paper-facing performance
evidence状态，不改变architecture、method claims或Sections 1--4；表格替换仍等待单独授权。

H5C prelaunch amendment：本轮只在H5B winner附近细化context/patch、LR、dropout、
weight decay与rank interactions，不改变method graph。54-trial matrix及success gate属于
paper-facing hyperparameter optimization，不产生新的architecture contribution或mechanism claim。

## 1. 核心术语

### 1.1 Forecast horizon、future time step 与 forecast target

给定长度为 $L$ 的历史观测

$$
\mathbf X
\in
\mathbb R^{L\times C},
$$

本文区分以下对象：

| Term | Symbol | Definition |
| --- | --- | --- |
| `forecast horizon` | $H$ | 一次请求覆盖的最大未来长度，例如 $96,192,336,720$ |
| `future time step` | $\tau$ | 第 $\tau$ 个未来时间位置，$1\leq\tau\leq H$；正文可简称 `future step` |
| `lead time` | $\tau$ 或物理时间间隔 | forecast origin 到目标时刻的距离；需要强调真实时间距离时使用 |
| `forecast target` | $(\tau,c)$ | 第 $\tau$ 个 future time step、第 $c$ 个变量对应的标量预测目标 |
| `future-step embedding` | $\phi(\tau)$ | decoder 内部表示 future-step position 的固定或学习描述符 |

`future coordinate` 不再作为 Introduction 和问题定义中的主术语。只有在讨论
数学坐标系或代码中的 `coordinate_field` 时，才允许使用
`future-step coordinate` 或 `future-step embedding`。

### 1.2 Future-region sharing-demand heterogeneity

问题层面的正式术语为：

> **future-region sharing-demand heterogeneity**

中文称“未来区间共享需求异质性”。其中 future region
$\mathcal B_b\subseteq\{1,\ldots,T\}$ 是预测域内部一组连续的 future time
steps，不是 requested forecast horizon。

这里的sharing demand指：在finite-capacity decoder中，一个由历史构造的
`history-conditioned latent state`适宜被多宽范围的future steps共同复用。
它不是future targets之间真实的probabilistic dependence，也不是某个future
step自身的属性。

问题的机制假设是：broad sharing通常提供更强的结构约束并降低估计variance，
但可能增加局部变化的approximation bias；fine-grained或step-specific
generation提供更强的局部自由度，但可能增加参数估计与优化难度。因此，对
future region $\mathcal B_b$ 和sharing extent $s$，可将有限容量风险概念化为：

$$
R_b(s)
=
\operatorname{Bias}_b^2(s)
+
\operatorname{Variance}_b(s)
+
\operatorname{Noise}_b.
$$

如果细粒度变化与平滑、宽范围轨迹成分的相对重要性随sample、variable与future
region改变，则不同regions的bias--variance optimum可能不同。这只是需要matched
evidence验证的建模假设，不是由“时间序列具有multi-scale structure”自动推出的
定理。

该术语描述task与finite-capacity output-side sharing pattern之间的mismatch，
不预先指定ISCF、scope arm、grouping或fusion。它也不表示requested horizon
改变了同一future step的Bayes conditional mean。

### 1.3 Region-dependent sharing-scale preference

问题的可检验表现称为：

> **region-dependent sharing-scale preference**

对一组 capacity-matched diagnostic predictors $D_s$，$s$ 只控制多个 future
steps 之间的 predictive-state sharing scale。定义 future region
$\mathcal B_b$ 上的风险：

$$
R_{b,s}
=
\mathbb E\left[
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_{c=1}^{C}
\left(
D_s(\mathbf X)_{\tau,c}-Y_{\tau,c}
\right)^2
\right].
$$

若不同 regions 的 matched risk curves 稳定交叉，或
$s_b^\star=\arg\min_sR_{b,s}$ 稳定不同，则支持 region-dependent
sharing-scale preference。这里的“preference”仅表示 matched empirical risk
更低，不是数据区间对某种架构的内在或绝对偏好。

### 1.4 Scope-indexed forecast field与latent-state sharing scope

ISCF正式解释为：

> **Independent Scope-Conditioned Forecasting**

它不是多个完整predictors的ensemble，而是在future-step与sharing-scope的乘积域
上定义一个统一的：

> **scope-indexed forecast field**

$$
\mathcal F_\theta:
(\mathbf X,\tau,c,s)
\mapsto
\hat y_{\tau,c}^{(s)}.
$$

对固定$s$，$\mathcal F_\theta(\cdot,\cdot,\cdot,s)$称为该field的一个
`scope-conditioned slice`，而不是独立forecasting model。各slices共享encoder、
future-step descriptors、future-step-specific synthesis vectors与训练目标，只在
scope-specific history projection及其latent-state sharing pattern上区分。

方法层面把decoder在多个future time steps之间复用latent state的范围称为：

> **cross-step latent-state sharing extent**

一个具体结构称为：

> **future-step latent-state sharing scope**

对scope size $s$，其结构语义为：

> 同一scope region内的$s$个future time steps在各自的step-specific synthesis
> 之前，共享一个由history representation与该region的future-step descriptor
> 共同构造的`history-conditioned, region-indexed latent state`，简称
> `scope-region latent state`。

ISCF当前使用：

$$
\mathcal S=\{1,48,144,360,720\}.
$$

其中$s=1$提供最细粒度的latent-state sharing，$s=720$提供full-domain
sharing。该概念属于output-side decoder structure，不等同于requested
forecast horizon、input receptive field、input temporal resolution、frequency
band 或 forecast targets 之间真实的 probabilistic dependence。

target-conditioned scope allocation定义为：

$$
\pi_\theta(s\mid\mathbf X,\tau,c),
\qquad
\sum_{s\in\mathcal S}
\pi_\theta(s\mid\mathbf X,\tau,c)=1.
$$

最终prediction通过沿scope轴进行weighted contraction得到：

$$
g_\theta(\mathbf X,\tau,c)
=
\sum_{s\in\mathcal S}
\pi_\theta(s\mid\mathbf X,\tau,c)
\mathcal F_\theta(\mathbf X,\tau,c,s).
$$

因此，`future-region sharing-demand heterogeneity`是问题，
`region-dependent sharing-scale preference`是可检验表现，
`future-step latent-state sharing scope`是ISCF的结构响应，
`target-conditioned scope allocation`负责在每个forecast target处整合不同
scope-conditioned slices。单个future step本身没有sharing scope；scope描述一组
future steps之间的latent-state sharing关系。

### 1.5 Horizon无关、future-step-indexed generation

本文把统一预测函数写成：

$$
g_\theta:
(\mathbf X,\tau,c)
\mapsto
\hat y_{\tau,c}.
$$

这里 $g_\theta$ 是 `horizon-agnostic`：同一个 future time step 的预测函数不把
requested horizon $H$ 作为语义输入。一个 $H$-step forecast 直接定义为：

$$
\hat{\mathbf Y}^{(H)}
=
\left[
g_\theta(\mathbf X,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

中文叙事统一使用“horizon无关”；英文架构表述使用 `horizon-agnostic`；
同一 future step 的预测不随 requested horizon 改变这一性质使用
`horizon-invariant`。不使用可能被误解为统计独立的 `horizon-independent`。

当前ISCF以future-step-specific synthesis vectors实现该接口：每个future
time step $\tau$具有自己的identity synthesis vector、nonlinear synthesis
vector与temporal bias。它们把对应scope-region latent state映射为
$\mathcal F_\theta(\mathbf X,\tau,c,s)$，再由target-conditioned scope
allocation沿scope轴进行weighted contraction。这里的`step-specific`不表示
future steps相互独立；它们仍可在synthesis之前通过不同sharing scopes共享
latent state。

### 1.6 Cross-horizon prefix consistency

本文将统一模型需要满足的 nested-output contract 称为：

> **cross-horizon prefix consistency (CHPC)**

固定 forecast origin 和完全相同的历史输入 $\mathbf X$。若
$H_1<H_2$，由上述 step-indexed function 定义的模型满足

$$
\Pi_{H_1}\hat{\mathbf Y}^{(H_2)}
=
\left[
g_\theta(\mathbf X,\tau,c)
\right]_{\tau=1,\ldots,H_1;\ c=1,\ldots,C}
=
\hat{\mathbf Y}^{(H_1)},
$$

则称其满足 CHPC，其中 $\Pi_H$ 表示保留前 $H$ 个 future time steps 的
prefix operator。CHPC 来自每个 future-step prediction 的 horizon-invariant
定义，不在 Introduction 中表述为“先完整生成 max-$T$ 再 crop”。

CHPC 是 task/system contract，不单独作为算法创新。本文的核心问题是：如何在
满足 CHPC、使用一个 unified model 的同时，避免 single fixed
cross-step sharing pattern 在整个 future domain 上形成表达折中。

### 1.7 明确弃用或限制的表述

| Avoid | Replacement / Boundary |
| --- | --- |
| `single-checkpoint multi-horizon forecaster` | Introduction 使用 `unified multi-horizon forecaster` |
| `same-origin cross-horizon prefix consistency` | 正式术语使用 CHPC；`fixed forecast origin` 写入定义 |
| `future-region predictive-structure heterogeneity` | 过于宽泛，无法直接推出sharing extent；问题使用 `future-region sharing-demand heterogeneity` |
| `Independent Scope-Coupled Fields` / multiple scope fields | ISCF=`Independent Scope-Conditioned Forecasting`；整体对象为单一`scope-indexed forecast field`，固定$s$只是一个slice |
| `future-generation scope heterogeneity` / `future-step coupling heterogeneity` | 问题使用 `future-region sharing-demand heterogeneity`；方法使用 `future-step latent-state sharing scope` |
| `future-step coupling granularity` / “某个future step具有某种coupling granularity” | 使用`latent-state sharing granularity/extent`；scope是多个future steps之间的共享关系 |
| `forecast-target-wise fusion policy` | 使用`target-conditioned scope allocation`；最终运算称`weighted contraction along the scope axis` |
| `history-conditioned generation state` | 问题层使用`history-conditioned latent state`；方法层使用`history-conditioned, region-indexed latent state`或`scope-region latent state` |
| `step-specific synthesis coefficients` | 使用`future-step-specific synthesis vectors`；代码对象是两个$K$维vector及bias，不是标量mixing coefficients |
| `future-step dependency heterogeneity` | 禁止；容易被误解为联合概率分布中的temporal dependence |
| `forecast step` / `future coordinate` | Introduction prose 使用 `future time step`；标量目标使用 `forecast target` |
| `horizon-independent` | 中文使用“horizon无关”；英文架构使用 `horizon-agnostic`，性质使用 `horizon-invariant` |
| “先生成max-$T$再crop”作为宏观定义 | 使用 horizon无关、future-step-indexed function 直接定义任意 $H$-step forecast |
| “independently generate different horizons” | 使用 `directly instantiate arbitrary horizons`；不同 horizons 是nested而非相互独立 |
| `temporal coherence` 表示 CHPC | 禁止；该术语已广泛用于 temporal-hierarchy reconciliation |
| `forecast stability` 表示 CHPC | 禁止；通常指不同 forecast origins/creation dates 之间的 revision |
| `strict multi-horizon gradient conflict` | 当前证据不支持；只允许写 heterogeneous demands 或 shared-decoder compromise |
| `all existing models lack prefix consistency` | 禁止；只能说标准 horizon-specific 多模型协议不提供 CHPC 保证 |

## 2. 论文总叙事

论文主线冻结为：

> 真实应用需要一个模型同时服务多个 nested forecast horizons。标准
> horizon-specific protocol 为每个 $H$ 独立训练模型，不能保证重叠 forecast
> horizons 中相同 future time steps 的预测一致，也带来重复训练与部署成本。horizon无关、
> future-step-indexed generation 可以直接实例化任意 horizon，并保证重叠
> future time steps 的预测一致；但许多direct multi-output forecasters从
> broad shared representation或单一固定output-generation pattern生成所有
> future steps。细粒度变化可能需要更强step-specific flexibility，平滑、宽范围
> 轨迹成分则可能受益于跨步复用共同latent state。单一共享范围未必能适合
> 所有samples、variables与future regions。ISCF通过多个independent
> scope-specific history projections构造具有不同cross-step latent-state
> sharing extents的scope-region states，并与共享的future-step-specific
> synthesis vectors共同形成一个scope-indexed forecast field；
> target-conditioned scope allocation在每个forecast target处沿scope轴整合该
> field；BSCA在不改变inference graph的前提下稳定scope slices与allocation的
> joint training。

该主线包含四层：

1. `system need`：一个 unified model 服务多个 nested horizons；
2. `system contract`：CHPC；
3. `modeling problem`：future-region sharing-demand heterogeneity；
4. `solution`：ISCF architecture + BSCA training。

## 3. 全文结构

```text
Abstract

1. Introduction

2. Related Work
   2.1 Fixed-Horizon Multi-Step Forecasting
   2.2 Unified and Varied-Horizon Forecasting
   2.3 Forecast Generation and Output-Side Modeling
   2.4 Multi-Scale Forecasting and Adaptive Allocation

3. Problem Formulation and Empirical Motivation
   3.1 Varied-Horizon Forecasting and Cross-Horizon Prefix Consistency
   3.2 Horizon-Specific Prefix Inconsistency
   3.3 Future-Region Sharing-Demand Heterogeneity

4. ISCF-BSCA: Prefix-Consistent Unified Multi-Horizon Forecasting
   4.1 Architecture Overview
   4.2 History State and Future Coordinate
   4.3 Generation of Scope-conditioned Forecasts
   4.4 Target-Adaptive Scope Allocation
   4.5 Balanced Scope Co-Adaptation

5. Experiments
   5.1 Experimental Setup
   5.2 Comparison with Horizon-Specific Forecasters
   5.3 One-Model-All-Horizons Evaluation
   5.4 Efficiency and System Cost
   5.5 Component and Training-Objective Ablations
   5.6 Forecast Consistency and Scope-Allocation Behavior
   5.7 Backbone Transferability

6. Discussion
   6.1 From Horizon-Specific Predictions to a Unified Forecasting System
   6.2 Output-Side Sharing as a Forecasting Design Dimension
   6.3 Limitations and Future Scope

7. Conclusion

Appendices
   A. Full Dataset-Horizon Results
   B. Additional Coupling-Scope Diagnostics
   C. Hyperparameter Sensitivity
   D. Reproducibility Details
```

standalone `Discussion`用于分离result observation、task/system interpretation与limitations。完整negative cells、secondary controls与敏感性结果放在Appendices。

### 3.1 Sections 5--7 author-fixed structural design

`docs/paper-drafts/iscf-bsca-sections-5-7-initial-design.md`的v0.2结构已由author确认并暂时固定。后续正文按`Experiments -> Discussion -> Conclusion`组织；Experiments依次承担system effectiveness、one-model capability、cost、matched attribution、consistency/allocation behavior与transfer。Qualitative example并入5.6，不另设case-study/failure-case subsection。

## 4. Introduction

当前clean manuscript正文以
`docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`
的`v0.9-author-refinement`为准。Introduction只保留一张由constructed
curves组成的双面板Figure 1，用于直观说明prefix disagreement与
future-region sharing-demand heterogeneity；不在Introduction展开dataset、
statistics、controls或sample-selection。两张approved real-data figures及其完整
proof protocol移至Section 3，暂按Figures 2--3组织。

### 4.1 Paragraph 1：multi-horizon need 与现行 horizon-specific protocol

**状态：v0.8 round1初步修订；varied-horizon literature定位待逐段确认。**

写作目标：

1. 从多个 planning ranges 的真实需求出发；
2. 定义本文研究的是 nested horizons 上的 unified multi-horizon forecasting；
3. 指出现行 benchmark 通常按 $H$ 分别训练模型；
4. 暂不讨论 ISCF、scope 或 loss。

建议正文逻辑：

> Multi-horizon forecasts support decisions over multiple planning ranges,
> from short-term control to long-term scheduling. Yet most long-term
> time-series forecasting models and benchmark protocols remain
> horizon-specific: a separate model is trained for each forecast horizon
> $H$, such as 96, 192, 336, and 720 steps. Recent work, including ElasTST and
> time-series foundation models such as TimesFM and Time-MoE, has begun to
> support varied or flexible forecasting horizons. Nevertheless, such efforts
> remain sparse relative to the extensive horizon-specific literature, and
> varied-horizon forecasting is still insufficiently developed as a unified
> problem with an explicit task definition, systematic problem analysis, and
> targeted decoder design. In this work, we formulate these requirements
> explicitly and investigate how a unified forecaster should organize
> output-side representations across different parts of the future domain.

这里的 `multi-horizon` 特指：同一个 forecasting system 服务多个请求长度
$H\in\mathcal H$，而不仅是一个模型一次输出多个 future steps。

### 4.2 Paragraph 2：horizon-specific systems 的系统割裂

**状态：v1.1 author-refined；Figure 1a用于说明horizon-specific systems的
prefix disagreement，并置于本段。**

本段只建立三项系统问题：

1. 相同历史、相同 future time step 可能因请求的 horizon 不同而产生不同预测；
2. 多套独立训练、存储与部署成本；
3. 不能自然解释为一个完整未来轨迹的 nested views。

对相同 $\mathbf X$ 和 $H_1<H_2$，独立模型通常不保证

$$
f_{\theta_{H_1}}(\mathbf X)
=
\Pi_{H_1}f_{\theta_{H_2}}(\mathbf X).
$$

建议正文：

> The horizon-specific protocol fragments forecasting into independent
> systems. As illustrated in Figure 1a, models trained for different horizons
> can assign different values to the same future time step, despite identical
> history and forecast origin. Their outputs therefore need not form coherent,
> nested views of one future trajectory. Serving multiple horizons also
> requires separate training, storage, deployment and maintenance.

对应中文：

> Horizon-specific协议将预测拆分为多个独立系统。如Figure 1a所示，即使
> history与forecast origin相同，针对不同horizon训练的模型仍可能对同一future
> time step给出不同值。因而，这些输出未必能构成同一future trajectory的一组
> 连贯nested views。同时，服务多个horizons还需要分别训练、存储、部署和维护
> 模型。

讨论过程中的细节：

1. 本段从七句压缩为四句，只保留system fragmentation、overlap disagreement与
   deployment redundancy。
2. Figure 1a只承担horizon-specific prefix disagreement的概念说明，因此保留在
   problem paragraph，不再用于解释“why CHPC is needed”。
3. 本段不声称horizon-specific models的accuracy必然更差。

### 4.3 Paragraph 3：unified forecaster 与 CHPC

**状态：v1.1 author-refined；采用future-step-indexed mapping正式定义CHPC，
不再承担Figure 1a的问题证据叙事。**

建议正文逻辑：

> We therefore formulate varied-horizon forecasting as learning a single
> horizon-agnostic mapping from observed history and future-step index to
> prediction. Under this formulation, a future-step prediction depends on the
> history and its step index, but not on the requested horizon. For any
> $H_1<H_2$, predictions over steps $1{:}H_1$ therefore remain identical under
> both requests. We call this basic requirement **cross-horizon prefix
> consistency (CHPC)**.

对应中文：

> 因此，我们将varied-horizon forecasting形式化为学习一个从observed history与
> future-step index到prediction的统一horizon无关映射。在这一形式化下，某个
> future step的预测由history及其step index决定，而不由requested horizon决定。
> 对于任意$H_1<H_2$，两种请求在$1{:}H_1$范围内的预测因此保持一致。我们将
> 这一基本要求称为
> **cross-horizon prefix consistency（CHPC）**。

讨论过程中的细节：

1. CHPC仍是varied-horizon forecasting system的basic requirement，不包装为
   算法创新。
2. 正文不采用“先生成max-$T$再crop”的实现叙事，只定义horizon无关、
   future-step-indexed prediction mapping。
3. Figure 1a引用已移回Paragraph 2；本段只负责formalization与CHPC naming，
   避免把problem illustration误写为requirement rationale。

### 4.4 Paragraph 4：naive unification 与 future-region sharing demand

**状态：v1.0 polished；以uniform output mechanism自然引出latent-state
sharing，Figure 1b嵌入problem命名句。**

建议正文：

> CHPC defines how forecasts at different horizons should relate, but not how
> a decoder should generate individual future regions. Most architectural
> advances focus on history encoding, while the output stage often uses one
> uniform mechanism to generate all future steps. Such a mechanism fixes how
> broadly a history-conditioned latent state is shared before step-specific
> prediction. Broad sharing can capture persistent trajectory structure, but
> may smooth local variations. Finer sharing offers greater step-specific
> flexibility, but provides weaker structural regularization. The preferred
> balance can vary across samples, variables and future regions, making one
> fixed sharing extent inadequate for the entire forecast domain. Figure 1b
> summarizes this intuition, which we term **future-region sharing-demand
> heterogeneity**. We examine this problem in greater detail in Section 3.

对应中文：

> CHPC规定了不同horizons的预测应当如何关联，但不决定decoder应如何生成各个
> future regions。多数架构进展集中在history encoding，而output stage通常沿用
> 一种统一机制生成全部future steps。该机制固定了history-conditioned latent
> state在step-specific prediction之前被共享的范围。Broad sharing有利于捕获
> 持续的trajectory structure，但可能平滑local variations；finer sharing提供
> 更强的step-specific flexibility，但结构正则更弱。这一平衡可能随sample、
> variable与future region变化，使固定sharing extent难以适配整个forecast
> domain。Figure 1b概括了这一直觉，我们称之为**future-region
> sharing-demand heterogeneity（未来区间共享需求异质性）**。我们将在Section
> 3进一步说明该问题。

讨论过程中的细节：

1. **本段的叙事作用。** Paragraph 3只建立horizon无关接口与CHPC；Paragraph
   4进一步指出prediction interface一致并不等于decoder设计已经解决，从system
   contract自然转入output-side sharing problem。
2. **问题、证据与方法必须分层。** 问题层术语是
   `future-region sharing-demand heterogeneity`；其可检验表现是
   `region-dependent sharing-scale preference`；`future-step latent-state
   sharing scope`只属于ISCF的方法层。不能用模型中的scope反过来定义问题。
3. **future region不是forecast horizon。** future region
   $\mathcal B_b\subseteq\{1,\ldots,T\}$ 是预测域内部一组连续future steps；
   forecast horizon $H$ 是一次预测请求覆盖的总长度。不同future regions的
   sharing demand可能不同，不表示同一future step会因requested horizon变化而
   改变预测目标。
4. **scope是关系而不是单点属性。** 单个future step本身没有sharing scope；
   scope描述多个future steps在target-specific synthesis之前共享latent state
   的范围。因此正文不写“different future steps have different coupling
   granularities”。
5. **从multi-scale intuition到sharing demand需要中间机制。** broad sharing
   可能以更强结构约束降低estimation variance，但增加local-detail bias；
   fine-grained generation可能提高局部flexibility，但增加参数估计与优化难度。
   因此，只有在不同future regions的bias--variance optimum确实不同时，才能
   推出region-dependent sharing demand。时间序列包含multi-scale成分本身并不
   足以完成该推导。
6. **现有decoder不是“global versus independent”的简单二分。**

   - iTransformer使用同一个variate-level representation经
     `Linear(d_model, pred_len)`生成全部future steps，属于global shared state
     加step-specific projection rows；
   - PatchTST将全部history-patch representations展平后经
     `Linear(d_model * patch_num, target_window)`输出，属于broad shared
     representation加step-specific rows；
   - DLinear使用`Linear(seq_len, pred_len)`，每个future step具有独立linear
     row，但共享输入分解与joint training，不等于独立nonlinear generation
     state；
   - N-HiTS使用hierarchical interpolation和multi-scale additive synthesis，
     已包含预定义多尺度trajectory sharing，但不是本文所研究的
     region-dependent adaptive sharing。

7. **经验命题而非普适规律。** “不同future regions可能受益于不同sharing
   scales”必须由第三章capacity-matched predictors的region-wise risk
   crossing、best-scale变化与相对best fixed scale的headroom支持。数据侧
   multi-scale energy只能作描述性辅助。
8. **理论与claim边界。** Bayes-target说明从Introduction正文删除，但内部边界
   保持不变：在fixed history、pointwise MSE且requested horizon
   不提供额外信息时，同一future step的Bayes conditional mean不依赖$H$。本文
   讨论的是finite-capacity decoder的sharing topology与inductive-bias
   mismatch，不是horizon-conditioned target变化；本段也不预设strict negative
   gradient conflict、short/long horizon与local/global scope的一一对应关系。
9. **弃用术语。** `predictive structure`缺少可计算边界，不能从其尺度差异直接
   推出sharing extent差异，因此不再用
   `future-region predictive-structure heterogeneity`命名问题。不使用
   `future-generation scope heterogeneity`或`future-step coupling
   heterogeneity`作为问题名称，因为两者预先嵌入方法设计；也不使用
   `future-step dependency heterogeneity`，以免被误解为联合概率分布中的
   temporal dependence。

#### Introduction Figure 1：constructed concept illustration

Figure 1采用一张双面板schematic-led composite：

- panel a使用三条constructed horizon-specific trajectories，分别终止于
  $H_1,H_2,H_3$，并在共同future step $\tau^\star$处标出不同预测值；
- panel b使用fine/intermediate/broad三条constructed risk curves，通过
  early/middle/late regions的最低曲线变化直观解释sharing-demand
  heterogeneity；
- 两panel使用彼此独立的restrained color families，不建立跨panel颜色映射，
  避免暗示forecast horizon与sharing extent存在一一对应关系；
- 图底保留`constructed curves; not empirical data`；caption不再重复该说明；
- Figure 1不报告dataset、metric、sample、NCHPD、MSE或headroom。Section 3
  Figures 2--3才承担正式problem evidence。

精简caption：

> **Figure 1 | Two challenges in varied-horizon forecasting.** **a**,
> Horizon-specific predictors may disagree at the same future time step
> $\tau^\star$ despite identical observed history. **b**, The sharing extent
> associated with the lowest risk can vary across future regions.

Canonical source artifact位于
`analysis/iscf_bsca_intro_concept_figure_20260730/`；状态为
`approved_for_manuscript_draft`，SVG/PDF/PNG/TIFF稳定副本位于
`paper-figures/`。该图仍仅作constructed conceptual illustration，不承担
empirical evidence。

### 4.5 Paragraph 5：ISCF-BSCA

**状态：v1.1 author-refined；显式加入single-scope decoder对照，Method Figure
4承担完整架构可视化。**

建议正文：

> Motivated by this heterogeneity, we propose ISCF-BSCA, an output-side decoder
> that integrates forecasts generated under different sharing extents.
> Independent Scope-Conditioned Forecasting (ISCF) represents each sharing
> extent through an independent history projection within a single
> scope-indexed forecast field. Each scope determines how broadly a
> history-conditioned latent state is reused before step-specific synthesis.
> A single-scope decoder applies one sharing extent throughout the forecast
> domain. ISCF instead integrates scope-conditioned forecasts through a
> target-conditioned allocation for each sample, variable and future step.
> The resulting forecast can adapt its sharing composition across future
> regions while retaining a single horizon-agnostic prediction function. To
> support joint learning, Balanced Scope Co-Adaptation (BSCA) supplies direct
> prediction signals to all scopes and discourages premature allocation
> concentration. BSCA operates only during training, adds no inference
> parameters or paths, and preserves CHPC across supported horizons.

对应中文：

> 受上述异质性启发，我们提出ISCF-BSCA，一个融合不同sharing extents所生成预测
> 的output-side decoder。Independent Scope-Conditioned Forecasting（ISCF）通过
> 独立history projection，在单一scope-indexed forecast field中表示每种sharing
> extent。每个scope决定history-conditioned latent state在step-specific
> synthesis之前被复用的范围。Single-scope decoder在整个forecast domain使用
> 一种sharing extent；ISCF则通过target-conditioned allocation，为每个sample、
> variable与future step整合scope-conditioned forecasts。由此，forecast能够跨
> future regions调整sharing
> composition，同时保持单一horizon无关prediction function。为支持joint
> learning，Balanced Scope Co-Adaptation（BSCA）向全部scopes提供直接prediction
> signals，并抑制过早的allocation concentration。BSCA仅在训练阶段生效，不增加
> inference parameters或paths，并在supported horizons上保持CHPC。

讨论过程中的细节：

1. **与Paragraph 4的因果衔接。** Paragraph 4的问题是不同future regions的
   sharing demand可能不同；因此Paragraph 5先提出整合multiple sharing
   extents，再把`scope`定义为实现每一种sharing extent的架构维度，最后由
   target-conditioned allocation完成逐target整合。问题定义中不预先出现scope。
2. **从multiple fields改为single field。** 全部scope-conditioned outputs共同
   构成$\mathcal F_\theta(\mathbf X,\tau,c,s)$；固定$s$只是该field沿scope轴的
   一个slice。该定义比“多个field预测后再融合”更贴合共享encoder、coordinate
   representation与synthesis vectors的真实计算图，也避免落入generic
   multi-predictor ensemble叙事。
3. **independent的精确定义。** independent只指各scope拥有独立的
   history-to-mode affine maps，在Introduction中概括为`independent history
   projections`。它们仍共享encoder、future-step representation、
   future-step-specific synthesis vectors与scope allocation；不表示不同
   horizons独立，也不表示多个完全独立models。
4. **scope-region latent state的精确定义。** state同时依赖history
   representation、scope-specific history projection与对应region的pooled
   future-step descriptor，因此完整名称为`history-conditioned,
   region-indexed latent state`。定义后简称`scope-region latent state`。
5. **step-specific flexibility没有被state sharing抹去。** latent-state
   sharing发生在synthesis之前；每个future time step仍通过自己的identity
   synthesis vector、nonlinear synthesis vector与bias形成prediction。因此
   ISCF不是把同一region中的future steps强制预测为相同值。
6. **allocation不是外接generic router。** allocation
   $\pi_\theta(s\mid\mathbf X,\tau,c)$与forecast field共享history和future-step
   coordinates，并在每个forecast target处沿scope轴完成normalized weighted
   contraction。正文不再使用`forecast-target-wise fusion policy`，但也不声称
   allocation为每个target找到了“最优scope”或形成universal specialization。
7. **tensor-level对应。** Method中展开
   `scope_modes:[B,C,S,D,K] -> scope_field:[B,C,T,S]`，
   `allocation:[B,C,T,S]`与`forecast:[B,T,C]`。其中`scope_field`和
   `allocation`分别对应现有代码的`arm_forecasts`与`policy`；这里只改变paper
   abstraction，不修改implementation。
8. **BSCA的机制边界。** target-conditioned allocation同时决定prediction
   combination与forecasting-loss gradient allocation。BSCA通过uniform
   slice-skill supervision和ramped `KL(uniform || allocation)`维持broad
   learning access，目标是缓解early concentration与scope-gradient starvation；
   uniform allocation只是proxy，不保证slices自动学习互不重叠的features。
9. **BSCA的贡献边界。** BSCA的论文价值来自
   `scope-indexed forecast field -> allocation-mediated gradient
   distribution -> train-only balanced co-adaptation`这一ISCF-specific链条，
   不来自generic KL、load balancing或expert training。
10. **CHPC表述延续Paragraph 3。** 模型以horizon无关、future-step-indexed
    prediction function直接实例化任意horizon；Introduction不写成先生成max-$T$
    再crop，也不声称independently generate horizons。CHPC仍是system property，
    不是Paragraph 5新增的算法模块。
11. **主图表示。** 横轴为future time step，纵轴为latent-state sharing scope；
    每个slice内部用不同宽度的共享state blocks展示sharing extent；allocation以
    同形heatmap覆盖scope--target平面；沿scope轴收缩后得到统一forecast
    trajectory。该图呈现一个结构化decoder，而不是五模型ensemble。
12. **claim边界。** canonical contiguous partition相对random partition的必要性
    未获支持，因此不使用`hierarchical/nested future lattice`命名框架，也不把
    temporal contiguity写成收益来源。multiple scales、multiple predictors与
    MoE原语已有prior art；novelty只能落在完整scope-indexed output-field链条。

#### Method Figure 4 author-fixed visual design：ISCF inference architecture

该方法总览图的visual design已由author暂时固定并放在Method开篇，不作为
Introduction中的第二张嵌入图。编号保持：Figure 1为Introduction concept；
Figures 2--3为Section 3 real-data evidence；Figure 4为method overview。当前只收到
review raster，stable SVG/PDF/TIFF bundle仍待author source同步，因此不以旧vector
bundle冒充author-fixed最终资产。

Figure 4当前包含：

1. **History path。** `History Series -> Patchify -> Encoder -> History State`，
   并由`Scope Projection`为每个sharing scope生成`Scope Matrix`。
2. **Scope-conditioned forecasting path。** `Future Coordinate`经region-wise pooling
   形成`Region Descriptor`；其与`Scope Matrix`收缩得到`Scope-region State`；共享的
   `Region-to-Step Forecast Generator`生成各`Scope-conditioned Forecast`。图中只画
   三个representative scopes，正式tensor仍为`scope_field:[B,C,T,S]`。
3. **Scope-probability path。** projected `History State`与未池化的target coordinate
   $\boldsymbol\phi_\tau$组成`Condition Vector`；`Allocation MLP`产生
   `Scope Probabilities:[B,C,T,S]`。Region Descriptor使用
   $\overline{\boldsymbol\phi}_g^{(s)}$，allocation使用$\boldsymbol\phi_\tau$，两条
   coordinate semantics不得混写。
4. **Varied-horizon output。** Scope-conditioned Forecasts按Scope Probabilities沿
   scope轴weighted contraction，得到一个`forecast:[B,T,C]` trajectory；不同
   requested horizons只返回该trajectory的nested prefixes。
5. **BSCA boundary。** Figure 4只显示ISCF inference graph；BSCA在Section 4.5用
   objective与gradient relation单独解释，不画成额外inference module。

旧Figure source、manifest与QA位于`analysis/iscf_bsca_method_figure_20260805/`，
旧stable manuscript assets位于`paper-figures/figure_iscf_bsca_method_overview.*`；
在author-fixed SVG/PDF/TIFF source到位前，这些旧资产不标记为当前Figure 4 final。
Introduction P5暂不因本轮术语同步而改写。

### 4.6 Paragraph 6：contributions

**状态：v1.1 author-refined provisional；结果句增强为预期paper-facing
conclusion，提交前必须由main/ablation/transfer tables逐项兑现。**

建议正文：

> Our contributions are threefold. First, we formulate varied-horizon
> forecasting as a unified system in which CHPC is a basic requirement, and
> identify future-region sharing-demand heterogeneity as an output-side
> challenge. Second, we introduce ISCF, which integrates forecasts generated
> under multiple sharing scopes through target-conditioned allocation. Third,
> we develop BSCA to support balanced scope learning without increasing
> inference-time complexity. Experiments across datasets from multiple
> application domains show that a single unified model outperforms separately
> trained horizon-specific forecasters. Component-wise ablations confirm the
> effectiveness of each component, while backbone transfer studies demonstrate
> decoder portability.

对应中文：

> 本文贡献主要包括三个方面。第一，我们将varied-horizon forecasting形式化为
> 一个以CHPC为基本要求的unified system，并将future-region sharing-demand
> heterogeneity识别为output-side challenge。第二，我们提出ISCF，通过
> target-conditioned allocation整合multiple sharing scopes生成的预测。第三，
> 我们提出BSCA，在不增加inference-time complexity的前提下支持balanced scope
> learning。多个应用领域数据集上的实验表明，单一unified model优于分别训练的
> horizon-specific forecasters。Component-wise ablations确认各组件的有效性，
> backbone transfer studies则证明decoder portability。

讨论过程中的细节：

1. **三项贡献对应三层论文逻辑。** Contribution 1是task/formulation与problem
   evidence；Contribution 2是output-side architecture；Contribution 3是
   ISCF-specific training机制与完整实证。这里不是为了凑三个条目而拆分组件，
   而是对应`problem -> architecture -> optimization/evidence`链条。
2. **CHPC不是单独的算法创新。** Contribution 1将CHPC放在forecasting-system
   formulation中；Contribution 2说明ISCF保留该性质。不能把CHPC再列成第四项
   method contribution，也不声称所有已有unified models都缺少CHPC。
3. **问题贡献采用可检验措辞。** 当前写`formalize ... as a testable
   finite-capacity challenge`，而不是把future-region sharing-demand
   heterogeneity写成未经Evidence III验证的普适事实。Problem/Motivation完成
   matched diagnostics后，最终稿可根据risk crossing覆盖度决定是否加强为
   `demonstrate`或继续保留条件性表述。
4. **架构创新按完整链条主张。** ISCF的claim不是“首个multi-scale decoder”
   或“首个multi-predictor fusion”，而是
   `heterogeneous output-side sharing demand -> independent history
   projections across latent-state sharing scopes -> scope-indexed forecast
   field -> target-conditioned scope allocation -> weighted contraction in a
   horizon-agnostic function`。TimeMixer、
   N-HiTS、FreqMoE与Moirai-MoE压缩了generic primitive-level novelty，但不
   自动覆盖该完整problem--mechanism链。
5. **BSCA不claim generic balancing novelty。** Contribution 3明确写成
   `ISCF-specific train-only objective`。现有三seed证据只支持相对ISCF-EQUAL
   的small but directionally robust gain，不支持universal gain、强
   specialization或所有datasets/horizons一致改善。
6. **结果句作为待兑现的paper-facing claim。** v0.9暂时保留作者强化句，不在
   本轮改写Introduction clean draft；但prelaunch protocol已冻结最终claim边界：
   DLinear/PatchTST primary rows通过时只能写`outperforms the evaluated
   standard-protocol horizon-specific baselines`，不能使用无定语的
   `outperforms horizon-specific forecasters`。TimePerceiver/SRSNet只作各自
   native single-seed point-estimate context。若任一结论未被完整test matrix
   支持，必须降级对应动词；未完成统计检验前仍不使用
   `statistically significant`。
7. **证据结构必须与贡献一一对应。** Contribution 1由baseline/simple matched
   diagnostics与CHPC disagreement支撑；Contribution 2由exact core ablations、
   same-backbone transfer controls、full test MSE/MAE和scope behavior支撑；
   Main II H720-prefix rows只是source-native one-model system benchmark，不是
   exact architecture attribution。Contribution 3由same-architecture objective
   control、three-seed official-test与internal health支撑。efficiency与
   transferability属于完整framework evidence，不能替代mechanism attribution。
8. **transferability是empirical scope，不是预先成立的贡献。** 在decoder迁移
   实验完成前，正文只写`evaluate ... decoder transferability`；若迁移结果不
   稳定，应将其降为analysis或limitation，而不是保留正向claim。
9. **round1 author response边界。** Introduction只轻量承认少量
   varied-horizon先例，不加入conceptual comparison table，也不展开与本文native
   decoder主线无关的结构路线。当前优先事项是P4 problem-existence evidence、
   ISCF/BSCA创新链的简洁表达与术语压缩；详细prior-art、parameterization、机制
   推导和controls分别进入Related Work、Method、Problem Formulation与
   Experiments。

BSCA 的 novelty 位于 ISCF-specific contribution chain，不 claim generic
KL、entropy regularization 或 load balancing 首创。

## 5. Related Work

当前v0.2 draft=`docs/paper-drafts/iscf-bsca-related-work-initial-draft.md`，
primary-source audit=`analysis/iscf_bsca_related_work_research_20260810/literature_design_and_source_audit.md`。
Section 2采用`fixed-horizon strategies -> unified/varied horizons -> forecast-generation
design -> multi-scale/adaptive allocation`的收束顺序，最终进入Section 3的
CHPC与future-region sharing-demand heterogeneity。

### 5.1 Fixed-Horizon Multi-Step Forecasting

覆盖：

- recursive：一个one-step model递归生成；
- classical direct：每个future step一个single-output model；
- MIMO：一个multi-output model一次生成完整future vector；
- DIRMO / block multi-output：多个models分别生成future blocks；
- benchmark 中按 horizon 分别训练的 protocol。

落点：

> 既有策略在一个预设horizon内组织multi-step outputs；本文进一步提出
> varied-horizon output strategy，由单一模型服务不同request endpoints，并返回
> 同一prediction trajectory的prefix-consistent views。

### 5.2 Unified and Varied-Horizon Forecasting

区分：

1. 一个 model family 支持不同输出长度；
2. 一个 forward 同时输出多个 future time steps；
3. 同一模型服务多个 requested horizons；
4. 不同 requested horizons 满足 CHPC。

不得声称所有已有模型都不 unified 或都不满足 CHPC。TimesFM等foundation models
已研究跨horizon generalization；Timer与Time-MoE的autoregressive generation支持
flexible output length。ElasTST更直接地以structured masks保证shared future outputs
对inference horizon不变，因此CHPC只作为本文的formal system contract，不能claim
horizon-invariance principle首创。ElasTST的multi-scale mechanism调整history-patch
resolution；本文的差异转向output-side、future-region-dependent state-sharing extent。

### 5.3 Forecast Generation and Output-Side Modeling

围绕 history representation 到 forecast sequence 的映射讨论：

- linear/global output heads；
- patch-based readouts；
- channel-independent decoding；
- basis、block、segment 与 implicit forecasting；
- shared 与 forecast-step-specific generation。

核心 comparison question：

> decoder 如何在多个 future time steps 之间分配 latent sharing，而不是 encoder
> 如何处理历史输入。

### 5.4 Multi-Scale Forecasting and Adaptive Allocation

明确区分：

- 既有 multi-scale 方法通常处理 input resolutions、frequency bands 或
  history features；
- 既有MoLE、FreqMoE、Time-MoE与Moirai-MoE已覆盖complete-expert mixing、
  frequency experts、sparse routing与token-level specialization；
- ISCF 的 scope 是 output-side latent-state sharing extent。

不把 primitive overlap 自动写成 novelty rejection；claim 落在
`unified nested-horizon problem -> CHPC contract -> future-region
sharing-demand heterogeneity -> scope-indexed forecast field and
target-conditioned scope allocation ->
balanced co-adaptation` 完整链上。

## 6. Problem Formulation and Empirical Motivation

Section 3 v0.7 author risk-definition refinement已落地：
`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`。当前状态为
`temporarily_frozen_usable`；Introduction v0.9未改动。后续默认不再改写Section 3，
除非Section 4或paper-facing evidence产生明确矛盾，并由用户显式同意解冻。

3.1--3.3不出现ISCF、BSCA、arm或production method名称，问题证据只使用已有
baseline或简单capacity-matched diagnostic heads。原3.3 naive-unified accuracy
audit与原3.5 design requirements已从manuscript Section 3删除：relative accuracy
留给Experiments，method identity从Section 4开始。Figures 2--3继续只作
validation-based problem evidence，不承担method effectiveness或learned
allocation claim。

Section 3以一个承上启下段进入两个相连问题：不同horizon requests应满足什么
一致性，以及unified decoder应如何组织future domain。全章narrative spine更新为
`CHPC task contract -> observed horizon-specific inconsistency ->
future-region sharing-demand heterogeneity -> decoder motivation`。3.1先定义CHPC，
再对比horizon-specific predictors，最后给出结构上满足CHPC的future-step-indexed
function。3.2使用`inconsistency`指违反CHPC的现象，保留CHPD/NCHPD作为量化
disagreement的统计量。公式只承担formalization或matched measurement，不替代
topic sentence与argument transition。

### 6.1 Manuscript 3.1：Varied-Horizon Forecasting and CHPC

Section 3 v0.7先从same-history / shared-target语义定义CHPC：

$$
\widehat y_{o+\tau,c}^{(H_i)}
=
\widehat y_{o+\tau,c}^{(H_j)},
\qquad H_i<H_j,\quad 1\leq\tau\leq H_i.
$$

随后以conventional horizon-specific formulation说明该一致性不受独立参数与优化
保证：


$$
\hat{\mathbf Y}^{(H)}
=
f_{\theta_H}(\mathbf X),
\qquad
H\in\mathcal H.
$$

最后给出结构上满足CHPC的unified formulation：

$$
\hat{\mathbf Y}^{(H)}
=
\left[
g_\theta(\mathbf X,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C},
$$

其中 $g_\theta$ 是horizon无关、future-step-indexed prediction function，$H$ 是
forecast horizon，$\tau$ 是future time step，$(\tau,c)$ 是forecast target。CHPC
不引入额外projection operator $\Pi_{H_i}$；3.1结论限定为不同horizon requests是
同一future trajectory的nested views。accuracy comparison不在Section 3展开。

### 6.2 Manuscript 3.2：Horizon-Specific Prefix Inconsistency

`inconsistency`表示horizon-specific predictions违反CHPC的现象；CHPD与NCHPD
继续作为量化该现象的prediction-disagreement statistics。

对 DLinear、PatchTST、iTransformer 的native horizon-specific implementations
分别训练$H=96,192,336,720$ 的模型。正式证据复用Main Results中的相同
checkpoints，不为motivation figure重复训练。五datasets、四horizons、三seeds
共180个main-baseline checkpoints，所有horizon pairs使用相同forecast origins。

定义 Cross-Horizon Prefix Disagreement：

$$
\operatorname{CHPD}(H_i,H_j)
=
\frac{1}{H_iC}
\sum_{\tau=1}^{H_i}
\sum_{c=1}^{C}
\left|
\hat y_{\tau,c}^{(H_i)}
-
\hat y_{\tau,c}^{(H_j)}
\right|,
\qquad
H_i<H_j.
$$

归一化版本按每个channel的train-split standard deviation计算：

$$
\operatorname{NCHPD}
=
\mathbb E_{o,c}
\left[
\frac{
\frac1{H_i}\sum_{\tau=1}^{H_i}
\left|
\hat y_{\tau,c}^{(H_i)}
-
\hat y_{\tau,c}^{(H_j)}
\right|
}{
\sigma^{\mathrm{train}}_c+\epsilon
}
\right].
$$

Section 3 Figure 2在formal definition之后展示：

- ETTh2/DLinear maximum-disagreement same-history example；
- 单一trajectory hero panel中的history、ground truth、四个horizon predictions与
  H720-relative mean $|\Delta|$摘要；
- 全validation origins/channels的紧凑$3\times3$ upper-triangular NCHPD
  heatmap。

正式analysis另报告per-origin distributions、squared disagreement与relative
disagreement amplitude。self-replay与同一unified-checkpoint replay必须得到
exact zero；origin timestamp、scaler roundtrip与prefix shape必须通过。

该证据只证明“不提供 CHPC 保证”和系统冗余，不证明 horizon-specific accuracy
更差，也不claim ElasTST等已有varied-horizon methods缺少invariance。D18/A6
artifacts只可用于evaluator smoke，不进入正式图表。

Full-search amendment：补齐DLinear × five datasets × seed2021的四个horizons，
只使用validation。在每个dataset的全部`origin × channel`联合单元上，按六个
horizon pairs的mean-over-overlap disagreement选择maximum；跨dataset再按同一
score排序。该选择必须在caption中标记为`maximum aggregate validation
disagreement`。mean-over-overlap防止单个future-step spike主导选择，但该图仍是
intentional strong example，不是representative case或prevalence evidence。

Final selected result：Section 3 Figure 2使用ETTh2/DLinear
origin=805、channel=0。
ETTh2的maximum joint score排名第二，但macro NCHPD为五dataset最高，macro RDA
排名第二，且shared-96 raw differences最清晰；因此visual audit选择ETTh2而非
maximum-cell ranking第一但overlay语义较弱的Weather。H96/H192/H336相对H720的
96-step mean absolute raw differences为2.51/2.16/2.40。正式三families、
五datasets、三seeds的prevalence evidence仍不由本次visualization search替代。
最终visual refinement将原来的trajectory与raw-difference上下两图合并为一个
hero panel，Section 3 Figure 2整体成为顶底对齐的two-panel layout。四个horizon不再依赖
dash pattern区分，而使用thin solid colors、sparse staggered marker shapes与
subtle white separation strokes；H720置于较低z-order，避免遮挡较短horizon
curves。预测主线缩至0.82--0.95 pt、marker间隔放宽至18 steps，避免四条高度
重合的predictions形成过粗色带。

### 6.3 Deferred Evidence Boundary：Naive Unified Forecasting

本段证据审计从Section 3 v0.5 manuscript删除，保留在architecture record中供
Experiments设计与后续claim audit使用；不得据此恢复Section 3中的accuracy claim。

把同一 baseline 改成horizon无关、future-step-indexed unified variant，再在
多个 requested horizons 上评估。定义：

$$
\operatorname{UP}_H
=
\frac{
\operatorname{MSE}^{\mathrm{unified}}_H
-
\operatorname{MSE}^{\mathrm{specific}}_H
}{
\operatorname{MSE}^{\mathrm{specific}}_H
}.
$$

若 $\operatorname{UP}_H>0$，说明该 baseline 的 naive unified adaptation
存在 performance compromise；若部分或全部 baseline 不出现正 penalty，则必须
收窄结论，不能宣称 unified forecasting 天然更难。

现有D18 audit不足以支持稳定正penalty：horizon specialists相对
`A6_MEASURE`的aggregate MSE差异仅0.1659%，只覆盖7/15 cells与2/5 datasets；
`A6_MEASURE`相对`A6_FULL`的measure-training差异反而达到1.7980%且覆盖15/15
cells。该confound大于并更稳定于horizon-specific/unified contrast。因此Section
3 v0.2只定义$\operatorname{UP}_H$与matched comparison要求，并明确：

- 不把naive unified performance compromise写成已证事实；
- unified motivation只由one-model service与CHPC contract承担；
- relative accuracy留给后续完整matched paper-facing scorecards；
- Introduction P6的unified superiority仍是provisional claim。

### 6.4 Manuscript 3.3：Future-Region Sharing-Demand Heterogeneity

这一节在提出ISCF之前建立问题证据，不使用ISCF、scope slice、BSCA或
target-conditioned scope allocation等方法术语。主要证据来自相同simple baseline上
end-to-end训练的capacity-matched **single-scale** diagnostic predictors。每次
training只含一个sharing extent，不含multiple scopes、fusion或allocation。

若使用 frozen representation，只能作为 secondary diagnostic，不能承担
problem-direction rejection。primary comparison应采用相同encoder class、数据、
objective、optimization、checkpoint rule与comparable initialization进行matched
end-to-end training。

Primary neutral tensor path为：

$$
\mathbf X:[B,L,C]
\rightarrow
\mathbf R:[B,C,D],
$$

$$
\mathbf U_{b,c,\tau}
=
G_\omega([\mathbf R_{b,c},\boldsymbol\phi_\tau])
\in\mathbb R^{D_z},
$$

$$
\mathbf Z^{(s)}_{b,c,g}
=
\operatorname{LayerNorm}
\left(
\frac1{|\mathcal G_{s,g}|}
\sum_{\tau\in\mathcal G_{s,g}}
\mathbf U_{b,c,\tau}
\right),
$$

$$
\hat y^{(s)}_{b,c,\tau}
=
\langle\mathbf a_\tau,
\mathbf Z^{(s)}_{b,c,g_s(\tau)}\rangle+b_\tau.
$$

$E_\psi,G_\omega,\phi_\tau,\mathbf a_\tau,b_\tau$在所有$s$下shape和parameter
count相同；所有$s$都计算完整$\mathbf U$，只改变parameter-free pooling与
latent-state sharing topology。每个future step始终保留自己的synthesis vector。

Diagnostic sharing grid冻结为
$S_{\mathrm{diag}}=\{1,8,32,128,720\}$，有意不复制最终ISCF的中间scope set。
Primary regions为12个等长60-step bins，不沿用requested horizons或method
boundaries。Primary objective为uniform full-domain pointwise MSE，避免
multi-prefix exposure把early-step reweighting混入sharing-demand证据。五
datasets、五scales、三seeds共75个end-to-end runs。

令 future region $\mathcal B_b$ 为预测域内的连续 future-step集合。对sharing
scale setting $s$ 定义：

$$
R_{b,s}
=
\mathbb E\left[
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_{c=1}^{C}
\left(
D_s(\mathbf X)_{\tau,c}-Y_{\tau,c}
\right)^2
\right].
$$

除region risks与crossover外，primary headroom必须由validation-selected
region schedule在official test上计算：

$$
s_{\mathrm{fixed}}^{\mathrm{val}}
=
\arg\min_s\sum_bw_bR_{b,s}^{\mathrm{val}},
\qquad
s_b^{\mathrm{val}}
=
\arg\min_sR_{b,s}^{\mathrm{val}},
$$

$$
\operatorname{CFH}
=
\frac{
R_{\mathrm{fixed}}^{\mathrm{test}}
-
R_{\mathrm{region\ schedule}}^{\mathrm{test}}
}{
R_{\mathrm{fixed}}^{\mathrm{test}}
}.
$$

这只是由多个single-scale models拼接的diagnostic upper bound，不是可部署
method，也不证明history-conditioned allocation可识别。

该问题得到支持需要同时观察：

1. matched risk curves在不同future regions稳定交叉；
2. validation-selected schedule在至少3/5 datasets使用不少于两个scales；
3. official-test macro CFH为正，至少3/5 datasets与2/3 seeds为正；
4. all matched/numeric controls通过。

主要展示：

- sharing-scale × future-region test risk heatmap，颜色相对
  validation-selected fixed scale；
- validation-selected best-scale ridge；
- $s=1,32,720$的sharing-scale crossover curves；
- validation-selected region schedule相对validation-selected fixed scale的
  official-test CFH。

Section 3 Figure 3紧跟sharing-demand definition与matched diagnostic
protocol，不提前画ISCF或BSCA。若要进一步claim temporal
contiguity specificity，primary support后必须追加group-size-matched random
grouping；该control不是basic sharing-demand existence gate。

data-side local fluctuation、block trend、frequency/scale energy与可预测性分析只作
描述性辅助，不能单独建立sharing-demand heterogeneity。单纯gradient
magnitude difference、单调lead-time difficulty或oracle label routing也不能
单独证明可学习的region-dependent sharing-scale preference。

正式设计、统计量、controls、failure attribution与Figure captions见
`analysis/iscf_bsca_intro_problem_evidence_design_20260729.md`。
该新证据矩阵按项目治理属于`test_informed`，不得描述为untouched holdout。

Full-search amendment：补齐neutral single-scale family在five datasets、
seed2021上的validation artifacts。对每个origin构造all-channel
`5 scales × 12 60-step regions`风险面，并按以下lexicographic order选择：

1. 赢得至少两个regions的scale数量；
2. distinct region winners数量；
3. winner histogram entropy；
4. 达到0.5% bidirectional margin的crossing pair数量；
5. mean best-versus-second-best region margin；
6. descriptive sample oracle headroom。

该顺序优先寻找由多个regions支持的分散scope winners，避免单region噪声胜者。
Final selected result：Section 3 Figure 3使用ETTm2 origin=4177。五个scales分别赢得
`2/2/2/3/3`个regions，全部10个scale pairs达到qualified bidirectional
crossing，mean winner margin=10.266%，descriptive region-oracle headroom=
8.112%。最终图使用region-best excess-prediction-risk heatmap与winner-colored
region-gain bars，不再展示高噪声step-wise curves。旧版fixed-s720 heatmap中
s720整行恒为0并显示为白色；新版编码消除该视觉歧义，同时在bar panel保留
fixed-s720 reference。maximum-heterogeneity validation role在caption中明确；
正式CFH继续deferred。图不使用ISCF/BSCA，不承担method effectiveness claim。

Section 3 v0.7将sample-level statistic $R_{o,b,s}$定义为future-region prediction
risk：给定origin、future region与sharing extent后，对该region全部targets聚合的
empirical squared-error loss。该quantity不是对data distribution取expectation的
population risk。validation-selected fixed/schedule与official-test CFH继续作为
architecture record中的future formal control，不进入当前manuscript。Figure 3的同一
validation-label winner与8.112% headroom只作descriptive oracle。完整neutral tensor
path与formal control继续保存在本architecture与canonical evidence design中。

### 6.5 Method Requirements Record（不作为Section 3 subsection）

Section 3 v0.7不再单设Design Requirements，也不在Section 3正文引出ISCF-BSCA。
以下内容保留为Section 4 architecture rationale与后续ablation claim map，不是当前
manuscript subsection。

由前三项证据导出：

1. 一个 unified model 服务全部 horizons；
2. 每个 future-step prediction 对 requested horizon 保持invariant，因而满足CHPC；
3. decoder 不应把单一固定cross-step sharing pattern强加给整个future domain；
4. 架构应在一个统一forecast field中提供多种future-step latent-state sharing
   scopes，并允许每个forecast target整合不同sharing extents；
5. scope-conditioned slices与scope allocation需要稳定joint training。

这些requirements在Section 4 architecture overview中映射到ISCF与BSCA，不再经由
Section 3.5引出；Figures 2--3仍不构成component effectiveness evidence。

v0.2进一步区分requirement来源：one model与CHPC来自task definition；Figure 3
直接支持within-sample future-region variation，因此只导出multiple extents与
future-step-varying integration；更细的sample/variable conditioning属于method
hypothesis，必须由后续ablation验证。stable joint learning同样只作为method必须
解决的optimization condition，不声称由Figures 2--3直接建立。

## 7. Method

Canonical manuscript draft：

`docs/paper-drafts/iscf-bsca-method-initial-draft.md`

当前状态=`v0.7-bsca-narrative-order-refinement`，等待author review。4.5保留dual-role总述，并将正文重排为`Uniform-Prefix Forecasting Loss -> probability-scaled multi-scope gradient problem -> Scope-Wise Forecasting Loss -> Allocation-Balance Regularizer`。Standalone 4.6继续留在Method之外；CHPC construction由4.4闭合，parameter/FLOPs/latency/memory进入Section 5 efficiency analysis。Reference implementation与实验状态不变，Stable vector-asset bundle仍待author source。

### 7.1 Architecture Overview

先以Figure 4引出两条协同路径。`Scope Forecasting Path`沿`History State/Future Coordinate -> Scope Matrix/Region Descriptor -> Scope-region State -> Scope-conditioned Forecasts:[B,C,T,S]`构建forecast field。`Target-Adaptive Allocation Path`沿`History State/Future Coordinate -> Condition Vector -> Allocation MLP -> Scope Probabilities:[B,C,T,S]`评估每个target的sharing-granularity preference。沿scope轴weighted contraction后形成`Varied-Horizon Forecasting` trajectory；BSCA不进入Figure 4 forward path。

### 7.2 History State and Future Coordinate

History encoder接口与fixed DCT-style future-step coordinate定义为：

$$
\mathbf X:[B,L,C]
\rightarrow
\mathbf Z:[B,C,P,D_e]
\rightarrow
\mathbf R:[B,C,PD_e],
$$

其中future-step coordinate只编码target identity，不使用future observation或requested horizon。

### 7.3 Generation of Scope-conditioned Forecasts

对每个$s\in\mathcal S$，独立`Scope Projection`从`History State`产生
`Scope Matrix`；`Future Coordinate`在scope region内平均得到`Region Descriptor`；
二者收缩形成`Scope-region State`。共享的`Region-to-Step Forecast Generator`将
state映射为`Scope-conditioned Forecast`，全部scopes共同形成：

$$
\mathcal F_\theta(\mathbf X)
\in
\mathbb R^{B\times C\times T\times S}.
$$

ISCF通过scope-specific projections与共享Encoder和Region-to-Step Forecast Generator
联合构造该field，而非集成独立训练的forecasters。该结构耦合跨scope的forecast
synthesis，同时避免重复encoder computation与generator parameters。

### 7.4 Target-Adaptive Scope Allocation

该路径以compact history summary表示sample-variable dynamics，以未池化
$\boldsymbol\phi_\tau$表示future target position。`Condition Vector`为
$[\mathbf u_{b,c};\boldsymbol\phi_\tau]$，`Allocation MLP`据此产生
target-wise `Scope Probabilities`：

$$
\boldsymbol\Pi
\in
\mathbb R^{B\times C\times T\times S}.
$$

每个probability vector表示target对不同sharing granularities的allocation；最终
prediction通过读取统一scope-indexed field并沿scope轴weighted contraction得到：

$$
\hat y_{b,\tau,c}
=
\sum_{s=1}^{S}
\pi_{b,c,\tau,s}
\mathcal F_{b,c,\tau,s}.
$$

请求$H$时只需激活与$1{:}H$相交的Scope-region States和$\tau\leq H$的
allocation entries。该property不改变shared targets的计算，因此保持CHPC；现有
reference implementation的full-field materialization边界保留在editorial audit。

### 7.5 Balanced Scope Co-Adaptation

BSCA同时处理两个训练要求：统一trajectory需要覆盖全部prefix endpoints，而多个scope
lines需要在allocation学习过程中保持可训练。叙事顺序固定为：先由
`Uniform-Prefix Forecasting Loss`定义varied-horizon主目标，再说明weighted contraction
导致的probability-scaled gradient问题，最后依次引出`Scope-Wise Forecasting Loss`与
`Allocation-Balance Regularizer`。Direct scope supervision提供allocation-independent
gradient path，uniform-reference KL则惩罚过度不均匀的Scope Probabilities，使
probability-mediated gradients在训练早期更均衡地分配。

$$
\mathcal L_{\mathrm{BSCA}}
=
\mathcal L_{\mathrm{prefix}}
+
\lambda_{\mathrm{scope}}\mathcal L_{\mathrm{scope}}
+
\lambda_{\mathrm{balance}}(u)\mathcal L_{\mathrm{balance}}.
$$

解释边界：

- `Uniform-Prefix Forecasting Loss`对所有prefix endpoints等权，形成varied-horizon主预测目标；
- `Scope-Wise Forecasting Loss`为每条scope line提供不依赖当前allocation probability的直接prediction-loss pathway；
- `Allocation-Balance Regularizer`以ramped uniform-reference KL惩罚过度不均匀的scope selection，促进更均衡的probability-mediated optimization；
- uniform reference不构成equal training、equal inference usage或semantic specialization保证；
- 当前证据支持 balanced co-adaptation，不支持更强的 universal conditional specialization claim。

Section 4在4.5结束。CHPC construction已由4.4的shared-target invariance与prefix output说明闭合；trainable parameters、FLOPs、latency、memory和one-model system cost统一转入Section 5.4 Efficiency Evaluation与后续analysis，不再单设Method 4.6。

## 8. Experiments

### 8.1 Experimental Setup

Main-result evidence universe覆盖8 datasets：ETTh1、ETTh2、ETTm1、ETTm2、
Weather、ECL、Solar、Exchange；当前Main I/II dense tables使用共同完整的前7个
datasets，Exchange分别作为limited companion/deferred extension。core ablation与
decoder transfer使用原5 datasets。统一报告
$\mathcal H=\{96,192,336,720\}$、seeds、validation-only checkpoint
selection、test-tuned hyperparameter selection、official-test MSE/MAE、
parameter/training budgets，以及
horizon-specific与unified两种protocol。

Main I/II中的论文方法必须是`ISCF-BSCA-MAIN-v1`：在frozen architecture
family内对8 datasets分别执行test-tuned HPO，每dataset选择一个profile
共同服务四个H。Exact `ISCF-BSCA-v1`及其现有超参数只用于ablation，不进入主表。
每个trial先由four-H mean validation MSE选择checkpoint。H4J起，dataset-level
profile由equal-weight joint MSE/MAE relative mean的1% guard后最大化MSE+MAE
leading cells选择；test不得选择epoch、checkpoint、seed、metric-specific或per-H
profile。全部trial结果必须保留并披露为`test_tuned/test_informed`。
TimeAlign encoder参数只作为source-audited search prior。当前完整矩阵先固定
seed2021；seeds2022/2023仅在时间允许时按完整experiment block扩展，且不得
result-selective扩展。

截至2026-08-04，H1--H4K共117个HPO trials已完成official test。H4K只使seven-dataset macro MSE/MAE改善0.0199%/0.0286%，joint selector相对frozen published targets仍为MSE 15/28、MAE 15/28、combined 30/56，未通过20/28、20/28、40/56 gates。合法selector与逐cell diagnostic oracle同为30/56，故当前结论仍是strong aggregate competitor和HPO partial pass，不是完整per-cell SOTA。

H4L冻结为ETTm2/Weather各24个wide space-filling profiles，覆盖此前基本未搜索的weight decay、ETTm2 `d_ff`、Weather `mode_rank`以及更宽的context/patch/capacity边界。四个profiles保留TimeAlign official encoder parameter coupling，再与ISCF-BSCA rank或optimizer regularization组合；TimeAlign head与alignment loss不进入本方法。48/48 training与test artifacts、192/192 standard-horizon rows、numeric health、provenance和checkpoint immutability均通过。165-trial joint selector将ETTm2切换为`h4l_wd1e3`并新增H720 MAE一个lead，Weather保持H4K selector；global MSE/MAE/combined=`15/28,16/28,31/56`。

H4M随后以24个新profiles扩大历史实验显示为高影响的interaction：ETTm2搜索`patch × low LR`并补rank/context，Weather搜索context/patch、low LR与rank组合；全部仍是one dataset profile shared by four H。24/24 frozen checkpoints、96/96 standard-horizon rows、numeric health、provenance和checkpoint immutability均通过。189-trial joint selector将ETTm2切换为`h4m_p6_lr5e5`，仍为0/4 MSE和1/4 MAE leads；Weather切换为`h4m_seq640_p20`并由2/8提高到4/8。Global MSE/MAE/combined=`17/28,16/28,33/56`，legal selector、unrestricted single-profile upper bound与per-cell diagnostic oracle均为33/56，故H4M仍是`performance_partial_pass_gate_fail`并回到Step 6 strategy decision。

并行TimeAlign ETTm2/Weather的8个official fixed-H reproduction jobs已完成artifact audit。ETTm2/Weather复现mean MSE/MAE分别为0.242889/0.302523和0.215800/0.244725，与paper three-run mean差异均低于0.26%；角色仅为native external baseline，不作matched mechanism attribution。H4O、selected-profile confirmation、其他baseline和3-seed仍未授权。

H4M后用户单独授权H4N继续扩大Weather HPO。H4N不改变architecture/objective/scales/inference graph，冻结40个seed2021 profiles并与189个历史profiles零重复：围绕`L512/p16/lr2e-5` MAE/joint frontier与`L640/p20/lr5e-5` MSE frontier，覆盖context×LR、LR外边界、patch geometry、mode rank及少量encoder capacity。选择规则以Weather four-H MSE/MAE relative mean最小为primary，只有0.1% near-tie才优先lead cells；one profile仍共享four H。统一预算120 epochs/patience24，training test=0，40/40 manifest后完整test已授权。该dataset-specific tuning只影响Main I performance profile，不扩张method claim；H4O、extra seeds及architecture/objective redesign未授权。

H4N exact commit=`ba17fc9`的40/40 resource smoke已通过，test=0且无numeric/runtime failure。Full Weather train/validation queue已于2026-08-05 10:47:16在GPU0--2启动；40/40 immutable checkpoint manifest通过前不访问official test。

2026-08-06 H4N 40/40 full train/validation artifacts与unique checkpoint hashes通过audit，training test保持0/40；40-row manifest SHA256=`a0f152f9172acc193fe512001123b71aeae6d6d3ab1028c915074f24d54c1ed4`。Formal test固定为完整160 standard-horizon rows且已获授权；该test只更新Weather dataset-level main-model profile，不改变architecture、method claim或ablation anchor。

H4N formal test现已40/40、160/160完成。Full-table selector选择`L608/p19/lr2e-5`，mean MSE/MAE=`0.214887/0.245821`，相对H4M current Weather profile为+0.063%/-0.608%，但full-table exact leads仍4/8且全部success gates失败。该结果不改变architecture freeze或method narrative，只把Weather main-profile evidence更新为partial MAE improvement；legacy 5-baseline 33/56 frontier与wide-table displayed 29/56均未提高。

2026-08-10用户重启HPO，但严格限定为Main II弱项`ETTh1/ECL/Solar`。H5A冻结
48个seed2021 profiles（每dataset 16），重点扩大历史高影响或欠搜索参数：ETTh1的
LR/context/patch/capacity，ECL的patch granularity与large-capacity optimizer边界，Solar的
patch×LR×rank interaction。Selector直接对冻结Main II七个external systems按共同三位小数
统计best cells，但要求four-H mean MSE和MAE均不比当前profile退化超过0.5%；每dataset仍
只能选择一个profile服务four H。当前三个datasets为`1/8,0/8,4/8` best，最低目标为
`2/8,1/8,5/8`，即全局Main II由24/56提高到至少27/56。Remote training与48/48
immutable manifest后的完整formal test已授权；H5B、extra seeds、architecture change及自动
修改Main I/Main II均未授权。Canonical prelaunch=`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/design_and_prelaunch_gate.md`。

H5A commit=`7544f76d`的48/48 resource smoke已通过，test=0、48 unique checkpoint
hashes且无OOM/numeric failure；full three-GPU train/validation queue已于2026-08-10
15:13:32启动，PID=`2375625`。48/48 immutable manifest完成前不得访问formal test。

H5A full train/validation与once-only formal test现已完成。48/48 checkpoints、192/192
standard rows、checkpoint immutability、dense artifact/provenance与numeric health全部通过。
Frozen Main II selector选择ETTh1=`h5a_lr3p5e4`、ECL=`h5a_seq336_p1`、Solar=
`h5a_seq512_p4_lr2p5e4`；best cells从`1/8,0/8,4/8`提高到`2/8,1/8,6/8`，
target total=`9/24`，projected global=`28/56`。三profile同时通过four-H mean MSE/MAE
0.5% guard，Solar保持4/4 MAE best，全部H5A gates通过。该结论仅为test-tuned
performance evidence，不改变architecture或method-attribution边界；Main II table仍等待
显式mutation授权，H5B/extra seeds/confirmation未启动。

### 8.2 Main Results I：Unified versus Horizon-Specific

每个baseline使用四个horizon-specific trained models；
`ISCF-BSCA-MAIN-v1`每dataset/seed使用一个tuned unified model。Main I按来源
分为published-transcribed与official-native reproduced两类：

| Family | Models | Result route |
| --- | --- | --- |
| local official/native reproduction | TimeAlign, QDF, AMD, SimpleTM | 七个dense datasets均使用本地复跑；QDF Solar使用ECL-derived source-informed preset；SimpleTM报告native repetitions mean |
| published fixed-H context | TVNet, iTransformer, TimeMixer, Leddam, ModernTCN, PatchTST, Crossformer, TimesNet, DLinear | 从TimeAlign Table 6逐cell转录并审计；不冒充matched local reproduction |
| paper method | ISCF-BSCA-MAIN-v1 | validation-selected checkpoints + test-tuned dataset profiles |

该表回答一个 unified model 能否与 separately optimized horizon-specific
models竞争，但不单独承担architecture attribution。Published-result primary
source为TimeAlign ICLR 2026 Table 6：当前表保留其9个published-context systems，
同时将TimeAlign自身一列替换为本地official-native复跑；该source缺Exchange。
AMD与SimpleTM在TimeAlign表中不存在，已用各自official repository复现当前7个
dense datasets。Exchange当前只保留ISCF-BSCA、TimeAlign与QDF companion；不得
把缺少其余11 systems的partial surface表述为完整8-dataset Main I。PDT固定`L=96`，
仅保留secondary cross-check。TimeAlign表存在lookback search
集合描述差异，且published values为3-seed mean；本地official reproduction
统一使用seed2021并披露差异。TimeAlign Exchange已按ETTh1-derived bootstrap
复跑，但因没有official Exchange preset，必须标记为source-informed而非official。
所有native/published rows都不进入matched mechanism attribution。

Table 6的140个目标published rows已完成PDF-coordinate transcription与渲染核验。源PDF存在5组逐horizon均值与reported Avg不一致，并分别在Table 1 caption、main-text implementation与Appendix E.1给出三种lookback grid；主表必须使用逐horizon原值并披露这些source-native protocol差异。

截至2026-08-06，TimeAlign 8 datasets × four H × seed2021与QDF `L=336` 8 datasets × four H × seed2023均已32/32完成并通过artifact/hash audit。QDF六个released datasets仅改变lookback并保留逐Hprofiles；Solar使用ECL-derived profile，Exchange使用ETTh1-derived profile。Main I的七数据集dense QDF block已原子替换为28/28本地L336 cells，Exchange companion扩为ISCF-BSCA/TimeAlign/QDF三系统。QDF在七数据集macro MSE/MAE为`0.287511/0.331426`，相对ISCF-BSCA高`9.541%/7.508%`，仅在ETTm2-H192 MSE领先ISCF；完整14-model表中ISCF-BSCA仍为27/56 best、19/56 second。上一版mixed-source L96 QDF仅作历史记录，不再进入当前Main I。

截至2026-08-08，AMD与SimpleTM的七数据集official-native复现已完成：14/14 units、AMD 28/28 + SimpleTM 82/82 raw rows、110 unique checkpoint hashes与56/56 table cells通过审计，旧失败root的partial rows永久excluded。Main I已从表头和数值层原子移除CMoS/TimeBase并加入AMD/SimpleTM。AMD macro MSE/MAE=`0.282132/0.328147`，SimpleTM=`0.292099/0.332380`；更新后的完整14-model表中ISCF-BSCA为29/56 best、19/56 second。AMD与SimpleTM只作为source-native fixed-H accuracy context，不进入matched mechanism attribution。

### 8.3 Main Results II：H720-trained One-Model-All-Horizons Benchmark

Main II在2026-08-08按用户要求重新冻结为H720-prefix system benchmark。对每个
external baseline和dataset，只使用一个由官方H720 training script得到的
fixed-H model；H96、H192和H336均由同一次H720 forecast裁剪前$H$ steps获得。
因此每个checkpoint的四个请求共享完全相同的H720 test origins，并形成一条nested
prediction trajectory。

| Model | # Models per Dataset | H96 Prefix | H192 Prefix | H336 Prefix | H720 | Avg. | H720 Main-I Audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TimeAlign | 1 |  |  |  |  |  | local exact |
| QDF | 1 |  |  |  |  |  | local exact |
| AMD | 1 |  |  |  |  |  | local exact |
| SimpleTM | 1 per native repeat |  |  |  |  |  | local exact after repeat mean |
| iTransformer | 1 |  |  |  |  |  | deviation from published mean |
| PatchTST | 1 |  |  |  |  |  | deviation from published mean |
| DLinear | 1 |  |  |  |  |  | deviation from published mean |
| ISCF-BSCA-MAIN-v1 (tuned) | 1 |  |  |  |  |  | local exact |

主表使用当前Main I共同完整的七个dense datasets：ETTh1、ETTh2、ETTm1、
ETTm2、Weather、ECL、Solar。Exchange只保留为deferred extension，因为冻结Main I
尚无AMD、SimpleTM、iTransformer、PatchTST、DLinear的完整Exchange H720 anchors；
不得为了得到eight-dataset表而把未审计结果混入。

Local H720 checkpoints已存在于ISCF-BSCA、TimeAlign、QDF、AMD和SimpleTM，
其Main II H720必须在相同native test contract下精确复现Main I；SimpleTM评估并
平均全部三个native repetitions。iTransformer、PatchTST和DLinear的Main I值是
published three-run means，而Main II首阶段是official-source single-seed复现，因此
只做signed deviation audit，不能强制等于published mean，也不能回写已冻结Main I。

该表回答one-model-all-horizons system competitiveness，但不作matched mechanism
attribution：各repository的lookback、objective、optimizer、selector、seed、参数量与
test-loader `drop_last`并不完全一致。正式decoder attribution仍由five-dataset
end-to-end core ablation和two-backbone transfer承担。Machine contract=
`configs/iscf_bsca_main_ii_h720_prefix_protocol.json`，prelaunch=
`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/design_and_prelaunch_gate.md`。

截至2026-08-09，Main II formal matrix已完整闭合：70个checkpoint evaluations、280个raw prefix rows、224个aggregate cells和448个MSE/MAE scalars全部通过，35个local H720 exact anchors无失败。ISCF-BSCA在28个dataset–horizon cells上的macro MSE/MAE为`0.262469/0.308281`，两项均在八个systems中排名第一；按共同三位小数显示口径为24/56 best、27/56 second，即51/56 metric cells位于前二。分dataset弱项仍完整保留：ETTh1 MAE与Solar MSE为rank 3，ETTh2/ECL双指标为rank 2。

该结果只把Main II推进为`paper_facing_effectiveness=pass`的one-model-for-all-horizons system benchmark。它不改变mechanism claim boundary：external source contracts并不matched，故BSCA/ISCF attribution与decoder portability仍必须由five-dataset end-to-end ablation、internal diagnostics和two-backbone transfer兑现。Canonical result=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/formal_results_20260809/result_and_table_audit.md`，LaTeX table=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/formal_results_20260809/table/table_iscf_bsca_main_ii.tex`。

Main II现与Main I使用同一展示契约：dataset顺序为ETTm1、ETTm2、ETTh1、
ETTh2、Weather、ECL、Solar；每dataset依次显示H96/H192/H336/H720与Avg.；
三位小数后best为red bold、second为blue underline；LaTeX间距与required packages
保持一致。两表只统一形式，不统一system集合或evidence role。当前全部paper-facing
experiment tables的状态、artifact与claim boundary统一登记于
`docs/iscf-bsca-paper-table-registry.md`；该整理不授权新的local patch、remote
training或formal test。

### 8.4 Efficiency Evaluation

以tuned `ISCF-BSCA-MAIN-v1`和可在同一环境复现的关键baselines报告：

- trained-model count；
- total stored parameters；
- training GPU-hours；
- single-request 与 all-horizon service latency；
- peak memory；
- CHPC guarantee。

`checkpoint count` 只允许出现在该实验/部署语境，不进入 Introduction 的宏观任务命名。

### 8.5 Ablation Studies

正文固定为五个matched end-to-end variants，并使用原5 datasets及exact `ISCF-BSCA-v1` hyperparameters，与Main I/II的tuned model严格分离：

| Variant | Frozen intervention | MSE | MAE |
| --- | --- | ---: | ---: |
| Full ISCF-BSCA | complete architecture and objective |  |  |
| w/o BSCA | retain Uniform-Prefix Forecasting Loss；remove Scope-Wise Forecasting Loss and Allocation-Balance Regularizer |  |  |
| w/o Target-Adaptive Allocation | replace learned Scope Probabilities with a matched non-adaptive fusion rule |  |  |
| Shared Scope Projection | replace scope-specific projections with one shared projection |  |  |
| Fixed Scope ($s=144$) | use only the preregistered middle scope；do not search for a best fixed scope |  |  |

不为Allocation-Balance Regularizer设置单独对照。Fixed Scope的$s=144$是budget-aware preregistered control，不是validation-selected optimum。random partition、scope count与$\lambda$ sensitivity仅在后续确有必要且获得独立授权时进入Appendix；当前不把它们纳入核心闭合矩阵。

### 8.6 Forecast Consistency and Scope-Allocation Behavior

展示：

1. ISCF-BSCA在shared targets上的exact CHPC/CHPD verification；
2. future-step或future-region × scope的Scope Probability map；
3. 各future regions/step bins上的aggregate scope utilization与scope-wise preference/error pattern；
4. 一条Full ISCF-BSCA相对frozen matched control提升最清晰的performance-selected trajectory，同时显示完整预测轨迹、nested horizon prefixes与对应Scope Probabilities。

Realized allocation value当前不纳入该节，以控制额外实验成本。Qualitative trajectory允许从提升最明显的样本中选择，但caption必须披露comparator、split与selection rule，并明确其是illustrative而非representative evidence。该节不设置failure-case panel；aggregate negative cells在5.2--5.3完整报告，并在Discussion中解释。

### 8.7 Backbone Transferability

选择结构不同的两个 backbones：

- lightweight linear/MLP backbone；
- patch/Transformer backbone。

| Backbone | Original Decoder | + ISCF | + ISCF-BSCA |
| --- | ---: | ---: | ---: |
| DLinear-style |  |  |  |
| PatchTST-style |  |  |  |

各backbone使用同一test-tuned原则选出的backbone-specific profile；不把
`ISCF-BSCA-v1` ablation hyperparameters机械迁移到transfer。
迁移实验用于判断 decoder 是否超越当前 encoder 的特定 co-adaptation，不能通过
只替换 frozen consumer 的不公平 probe 得出方向级结论。

## 9. Discussion

### 9.1 From Horizon-Specific Predictions to a Unified Forecasting System

解释将不同request endpoints视为同一预测轨迹的nested views后，forecasting system的定义、CHPC要求、accuracy与system cost之间的关系。该讨论只解释Main I、Main II、CHPC与efficiency evidence能够支持的系统层结论，不从CHPC推导accuracy，也不把所有flexible-horizon models概括为不一致。

### 9.2 Output-Side Sharing as a Forecasting Design Dimension

将Section 3的future-region sharing-demand heterogeneity与scope-indexed forecast generation、target-adaptive allocation、core ablation、allocation analysis和decoder transfer连接起来。Scope Probabilities或region-wise variation本身不等价于causal specialization；正向机制表述必须等待matched ablation与aggregate behavior共同支持。

### 9.3 Limitations and Future Scope

集中说明external baseline protocol heterogeneity、test-informed selection、seed coverage、deterministic point forecasting边界、prefix-bounded execution的implementation/evidence差距，以及最终结果中的negative dataset/horizon cells。这里只保留final artifacts实际支持的limitations，不使用泛化的future-work套话。

## 10. Conclusion

保持两段：

1. horizon-specific predictors 不构成带 CHPC 保证的 unified forecasting
   system，并概括ISCF-BSCA的scope-indexed forecast field、target-adaptive scope allocation与balanced co-adaptation；
2. 只总结最终已由artifacts支持的performance、efficiency、mechanism与transfer结论，以最窄的material boundary收束。

Conclusion不重复展开limitations；完整解释保留在Section 6.3。

## 11. Claim Boundary

### 10.1 允许的 claims

- 标准 horizon-specific multi-model protocol 不保证 CHPC；
- ISCF-BSCA 通过 horizon无关、future-step-indexed prediction field 满足 exact CHPC；
- matched evidence支持future regions存在region-dependent sharing-scale
  preference时，可表述为future-region sharing-demand heterogeneity；
- 一个scope-indexed forecast field提供多种future-step latent-state sharing
  scopes，作为对该问题的架构响应；
- independent scope-specific history projections相对near-matched shared-width
  projection有稳定收益；
- BSCA 相对 same-architecture ISCF-EQUAL 有 small but three-seed directionally
  robust gain；
- BSCA 不改变 inference graph。

### 10.2 当前禁止的 claims

- 所有既有 multi-horizon models 都不满足 CHPC；
- CHPC 本身是独立算法创新；
- unified forecasting 必然弱于 horizon-specific forecasting；
- multi-horizon training 存在普遍 negative-gradient conflict；
- canonical contiguous scopes 是 ISCF 收益的必要来源；
- policy 已学习到 universal sample-wise scope specialization；
- generic KL、entropy regularization、load balancing 或 expert training 是本文首创；
- BSCA 对所有 datasets/horizons 都有提升。
- 在seeds2022/2023可选扩展完成前，Main I/II、new ablations或transfer具有
  cross-seed robustness。

## 12. Primary-Source Terminology Audit

Search date：`2026-07-23` 至 `2026-07-24`；P5--P6 novelty-boundary refresh：
`2026-07-28`；Section 2 primary-source refresh：`2026-08-10`。

Topic scope：

- multi-step/direct/recursive/MIMO terminology；
- future time step、forecast step、lead time 与 horizon；
- multi-output block strategies；
- common versus step-specific future features；
- multi-scale forecast synthesis；
- probabilistic dependency across future time steps与deterministic
  predictive-state sharing的边界；
- forecast stability across creation dates；
- temporal forecast coherence/reconciliation；
- unified/multi-horizon models；
- output-side decoder modeling；
- multi-scale/multi-predictor forecasting、time-series MoE与generic expert
  balancing相对ISCF/BSCA的novelty boundary。

Primary sources：

1. Ben Taieb and Hyndman, *Boosting Multi-Step Autoregressive Forecasts*，
   ICML/PMLR direct-recursive strategy 与 bias--variance context：
   <https://proceedings.mlr.press/v32/taieb14.html>
2. Green et al., *Stratify: Unifying Multi-Step Forecasting Strategies*，
   RecMO/DirMO/DirRecMO 与 block-size parameterization：
   <https://link.springer.com/article/10.1007/s10618-025-01135-1>
3. Kan et al., *Multivariate Quantile Function Forecaster*，multi-horizon
   sequence models 与 dependency across future time steps：
   <https://proceedings.mlr.press/v151/kan22a.html>
4. Nguyen et al., *ClimateLearn*，direct、continuous 与 iterative forecasting
   的 lead-time terminology：
   <https://proceedings.neurips.cc/paper_files/paper/2023/file/ed73c36e771881b232ef35fa3a1dec14-Paper-Datasets_and_Benchmarks.pdf>
5. Das et al., *A Decoder-Only Foundation Model for Time-Series Forecasting
   (TimesFM)*，跨 datasets、granularities 与 horizons 的统一模型：
   <https://proceedings.mlr.press/v235/das24c.html>
6. *Forking Sequences*，forecast creation dates 上的 forecast-stability
   问题，与本文 fixed-origin CHPC 区分：
   <https://openreview.net/forum?id=dXdycy7WCX>
7. Girolimetto et al., *Cross-Temporal Probabilistic Forecast
   Reconciliation*，coherence 表示 aggregation constraints：
   <https://arxiv.org/abs/2303.17277>
8. Li et al., *Towards Accurate Time Series Forecasting via Implicit
   Decoding*，output-side forecasting phase 与 joint future pattern
   generation：
   <https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html>
9. Challu et al., *N-HiTS: Neural Hierarchical Interpolation for Time Series
   Forecasting*，multi-scale forecast synthesis与different frequency/scale
   components：
   <https://ojs.aaai.org/index.php/AAAI/article/view/25854>
10. Zhang et al., *TFFS: A Trainable Federal Fusion Strategy for Multistep
    Time Series Forecasting*，future steps之间的common feature sharing与
    step-specific information：
    <https://www.sciencedirect.com/science/article/pii/S0020025524010405>
11. Zeng et al., official DLinear implementation，`Linear(seq_len,
    pred_len)`中的step-specific output rows：
    <https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py>
12. Nie et al., official PatchTST implementation，flattened
    history-patch representation与`Linear(nf, target_window)`forecast head：
    <https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py>
13. Liu et al., official iTransformer implementation，shared
    variate-level representation与`Linear(d_model, pred_len)`projection：
    <https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py>
14. Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series
    Forecasting*，Past-Decomposable-Mixing与Future-Multipredictor-Mixing已覆盖
    generic multiscale predictor ensemble：
    <https://openreview.net/forum?id=7oLshfEIC2>
15. Liu, *FreqMoE: Enhancing Time Series Forecasting through Frequency
    Decomposition Mixture of Experts*，frequency-band experts与dynamic gating
    已覆盖frequency-side decomposition-and-fusion原语：
    <https://proceedings.mlr.press/v258/liu25i.html>
16. Liu et al., *Moirai-MoE: Empowering Time Series Foundation Models with
    Sparse Mixture of Experts*，token-level specialization说明generic
    time-series MoE不能作为本文的component novelty：
    <https://proceedings.mlr.press/v267/liu25an.html>
17. Guo et al., *Advancing Expert Specialization for Better MoE*，指出常见
    auxiliary load balancing可能造成expert overlap与过度均匀routing，支持
    BSCA只claim broad learning access而不claim forced specialization：
    <https://proceedings.neurips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html>
18. Ni et al., *Mixture-of-Linear-Experts for Long-term Time Series
    Forecasting*，多个完整linear-centric experts与router output mixing已覆盖
    generic multiple-predictor fusion叙事：
    <https://proceedings.mlr.press/v238/ni24a.html>
19. Shi et al., *Time-MoE: Billion-Scale Time Series Foundation Models with
    Mixture of Experts*，multi-resolution forecasting heads与dynamic
    scheduling已覆盖generic flexible-horizon multi-head叙事：
    <https://openreview.net/forum?id=e1wDDFmlVu>

Coverage boundary：

- 本轮术语检索确认`forecast step`、`lead time`等文献用法；Introduction为可读性
  使用`future time step`，Method允许在引用既有工作时保留原术语；
- `future-region sharing-demand heterogeneity`是本文的问题层术语；
  `region-dependent sharing-scale preference`是其matched empirical
  manifestation；`future-step latent-state sharing scope`是方法层术语；
- `predictive structure`边界过宽，multi-scale component importance不能直接推出
  cross-step sharing extent；当前问题定义显式加入finite-capacity
  bias--variance mechanism与matched risk-crossing requirement；
- DLinear、PatchTST与iTransformer不能简单标成“independent versus global”：
  需要分别区分step-specific rows、shared history representation与shared
  variate-level state；
- probabilistic literature中的`dependency across future time steps`不等同于
  deterministic decoder中的predictive-state sharing，因此不使用
  `future-step dependency heterogeneity`命名本文问题；
- 不claim“未来序列存在multi-scale structure”本身是新发现；novelty必须落在
  prefix-consistent unified setting、region-dependent sharing demand及其
  architecture/evidence chain上；
- TimeMixer、N-HiTS、FreqMoE与Moirai-MoE已显著压缩multi-scale、
  multi-predictor与MoE原语的novelty空间；ISCF只能在完整的output-side
  sharing-demand response chain上主张architecture contribution；
- MoLE与Time-MoE进一步说明multiple complete predictors、router output
  mixing与multi-resolution heads均不能作为ISCF的核心抽象；v0.6因此使用一个
  `scope-indexed forecast field`及其scope-conditioned slices描述共享计算图；
- `forecast field`本身只是对离散$(\tau,s)$乘积域上tensor function的paper
  abstraction，不claim neural-field primitive或“field”术语首创；
- generic load balancing既非BSCA的新原语，也不天然产生specialization；BSCA
  只能作为ISCF中policy-mediated gradient allocation的train-only
  co-adaptation机制陈述；
- 2026-07-28 Zotero connector返回`connection refused`，因此新增sources
  14--19均记为external primary-source discovery，是否已存在于用户Zotero
  `FSA` subset尚未核验；这是一项coverage gap，不影响本轮primary-source
  边界判断；
- 投稿前仍需针对最终 title、method naming 与 2026 最新 decoder work 再做一次
  freshness search。

## 13. 逐段讨论记录

| Date | Section | Consensus | Remaining Question |
| --- | --- | --- | --- |
| 2026-07-24 | Full paper structure | 六章正文、无独立 Discussion、problem evidence 前置 | 各章篇幅与图表编号待定 |
| 2026-07-24 | Introduction P1--P3 v0.1 | `forecast step`、full-trajectory prefix formulation | 由v0.2取代 |
| 2026-07-24 | Introduction P1--P3 v0.2 | `future time step`、horizon无关、future-step-indexed generation、CHPC | 英文最终措辞在全文写作阶段润色 |
| 2026-07-24 | Central scope terminology v0.2 | problem=`future-step coupling granularity`；instance=`future-step coupling scope` | 由v0.3的问题—证据—方法三层术语取代 |
| 2026-07-24 | Introduction P4 v0.3 | problem=`future-region predictive-structure heterogeneity`；缺少从multi-scale importance到sharing extent的逻辑桥 | 由v0.4取代 |
| 2026-07-24 | Introduction P4 v0.4 | problem=`future-region sharing-demand heterogeneity`；加入generation-state sharing的bias--variance mechanism与baseline decoder taxonomy | 第三章具体baseline、regions与matched controls待实验计划冻结 |
| 2026-07-24 | Introduction P5--P6 | 保留 provisional narrative | 由v0.5讨论稿取代 |
| 2026-07-28 | Introduction P5 v0.5 | ISCF完整方法概览；BSCA gradient-allocation机制；CHPC延续 | 待用户逐段确认英文强度与术语 |
| 2026-07-28 | Introduction P6 v0.5 | problem--architecture--training/evidence三项贡献；不提前写未完成主表结果 | 待用户确认贡献切分与最终结果句策略 |
| 2026-07-28 | ISCF framework + Introduction P5 v0.6 | ISCF=`Independent Scope-Conditioned Forecasting`；单一`scope-indexed forecast field`；`target-conditioned scope allocation`；P5重写 | 暂时冻结；P6下一轮按新框架重写 |
| 2026-07-29 | Introduction v0.2 round1 author response | varied-horizon仍欠充分系统发展；CHPC=basic property；P4 evidence pending；P5术语压缩；qualitative advantage only | 先确认P3，再冻结P4 diagnostic与Figure 1 |
| 2026-07-29 | Introduction problem-evidence v1 | 两项独立实验：native baseline NCHPD；neutral single-scale sharing-risk/CFH；Figure 1置于P2后，Figure 2置于P4后 | Step7A implementation与后续training/test均需新授权 |
| 2026-07-29 | Introduction evidence visualization pilot | Weather/seed2021共9 runs；85% disagreement quantile；validation only；formal matrices deferred | Step7B pass；remote initial pilot authorized |
| 2026-07-29 | Visualization candidate search extension | figure screening不作architecture rejection；prefix difference view retained；ETTm1 neutral 5-run screen | thresholds frozen；ETTh2/formal test未授权 |
| 2026-07-29 | Two problem-evidence figures selected | Weather prefix-difference/NCHPD；ETTm1 sharing-risk crossover/region contrast | integrate captions and Problem Formulation definitions |
| 2026-07-30 | Five-dataset full visualization search | maximum joint prefix selection；sample-level supported multi-scope selection；31 missing runs；global dynamic GPU queue | remote results与最终视觉样式待审计 |
| 2026-07-30 | Full-search figure selection | ETTh2 prefix + ETTm2 sharing；SVG/PDF/PNG/TIFF；Nature QA 13 pass/1 warn/0 fail | integrate captions；formal prevalence/CFH仍deferred |
| 2026-07-30 | Nature figure refinement | exact 183 mm；muted semantic palettes；compact prefix triangle；region-best excess-risk sharing heatmap | captions disclose maximum validation selection；formal prevalence/CFH仍deferred |
| 2026-07-30 | Prefix two-panel refinement | merge trajectory+difference；solid colors + staggered shapes + separation strokes；mean-difference inset | preserve raw statistics；caption更新为a/b两panel |
| 2026-07-30 | Prefix trajectory density refinement | forecast lines 0.82--0.95 pt；marker stride 18；subtle 0.38-pt under-stroke | reduce visual ink without changing data or evidence structure |
| 2026-07-30 | Introduction figures accepted for draft | Prefix disagreement + sharing-demand heterogeneity；SVG/PDF/PNG/TIFF centralized in `paper-figures/` | current manuscript-ready copies；validation-only claim boundary unchanged |
| 2026-07-30 | Introduction v0.5 evidence integration | Figures 1--2 embedded；aggregate statistics、selection disclosure与formal captions写入draft | problem-evidence prose complete；headline method results pending main tables |
| 2026-07-30 | Introduction v0.6 compact figure layout | one constructed Figure 1；real-data figures relocate to Section 3 Figures 2--3 | visual-review item resolved by v0.7；detailed proof removed from Introduction |
| 2026-07-30 | Introduction v0.7 figure approval and reflow | constructed Figure 1 approved and copied to `paper-figures/`；P1--P6 and caption use one physical line per natural paragraph | freeze Section 3 definitions and Figures 2--3 captions |
| 2026-07-30 | Introduction v0.8 structural polish | P2 concise；P3 CHPC + Figure 1a；P4 uniform output mechanism to sharing demand；P5 demand-to-scope bridge；P6 evidence-bounded evaluation sentence | main/transfer results decide final positive result wording |
| 2026-07-30 | Introduction v0.9 author refinement | Figure 1a moved to P2；caption shortened；single-scope contrast added；P6 strengthened；highlighted review copy created | Method Figure 4 planned；positive result sentence remains table-contingent |
| 2026-07-31 | Introduction v0.9 temporary freeze and paper-writing handoff | clean draft frozen usable；new authoritative reading order and startup prompt created | Section 3 integration is next；old D22 handoff becomes historical |
| 2026-07-31 | Paper-experiments parallel handoff | experiment-specific reading order、claim-to-table audit、baseline roles、prelaunch deliverables与copy-ready prompt冻结 | E0 artifact inventory active；remote training/formal test仍未授权 |
| 2026-07-31 | Paper-facing experiment consolidation v1 | exact checkpoint/hash audit；minimal baseline set；345 checkpoint slots，45 completed metric-evidence records/300 new；binary reuse unverified；Main I/II、ablation、transfer、efficiency与four-layer gates冻结 | E2 conditional pass；request Tier A local patch only；remote training/formal test仍false |
| 2026-08-03 | Section 3 v0.3 concise polish | manuscript body由2,308词压缩至1,576词；合并重复释义与claim boundary；精简Figures 2--3 captions和future CFH protocol；保留CHPC/CHPD/NCHPD/$\operatorname{UP}_H$/$R_{o,b,s}$/CFH、matched controls与validation-only边界 | author review；Introduction与并行experiment cursor不变 |
| 2026-08-03 | Section 3 v0.4 field-style alignment | 参照iTransformer、TimeMixer、TimeXer与TimeMixer++官方论文校准时序预测顶会语体；P1改为单一承接段，删除meta roadmap P2；3.1--3.5改为连续setting-to-evidence叙事 | author review；definitions、numbers、claim boundaries、Introduction与experiment cursor不变 |
| 2026-08-04 | Section 3 v0.5 author structure refinement | 3.1改为shared-target→CHPC→horizon-specific contrast→unified function；3.2区分inconsistency现象与CHPD/NCHPD统计并重写Figure 2叙事；删除naive-unified accuracy与Design Requirements subsections；future-region sharing renumber为3.3并按panel a/b重写 | author review；Figure 2 selection与Figure 3 oracle boundary压缩保留；Introduction、figures与experiment cursor不变 |
| 2026-08-04 | Section 3 v0.6 terminology and flow refinement | 3.1明确主语为varied-horizon forecaster；3.2以trajectory-level与aggregate evidence合并收束；3.3将经验量$R_{o,b,s}$统一为region-wise $\operatorname{MSE}_{o,b,s}$，Figure 3a同步改为excess MSE | author review；统计值、figure布局、claim boundary、Introduction与experiment cursor不变 |
| 2026-08-04 | Section 3 v0.7 author risk-definition refinement | 保留3.1 varied-horizon subject；3.2改为DLinear observation→horizon-specific structural limitation；3.3撤销region-wise MSE命名并正式定义future-region prediction risk $R_{o,b,s}$ | author review；Figure 3a同步为excess prediction risk；数据、统计值、claim boundary、Introduction与experiment cursor不变 |
| 2026-08-04 | Section 3 v0.7 temporary freeze | 用户确认当前版本基本满意并暂时固定为论文可用Section 3 | body、terminology、equations、Figures 2--3 integration与captions冻结；后续明确矛盾 + author approval才解冻；writing cursor转向Section 4 pending direction |
| 2026-08-05 | Section 4 v0.1 and Method Figure 4 | 六段Method computation flow、exact tensor/coordinate/scope/allocation/BSCA/complexity公式与四panel Figure 4 initial bundle完成 | author review；Introduction/Section 3与frozen implementation不变；main/ablation/transfer claims仍由Section 5 tables决定 |
| 2026-08-07 | Section 4 v0.2 main-figure alignment | Figure 4 visual design暂时固定为单一ISCF inference schematic；正文、subsection names、terminology ledger、caption与editorial audit按History State、Scope Matrix、Region Descriptor、Scope-region State、Region-to-Step Forecast Generator、Scope-conditioned Forecast、Condition Vector、Allocation MLP、Scope Probabilities和Varied-Horizon Forecasting同步 | author text review；stable SVG/PDF/TIFF source待同步；Introduction/Section 3、implementation与claim boundary不变 |
| 2026-08-07 | Section 4 v0.3 author refinement through 4.3 | 按author逐项反馈重写Section开头、4.1--4.3；强化decoder-side framing、Encoder interface、Future Coordinate rationale、per-scope information pool与region-local generation chain | continued author review；prefix-bounded execution仅作为architecture-supported property，reference implementation full-field materialization边界写入editorial audit；Introduction/Section 3、implementation与experiment authorization不变 |
| 2026-08-08 | Section 4 v0.4 path and allocation refinement | 固定`Scope Forecasting Path`与`Target-Adaptive Allocation Path`；精简Figure 4 caption；4.3强化unified field与parameter sharing；4.4重构target-adaptive granularity allocation；4.5--4.6同步optimization/CHPC/complexity | author review；新增v0.4 highlighted review；prefix-bounded execution仍只作architecture property，current full-field implementation与performance/efficiency claim boundaries不变 |
| 2026-08-08 | Section 4 v0.5 BSCA objective refinement | 精简4.3 unified-field总结；4.5按multi-scope gradient imbalance问题重构；统一`Uniform-Prefix Forecasting Loss`、`Scope-Wise Forecasting Loss`与`Allocation-Balance Regularizer`命名和符号 | author review；三项均为training objectives；uniform KL仅拓宽early gradient access，不保证equal usage、sufficient training或specialization；implementation与实验状态不变 |
| 2026-08-10 | Section 4 v0.6 BSCA training refinement | 4.5显式区分varied-horizon objective与multi-scope stabilization；重写balance作用为penalizing non-uniform allocation并平衡probability-mediated gradients；删除4.6 | author review；CHPC construction保留在4.4；complexity转入Section 5.4 efficiency/analysis；不新增mechanism或effectiveness claim |
| 2026-08-10 | Section 4 v0.7 BSCA narrative-order refinement | 保留4.5首段dual-role总述；先完整定义uniform-prefix objective，再说明multi-scope probability-scaled gradient问题并引出scope-wise与balance terms | author review；删除突兀的`second objective`前置解释；公式、objective、implementation、experiments与claim boundaries不变 |
| 2026-08-10 | Section 4 v0.7 temporary freeze and Section 2 v0.1 Related Work initial draft | 用户确认Section 4暂时敲定；按四段funnel完成Related Work primary-source refresh、subsection design与英文初稿 | Section 4 body/terms/equations/Figure 4 integration frozen usable；Section 2 v0.1 pending author review；ElasTST horizon-invariance prior显式承认；experiment cursor不变 |
| 2026-08-10 | Section 2 v0.2 author structure refinement | 按author反馈严格区分recursive/direct/MIMO/DIRMO，重构foundation-model、forecast-generation、output-side multi-scale与allocation叙事 | 四个subsection标题同步；ElasTST系统贡献完整承认，差异收紧到history-patch resolution与output-side state-sharing extent；Introduction/Sections 3--4与experiment cursor不变 |
| 2026-08-10 | Section 2 v0.2 temporary freeze | 2.3 opening改为`Beyond shallow output projections`，移除`A smaller body of work`的数量判断 | Author确认其余内容；Section 2正文、结构、citations与claim boundaries暂时冻结；next manuscript section pending direction；experiment cursor不变 |
| 2026-08-11 | Sections 5--7 structural design v0.1 | 基于frozen Sections 1--4、table registry与claim boundaries设计Experiments、Discussion、Conclusion及Appendix evidence ladder | 新增独立设计稿；standalone Discussion、case-study routing与5.6 split/merge均pending author discussion；不填result prose、不改实验授权 |
| 2026-08-12 | Sections 5--7 structural design v0.2 temporary freeze | Author确认七章结构与standalone Discussion；Core-Ablation固定为Full、w/o BSCA、w/o Target-Adaptive Allocation、Shared Scope Projection与Fixed Scope $s=144$；qualitative并入5.6 | 不设balance-only或failure-case；realized allocation value移出当前计划；performance-selected example必须披露selection；不新增实验授权 |
