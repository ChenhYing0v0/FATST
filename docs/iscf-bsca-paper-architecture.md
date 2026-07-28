# ISCF-BSCA 论文结构与叙事共识稿

## 文档状态

| Field | Content |
| --- | --- |
| `document_role` | ISCF-BSCA 论文全文结构、术语、claim 与实验布局的权威讨论稿 |
| `version` | `v0.5` |
| `last_updated` | `2026-07-28` |
| `paper_candidate` | `ISCF-BSCA-v1` |
| `current_review_cursor` | Introduction 第5--6段讨论稿已形成，等待逐段确认 |
| `frozen_consensus` | 论文六章结构；Introduction 第1--4段的任务定义、CHPC 与 future-region sharing-demand formulation |
| `provisional_content` | Introduction 第5--6段v0.5讨论稿；Related Work、Problem/Motivation 的具体实验、Method、Experiments、Conclusion |
| `not_authorized_by_this_document` | 新模型实现、remote training、formal test、按结果调参 |

本文档用于逐段讨论论文，而不是宣告全文已经定稿。标记为
`frozen_consensus` 的内容在出现新证据或明确讨论结论前保持不变；
`provisional_content` 只表示当前最佳结构，后续按章节继续修订。

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
`history-conditioned generation state`适宜被多宽范围的future steps共同复用。
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

### 1.4 Future-step coupling granularity and scope

方法层面把 decoder 在生成多个 future time steps 时采用的 latent-state sharing
粒度称为：

> **future-step coupling granularity**

一个具体的共享范围称为：

> **future-step coupling scope**

对 scope size $s$，其结构语义为：

> $s$ 个 future time steps 在 target-specific synthesis 之前共享一个由历史
> representation 与该组 future-step descriptor 构造的 latent generation
> state。

ISCF 当前使用：

$$
\mathcal S=\{1,48,144,360,720\}.
$$

其中 $s=1$ 提供最细粒度的 latent-state sharing，$s=720$ 提供 full-domain
sharing。该概念属于 output-side decoder structure，不等同于 requested
forecast horizon、input receptive field、input temporal resolution、frequency
band 或 forecast targets 之间真实的 probabilistic dependence。

因此，`future-region sharing-demand heterogeneity` 是问题，
`region-dependent sharing-scale preference` 是可检验表现，
`future-step coupling scope` 才是 ISCF 对该问题的架构响应。单个 future step
本身没有 coupling scope；scope 描述的是一组 future steps 之间的共享关系。

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

当前 ISCF 以step-specific synthesis coefficients实现该接口：每个 future
time step $\tau$具有自己的identity synthesis row、nonlinear synthesis row与
temporal bias，并由target-wise policy融合多个scope fields。这里的
`step-specific`不表示future steps相互独立；它们仍可通过不同coupling scopes
共享latent generation state。

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
| `future-generation scope heterogeneity` / `future-step coupling heterogeneity` | 问题使用 `future-region sharing-demand heterogeneity`；方法使用 `future-step coupling scope` |
| “某个future step具有某种coupling granularity” | scope是多个future steps之间的共享关系；用 `region-dependent sharing-scale preference` 描述经验现象 |
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
> 轨迹成分则可能受益于跨步复用共同generation state。单一共享范围未必能适合
> 所有samples、variables与future regions。ISCF通过多个independent
> scope-coupled fields提供不同future-step coupling scopes，并对每个forecast
> target融合这些fields；BSCA在不改变inference graph的前提下稳定其joint
> training。

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
   2.1 Multi-Step and Horizon-Specific Forecasting
   2.2 Unified Multi-Horizon Forecasting and Consistency
   2.3 Forecast Decoders and Output-Side Temporal Modeling
   2.4 Multi-Scale Temporal Modeling

3. Problem Formulation and Empirical Motivation
   3.1 Horizon-Specific and Unified Multi-Horizon Forecasting
   3.2 Cross-Horizon Disagreement of Horizon-Specific Models
   3.3 Performance Compromise in Naive Unified Forecasting
   3.4 Future-Region Sharing-Demand Heterogeneity
   3.5 Design Requirements

4. ISCF-BSCA: Prefix-Consistent Unified Multi-Horizon Forecasting
   4.1 Architecture Overview
   4.2 Horizon-Agnostic Future-Step Generation and CHPC
   4.3 History Encoding
   4.4 Independent Scope-Coupled Fields
   4.5 Forecast-Target-Wise Scope Fusion
   4.6 Balanced Scope Co-Adaptation
   4.7 Complexity and Structural Properties

5. Experiments
   5.1 Experimental Setup
   5.2 One Unified Model versus Horizon-Specific Models
   5.3 Unified Multi-Horizon Benchmark
   5.4 Efficiency Evaluation
   5.5 Ablation Studies
   5.6 Alleviating the Unified Forecasting Problem
   5.7 Decoder Transferability
   5.8 Case Studies

6. Conclusion

Appendices
   A. Full Dataset-Horizon Results
   B. Additional Coupling-Scope Diagnostics
   C. Hyperparameter Sensitivity
   D. Reproducibility Details
```

不设置独立 `Discussion`。必要 limitations 放在 Conclusion 末段，完整 negative
cells、secondary controls 与敏感性结果放在 Appendices。

## 4. Introduction

### 4.1 Paragraph 1：multi-horizon need 与现行 horizon-specific protocol

**状态：第二轮共识。**

写作目标：

1. 从多个 planning ranges 的真实需求出发；
2. 定义本文研究的是 nested horizons 上的 unified multi-horizon forecasting；
3. 指出现行 benchmark 通常按 $H$ 分别训练模型；
4. 暂不讨论 ISCF、scope 或 loss。

建议正文逻辑：

> Multi-horizon forecasting is essential in applications that require
> predictions over several planning ranges, from short-term control to
> long-term scheduling. However, the standard long-term forecasting protocol
> typically trains a separate model for each forecast horizon $H$, such as
> 96, 192, 336, and 720 steps. Although each model is individually optimized
> for its designated horizon, the resulting collection of horizon-specific
> predictors does not constitute a unified multi-horizon forecasting system.

这里的 `multi-horizon` 特指：同一个 forecasting system 服务多个请求长度
$H\in\mathcal H$，而不仅是一个模型一次输出多个 future steps。

### 4.2 Paragraph 2：horizon-specific systems 的三个不足

**状态：第二轮共识。**

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

建议正文逻辑：

> For the same observed history, independently trained horizon-specific
> models may produce different predictions for the same future time step. In
> particular, the first $H_1$ steps predicted by an $H_2$-step model are not
> guaranteed to agree with the output of a separately trained $H_1$-step
> model, even when $H_1<H_2$. Such horizon-dependent disagreement prevents
> the forecasts from being interpreted as nested views of one future
> trajectory. Moreover, maintaining separate models for different horizons
> multiplies training, storage, and deployment costs.

本段不声称 horizon-specific models 的 accuracy 必然更差。

### 4.3 Paragraph 3：unified forecaster 与 CHPC

**状态：第二轮共识。**

建议正文逻辑：

> We therefore formulate unified multi-horizon forecasting through a
> horizon-agnostic prediction function indexed by future time step. Given an
> observed history, the model directly defines a prediction for each future
> step, and an $H$-step forecast is instantiated by evaluating the
> corresponding sequence of future steps. Since the prediction at each step
> is determined by the observed history and its future-step index, rather
> than by the requested horizon, predictions at overlapping future steps
> remain identical across horizons. We refer to this property as
> cross-horizon prefix consistency. Our architecture realizes this
> step-indexed interface through step-specific synthesis coefficients,
> allowing arbitrary horizons to be instantiated without horizon-specific
> prediction heads.

CHPC 在 Introduction 中作为 forecasting-system desideratum，而不是单独包装为
method novelty。该 formulation 不把不同 horizons 称为相互独立；它们是同一个
horizon无关、step-indexed field 的nested outputs。

### 4.4 Paragraph 4：naive unification 与 future-region sharing demand

**状态：v0.4共识冻结。**

建议正文：

> Although a horizon-agnostic prediction field provides a consistent
> interface across horizons, it does not make unified forecasting inherently
> easy. Many direct multi-output forecasters generate all
> future steps either from a broadly shared latent representation or through
> one fixed output-generation pattern. Such a fixed pattern imposes the same
> extent of cross-step sharing throughout the forecast domain. However,
> fine-scale variations may require greater step-specific flexibility,
> whereas smoother and broader trajectory components may benefit from
> reusing a common history-conditioned generation state across multiple
> steps. Because their relative importance can vary across samples,
> variables, and future regions, no single sharing extent is guaranteed to
> be uniformly suitable. We refer to this challenge as **future-region
> sharing-demand heterogeneity**.

对应中文：

> 尽管horizon无关的prediction field为不同horizons提供了一致的预测接口，但它
> 本身并不能使unified forecasting天然变得容易。许多direct multi-output
> forecasters从一个宽范围共享的latent representation生成全部future
> steps，或者在整个预测域采用一种固定的output-generation pattern。这种设计
> 对所有future regions施加了相同的cross-step sharing extent。然而，细粒度
> 变化可能需要更强的step-specific flexibility，而平滑、宽范围的轨迹成分则
> 可能受益于在多个future steps之间复用共同的history-conditioned generation
> state。由于这些成分的相对重要性可能随sample、variable与future region
> 变化，单一sharing extent未必能在整个预测域内始终适用。我们将这一挑战称为
> **future-region sharing-demand heterogeneity（未来区间共享需求异质性）**。

讨论过程中的细节：

1. **本段的叙事作用。** Paragraph 3只建立horizon无关接口与CHPC；Paragraph
   4进一步指出prediction interface一致并不等于unified forecasting已经解决，
   从system contract自然转入decoder modeling problem，为Paragraph 5引出ISCF。
2. **问题、证据与方法必须分层。** 问题层术语是
   `future-region sharing-demand heterogeneity`；其可检验表现是
   `region-dependent sharing-scale preference`；`future-step coupling scope`
   只属于ISCF的方法层。不能用模型中的scope反过来定义问题。
3. **future region不是forecast horizon。** future region
   $\mathcal B_b\subseteq\{1,\ldots,T\}$ 是预测域内部一组连续future steps；
   forecast horizon $H$ 是一次预测请求覆盖的总长度。不同future regions的
   sharing demand可能不同，不表示同一future step会因requested horizon变化而
   改变预测目标。
4. **scope是关系而不是单点属性。** 单个future step本身没有coupling
   granularity或scope；scope描述多个future steps在target-specific synthesis
   之前共享predictive state的范围。因此正文不写“different future steps have
   different coupling granularities”。
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
8. **理论与claim边界。** 在fixed history、pointwise MSE且requested horizon
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

### 4.5 Paragraph 5：ISCF-BSCA

**状态：v0.5讨论稿，待用户确认。**

建议正文：

> To address these heterogeneous sharing demands, we propose ISCF-BSCA, a
> unified multi-horizon forecaster built on Independent Scope-Coupled Fields
> (ISCF). ISCF constructs multiple scope-coupled forecast fields. Each field
> defines predictions across the future domain, but has its own
> history-to-mode mapping and reuses a history-conditioned generation state
> according to a distinct cross-step sharing extent; step-specific synthesis
> coefficients
> retain the flexibility to produce individual forecast targets. A
> forecast-target-wise fusion policy then combines the field predictions for
> each sample, variable, and future time step, allowing multiple sharing
> extents to contribute within one horizon-agnostic prediction function.
> Because this policy also determines how the fused forecasting loss allocates
> gradients to the fields, we further introduce Balanced Scope Co-Adaptation
> (BSCA), a train-only objective that maintains broad learning access across
> fields and mitigates premature policy concentration during joint
> optimization. BSCA adds neither inference parameters nor an additional
> inference path. Together with the future-step-indexed interface, the
> resulting architecture directly instantiates arbitrary forecast horizons
> while retaining cross-horizon prefix consistency.

对应中文：

> 为响应上述异质性共享需求，我们提出ISCF-BSCA：一个以Independent
> Scope-Coupled Fields（ISCF）为核心的unified multi-horizon forecaster。
> ISCF构造多个scope-coupled forecast fields。每个field都在预测域上为各个
> future time steps定义prediction，但拥有独立的history-to-mode mapping，并
> 按照一种不同的cross-step sharing extent复用history-conditioned generation
> state；与此同时，
> step-specific synthesis coefficients保留了生成单个forecast target所需的
> 灵活性。随后，forecast-target-wise fusion policy针对每个sample、variable
> 和future time step组合各field的prediction，使多种sharing extents能够在
> 同一个horizon无关prediction function中共同发挥作用。由于该policy还决定
> fused forecasting loss如何向各fields分配gradient，我们进一步提出Balanced
> Scope Co-Adaptation（BSCA）：一种仅在训练期使用的objective，用于维持各
> fields获得广泛学习信号的机会，并缓解joint optimization中过早的policy
> concentration。BSCA既不增加inference parameters，也不引入额外的inference
> path。结合future-step-indexed interface，该架构可以直接实例化任意forecast
> horizon，同时保持cross-horizon prefix consistency。

讨论过程中的细节：

1. **与Paragraph 4的因果衔接。** Paragraph 4的问题是不同future regions的
   sharing demand可能不同；Paragraph 5才引入`future-step coupling scope`作为
   架构响应。正文使用“address these heterogeneous sharing demands”，不将
   coupling scope反写成问题定义。
2. **field覆盖整个预测域。** 每个scope-coupled field都为各future time steps
   提供prediction；“scope”描述field内部的generation state在多宽的future-step
   span内被复用，不表示一个field只预测某个forecast horizon或某个固定future
   region。
3. **independent的精确定义。** independent只指各scope拥有独立的
   history-to-mode affine maps；它们仍共享encoder、future-step representation、
   step-specific synthesis与final fusion policy。它不表示不同horizons独立，也
   不表示多个完全独立模型。
4. **step-specific flexibility没有被scope sharing抹去。** scope-level state
   sharing发生在target-specific synthesis之前；每个future time step仍通过独立
   synthesis coefficients形成forecast target。因此ISCF不是把一个group中的
   future steps强制预测为相同值。
5. **tensor-level压缩表述。** Method中再展开
   `scope_modes:[B,C,S,D,K] -> arm_forecasts:[B,C,T,S]`，
   `policy:[B,C,T,S]`与`fused_forecast:[B,T,C]`。Introduction只保留
   history-to-mode mapping、sharing extent、step-specific synthesis与
   target-wise fusion四个必要概念。
6. **fusion的允许解释。** forecast-target-wise fusion使不同sharing extents
   能够按sample、variable与future time step产生不同贡献；在没有更强mechanism
   evidence前，不写成“policy为每个target选择最优scope”，也不声称已经形成
   universal conditional specialization。
7. **BSCA的机制边界。** direct policy同时参与prediction mixture与fused-loss
   gradient allocation。BSCA通过uniform arm-skill supervision和ramped
   `KL(uniform || policy)`维持broad learning access，目标是缓解early
   concentration与scope-gradient starvation。uniform policy只是这一机制的
   proxy，不保证fields自动学习互不重叠的features。
8. **BSCA的贡献边界。** BSCA的论文价值来自
   `scope-coupled fields -> policy-mediated gradient allocation ->
   train-only balanced co-adaptation`这一ISCF-specific链条，不来自generic
   KL、load balancing或expert training。已有MoE研究也表明过强均匀路由可能
   抑制specialization，因此正文使用`maintains broad learning access`与
   `mitigates premature concentration`，不使用“forces balanced experts”。
9. **CHPC表述延续Paragraph 3。** 模型以horizon无关、future-step-indexed
   prediction function直接实例化任意horizon；Introduction不写成先生成max-$T$
   再crop，也不声称independently generate horizons。CHPC仍是system property，
   不是Paragraph 5新增的算法模块。
10. **不进入正文的负边界。** canonical contiguous partition相对random
    partition的必要性未获支持；multiple scales、multiple predictors和MoE原语
    也已有充分prior art。这些限制保留在Related Work、ablation解释和claim
    boundary中，不塞入Introduction方法概览。

### 4.6 Paragraph 6：contributions

**状态：v0.5讨论稿，待用户确认。**

建议正文：

> Our contributions are threefold. First, we formulate prefix-consistent
> unified multi-horizon forecasting as a forecasting-system problem and
> formalize future-region sharing-demand heterogeneity as a testable
> finite-capacity challenge: different samples, variables, and future regions
> may favor different extents of cross-step state sharing. We operationalize
> this hypothesis through capacity-matched region-wise diagnostics, separating
> evidence for the task problem from the scope-based mechanism used to address
> it. Second, we develop ISCF, an output-side forecasting architecture that
> combines independent history-to-mode mappings, multiple future-step coupling
> scopes, step-specific synthesis, and forecast-target-wise fusion within a
> horizon-agnostic prediction function. This design responds to heterogeneous
> sharing demands while retaining cross-horizon prefix consistency without
> horizon-specific prediction heads. Third, we introduce BSCA, an ISCF-specific
> train-only objective for stabilizing the co-adaptation of scope-coupled
> fields and their fusion policy without changing the inference graph. We
> evaluate the complete framework against horizon-specific systems, matched
> unified forecasters, and architecture/objective controls, covering
> predictive accuracy, system efficiency, cross-horizon consistency, mechanism
> attribution, and decoder transferability.

对应中文：

> 本文的贡献主要包括三个方面。第一，我们将具有prefix consistency的unified
> multi-horizon forecasting形式化为一个forecasting-system problem，并将
> future-region sharing-demand heterogeneity形式化为一个可检验的有限容量
> 挑战：不同sample、variable与future region可能偏好不同程度的cross-step
> state sharing。我们通过capacity-matched的region-wise diagnostics对该假设
> 进行操作化检验，从而将task problem的存在性证据与用于解决该问题的
> scope-based mechanism区分开来。第二，我们提出ISCF，一个output-side
> forecasting architecture；它在同一个horizon无关prediction function中结合
> independent history-to-mode mappings、multiple future-step coupling scopes、
> step-specific synthesis与forecast-target-wise fusion。该设计在响应异质性
> sharing demand的同时，无需horizon-specific prediction heads即可保持
> cross-horizon prefix consistency。第三，我们提出BSCA，一个ISCF-specific、
> train-only objective，用于稳定scope-coupled fields及其fusion policy的
> co-adaptation，并且不改变inference graph。我们将通过horizon-specific
> systems、matched unified forecasters以及architecture/objective controls
> 对完整framework进行评估，覆盖predictive accuracy、system efficiency、
> cross-horizon consistency、mechanism attribution与decoder transferability。

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
   `heterogeneous output-side sharing demand -> independent
   history-to-mode maps across coupling scopes -> step-specific synthesis ->
   forecast-target-wise fusion in a horizon-agnostic field`。TimeMixer、
   N-HiTS、FreqMoE与Moirai-MoE压缩了generic primitive-level novelty，但不
   自动覆盖该完整problem--mechanism链。
5. **BSCA不claim generic balancing novelty。** Contribution 3明确写成
   `ISCF-specific train-only objective`。现有三seed证据只支持相对ISCF-EQUAL
   的small but directionally robust gain，不支持universal gain、强
   specialization或所有datasets/horizons一致改善。
6. **结果句暂不超前。** 当前段落只冻结evaluation dimensions，不提前写“一个
   unified model优于四个horizon-specific models”或“state-of-the-art”。只有
   horizon-specific主表、matched unified主表与modern baselines完整后，才能在
   最终Introduction加入具体性能结论。
7. **证据结构必须与贡献一一对应。** Contribution 1由baseline/simple matched
   diagnostics与CHPC disagreement支撑；Contribution 2由matched architecture
   controls、full test MSE/MAE和scope behavior支撑；Contribution 3由same-
   architecture objective control、three-seed official-test与internal health
   支撑。efficiency与transferability属于完整framework evidence，不能替代
   mechanism attribution。
8. **transferability是empirical scope，不是预先成立的贡献。** 在decoder迁移
   实验完成前，正文只写`evaluate ... decoder transferability`；若迁移结果不
   稳定，应将其降为analysis或limitation，而不是保留正向claim。

BSCA 的 novelty 位于 ISCF-specific contribution chain，不 claim generic
KL、entropy regularization 或 load balancing 首创。

## 5. Related Work

### 5.1 Multi-Step and Horizon-Specific Forecasting

覆盖：

- recursive；
- direct；
- direct-recursive hybrid；
- MIMO / multi-output；
- DIRMO / block multi-output；
- benchmark 中按 horizon 分别训练的 protocol。

落点：

> 既有研究主要讨论 error accumulation、bias--variance trade-off 与 future-step
> dependencies；本文进一步关注多个 requested horizons 是否来自同一个
> CHPC forecasting system。

### 5.2 Unified Multi-Horizon Forecasting and Consistency

区分：

1. 一个 model family 支持不同输出长度；
2. 一个 forward 同时输出多个 future time steps；
3. 同一模型服务多个 requested horizons；
4. 不同 requested horizons 满足 CHPC。

不得声称所有已有模型都不 unified 或都不满足 CHPC。TimesFM 等模型已研究跨
horizon generalization；forking-sequence models 生成 multi-horizon grids，但其
forecast-origin stability 与本文 CHPC 不同。

### 5.3 Forecast Decoders and Output-Side Temporal Modeling

围绕 history representation 到 forecast sequence 的映射讨论：

- linear/global output heads；
- patch-based readouts；
- channel-independent decoding；
- basis、block、segment 与 implicit forecasting；
- shared 与 forecast-step-specific generation。

核心 comparison question：

> decoder 如何在多个 future time steps 之间分配 latent sharing，而不是 encoder
> 如何处理历史输入。

### 5.4 Multi-Scale Temporal Modeling

明确区分：

- 既有 multi-scale 方法通常处理 input resolutions、frequency bands 或
  history features；
- ISCF 的 scope 是 output-side latent-state sharing extent。

不把 primitive overlap 自动写成 novelty rejection；claim 落在
`unified nested-horizon problem -> CHPC contract -> future-region
sharing-demand heterogeneity -> multiple coupling scopes and fusion ->
balanced co-adaptation` 完整链上。

## 6. Problem Formulation and Empirical Motivation

本章不出现 ISCF、BSCA、arm 或 production method 名称。问题证据使用已有
baselines 或简单 capacity-matched diagnostic heads。

### 6.1 Horizon-Specific and Unified Multi-Horizon Forecasting

Horizon-specific formulation：

$$
\hat{\mathbf Y}^{(H)}
=
f_{\theta_H}(\mathbf X),
\qquad
H\in\mathcal H.
$$

Unified formulation：

$$
\hat{\mathbf Y}^{(H)}
=
\left[
g_\theta(\mathbf X,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C},
$$

其中 $g_\theta$ 是horizon无关、future-step-indexed prediction function。
随后定义 CHPC，并说明 $H$ 是 forecast horizon、$\tau$ 是 future time step、
$(\tau,c)$ 是 forecast target。

### 6.2 Evidence I：Cross-Horizon Disagreement

对 DLinear、PatchTST、iTransformer 等已有 baseline 分别训练
$H=96,192,336,720$ 的模型。定义 Cross-Horizon Prefix Disagreement：

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

归一化版本：

$$
\operatorname{NCHPD}
=
\frac{\operatorname{CHPD}}{\operatorname{Std}(Y)}.
$$

计划展示：

- horizon-pair NCHPD heatmap；
- same-history overlapping future-step overlay；
- disagreement versus forecast error；
- model count、training cost、storage 与 deployment cost。

该证据只证明“不提供 CHPC 保证”和系统冗余，不证明 horizon-specific accuracy
更差。

### 6.3 Evidence II：Naive Unified Forecasting

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

### 6.4 Evidence III：Future-Region Sharing-Demand Heterogeneity

这一节在提出 ISCF 之前建立问题证据，不使用 ISCF、scope arm、BSCA 或
target-wise fusion 等方法术语。主要证据来自相同 simple baseline 上
end-to-end训练的 capacity-matched diagnostic predictors：

- fine-grained cross-step sharing；
- local/block-level sharing；
- broad/global sharing。

若使用 frozen representation，只能作为 secondary diagnostic，不能承担
problem-direction rejection。primary comparison应采用相同encoder class、数据、
objective、optimization、checkpoint rule与comparable initialization进行matched
end-to-end training。

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

该问题得到支持需要同时观察：

1. matched risk curves在不同future regions稳定交叉；
2. $s_b^\star=\arg\min_sR_{b,s}$ 随region稳定变化，而不是单一scale全域支配；
3. region-wise best scale相对best single fixed scale留下稳定headroom；
4. 若主张连续future regions有意义，ordered contiguous grouping必须与
   parameter-matched random grouping区分。

主要展示：

- sharing-scale × future-region risk heatmap；
- region-wise best-sharing-scale map；
- sharing-scale crossover curves；
- region-specific oracle相对best fixed scale的headroom；
- ordered versus random grouping control。

data-side local fluctuation、block trend、frequency/scale energy与可预测性分析只作
描述性辅助，不能单独建立sharing-demand heterogeneity。单纯gradient
magnitude difference、单调lead-time difficulty或oracle label routing也不能
单独证明可学习的region-dependent sharing-scale preference。

### 6.5 Design Requirements

由前三项证据导出：

1. 一个 unified model 服务全部 horizons；
2. 每个 future-step prediction 对 requested horizon 保持invariant，因而满足CHPC；
3. decoder 不应把单一固定cross-step sharing pattern强加给整个future domain；
4. 架构应提供多种future-step coupling scopes，使不同forecast targets可组合
   不同sharing extents；
5. 多个 scope paths 需要稳定 joint training。

## 7. Method

### 7.1 Architecture Overview

先给出完整 tensor flow，再解释直觉。

### 7.2 Horizon-Agnostic Future-Step Generation and CHPC

对每个 future time step 与变量，模型定义：

$$
\hat y_{b,\tau,c}
=
g_\theta(\mathbf X_b,\tau,c),
$$

其中 requested horizon 不进入 $g_\theta$ 的预测语义。一个 $H$-step forecast
直接由 $\tau=1,\ldots,H$ 的step-indexed predictions组成；这些 future steps
可以批量并行计算，但架构定义不依赖预先固定的 requested horizon。

### 7.3 History Encoding

$$
\mathbf X:[B,L,C]
\rightarrow
\mathbf Z:[B,C,P,D]
\rightarrow
\mathbf R:[B,C,PD].
$$

### 7.4 Independent Scope-Coupled Fields

对每个 $s\in\mathcal S$，使用独立的 history-to-mode map 生成
scope-specific modes。每个 field 为全部可查询 future steps 定义
scope-specific predictions：

$$
\hat{\mathbf Y}_s
\in
\mathbb R^{B\times C\times T}.
$$

scope size 只规定 future-step latent sharing extent，不是 requested horizon。

### 7.5 Forecast-Target-Wise Scope Fusion

policy weights：

$$
\mathbf P
\in
\mathbb R^{B\times C\times T\times S}.
$$

最终预测：

$$
\hat y_{b,c,\tau}
=
\sum_{s=1}^{S}
p_{b,c,\tau,s}
\hat y_{b,c,\tau,s}.
$$

### 7.6 Balanced Scope Co-Adaptation

$$
\mathcal L
=
\mathcal L_{\mathrm{fused}}
+
\mathcal L_{\mathrm{equal\text{-}skill}}
+
\lambda(u)
\frac{
D_{\mathrm{KL}}(q\Vert p)
}{
\log S
}.
$$

解释边界：

- equal-skill 为各 scope fields 提供直接 predictive supervision；
- uniform anchor 避免 policy 过早关闭部分 scope 的 fused-gradient access；
- ramp 允许 prediction paths 先建立基本能力；
- BSCA 不增加 inference 参数或路径；
- 当前证据支持 balanced co-adaptation，不支持更强的 universal conditional
  specialization claim。

### 7.7 Complexity and Structural Properties

报告：

- trainable parameters；
- inference FLOPs、latency 与 memory；
- one-model versus multi-model system cost；
- CHPC；
- BSCA train-only property。

## 8. Experiments

### 8.1 Experimental Setup

报告 datasets、$\mathcal H=\{96,192,336,720\}$、seeds、validation-only
checkpoint selection、official-test MSE/MAE、parameter/training budgets，以及
horizon-specific 与 unified 两种 protocol。

### 8.2 Main Results I：Unified versus Horizon-Specific

每个 baseline 使用四个 horizon-specific trained models；ISCF-BSCA 使用一个
unified trained model。

| Model | # Trained Models | H96 | H192 | H336 | H720 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear-Specific | 4 |  |  |  |  |  |
| PatchTST-Specific | 4 |  |  |  |  |  |
| iTransformer-Specific | 4 |  |  |  |  |  |
| TimeMixer-Specific | 4 |  |  |  |  |  |
| ISCF-BSCA | 1 |  |  |  |  |  |

该表回答一个 unified model 能否与 separately optimized horizon-specific
models 竞争，但不单独承担 architecture attribution。

### 8.3 Main Results II：Unified Multi-Horizon Benchmark

把相同 baselines 改成 horizon无关、future-step-indexed unified variants，
并在相同 requested horizons 上评估。使用一致的 checkpoint selector 与尽可能
matched 的 supervision protocol。

| Unified Model | # Trained Models | H96 | H192 | H336 | H720 | Avg. | Gap to Native Specific |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear-Unified | 1 |  |  |  |  |  |  |
| PatchTST-Unified | 1 |  |  |  |  |  |  |
| iTransformer-Unified | 1 |  |  |  |  |  |  |
| TimeMixer-Unified | 1 |  |  |  |  |  |  |
| ISCF-BSCA | 1 |  |  |  |  |  |  |

Table I 证明实际 system competitiveness；Table II 隔离 unified setting 下的
decoder effectiveness。两表不能相互替代。

### 8.4 Efficiency Evaluation

报告：

- trained-model count；
- total stored parameters；
- training GPU-hours；
- single-request 与 all-horizon service latency；
- peak memory；
- CHPC guarantee。

`checkpoint count` 只允许出现在该实验/部署语境，不进入 Introduction 的宏观任务命名。

### 8.5 Ablation Studies

正文只保留主要组件的 with/without：

| Variant | Independent Fields | Target-Wise Fusion | BSCA | MSE | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full ISCF-BSCA | yes | yes | yes |  |  |
| w/o BSCA | yes | yes | no |  |  |
| w/o Independent Fields | no | yes | yes |  |  |
| w/o Target-Wise Fusion | yes | no | yes |  |  |
| w/o Multiple Coupling Scopes | no | no | no |  |  |

random partition、scope count 与 $\lambda$ sensitivity 放 Appendix。由于
canonical versus random partition 当前无稳定正向归因，不把 canonical grouping
包装为核心 positive ablation。

### 8.6 Alleviating the Unified Forecasting Problem

展示：

1. naive unified、ISCF-EQUAL 与 ISCF-BSCA 的 unified penalty；
2. horizon-specific systems 的 NCHPD 与 ISCF-BSCA 的 exact-zero CHPD；
3. future-step × coupling-scope utilization maps；
4. per-scope forecast error、fused error、prediction diversity 与 oracle
   headroom；
5. 不同 future-step bins 上的 improvement。

该节回答“前文问题被缓解多少”，不负责首次证明问题存在。

### 8.7 Decoder Transferability

选择结构不同的两个 backbones：

- lightweight linear/MLP backbone；
- patch/Transformer backbone。

| Backbone | Original Decoder | + ISCF | + ISCF-BSCA |
| --- | ---: | ---: | ---: |
| DLinear-style |  |  |  |
| PatchTST/iTransformer-style |  |  |  |

迁移实验用于判断 decoder 是否超越当前 encoder 的特定 co-adaptation，不能通过
只替换 frozen consumer 的不公平 probe 得出方向级结论。

### 8.8 Case Studies

可视化：

- 同一历史的 horizon-specific overlapping forecasts；
- unified full trajectory 与多个 prefixes；
- coupling-scope weights 随 future time step 的变化；
- difficult samples 上 field forecasts 与 fused forecast。

## 9. Conclusion

保持三段：

1. horizon-specific predictors 不构成带 CHPC 保证的 unified forecasting
   system；
2. ISCF-BSCA 以 multiple future-step coupling scopes、target-wise fusion 与
   balanced co-adaptation 构建统一模型；
3. 总结 performance、efficiency、mechanism evidence 与 limitations。

当前 limitations 至少包括：

- BSCA 相对 EQUAL 的增益 small but directionally robust，不是 universal gain；
- ETTm2 存在负向 dataset effect；
- canonical contiguous grouping 的必要性未被 matched random control 支持；
- 当前结论限定于 deterministic point forecasting；
- official test 已按治理规则明确标记为 test-informed。

## 10. Claim Boundary

### 10.1 允许的 claims

- 标准 horizon-specific multi-model protocol 不保证 CHPC；
- ISCF-BSCA 通过 horizon无关、future-step-indexed prediction field 满足 exact CHPC；
- matched evidence支持future regions存在region-dependent sharing-scale
  preference时，可表述为future-region sharing-demand heterogeneity；
- multiple fields提供不同future-step coupling scopes，作为对该问题的架构响应；
- independent fields 相对 near-matched shared-width field 有稳定收益；
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

## 11. Primary-Source Terminology Audit

Search date：`2026-07-23` 至 `2026-07-24`；P5--P6 novelty-boundary refresh：
`2026-07-28`。

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
   <https://openreview.net/forum?id=gqoeQPhQcE>
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

Coverage boundary：

- 本轮术语检索确认`forecast step`、`lead time`等文献用法；Introduction为可读性
  使用`future time step`，Method允许在引用既有工作时保留原术语；
- `future-region sharing-demand heterogeneity`是本文的问题层术语；
  `region-dependent sharing-scale preference`是其matched empirical
  manifestation；`future-step coupling scope`是方法层术语；
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
- generic load balancing既非BSCA的新原语，也不天然产生specialization；BSCA
  只能作为ISCF中policy-mediated gradient allocation的train-only
  co-adaptation机制陈述；
- 2026-07-28 Zotero connector返回`connection refused`，因此新增sources
  14--17均记为external primary-source discovery，是否已存在于用户Zotero
  `FSA` subset尚未核验；这是一项coverage gap，不影响本轮primary-source
  边界判断；
- 投稿前仍需针对最终 title、method naming 与 2026 最新 decoder work 再做一次
  freshness search。

## 12. 逐段讨论记录

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
