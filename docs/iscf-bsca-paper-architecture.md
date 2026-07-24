# ISCF-BSCA 论文结构与叙事共识稿

## 文档状态

| Field | Content |
| --- | --- |
| `document_role` | ISCF-BSCA 论文全文结构、术语、claim 与实验布局的权威讨论稿 |
| `version` | `v0.2` |
| `last_updated` | `2026-07-24` |
| `paper_candidate` | `ISCF-BSCA-v1` |
| `current_review_cursor` | Introduction 前半部分（第1--3段）完成第二轮共识 |
| `frozen_consensus` | 论文六章结构；Introduction 第1--3段的 horizon无关、future-step-indexed formulation 与术语体系 |
| `provisional_content` | Introduction 第4--6段、Related Work、Problem/Motivation、Method、Experiments、Conclusion |
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

### 1.2 Future-step coupling granularity

本文把 decoder 在生成多个 future time steps 时采用的 latent-state sharing
粒度称为：

> **future-step coupling granularity**

它描述的是多个 future time steps 在 target-specific synthesis 之前，以多细或
多粗的范围共享预测状态。该概念属于 output-side decoder structure，不等同于：

- requested forecast horizon；
- input receptive field；
- input temporal resolution；
- frequency band；
- forecast targets 之间真实的 probabilistic dependence。

### 1.3 Future-step coupling scope

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
sharing。首次定义后，正文可简称 `coupling scope` 或 `scope`。

问题层面统一使用：

> **future-step coupling heterogeneity**

其含义是：适合的 output-coupling granularity 可能随 sample、variable 与
future time step 变化。本文不预设“short horizon 必然偏好 local scope”或
“long horizon 必然偏好 global scope”。

### 1.4 Horizon无关、future-step-indexed generation

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

### 1.5 Cross-horizon prefix consistency

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
满足 CHPC、使用一个 unified model 的同时，避免 fixed-granularity decoder
在多个 future time steps 上形成表达折中。

### 1.6 明确弃用或限制的表述

| Avoid | Replacement / Boundary |
| --- | --- |
| `single-checkpoint multi-horizon forecaster` | Introduction 使用 `unified multi-horizon forecaster` |
| `same-origin cross-horizon prefix consistency` | 正式术语使用 CHPC；`fixed forecast origin` 写入定义 |
| `future-generation scope heterogeneity` | 使用 `future-step coupling heterogeneity` |
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
> future time steps 的预测一致；但同一个 fixed-granularity decoder 仍需承担
> 不同 samples、variables 与 future time steps 的异质输出生成需求。
> ISCF 通过多个 independent scope-coupled fields 表示不同
> future-step coupling granularities，并对每个 forecast target 融合这些
> fields；BSCA 在不改变 inference graph 的前提下稳定其 joint training。

该主线包含四层：

1. `system need`：一个 unified model 服务多个 nested horizons；
2. `system contract`：CHPC；
3. `modeling problem`：future-step coupling heterogeneity；
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
   3.4 Heterogeneous Future-Step Coupling Granularities
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

### 4.4 Paragraph 4：naive unification 与 coupling heterogeneity

**状态：待继续讨论。**

当前建议：

> Horizon-agnostic, future-step-indexed generation establishes a consistent
> interface across horizons, but it does not by itself ensure accurate
> unified forecasting. A single fixed-granularity decoder must use one
> output-generation structure for all samples, variables, and future time
> steps. This forces the decoder to balance fine-grained flexibility against
> broader latent sharing, although the appropriate future-step coupling
> granularity may vary across forecast targets.

本段禁止预设 strict negative gradient conflict，也不预设某个 horizon 与某个
scope 的一一对应关系。

### 4.5 Paragraph 5：ISCF-BSCA

**状态：待继续讨论。**

当前建议依次介绍：

1. ISCF 建立多个 independent scope-coupled fields；
2. 每个 field 对应一种 future-step coupling scope，并定义各 future steps 的预测；
3. forecast-target-wise fusion 对每个 $(\tau,c)$ 组合 fields；
4. BSCA 在训练期促进 fields 与 fusion policy 的 balanced co-adaptation；
5. inference graph 不因 BSCA 增加参数或路径；
6. horizon无关、future-step-indexed definition 使 CHPC by construction。

不声称 canonical contiguous partition 已被证明优于 random partition；不声称
policy 学得 universal conditional specialization。

### 4.6 Paragraph 6：contributions

**状态：待继续讨论。**

当前三项 contribution：

1. **Problem and formulation.** 定义 nested-horizon unified forecasting 与
   CHPC，系统分析 horizon-specific systems 的跨 horizon disagreement、系统
   冗余，以及 naive unified decoder 的输出粒度折中。
2. **Architecture.** 提出 ISCF，用多个 independent scope-coupled fields 与
   forecast-target-wise fusion 表示 heterogeneous future-step coupling
   granularities。
3. **Training and evidence.** 提出 ISCF-specific BSCA，在 inference graph
   不变的情况下改善 balanced co-adaptation，并通过 horizon-specific、
   unified、matched-control 与 mechanism-health surfaces 评估。

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
`unified nested-horizon problem -> CHPC contract -> heterogeneous future-step
coupling -> independent fields and fusion -> balanced co-adaptation` 完整链上。

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

### 6.4 Evidence III：Heterogeneous Future-Step Coupling Granularities

在相同 simple/frozen baseline representation 上连接 capacity-matched
diagnostic heads：

- fine-grained sharing；
- block-level sharing；
- global sharing。

对 scope setting $s$ 与 future-step bin $b$ 定义：

$$
E_{s,b}
=
\operatorname{MSE}
\left(
\text{diagnostic head with sharing scope }s,
\text{future-step bin }b
\right).
$$

展示：

- coupling-granularity × future-step-bin heatmap；
- best-granularity map；
- scope crossover curves；
- data-side local fluctuation、block trend 与 scale-dependent
  predictability 作为描述性辅助。

方向级证据要求多个 scope heads 的 relative performance 随 sample、variable
或 future-step region 稳定交叉。单纯 gradient magnitude difference、单调
lead-time difficulty 或 data-side energy 不能单独建立该问题。

### 6.5 Design Requirements

由前三项证据导出：

1. 一个 unified model 服务全部 horizons；
2. 每个 future-step prediction 对 requested horizon 保持invariant，因而满足CHPC；
3. decoder 支持多种 future-step coupling granularities；
4. 不同 forecast targets 可组合不同 coupling scopes；
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
- multiple fields 表示不同 future-step coupling granularities；
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

Search date：`2026-07-23` 至 `2026-07-24`。

Topic scope：

- multi-step/direct/recursive/MIMO terminology；
- future time step、forecast step、lead time 与 horizon；
- multi-output block strategies；
- forecast stability across creation dates；
- temporal forecast coherence/reconciliation；
- unified/multi-horizon models；
- output-side decoder modeling。

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

Coverage boundary：

- 本轮术语检索确认`forecast step`、`lead time`等文献用法；Introduction为可读性
  使用`future time step`，Method允许在引用既有工作时保留原术语；
- `future-step coupling granularity` 与 CHPC 是为本文问题链给出的明确术语，
  不是宣称既有文献从未使用相近概念；
- 投稿前仍需针对最终 title、method naming 与 2026 最新 decoder work 再做一次
  freshness search。

## 12. 逐段讨论记录

| Date | Section | Consensus | Remaining Question |
| --- | --- | --- | --- |
| 2026-07-24 | Full paper structure | 六章正文、无独立 Discussion、problem evidence 前置 | 各章篇幅与图表编号待定 |
| 2026-07-24 | Introduction P1--P3 v0.1 | `forecast step`、full-trajectory prefix formulation | 由v0.2取代 |
| 2026-07-24 | Introduction P1--P3 v0.2 | `future time step`、horizon无关、future-step-indexed generation、CHPC | 英文最终措辞在全文写作阶段润色 |
| 2026-07-24 | Central scope terminology | problem=`future-step coupling granularity`；instance=`future-step coupling scope` | Method title 与 ISCF acronym expansion 后续单独审议 |
| 2026-07-24 | Introduction P4--P6 | 保留 provisional narrative | 下一轮逐段讨论 |
