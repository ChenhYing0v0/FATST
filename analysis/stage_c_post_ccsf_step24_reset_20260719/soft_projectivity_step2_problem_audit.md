# Soft Projectivity Step 2 问题审计与 D18 设计

## 1. 为什么出现这条候选问题

exact projectivity要求：

$$
F_H(x)=P_HF_K(x),\qquad H\leq K.
$$

它等价于所有horizons共享一个full-domain predictor：

$$
F_H(x)=P_HF_T(x).
$$

这带来一个清晰的no-go：若shared-prefix prediction必须完全相同，requested horizon就不可能产生任何统计适配。
因此过去“exact projectivity + horizon-adaptive accuracy”的目标在数学上过度约束。

D17又表明，保留exact projectivity后对frozen full-domain draft做prefix-safe correction并不能跨validation→test
稳定迁移。该结果不否定E2E future-context，却没有给继续实现它的正向problem evidence。

新的Step 2问题不是立即引入horizon embedding，而是先问：

> 相对一个强measure-trained unified model，分别为H96/H192/H336优化的同架构models，是否在各自horizon上存在
> 稳定、跨dataset的accuracy headroom？

若不存在，soft projectivity没有研究必要；若存在，才说明exact shared predictor可能以accuracy换取一致性。

## 2. Soft projectivity的数学位置

可将未来候选问题写成：

$$
\min_\theta
\mathbb E_{H\sim\mu}\mathcal R_H(F_H(x),y_{1:H})
+\lambda\,
\mathbb E_{H<K}
D\!\left(F_H(x),P_HF_K(x)\right).
$$

- $\lambda\rightarrow\infty$：退化为exact projectivity，$H$在shared prefix上冗余；
- $\lambda=0$：退化为horizon-specific predictors，没有统一一致性；
- 有限$\lambda$：允许模型在accuracy收益足够时产生受控prefix deformation。

这给出一个potential paper problem：学习accuracy–consistency Pareto frontier，而不是把exact invariance当作无需验证
的绝对正确约束。

[Self-critique] finite-$\lambda$本身只是generic consistency regularization。若没有连续、低自由度、可解释的
horizon deformation operator，以及与其耦合的training principle，它不足以形成SCI贡献。

## 3. External primary-source boundary

检索日期：2026-07-19。external-first，Zotero presence未作为novelty证据。

| Source | Primary evidence | Boundary |
| --- | --- | --- |
| ElasTST, NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html | 直接主张horizon-invariant output并使用horizon reweighting；soft projectivity若成立，必须明确挑战rigid invariance而非重复它 |
| Temporal horizons in forecasting, OpenReview submission 2025 | https://openreview.net/forum?id=BeudQIxT1R | 说明training horizon影响AR model的performance/learnability landscape；支持“horizon有优化代价”，但设定为autoregressive dynamical forecasting |
| Loss Shaping, ICML 2024 | https://openreview.net/forum?id=9CCoVyFuEp | 已覆盖per-step constraint与primal-dual shaping；generic Lagrangian不计novelty |
| When Rigid Coherency Hurts, OpenReview submission | https://openreview.net/forum?id=YsNlFsG-jj | 在probabilistic hierarchical forecasting中讨论rigid coherency的代价；提供概念邻近prior，但不是fixed-past horizon projectivity |
| ElasTST / TimesFM / AutoTimes | official proceedings pages | 已覆盖varied-length inference、decoder-only arbitrary horizon与AR arbitrary length；“一个模型支持多个长度”不能单独claim |

本次没有找到直接覆盖
`fixed-past horizon-specialization headroom -> controlled shared-prefix deformation -> accuracy-consistency frontier`
的primary work，属于[Medium-confidence absence finding]，不是novelty证明。

## 4. SC-D18-SPC：Soft-Projectivity Cost diagnostic

### 4.1 诊断目的

D18只测problem existence，不实现soft-projective model：

> exact shared full-domain predictor相对horizon-specialized training是否付出稳定accuracy代价？

### 4.2 Matched carrier

统一使用A6-LBF-natural architecture与五个dataset-aware profiles。所有arms：

- output domain固定T=720；
- Encoder、decoder、parameter count与initialization class一致；
- 不输入requested H；
- 只改变training loss mask与checkpoint selector。

已有controls：

- `A6_MEASURE`：对H96/H192/H336/H720的统一measure训练与mean-four-H checkpoint；
- `A6_FULL`：full-H720 loss control。

新增diagnostic arms：

- `A6_SPEC96`：只优化prefix H96，checkpoint按validation H96；
- `A6_SPEC192`：只优化prefix H192，checkpoint按validation H192；
- `A6_SPEC336`：只优化prefix H336，checkpoint按validation H336。

T720以后的输出在short-specific arms中不作performance claim；保留full-domain head只为确保architecture与parameter
count一致。D18的preregistered test cells仅是各specific arm的own horizon，以及两个统一controls对应horizon。

### 4.3 Matrix

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- seed：2021 problem screen；
- new training：3 specific arms × 5 datasets = 15 runs；
- reused frozen controls：A6_MEASURE、A6_FULL；
- MSE与MAE逐dataset报告；
- 保存same-input prediction probes，计算specific arm与A6_MEASURE在shared prefix上的NRMSE。

test仍是`test_informed problem diagnostic`。candidate、loss masks、checkpoint rules、profiles、seed、cells与gates必须
在launch前冻结；不得按dataset选择specific horizon或loss weight。

### 4.4 预冻结problem gates

只有全部通过，soft projectivity才进入Step 4：

1. three-horizon macro own-H MSE gain over A6_MEASURE $\geq0.5\%$；
2. 至少2/3 specific horizons为正；
3. 至少4/5 datasets的three-horizon aggregate为正；
4. 至少10/15 dataset-horizon cells为正；
5. each-horizon macro不允许低于`-0.5%`；
6. shared-prefix prediction NRMSE非零，且dataset-horizon gain不能完全由A6_FULL弱control解释；
7. 无NaN/Inf、>100% degradation或checkpoint/protocol mismatch。

### 4.5 Decision map

- 若相对A6_MEASURE不通过：exact projectivity没有可测accuracy cost，soft route关闭，回Step 2重审整篇fixed-past
  multi-horizon主线；
- 若只相对A6_FULL通过：`measure training explains`，不需要soft architecture；
- 若specific arms稳定超过A6_MEASURE：problem升级`problem_supported`，进入Step 4设计连续低秩deformation
  operator与matched exact/independent controls；
- 即使D18通过，也不能把separate horizon-specific models写成贡献，它们只是oracle/problem controls。

## 5. 当前授权边界

当前只完成Step 2/3 protocol design：

- method implementation：false；
- remote training：false，需下一轮完成static prelaunch与user continuation；
- official test access：false until frozen machine-readable config；
- Contribution 1 / Contribution 2：仍未形成。
