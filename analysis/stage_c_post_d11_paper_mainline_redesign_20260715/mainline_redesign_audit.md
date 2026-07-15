# StageC Post-D11 Paper Mainline Redesign Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `search_date` | 2026-07-15 |
| `current_step` | StageC joint rollback to Step 2-3；先验证新的architecture-training problem，不直接实现method |
| `D11_scope` | 只否定“short/long directional gradient conflict”作为SC1主问题；没有否定Contribution 1 slot、D6 crossing或projective future representation |
| `old_SC1` | PMFO/PAF/JAPO exact implementations保持关闭；RGNB/projectivity/locality evidence保留 |
| `old_SC2` | `SC2-MIPR retired_as_core_candidate`；raw measure risk只保留为protocol/control |
| `new_thesis` | unified multi-horizon forecasting是一个共享future function在nested prefix-risk family下的低秩可预测逼近问题 |
| `new_C1` | provisional `PRISM`：Prefix-Risk Isometric Synthesis Module |
| `new_C2` | provisional `CAPE`：Cross-fitted Adaptive Predictable-Energy frame learning |
| `method_authorization` | false；只授权`D12 predictable-frame feasibility`的train/validation diagnostic；test=false |
| `rollback` | D12若不能跨dataset同时支持predictable compaction与prefix localization，回Step 2并关闭forecast-frame主线 |

## 1. D11到底否定了什么

[Fact] D11在五个datasets、三个A6 checkpoints上没有发现strict short/long directional conflict：formal
total-gradient gate为`0/5`，support-specific component gate为`2/5`，same-component跨regime negative为`0`。
因此不能再把“短期与长期梯度互相打架”写成论文问题，也不能据此引出PCGrad、GradNorm、component loss或
gradient surgery。

[Fact] D11仍观测到nested support的coverage差异：short prefix对RGNB最后两个groups严格没有gradient，long
measure才覆盖这些groups；responsibility redistribution为`3/5`。这是一条监督可达性观察，不是conflict证据。

[Strong Evidence] Contribution 1没有被否定。以下独立证据仍成立：

1. D6在disjoint validation window上复现local b144与global DCT的short-positive/long-negative crossing，
   12/15 units通过；
2. D7/D8均表明canonical local geometry优于permuted/random geometry；
3. PMFO的conservation相对ablation有正向作用，而失败集中在rigid transition/readout；
4. A6已经证明free rank-256 history-to-future operator是一条强carrier，且$H$只作row crop即可获得exact prefix
   consistency。

[Decision] 正确动作不是“放弃Contribution 1，直接做Contribution 2”，而是把SC1从已经失败的
`tree/router/expert/conflict`实现抽象中退出来，重新寻找能解释D6、同时不破坏A6自由度的future representation。

## 2. External Primary-Source Audit

### 2.1 Search protocol

本轮遵循项目规则：Zotero只作seed，不用于判断新颖性或检索完整性。外部检索优先使用会议论文集、OpenReview、
arXiv与official code。主题覆盖：dynamic/arbitrary horizon、functional/basis/implicit decoder、hierarchical
interpolation、future-label transform、quadratic objective、per-step constraints、importance/task sampling及
function-space discretization consistency。

### 2.2 Prior-art boundary

| Work | Primary source | Direct pressure | Remaining gap |
| --- | --- | --- | --- |
| N-HiTS, AAAI 2023 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/25854) | hierarchical interpolation、multi-rate sampling、coarse-to-fine synthesis已被覆盖 | 未按nested prefix-risk family设计rank-limited output frame |
| BasisFormer, NeurIPS 2023 | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html) | learned/interpretable basis与history-basis coefficient matching已被覆盖 | 未区分future variance与history-predictable future variance，也未处理prefix-risk locality |
| TimesFM, ICML 2024 | [paper](https://proceedings.mlr.press/v235/das24c.html) | variable context/prediction lengths不能作为独立claim | autoregressive foundation-model目标与本项目single shared projective forecast不同 |
| DAM, 2024 preprint | [paper](https://arxiv.org/abs/2407.17880) | adjustable continuous basis与non-fixed horizons形成直接压力 | continuous basis本身不再可claim；prefix-risk frame仍是不同问题 |
| Implicit Forecaster, NeurIPS 2025 | [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | constituent waves与implicit future synthesis已被覆盖 | 不以frequency/amplitude/phase作为本项目创新 |
| FlowState, 2025 workshop | [OpenReview](https://openreview.net/forum?id=R50AT6nAsM) | functional basis decoder、sampling-rate invariance与dynamic horizons已被覆盖 | flexible length/resolution不是本项目claim；nested risk geometry仍未被直接覆盖 |
| TV-INR, 2025 preprint/TMLR submission | [paper](https://arxiv.org/abs/2506.01544) | continuous INR与随机forecast-length training增加generic neural-field压力 | 不再采用“continuous function”作为独立贡献 |
| Time-o1, NeurIPS 2025 | [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0cd62dea69635f4c5b569848267fe5a8-Abstract-Conference.html) | PCA transformed-label alignment、label autocorrelation与task overload已被覆盖 | 不能把future transform/component loss写成新贡献 |
| QDF, ICLR 2026 | [OpenReview](https://openreview.net/forum?id=vpO8n9AqEG) | full quadratic weighting、off-diagonal label correlation与heterogeneous step weights已被覆盖 | generic quadratic/step weighting不能成为SC2；output-frame estimation是不同干预点 |
| Loss Shaping Constraints, 2024 | [OpenReview](https://openreview.net/forum?id=9CCoVyFuEp) | per-step loss constraints已被覆盖 | 不做per-step bound或dynamic loss weighting claim |
| Importance Sampling, ICML 2018 | [paper](https://proceedings.mlr.press/v80/katharopoulos18a.html) | unbiased sampling与gradient-variance reduction是generic primitive | coverage-aware horizon sampling单独不足以成为贡献 |
| MTO audit, NeurIPS 2022 | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract-Conference.html) | task sampling和hyperparameter protocol可解释许多复杂MTO收益 | SC2必须与simple sampling/weighting作严格controls |

[Decision] 已经被占据或创新边界过弱的叙事包括：

- “第一个支持arbitrary horizon的decoder”；
- “用basis/wavelet/implicit function生成future”；
- “future steps相关，所以做PCA/component/quadratic loss”；
- “不同horizon训练机会不同，所以做importance sampling”；
- “短长期梯度冲突，所以做gradient surgery”。

[Provisional Novelty] 本轮未发现已有工作同时研究以下完整链条：

> nested prefix deployment risk → rank-limited future frame的conditioning-locality tradeoff →
> conditional-mean/predictable-energy frame estimation → exact domain-only projective synthesis。

这只是截至检索日的中等置信度判断，不是absence proof。投稿前仍需citation chaining与2026 freshness audit。

## 3. New Paper Mainline

### 3.1 Core problem

令最大future domain为$T$，requested horizon为$H$，prefix mask为$M_H$。一个共享模型预测同一个future
function $\hat y(x)\in\mathbb R^T$，部署时只返回$\hat y_{1:H}$。若deployment horizon distribution为
$\mu(H)$，平均prefix MSE为

$$
\mathcal R_\mu
=\mathbb E_{H\sim\mu}\left[\frac1H\|M_H(\hat y-y)\|_2^2\right]
=(\hat y-y)^TW_\mu(\hat y-y),
$$

$$
W_\mu=\mathbb E_{H\sim\mu}[M_H/H].
$$

[Fact] $W_\mu$只是exact risk identity，本身不是创新。它说明multi-horizon统一训练不是四个离散task IDs，而是
同一个future function在一族nested domains上的风险积分。

当decoder只有$r=256<T=720$个future degrees of freedom时，真正的问题是：

1. global basis容易压缩future covariance，却会把short-prefix误差与long-domain结构耦合；
2. local basis容易服务short prefix，却损失long-domain coherence；
3. raw-label PCA优先保留高方差方向，但其中可能包含history无法预测的noise；
4. 因此，frame应同时考虑`where accuracy is requested`与`what history can predict`。

这把D6的conditioning-locality crossing与A6的rank-256 carrier统一到一个可推导问题中。

### 3.2 Proposed paper statement

> Unified multi-horizon forecasting should learn one projective future function in a forecast frame that is
> localized by the deployment prefix measure and concentrated on future variation predictable from history.

中文表述：一个统一模型不应识别“这是H96还是H720”，而应先学习同一个future field；decoder负责让有限
capacity在nested prefix风险下分布合理，training strategy负责只把capacity分配给history真正可预测的future
directions。

## 4. Contribution 1 Candidate: PRISM Decoder

工作名：`PRISM = Prefix-Risk Isometric Synthesis Module`。

### 4.1 Tensor path

保留A6 Encoder与free coefficient generator：

$$
M[B,C,P,D]\rightarrow h[B,C,PD]\rightarrow a[B,C,r],\qquad r=256.
$$

PRISM只重构future synthesis：

$$
U_\mu[T,r],\quad U_\mu^TW_\mu U_\mu=I,
$$

$$
\hat y_H[B,C,H]=U_\mu[:H,:]\,a[B,C,r].
$$

$H$只裁剪$U_\mu$的rows，不进入Encoder、coefficient head、query、router或expert，因此任意两个horizons的
共同prefix严格一致。coefficient head的shape与rank保持A6一致，避免再犯PAF/JAPO用受限shared map替换A6
自由operator的问题。

### 4.2 Prefix-risk localization

对$W_H=M_H/H$，定义frame在prefix family中的cross-component leakage：

$$
\mathcal L_{prefix}(U)
=\mathbb E_{H\sim\mu}\left[
\|\operatorname{offdiag}(U^TW_HU)\|_F^2
\right].
$$

- 只优化future covariance时，frame趋向global PCA/DCT式compaction；
- 只优化$\mathcal L_{prefix}$时，frame趋向time-local supports；
- 两者的Pareto解正对应D6已经观测到的global conditioning与prefix locality tradeoff。

PRISM不是固定balanced intervals、recursive tree、exchangeable experts或explicit-H routing。它把
`prefix measure family`直接写入future frame的几何约束，同时保持一个free coefficient path。

### 4.3 Contribution boundary

可claim的不是“basis decoder”或“orthogonal basis”，而是：

1. 从nested prefix risks推导frame metric与localization functional；
2. 在同一rank-limited frame中连续求解predictive compaction/locality Pareto frontier；
3. domain-only crop带来exact projectivity；
4. 与CAPE的predictable-energy estimator形成architecture-training co-design。

### 4.4 Feasibility and risk

| Aspect | Assessment |
| --- | --- |
| algebra | high：$T=720,r=256$的weighted Stiefel/eigendecomposition规模很小 |
| implementation | high：替换A6 `basis [720,256]`，coefficient head/Encoder不变 |
| optimization | medium：locality约束可能过强；必须先做Pareto diagnostic而非直接E2E |
| novelty | medium：primitive与basis/PCA/orthogonality重叠，但完整problem-mechanism chain目前未见直接覆盖 |
| performance | medium：D6支持tradeoff存在，但尚未证明Pareto frame能超过free learned A6 basis |

[Self-Critique] 如果PRISM只是给A6 basis加一个orthogonality regularizer，它不足以成为SCI contribution。只有
`prefix-risk-derived geometry + predictable-energy construction + exact projective decoder`的完整机制通过controls，
才能升为paper core。

## 5. Contribution 2 Candidate: CAPE Frame Learning

工作名：`CAPE = Cross-fitted Adaptive Predictable-Energy frame learning`。

### 5.1 Why raw future covariance is the wrong object

在先减去train-only future mean（或等价地为decoder保留intercept）后，令
$m(x)=\mathbb E[y\mid x]$。对任意满足$U^TW_\mu U=I$的rank-$r$ frame和任意coefficient predictor
$f(x)$：

$$
\min_f\mathbb E\|y-Uf(x)\|_{W_\mu}^2
=\mathbb E\|y\|_{W_\mu}^2
-\operatorname{tr}(U^TW_\mu\Sigma_mW_\mu U),
$$

其中$\Sigma_m=\operatorname{Cov}(m(x))$。最优$f^*(x)=U^TW_\mu m(x)$。

[Theory Implication] rank受限decoder应该捕获conditional mean的covariance，即history可预测的future energy，
而不是raw label covariance：

$$
\operatorname{Cov}(y)=\Sigma_m+\mathbb E[\operatorname{Cov}(y\mid x)].
$$

raw PCA可能把有限rank浪费在高方差但不可预测的noise上。Time-o1/QDF关注label transform或loss metric；CAPE的
干预点是**如何估计并构造decoder output subspace**。

[Verification] 随机weighted frame的数值sanity check得到
$\max|U^TW_\mu U-I|=1.55\times10^{-15}$，上述centered risk identity gap为
$2.84\times10^{-14}$。这只验证代数实现，没有替代D12对真实数据假设的检验。

### 5.2 Cross-fitted estimator

真实$m(x)$不可见。CAPE只使用training split：

1. 将train windows划分为K folds；
2. pilot forecaster在K-1 folds训练，在held-out fold产生out-of-fold prediction $\tilde m_i$；
3. 合并所有OOF predictions估计$\widehat\Sigma_m=\operatorname{Cov}(\tilde m)$；
4. 用$\widehat\Sigma_m$而非$\operatorname{Cov}(y)$构造frame；
5. 最终forecaster从头训练，不复用pilot Encoder或decoder weights。

cross-fitting避免用同一样本的in-sample拟合noise伪造predictable energy。pilot首轮使用A6 natural carrier；后续必须
以简单linear/DLinear pilot作robustness control，避免“只复制A6偏好”。

### 5.3 Joint frame problem

PRISM-CAPE的核心优化为：

$$
\max_{U^TW_\mu U=I}
\operatorname{tr}(U^TW_\mu\widehat\Sigma_mW_\mu U)
-\lambda\mathcal L_{prefix}(U).
$$

- $\lambda=0$：CAPE-only predictable global frame；
- raw $\operatorname{Cov}(y)$ + $\lambda>0$：PRISM-only prefix-localized frame；
- predictable covariance + $\lambda>0$：joint PRISM-CAPE；
- raw covariance + $\lambda=0$：PCA-like matched control。

这形成自然`2x2` factorial，而不是把两个名称绑定成无法归因的一次改动。

### 5.4 Why current MIPR should not be retained

1. D11没有支持cross-component gradient conflict；
2. benchmark-horizon measure下MIPR off-block energy只有`0.002480`，历史problem headroom很弱；
3. QDF、Time-o1、Loss Shaping已显著占据quadratic/component/step-weighting叙事；
4. 若完整$T=720$ label可用，raw multi-horizon risk $e^TW_\mu e$可以一次forward精确计算，不需要用importance
   sampling制造一个本可消除的variance问题；
5. MIPR删除cross-scale terms，是biased surrogate，且D11没有证明删除这些terms有必要。

[Decision] `MIPR retired_as_core_candidate`。$W_\mu$继续作为evaluation/training protocol与same-measure raw
control；coverage-aware sampling只保留为将来compute-constrained场景的backup diagnostic，不列为当前创新点。

### 5.5 Feasibility and risk

| Aspect | Assessment |
| --- | --- |
| theorem | high：weighted low-rank approximation可直接推导；需补充center/intercept与estimator consistency |
| computation | medium-high：3-fold OOF增加pilot训练，但当前训练速度可接受，且只发生在train stage |
| leakage control | high：OOF + train-only provenance可审计；validation/test不进入frame estimation |
| novelty | medium-high for full chain；conditional-mean covariance本身是经典对象，claim必须落在prefix-risk forecast-frame learning |
| performance | medium：若pilot过弱或future几乎不可预测，$\widehat\Sigma_m$会不稳定；D12必须先验证 |

## 6. Why The Two Contributions Form One Story

| Paper question | PRISM | CAPE |
| --- | --- | --- |
| one model如何服务任意prefix？ | $H$只crop同一$U_\mu$，exact projectivity | frame estimation完全不读取requested $H$ ID |
| rank-256 capacity放在哪里？ | prefix-risk locality决定“where accuracy matters” | predictable covariance决定“what history can predict” |
| 如何解释D6 crossing？ | $\mathcal L_{prefix}$对抗global compaction | predictable energy避免locality以捕获noise为代价 |
| 如何避免旧SC1失败？ | 保留A6 free coefficient head；无tree transition、separable PAF、MoE | final model从头训练；不freeze/replace co-adapted A6 Encoder |
| 如何避免旧SC2/prior-art重叠？ | architecture geometry来自prefix family | 不改写loss weights；估计decoder subspace而非做component loss |

论文可以沿一条逻辑线展开：

1. fixed-vector视角把multi-horizon误写成多个horizon-specific tasks；
2. function-space视角把它写成一个future function上的nested prefix-risk family；
3. rank-limited decoder因此需要risk-localized frame（PRISM）；
4. frame不应拟合不可预测label noise，因此需要predictable-energy estimation（CAPE）；
5. 二者联合得到同一个H-agnostic、projective、capacity-aware forecaster。

## 7. D12 Step 2-3 Diagnostic Plan

### 7.1 Role and data boundary

- status：`diagnostic_only`；不是method effectiveness gate；
- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- train split：构造OOF predictions与frame；
- validation split：只评估offline/probe generalization；
- test：禁止；
- horizons：equal mass over integer $H\in\{48,\ldots,720\}$为primary dense measure；log-uniform为
  robustness，benchmark horizons为secondary；
- rank：固定$r=256$；dataset natural profiles继续冻结；params差异不参与选择。

### 7.2 D12-A: existence and estimator audit

1. 比较$\operatorname{Cov}(y)$、$\operatorname{Cov}(\tilde m)$与residual covariance的spectrum/subspace angles；
2. 检查OOF folds/seeds间predictable subspace稳定性；
3. 审计train-only provenance、PSD、centering与$W_\mu$ conditioning；
4. 用linear/DLinear pilot作至少一个model-bias control；
5. 若predictable covariance退化、跨fold不稳定或等价于raw PCA，则CAPE不进入Step4。

### 7.3 D12-B: PRISM Pareto audit

比较raw-PCA、predictable-global、local DCT/RGNB、PRISM-raw与PRISM-CAPE的小型预注册$\lambda$ frontier。
$\lambda$使用dimensionless normalization并在所有datasets共享同一小集合；只按train/offline Pareto rule选择，
不按每个dataset的validation MSE精调：

- predictable-energy capture；
- prefix leakage $\mathcal L_{prefix}$；
- short/long oracle projection error；
- matched history-to-coefficient linear probe validation MSE；
- D6-style short/long crossing是否缩小；
- random/permuted frame controls。

frozen A6 memory probe只作conditional diagnostic，所有frame使用同一memory、rank、fit split与optimizer；它不能
直接决定method有效性。

### 7.4 Advancement gate

只有joint PRISM-CAPE在至少3/5 datasets上形成不被raw PCA、DCT/RGNB、random/permutation解释的
`predictable capture–prefix locality` Pareto improvement，并在validation probe上没有系统性long-domain collapse，
才授权Step 4-6 source/theory/narrative design。

失败归因：

- predictable covariance不稳定：`hypothesis_false_for_CAPE`，CAPE关闭；
- locality frontier存在但probe无收益：`representation_geometry_not_predictively_useful`；
- probe收益仅来自rank/capacity：`capacity_control_explains`；
- frozen probe失败但oracle/Pareto通过：`diagnostic_inconclusive_for_direction`，不得方向级否决；
- joint frontier整体失败：rollback Step 2，关闭basis/frame主线，备选回到query-set-invariant neural operator，但其
  prior-art压力更高。

## 8. Later Experiment Stages If D12 Passes

1. **Step 4-6**：补全conditional-mean theorem、cross-fit consistency、full-text prior-art与claim boundary；冻结
   $\mu$、$\lambda$ rule、frame solver及controls。
2. **Step 7A**：实现PRISM/CAPE production path；验证shape、prefix equality、$W_\mu$-orthonormality、train-only
   provenance、gradient、parameter budget及code-theory consistency。
3. **Step 8 seed2021 screen**：五datasets validation-only；A6、raw-global、CAPE-global、PRISM-raw、
   PRISM-CAPE、DCT/RGNB与random controls；全部从可比initialization class训练。
4. **Step 9-10 staged confirmation**：只有one-seed gate通过才补seed2022/2023；报告dense-horizon curves、
   benchmark horizons、short/long segments、stability与frame diagnostics。
5. **Paper factorial**：`localization on/off × predictable/raw covariance`验证两个main effects与interaction；若只有
   joint arm有效，论文应诚实写成co-designed method的两个技术组件，不伪称两个独立普适算法。
6. **Generality**：主方法通过后再接第二backbone；不在problem gate前同步修改Encoder。
7. **Test**：所有architecture、measure、hyperparameters、seeds与claims冻结后才读取。

## 9. Final Research Judgment

[Decision] D11没有把论文逼到“只剩Contribution 2”。它实际上帮助我们移除了一个错误动机，并暴露了旧
SC2同样不够强。当前最值得投入的不是继续修JAPO或实现coverage loss，而是验证一个更基础、也更统一的命题：

> 多horizon统一预测的decoder capacity，应由prefix deployment risk与history-predictable future variation共同
> 决定。

PRISM与CAPE在数学上可实现、工程上与A6 shape兼容、叙事上相互闭合，并且都有明确的低成本falsification
路径。但它们现在只是`proposed_step2_3` candidates，不是已经成立的论文创新。下一步必须先完成D12，不能因
论文需要两个创新点就跳过problem gate。
