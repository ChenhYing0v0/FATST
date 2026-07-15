# StageC Post-D12 系统复盘与论文主线重构

日期：2026-07-15
状态：Step 1-3 evidence synthesis completed；Step 4-6 provisional narrative audit completed；
method implementation / remote training / test 均未授权。

## 0. 决策摘要

[Decision] D12并未否定 unified multi-horizon forecasting 的论文价值；它否定的是“用
conditional-mean covariance重新分配一个固定rank-256 forecast frame”这一条具体问题链。

[Strong Evidence] D3-D8反复支持的跨dataset事实只有两项：

1. future coordinate/support geometry会影响预测生成；
2. short prefix偏好更local的support，而long domain偏好更global的coherence。

[Strong Evidence] D9-D12连续排除了三种看似自然、但证据不足的解释：

1. history scale与future support存在统一可路由对应关系；
2. short/long supervision在future components上存在directional gradient conflict；
3. A6的主要瓶颈是predictable subspace上的rank allocation错误。

[Decision] 因此，下一条主线不再围绕“为720步输出寻找更好的固定basis、scale router或frame”，
而改写为：

> Unified multi-horizon forecasting不只是一次生成一条future row；它应学习同一future target在
> information set逐步扩展时，预测如何被合理地修订。模型的基本对象应是causal forecast-revision
> surface，而不是彼此独立的horizon-specific forecasts。

暂定论文标题：

> Learning Forecasts That Evolve with Information: Causal Revision Surfaces for Unified
> Multi-Horizon Forecasting

两项provisional contribution：

1. Contribution 1：Nested-Information Forecast Revision Operator，简称 NIFRO；
2. Contribution 2：Innovation-Accounted Revision Learning，简称 IARL。

两项都只处于 Step 2-3 proposed 状态。它们必须先通过 D13 rolling-origin problem diagnostic，
才允许进入正式 Step 4-6 source-informed theory/design gate。

## 1. 内部实验的系统复盘

### 1.1 已确认的baseline边界

A6-LBF-natural-baseline的forward contract为：

$$
X\in\mathbb R^{B\times720\times C}
\rightarrow
M\in\mathbb R^{B\times C\times P\times D}
\rightarrow
h=\operatorname{flatten}(M)\in\mathbb R^{B\times C\times PD}
$$

$$
a=W_a h\in\mathbb R^{B\times C\times256},
\qquad
\hat Y_H=B_{1:H}a+b_{1:H}\in\mathbb R^{B\times C\times H}.
$$

其中dataset-specific natural profiles允许
$P\in\{12,24,48\}$、$D\in\{32,64\}$；requested horizon $H$只裁剪输出前缀。

[Fact] flatten本身是bijective reshape，不会丢失元素。

[Strong Evidence] 但A6在readout入口把所有patch一次性压到256个coefficients，之后所有future
coordinates只读取这同一组coefficients。因此真正的问题不是“flatten丢了数据”，而是：

1. patch-level信息没有direct target-specific access path；
2. 所有future targets共享一次global compaction；
3. 模型只学习单个origin的一条forecast row，没有表示forecast随新信息到来如何演化。

### 1.2 D3-D8留下的正证据

| Evidence | Result | 允许的结论 | 不允许的结论 |
| --- | --- | --- | --- |
| D3 basis main effect | macro MSE +2.9174%，5/5 interaction guard | output support geometry真实影响probe | balanced basis本身已是method |
| D4 structured basis | locality +1.6324%；DCT/PCA优于balanced | locality是有效解释 | exact balanced midpoint有独特性 |
| D6 disjoint confirmation | short +1.1964%，long -1.2675%，12/15 crossing | local/global support需求随forecast distance变化 | history scale可据此被路由 |
| D7 descriptor probe | true geometry较perm/random约+12.84%至+13.80% | canonical geometry在conditional probe中有信息 | frozen free gap可判E2E方法 |
| D8 E2E | geometry较matched +14.33%；较A6 -28.10% | geometry main effect仍在 | shared-latent PAF可替代A6 free head |

这组结果的共同含义是：

> future support值得建模，但不能通过一个rigid separable basis/query bottleneck牺牲A6的自由函数类。

### 1.3 D9-D12关闭的解释

| Hypothesis | Result | Failure attribution |
| --- | --- | --- |
| history-scale × future-support routing | D9 macro rho 0.173810；D10 best mapping 0/5 | hypothesis_false |
| future-component directional conflict | D11 strict conflict 0/5；support-specific 2/5 | hypothesis_false |
| predictable covariance frame allocation | D12 risk-aligned v2 support 1/5 | hypothesis_false for cross-dataset mainline |

这些negative results非常重要，因为它们阻止我们继续把“scale”“conflict”“predictable rank”
作为未经支持的中间变量。

### 1.4 前一阶段最关键的架构教训

1. 固定tree、shared separable PAF、weak expert mixing都可能保留漂亮的数学性质，却缩小实际
   forecast function class；
2. frozen encoder replacement只能说明与该representation的compatibility，不能判定paper-core方法；
3. end-to-end结果表明A6 free operator是必须尊重的capacity boundary；
4. patch-level memory access仍未被公平测试，但“直接用generic atom query cross-attend全部patch”
   也没有内部problem evidence，不能直接升级为method；
5. 新方法必须先从可观测问题出发，再设计信息路径，不能先发明decoder再寻找解释。

## 2. 外部调研与novelty边界

检索日期：2026-07-15。
检索来源：外部primary/official sources为主；Zotero本轮未用于discovery，是否已收录未核验。
因此“未发现完全相同工作”只能标为中等置信度，不能写成绝对first claim。

### 2.1 直接相关工作

| Work | Source | 已覆盖内容 | 对本项目的约束 |
| --- | --- | --- | --- |
| A Multi-Horizon Quantile Recurrent Forecaster | arXiv 1711.11053 | 2017年已提出forking-sequences，在多个forecast creation dates放置共享decoder | 多origin训练或forecast grid不是新贡献 |
| Forking-Sequences | arXiv 2510.04487 / OpenReview | 系统化整个FCD × horizon grid、gradient variance、ensembling与stability | 不能声称首次jointly forecast all origins |
| Improving forecast stability using deep learning | IJF 2023 | N-BEATS-S把instability加入loss | 简单revision penalty不是新贡献 |
| On forecast stability | IJF 2025 | 区分vertical/horizontal stability，并研究post-processing/Pareto | stability分类与平滑已有成熟边界 |
| Beyond Accuracy | arXiv 2601.10863 | accuracy-coherence metric与differentiable objective | accuracy + stability加权不是新贡献 |
| Forecast Rationality Tests Based on Multi-Horizon Bounds | JBES 2012 | conditional mean下的revision、MSE和covariance bounds | martingale/rational-revision数学本身不是新理论 |
| TimeFlow | TMLR 2024 | conditional INR连续时间imputation/forecasting | continuous target query不是新贡献 |
| Shifting Time / KRNO | ICML 2025 | continuous time-shift neural operator | generic neural operator叙事已拥挤 |

Primary URLs：

- https://arxiv.org/abs/1711.11053
- https://arxiv.org/abs/2510.04487
- https://openreview.net/forum?id=dXdycy7WCX
- https://www.sciencedirect.com/science/article/abs/pii/S016920702200098X
- https://www.sciencedirect.com/science/article/pii/S0169207025000068
- https://arxiv.org/abs/2601.10863
- https://www.tandfonline.com/doi/full/10.1080/07350015.2012.634337
- https://openreview.net/forum?id=P1vzXDklar
- https://openreview.net/forum?id=emkdmORaj4

### 2.2 本轮排除的替代主线

#### A. Continuous future field / INR

[Decision] 不作为主线。

理由：TimeFlow、KRNO及其他neural operator工作已经直接覆盖continuous coordinate forecasting；
内部D8又显示descriptor-generated shared query head存在明显function-class风险。除非以后发现
irregular-grid或super-resolution是核心benchmark问题，否则该路线的novelty/performance比不优。

#### B. Generic forecast stability regularization

[Decision] 不作为主线。

理由：2023-2026已有直接loss、metric、dynamic weighting和post-processing工作。简单惩罚
$\|\hat y_{o+1,\tau}-\hat y_{o,\tau}\|^2$不仅prior-art拥挤，还会同时压制有用与无用revision。

#### C. Forking-sequences / forecast grid

[Decision] 只作为mandatory baseline与training primitive，不作为贡献。

理由：MQ-RNN及Forking-Sequences已经明确产生FCD × horizon grid。新架构必须证明它不是“把A6
改成forking训练”，而是显式建模nested information projection和patch-conditioned revision。

#### D. Latent semigroup / future-state transition

[Decision] 暂不推进。

理由：内部B13/PMFO transition证据弱，外部latent-state与continuous operator方向拥挤；再次引入
latent future evolution容易重复“理论漂亮、实际head受限”的失败。

## 3. 新论文主线叙事

### 3.1 核心问题

传统multi-horizon model在origin $o$上学习：

$$
\hat{\mathbf y}_o=
\left[
\hat y_{o,o+1},\ldots,\hat y_{o,o+T}
\right].
$$

这只是一条row。部署时，随着新观测到来，模型会在$o+1,o+2,\ldots$重新预测同一target $\tau$，
形成：

$$
F(o,\tau)=\mathbb E[Y_\tau\mid\mathcal F_o],
\qquad o<\tau.
$$

所有$(o,\tau)$组成一个上三角forecast surface：

- 固定$o$的一行是usual multi-horizon forecast；
- 固定$\tau$的一列是同一target的forecast revision path；
- requested $H$只是latest-origin row的裁剪，不是模型condition。

[Hypothesis] 当前window-sampled unified models虽然可在任意$H\le720$输出prefix，却没有约束不同
information sets下的forecast revision是否合理。这可能导致两类浪费：

1. 新信息到来后，预测发生大幅变化，却没有得到相应accuracy gain；
2. patch-level新信息本可支持target-specific correction，却在一次global compaction中被弱化。

### 3.2 论文thesis

> 真正的unified multi-horizon forecasting应同时统一两个轴：future target axis与information-update
> axis。模型不仅要一次预测多个未来位置，还要使这些预测在nested information sets下以可解释、
> 有效的innovation方式演化。

### 3.3 两项contribution如何闭环

1. NIFRO定义“预测如何随information set变化”的architecture；
2. IARL定义“什么样的变化是有价值的”的training principle；
3. NIFRO产生的origin-target surface为IARL提供revision pairs；
4. IARL反过来约束NIFRO的revision path，避免它退化为多个独立forecast heads；
5. inference只读取latest-origin row，仍保持one model for arbitrary prefixes。

因此两项贡献不是decoder + unrelated loss，而是同一个数学对象的representation与learning rule。

## 4. Contribution 1：NIFRO

全名：Nested-Information Forecast Revision Operator。
角色：decoder/operator为主，causal patch encoder为辅。
当前状态：proposed_step2_3，D13前不进入实现。

### 4.1 设计目标

NIFRO不把requested horizon作为输入，也不建立horizon-specific experts。它沿information arrival轴
组织计算，并让target coordinate只表示“要预测哪个未来位置”。

目标tensor path：

$$
X\in\mathbb R^{B\times C\times720}
\rightarrow
M\in\mathbb R^{B\times C\times P\times D}
\rightarrow
\Delta\in\mathbb R^{B\times C\times P\times T}
\rightarrow
F\in\mathbb R^{B\times C\times P\times T}.
$$

其中：

- $M_p$是只使用第$p$个patch及其之前信息的causal prefix memory；
- $\Delta_{p,\tau}$是新patch到来后对target $\tau$的forecast innovation；
- $F_{p,\tau}$是处理到第$p$个patch后的forecast；
- 最终预测是$F_{P,1:H}$。

### 4.2 核心operator

先构造causal patch memories：

$$
M_p=E(X_{\le p}),
$$

再由共享revision operator产生：

$$
\Delta_{p,\tau}
=
G_\theta\left(M_{p-1},\,Z_p,\,q_\tau\right),
\qquad
F_{p,\tau}=F_{p-1,\tau}+\Delta_{p,\tau}.
$$

$Z_p$是新到达patch的local token，$q_\tau$是target coordinate token。所有$p,\tau$共享$G_\theta$；
不存在requested-$H$ embedding。

工程上可把$P\times T$ revision tensor并行计算，再沿origin axis做prefix scan；无需Python循环。

### 4.3 与A6的function-class关系

A6的线性readout可写成：

$$
\hat Y=b+W\operatorname{vec}(M)
=b+\sum_{p=1}^{P}W_pM_p.
$$

因此，若NIFRO的linear control令$\Delta_p=W_pM_p$，其latest-origin row可exactly reproduce
A6 linear readout。这个decomposition不是额外dense bypass，而是把同一个map按patch information
increments重新参数化。

正式method只能在此matched linear control上增加有明确必要性的context-dependent revision，例如：

$$
\Delta_{p,\tau}
=
W_{p,\tau}Z_p
+
g_\theta(M_{p-1},Z_p,q_\tau)\odot V_\theta(Z_p,q_\tau).
$$

是否需要第二项，必须由D13-B证明new-patch information对ideal correction具有跨dataset
predictability。若该证据不存在，只保留linear decomposition不能构成paper-core novelty。

### 4.4 Exact invariants

Step 5必须证明并由Step 7A验证：

1. Causality：$F_{p,\tau}$不读取$p$之后的patch；
2. origin-prefix equality：独立运行前$p$个patch与full surface第$p$行一致；
3. target-prefix equality：$H_1<H_2$时，$F_{P,1:H_1}$完全一致；
4. no requested-H conditioning：$H$只进入slice；
5. A6 linear containment：matched linear arm在同一memory上数值等价；
6. direct patch-to-target path：每个有效patch对至少一组future targets具有非零gradient；
7. causal leakage guard：validation/test future cells不进入encoder或training mask。

### 4.5 为什么它可能解决当前瓶颈

1. 它不再把history一次性压缩成一组共享coefficients后才预测所有targets；
2. 它保留每个patch对不同targets的直接贡献；
3. local/global support差异不需要手工scale mapping，可由不同information increments对不同targets的
   贡献自然形成；
4. 它把multi-horizon叙事从benchmark horizons提升为nested-information forecast surface；
5. 它避开D9/D10已否定的history-scale router，也不依赖D11已否定的gradient conflict。

### 4.6 Novelty边界

不能claim：

- 首次产生multi-origin forecast grid；
- 首次使用forking-sequences；
- 首次用target query或causal attention；
- martingale forecast理论本身新。

允许探索的完整贡献链是：

> unified varied-horizon deployment
> -> nested information sets
> -> exact causal patch-to-target revision operator
> -> dual origin/target projectivity
> -> latest-row arbitrary-prefix inference
> -> innovation-accounted learning.

Novelty风险：中等。Forking-Sequences是最强近邻；正式Step 4必须逐项对比其architecture、mask、
decoder sharing与loss，证明NIFRO不是其重新命名。

### 4.7 可行性评估

| Dimension | Judgment | Basis |
| --- | --- | --- |
| Engineering | medium-high | masked attention、prefix scan、target chunking均为标准可实现操作 |
| 3090 memory | medium | $P\le48,T=720$；需target chunk和2-4 sampled origins |
| Optimization | medium | linear containment降低起点风险，但nonlinear revision可能collapse |
| Performance | medium | direct patch access有合理路径，但尚无D13 problem evidence |
| Narrative | medium-high if D13 passes | architecture与training自然共用forecast surface |
| Novelty | medium | 必须正面超过Forking-Sequences边界 |

## 5. Contribution 2：IARL

全名：Innovation-Accounted Revision Learning。
角色：training loss and strategy。
当前状态：proposed_step2_3，D13前不进入实现。

### 5.1 理论出发点

对同一target $\tau$，令旧、新information sets满足
$\mathcal F_o\subset\mathcal F_{o+1}$：

$$
m_o=\mathbb E[Y_\tau\mid\mathcal F_o],
\qquad
m_{o+1}=\mathbb E[Y_\tau\mid\mathcal F_{o+1}].
$$

定义revision与new error：

$$
\Delta=m_{o+1}-m_o,
\qquad
e_{new}=Y_\tau-m_{o+1}.
$$

conditional projection给出：

$$
\mathbb E[e_{new}\Delta]=0.
$$

又因为$e_{old}=e_{new}+\Delta$：

$$
\mathbb E[e_{old}^2-e_{new}^2]
=
\mathbb E[\Delta^2]
+2\mathbb E[e_{new}\Delta].
$$

理想条件下：

$$
\underbrace{\mathbb E[e_{old}^2-e_{new}^2]}_{\text{accuracy gain }G}
=
\underbrace{\mathbb E[\Delta^2]}_{\text{revision energy }R}.
$$

直观解释：一次revision移动了多远，就应换回相应的squared-error改善；否则其中有一部分变化没有被
新信息“挣回来”。

### 5.2 为什么不是普通stability loss

普通stability loss最小化$R=\mathbb E[\Delta^2]$，会把两类revision一起压小：

1. 对accuracy有帮助的useful revision；
2. 只制造波动的harmful/excess revision。

IARL不追求“预测尽量不变”，而追求：

> revision可以大，但必须在统计上由新information带来的accuracy improvement解释。

### 5.3 Provisional objective

基础surface accuracy：

$$
\mathcal L_{point}
=
\sum_{(o,\tau)\in\Omega}
w_{o,\tau}\ell\left(Y_\tau,F(o,\tau)\right).
$$

revision moment：

$$
C_g
=
\operatorname{mean}_{(o,\tau)\in g}
\left[
\left(Y_\tau-F(o+1,\tau)\right)
\left(F(o+1,\tau)-F(o,\tau)\right)
\right],
$$

其中$g$为batch × channel × target-distance bin，避免单sample高噪声。

normalized orthogonality loss：

$$
\mathcal L_{orth}
=
\sum_g
\frac{C_g^2}
{\left(E_g[e_{new}^2]+\epsilon\right)
 \left(E_g[\Delta^2]+\epsilon\right)}.
$$

等价的revision-efficiency diagnostic为：

$$
\eta_g=\frac{G_g}{R_g+\epsilon}
=1+\frac{2C_g}{R_g+\epsilon}.
$$

初版method contract只应使用一个moment penalty，避免把代数等价的$G-R$与$C$重复计权。

### 5.4 防止trivial solution

IARL存在$\Delta=0$的trivial moment solution，必须通过以下约束阻断：

1. 每个valid origin-target cell都有point loss；
2. no-revision arm必须作为mandatory control；
3. newer-origin accuracy必须单独报告，不能只报告moment；
4. loss weight不做per-dataset精调，使用一个global small grid或dual update；
5. 对$C_g$的gradient path要在Step 5审计，防止通过增大$e_{new}$抵消moment；
6. 必须比较full-gradient、stop-gradient单侧variant与no-moment control，但只允许一个预注册primary。

### 5.5 Novelty边界

不能claim：

- forecast revision / forecast rationality理论新；
- 首次优化forecast stability；
- 首次使用multi-origin loss。

允许探索的claim是：

> 把nested conditional-projection的revision moment，从econometric after-the-fact test转化为
> jointly generated neural forecast surface上的differentiable training principle；其目标不是
> suppress revision，而是account for revision by realized accuracy gain。

Novelty风险：中等。现有rationality tests与stability losses分别很接近，但本轮未发现它们在deep
multi-horizon model中以该exact coupling联合实现。正式novelty claim需要更广的formula-level检索。

### 5.6 可行性评估

| Dimension | Judgment | Basis |
| --- | --- | --- |
| Data | high | rolling windows天然提供same-target multi-origin pairs |
| Compute | high | 只增加surface cells上的统计量 |
| Numeric stability | medium | 小$R$、batch cancellation与denominator需处理 |
| Optimization | medium-low to medium | trivial no-revision与gradient gaming是主要风险 |
| Performance | unknown | 只有D13证明A6存在excess revision后才有headroom |
| Narrative | high if paired with NIFRO | loss直接约束architecture产生的revision surface |

## 6. D13：下一步必须先做的problem diagnostic

名称：D13 Rolling-Origin Revision Efficiency Audit。
角色：diagnostic_only。
允许数据：train用于拟合controls，validation用于最终gate，test=false。
模型：现有A6 natural checkpoints，5 datasets × 3 seeds；不训练新forecast method。

### 6.1 要回答的问题

1. A6在新信息到来后是否确实提高同一target的accuracy？
2. revision energy是否与accuracy gain匹配？
3. 简单train-fit revision calibration能否在validation稳定改善A6？
4. 新到达patch是否包含可预测的ideal correction信息？
5. 该现象是否跨ETTh1、ETTh2、ETTm1、ETTm2、Weather，而非单dataset偶然性？

### 6.2 数据构造

共同origin gaps固定为：

$$
\delta\in\{15,30,60\},
$$

分别对应720-step context的$1/48,1/24,1/12$，同时覆盖当前natural profile的三种patch length。

对new-origin horizon：

$$
h\in\{48,96,144,192,288,336,512\},
$$

只保留$h+\delta\le720$的same-target pairs：

$$
\hat y_{old}=\hat y_{o,\tau},
\quad
\hat y_{new}=\hat y_{o+\delta,\tau},
\quad
\tau=o+\delta+h.
$$

每个origin都用其自身720-step history独立运行A6，避免future leakage。D13不把frozen replacement
用于method判定；它只审计现有baseline的problem headroom。

### 6.2.1 Fixed-window非严格nested的理论边界

[Fact] A6每个origin只读取最近720 points。origin从$o$移动到$o+\delta$时，input既加入最新
$\delta$ points，也移除最旧$\delta$ points。因此A6的effective model inputs不是严格nested
sigma-algebras；conditional-mean martingale identity是full-information ideal reference，不是A6必须满足的
theorem。

D13必须增加window-expiry attribution：

1. 分别记录added block与expired block；
2. 用added-only、expired-only及二者联合的train-only probe解释revision；
3. 在可行时增加long-context/no-expiry control，或至少增加shared-middle-context control；
4. 若revision inefficiency主要由expired block解释，而非new information，则不能支持NIFRO/IARL的
   nested-information叙事；
5. NIFRO名称与exact nested-state contract必须在formal Step 4重新审计。若只能得到rolling-window
   operator，应改名并收紧理论claim。

### 6.3 Primary statistics

对每个dataset/seed/gap/horizon：

$$
\Delta=\hat y_{new}-\hat y_{old},
\quad
R=E[\Delta^2],
\quad
G=E[(y-\hat y_{old})^2-(y-\hat y_{new})^2],
\quad
C=E[(y-\hat y_{new})\Delta].
$$

并报告：

1. revision efficiency $\eta=G/(R+\epsilon)$；
2. harmful revision fraction：revision后point error增大的比例；
3. optimal scalar revision coefficient
   $\alpha^*=E[(y-\hat y_{old})\Delta]/(R+\epsilon)$；
4. train-fit $\alpha^*$在validation上的MSE变化；
5. vertical volatility；
6. $G-R-2C$ numerical identity误差。

### 6.4 Mandatory controls

1. no-revision：$\hat y_{new}=\hat y_{old}$；
2. raw A6 revision：$\alpha=1$；
3. train-fit scalar calibration：只在train估计$\alpha$，validation固定；
4. origin-shuffled pairing：破坏same-target information nesting；
5. target-shift pairing：保留marginals但破坏target alignment；
6. window-expiry attribution：added / expired / shared-middle features；
7. long-context/no-expiry control若数据与checkpoint接口允许；否则明确标记unavailable并降低结论等级；
8. per-gap and pooled：防止某一个$\delta$支配；
9. L1 replication：只作robustness，不改变MSE theory primary。

### 6.5 D13-A problem gate

一个dataset支持revision-efficiency problem，需同时满足：

1. raw A6的$G>0$，说明更新整体有用；
2. 至少2/3 seeds的$\eta$同方向偏离1，且$|\eta-1|\ge0.10$；
3. train-fit scalar calibration在validation改善MSE至少0.3%，且至少2/3 seeds为正；
4. no-revision的new-origin MSE差于raw A6，排除“不更新即可”；
5. aligned effect显著强于origin-shuffled与target-shift controls；
6. signal不能主要由expired-window block解释。

总体pass：至少3/5 datasets支持，macro validation MSE calibration为正，且不存在>5%的dataset级
严重退化。

### 6.6 D13-B patch-information gate

只有D13-A通过才执行。

用train-only ridge/linear probe比较：

1. old-state-only features；
2. new-patch-only features；
3. old-state + new-patch features；
4. time-shifted new-patch control。

target为ideal correction $y-\hat y_{old}$或A6 revision error。若old + new在validation相对old-only
稳定改善，且超过shifted control，才支持NIFRO的direct patch-to-target intervention。

总体pass：至少3/5 datasets，2/3 seeds方向一致，并超过预注册small effect gate。

### 6.7 Failure attribution

- D13-A失败：revision inefficiency problem不跨dataset；NIFRO/IARL主线关闭，rollback Step 2；
- D13-A通过、D13-B失败：training objective可继续评估，但patch-direct architecture缺乏依据；
- D13-A/B通过：只授权NIFRO/IARL正式Step 4-6，不直接授权implementation；
- numeric pathology：标记diagnostic_invalid_for_direction_rejection，修复protocol后重跑；
- scalar calibration解释全部收益：只支持training problem，不支持复杂architecture。

## 7. 后续11-step研究计划

### Phase C-R0：D13 problem verification

- current step：Step 2-3；
- gate：D13-A与conditional D13-B；
- rollback：Step 2；
- method/test：false。

### Phase C-R1：source-informed Step 4

仅在D13通过后：

1. 逐公式对照Forking-Sequences、MQ-RNN/MQTransformer、N-BEATS-S、forecast rationality tests；
2. 检查official code中的causal mask、decoder parameter sharing、FCD construction；
3. 审计NIFRO是否真的需要new-patch direct path；
4. 审计IARL moment是否与现有stability loss或optimal revision regression等价。

Gate：完整problem -> constraint -> mechanism -> implementation -> claim链仍有非平凡边界。

### Phase C-R2：Step 5 theory feasibility

1. NIFRO A6 linear containment证明；
2. origin/target dual projectivity；
3. leakage-free triangular mask；
4. IARL identity、gradient与trivial-solution analysis；
5. synthetic DGP：有用revision、excess revision、no-new-information三种case。

Gate：所有exact identities通过；不存在由control完全解释的method class。

### Phase C-R3：Step 6 design freeze

冻结：

1. method tensor contract；
2. A6、forking-A6、linear NIFRO、nonlinear NIFRO controls；
3. point-only、generic stability、IARL objectives；
4. parameter/FLOP matching与shared hyperparameter policy；
5. five-dataset seed2021 screen和three-seed confirmation规则。

### Phase C-R4：Step 7A local implementation

只做shape、causality、prefix、gradient、memory、dry-run gates。

### Phase C-R5：Step 8-10 remote effectiveness

1. seed2021五dataset broad screen；
2. 通过后补seeds2022/2023；
3. primary仍为dense-horizon MSE/MAE；
4. secondary为revision efficiency、vertical stability、harmful revision fraction；
5. 2×2 factorial：
   - A6/forking control；
   - NIFRO + point loss；
   - matched operator + IARL；
   - NIFRO + IARL。

只有architecture main effect、training main effect和joint interaction均可归因时，才形成两项
paper-core contributions。

## 8. 反方审查

### 8.1 最强反对意见一：这只是Forking-Sequences换名

[Valid Concern] 是当前最大novelty风险。多FCD grid本身完全不能claim。

应对：D13后必须证明贡献落在nested-information factorization、exact dual projectivity、
patch-conditioned revision path与innovation accounting的完整组合；如果只剩“多origin一起训练”，
立即关闭NIFRO。

### 8.2 最强反对意见二：revision stability不一定改善benchmark MSE

[Valid Concern] benchmark只看latest-origin point accuracy，surface regularization可能只是deployment
quality改进。

应对：paper-core effectiveness gate仍以MSE/MAE为primary。若只改善stability而不保持/改善accuracy，
该路线最多成为应用型secondary contribution，不满足当前论文目标。

### 8.3 最强反对意见三：IARL可被trivial no-revision解释

[Valid Concern] moment condition本身允许$\Delta=0$。

应对：point loss across origins、no-revision control、new-origin accuracy和D13的positive $G$是必要条件。
若scalar shrinkage已解释全部收益，则不实现复杂IARL。

### 8.4 最强反对意见四：causal prefix encoder会破坏A6特调carrier

[Valid Concern] 这是真实风险，不能把失败自动归因于hyperparameter。

应对：

1. 先用独立A6 rolling inference做D13，不动carrier；
2. Step 6预注册forking-A6和linear containment control；
3. paper-core比较必须matched end-to-end joint training；
4. 若causal encoder control本身显著退化，先回Step 4修正intervention point，不堆decoder/loss。

## 9. 最终判断

[Strong Evidence] 旧Contribution 1/2不是因为“创新点不够”而失败，而是其假设中的中间变量没有获得
跨dataset支持。继续改basis、rank、scale router或component loss，大概率只会重复局部正信号与整体
E2E失败。

[Hypothesis] forecast-revision surface是目前最有希望的新problem formulation，因为它：

1. 直接面向multi-horizon unified forecasting，而非把generic operator移植到forecasting；
2. 同时容纳decoder、training strategy和辅助encoder改造；
3. 对用户担心的patch-level信息压缩给出direct、可诊断的计算路径；
4. 不依赖D9-D12已否定的scale/conflict/frame assumptions；
5. 两项贡献共享同一理论对象，论文叙事天然完整。

[Uncertainty] 当前尚没有内部证据证明A6存在跨dataset revision inefficiency，也没有证明new patch
能够预测ideal correction。因此NIFRO/IARL是“高叙事潜力、中等可行性、需先诊断”的候选，而不是
已成立创新点。

[Decision] 下一步只执行D13-A；通过后才执行D13-B。二者通过后，回到Step 4-6做正式source-informed
redesign，不直接进入method coding或remote training。
