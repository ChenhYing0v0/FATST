# StageC Contribution 1 Post-JAPO Systematic Review

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | Contribution 1 rollback to Step 4 source-informed redesign audit |
| `problem` | short-prefix locality与long-domain coherence存在稳定crossed interaction，但既有structured decoders要么限制A6 operator freedom，要么引入不可辨识的expert mixture |
| `existence_evidence` | D6 support×horizon crossing、D7/D8 canonical geometry effect、D1 ordered memory sufficiency |
| `idea` | support-identifiable local-global projective operator；不是正式candidate，只是Step4搜索方向 |
| `theory_check` | RGNB/projectivity/conservation可保留；operator identifiability、history-scale alignment与optimization accessibility待验证 |
| `design` | 先做A6 history-support Jacobian/scale audit；未通过前不实现新decoder |
| `narrative_gate` | pending；generic basis/wavelet/local-global/FiLM均不得单独作claim |
| `effectiveness_gate` | pending；JAPO exact v1已失败，seed2023停止 |
| `artifacts` | D1、PMFO、D2-D8、JAPO theory/design/two-seed reports |
| `decision` | 保留problem与geometry scaffold，关闭expert-mixture abstraction；下一步执行history-support operator evidence audit |

## 1. Paper-Level Thesis

StageC的研究问题不是让模型识别几个benchmark horizon IDs，而是：

> 一个共享模型如何表示同一个可限制、可细化的future function，使任意requested prefix都来自同一operator，
> 并用deployment horizon measure一致地训练该函数族？

requested $H$必须只定义output/loss domain，不进入learned semantic path。architecture contribution负责
future function的projective representation与history-conditioned generation；training contribution负责不同prefix
domains在deployment measure下的风险与gradient responsibility。

## 2. Evidence Path Review

### 2.1 A6 establishes the accepted freedom floor

A6执行

$$
M[B,C,P,D]\rightarrow h[B,C,PD]\rightarrow z[B,C,256]
\rightarrow B_{[:H]}[H,256]\rightarrow \widehat y_H[B,C,H].
$$

它已经满足domain-only horizon与prefix consistency。其优势来自直接、自由且易优化的history-to-temporal
operator；不足是single global learned table没有nested/local support、resolution语义或operator-aligned risk。

D1-v2说明ordered patch memory被frozen A6 head实际使用，shuffle/collapse均显著增加SSE；因此首轮没有证据要求
先替换Encoder。但ETTh2 coarse/mid linear probe为负，说明“信息存在”不等于“简单multiscale readout可访问”。

### 2.2 PMFO-RCT: projectivity helped, rigid replacement failed

PMFO-RCT相对A6 macro为`-1.0955%`。conservation相对no-conservation为`+2.3393%`，而transition相对
no-transition仅`+0.0486%`。因此projective/conservative synthesis是可保留原则，fixed recursive tree transition
不是被支持的机制。失败发生在以刚性hierarchical readout整体替换A6自由operator。

### 2.3 D2-D6: locality is real, but conditioning dominates

D4在matched head下得到：balanced interval相对permuted interval为`+1.6324%`，证明contiguous locality有效；
但相对DCT/PCA为`-0.8609%/-1.5050%`，random interval tree与balanced近似等价。accuracy排序主要跟随
decorrelation与energy compaction，而不是active atom count。

D5/D6在disjoint validation windows复现固定b144 local DCT的crossing：short horizons相对global DCT
`+1.1964%`，long horizons`-1.2675%`，12/15 units同时short-positive/long-negative，crossing约在H144-192。

[Strong Evidence] multi-horizon unified forecasting的真实problem是conditioning-locality tradeoff：short prefix需要
local support，long domain需要global coherence；不能用一个固定basis或horizon-specific head解决。

### 2.4 PLGO/PAF: geometry works, separable operator does not

RGNB构造通过stable reconstruction、A6 morphism与exact prefix restriction。D7 frozen-memory中canonical geometry
相对PERM/RANDOM提升`12.84%-13.80%`、5/5 datasets；D8 fair end-to-end中geometry相对matched controls仍为
`+14.33%`、5/5，且short horizon gain最强。

但D8 GEO相对A6为`-28.10%`，m694 width只回收`+0.58%`。因此geometry是真实inductive bias，
$\alpha_j=\psi(d_j)^TAh$却是过强的shared/separable history-to-atom restriction。flatten本身是bijective reshape，
不是information compression；真正问题是scale structure没有被显式暴露给operator。

### 2.5 JAPO: stronger function class without identifiable responsibility

JAPO theory通过A6 containment、exact projectivity、continuity与strict non-affine witness。它证明了joint
history-atom gate可以表达fixed PAF之外的函数，但这只是existence theorem。

two-seed 70/70 gate得到：JOINT相对A6 `-1.2435%`、0/5，relative same-bank median `-0.1175%`、1/5；
UNIFORM/HISTORY/ATOM均在macro上优于JOINT。两个seed的normalized router entropy都接近1。

[Strong Evidence] independent initialization避开了严格symmetry trap，却没有解决mixture non-identifiability：
两个full-capacity experts可以吸收相同forecast map，router没有必须形成history-support interaction的optimization
pressure。JAPO exact v1关闭，不能通过temperature、auxiliary specialization loss或seed2023继续修补。

## 3. What Is Retained And What Is Closed

### Retained

1. `problem evidence`：support scale × horizon crossing跨window成立；
2. `future geometry`：canonical RGNB descriptors相对PERM/RANDOM稳定有效；
3. `projective scaffold`：RGNB restriction、conservation与domain-only $H$；
4. `current Encoder as first carrier`：ordered patch memory包含可用信息；
5. `continuous dense-horizon task`：比四个benchmark horizons更能支撑measure-aligned training叙事。

### Closed

1. fixed tree transition；
2. fixed basis change、full-affine grouping或random-equivalent factorization；
3. descriptor-generated shared/separable PAF；
4. geometry-only expert mixture；
5. two exchangeable full-capacity experts + weak softmax routing；
6. explicit horizon embedding、atom-to-patch retrieval、post-failure auxiliary-loss repair。

## 4. Most Plausible Mechanism Class

下一候选不应是MoE，而应是一个**support-identifiable projective operator**。该短语只是Step4方向标签，尚未
通过narrative/theory gate。

### 4.1 Future-side direct-sum contract

future function使用RGNB或等价stable direct sum：global root负责long-domain smooth/coherent structure，
local detail subspaces负责prefix-local variations。各subspace正交或具有明确non-overlap responsibility，避免
exchangeable experts相互吸收。

### 4.2 History-side multiresolution accessibility

保留`memory [B,C,P,D]`，但从patch axis构造global/coarse/mid/local history states。这里不是future atom去检索
某个history patch；alignment只发生在scale/resolution层，future location继续由synthesis geometry承担。

### 4.3 Non-exchangeable operator modulation

保留一个A6-equivalent shared coefficient state，再由history-scale state与support geometry对其进行连续、
non-exchangeable feature transport。constant/identity transport必须在同一参数化中容易到达A6-equivalent operator；
dynamic transport提供history-dependent nonlinear map。不得为每个scale配置独立full affine map，也不得依赖
expert softmax来分配职责。

### 4.4 Training-side horizon measure

当architecture contract稳定后，Contribution 2应优先研究exact prefix-measure risk：$H$只裁剪loss domain，
不作为模型输入。若operator direct-sum成立，再评估measure-induced block risk是否能让local/global components按
deployment probability获得匹配gradients。MIPR在benchmark measure下证据弱，因此主场景必须是continuous
dense-horizon deployment，并保留same-measure raw weighting control。

## 5. External Source Boundary

- search date：2026-07-15；
- source policy：external primary sources优先，Zotero仅作seed；
- topics：functional basis forecasting decoder、implicit decoding、multiresolution/local-global neural operator、
  restriction/prolongation与dynamic horizons；
- verified primary sources：arXiv/OpenReview pages；部分OpenReview full text受challenge阻断，只使用其官方
  metadata/abstract，降低claim confidence。

Relevant prior art：

1. [FlowState](https://openreview.net/forum?id=R50AT6nAsM)已经使用functional basis decoder支持dynamic horizons；
2. [Implicit Forecaster](https://openreview.net/forum?id=gqoeQPhQcE)已经从frequency/amplitude/phase constituent
   waves设计forecast decoder；
3. [M2NO](https://arxiv.org/abs/2406.04822)以multiwavelet restriction/prolongation构造multiresolution operator；
4. [LGNO](https://arxiv.org/abs/2606.18221)组合global operator与local multiresolution branch；
5. [TimeStacker](https://openreview.net/forum?id=5RYSqSKz9b)已在time-series representation中强调global/local
   multilevel observation。

[Decision] generic basis decoder、wavelet、local-global branch、FiLM或dynamic horizon都不能单独构成创新。
可辩护边界必须位于完整链条：

> unified prefix-domain problem → projective direct-sum constraint → scale-identifiable history coupling →
> non-exchangeable operator modulation → horizon-measure risk。

## 6. Self-Critique And Falsification

1. history-scale alignment仍是hypothesis；flatten保留信息不证明scale-aware interface有必要；
2. direct-sum responsibility可能只是人为结构，若A6 Jacobian不呈现scale pattern，应停止该方向；
3. identity/A6 containment可能再次只是存在性证明；必须审计optimization accessibility；
4. dynamic feature transport可能退化为identity或generic nonlinearity，必须有support-shuffle与history-scale-collapse
   controls；
5. MIPR仅在明确deployment measure、L2 algebra和raw measure controls下具有理论价值；不得先于SC1实现。

## 7. Immediate Step 4 Plan

### SC1-D9 History-Support Operator Evidence Audit (`diagnostic_only`)

目的：判断A6是否存在可供下一operator利用的history-scale ↔ future-support structure，而不是再次凭叙事设计网络。

计划路径：

1. D9-A先从frozen natural A6 checkpoint精确恢复$W=BC$，不读取数据、不做finite difference；
2. 将$W[720,P,D]$投影到future RGNB support groups与history patch-axis DCT scales；
3. 以scale correlation、fine-global contrast、atom-label permutation与random orthogonal history bases作matched gates；
4. D9-A通过后才授权D9-B sample-dependent input-Jacobian/JVP confirmation；
5. 分离future support scale与history recency，禁止atom-to-patch retrieval claim；
6. canonical scale coupling若不能跨5 datasets/3 seeds超过controls，停止该方向并回Step2/3；
7. D9-B通过也只授权Step4-5 candidate/theory，不等于method effectiveness。

冻结设计见`analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/d9_diagnostic_design.md`。

当前不授权新model code、remote method training、test、SC2或joint factorial。
