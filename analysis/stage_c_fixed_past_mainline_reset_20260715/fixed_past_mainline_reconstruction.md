# StageC Fixed-Past 主线重构与创新点审计

日期：2026-07-15
状态：Step 1 external audit completed；Step 2-3 problem formulation；method implementation / remote training /
test均未授权。

## 0. 决策摘要

[Decision] 当前论文的研究对象重新固定为：

> 给定同一段fixed past，一个共享模型如何生成任意requested horizon的统一future prefix，并在不把$H$
> 变成learned semantic condition的前提下，同时保持global trajectory coherence与target-specific historical
> evidence access？

forecast-revision surface是有吸引力但不同的连续预测问题，已独立存档在`New-idea.md`。它不再占用当前论文
Contribution slots，D13也不再是active protocol。

[Strong Evidence] A6中真正尚未被正面验证的信息边界不是`flatten`，而是：

$$
M[B,C,P,D]\rightarrow \operatorname{vec}(M)[B,C,PD]
\rightarrow g[B,C,256]\rightarrow \hat Y[B,C,H].
$$

`flatten`是bijective reshape；潜在的信息限制发生在`PD -> 256`。全部future coordinates随后只共享同一个
global coefficient state $g$。但内部实验尚未证明完整patch memory $M$包含超越$g$的、可泛化的conditional
predictive information。

因此，本轮提出一条严格受problem gate约束的新主线：

> Beyond Global Compression：模型同时保留compact global state与uncompressed patch memory；只有当
> patch memory在给定global state后仍带来可验证的conditional predictive gain时，target-specific direct
> information path才被允许改变预测。

两项provisional contributions为：

1. `CADMO`：Compression-Aware Dual-Memory Operator；
2. `CPGA`：Conditional Predictive-Gain Accounting。

二者均为Step 2-3 provisional，尚不是论文贡献。下一步只授权`D14 Conditional Patch-Memory Headroom
Audit`。D14通过后才返回formal Step 4-6。

## 1. 当前fixed-past baseline的tensor事实

### 1.1 A6 forward contract

对fixed history：

$$
X\in\mathbb R^{B\times720\times C}
\rightarrow
M\in\mathbb R^{B\times C\times P\times D}
\rightarrow
h=\operatorname{vec}(M)\in\mathbb R^{B\times C\times PD},
$$

$$
g=W_gh+b_g\in\mathbb R^{B\times C\times256},
\qquad
\hat Y_H=B_{1:H}g+b_{1:H}\in\mathbb R^{B\times C\times H}.
$$

其中natural profiles允许dataset-specific $P\in\{12,24,48\}$与$D\in\{32,64\}$。requested $H$只裁剪
`basis[:H]`，所以A6已经具备：

- one model for multiple prefix horizons；
- exact prefix equality；
- $H$不进入Encoder、router或decoder state；
- output-side $O(HK)$直接生成，而非强制先写出720步。

因此，新主线不能再把上述性质包装为创新。

### 1.2 信息压缩的正确表述

[Fact] $M\rightarrow\operatorname{vec}(M)$不丢元素。

[Fact] $g=C(M)$是$M$的deterministic compressed representation；所有future coordinates只能通过$g$访问
history。

[Hypothesis] 对某些target coordinate $\tau$，$M$中可能存在没有被单一global state $g$保留的conditional
predictive information：

$$
\mathbb E[Y_\tau\mid g,M]\neq\mathbb E[Y_\tau\mid g].
$$

这不是既定事实。D2 full-affine只获得弱、非统一收益；B14 unit-specific retrieval也形成负证据。因此必须先
用D14检验，而不能把“patch更清晰”当作显然正确。

## 2. 内部历史证据的深度复盘

### 2.1 可保留的正证据

| Evidence | Result | 对新主线的含义 |
| --- | --- | --- |
| D1-v2 ordered-memory probe | frozen A6 memory在3 datasets上含有head可使用信息 | $M$不是无结构噪声；但未证明相对$g$的增量价值 |
| D3 basis main effect | MSE `+2.9174%`，5/5 | future output geometry影响预测，但不是target-specific history access证据 |
| D4 locality | balanced vs permuted interval `+1.6324%` | temporal locality可能有条件价值；DCT/PCA更强，不能claim exact balanced basis |
| D6 crossing | short `+1.1964%`，long `-1.2675%`，12/15 | target distance对应不同support preference；不等于history-scale routing |
| D8 geometry attribution | PAF geometry vs matched `+14.33%` | structured target coordinates有作用；rigid replacement仍远差于A6 |

### 2.2 必须尊重的负证据

| Evidence | Result | 禁止的捷径 |
| --- | --- | --- |
| D2 full-affine / nonlinear | rank expansion约`+0.678%`；dense nonlinear约`-6.47%` | 不能用generic bigger head或full-affine bypass解决 |
| D8 E2E | PAF vs A6 `-28.10%`，5/5负 | 不能用shared-latent query/basis整体替换A6 free head |
| JAPO | two seeds、0/5胜A6；router近uniform | 不能再堆atom experts或weak router |
| D9-D10 | 无跨dataset history-scale→future-support mapping | 不能构造scale router主线 |
| D11 | directional conflict 0/5 | 不能用conflict-aware loss作Contribution 2 |
| D12 | predictable frame support 1/5 | 不能继续围绕rank-256 frame reallocation |
| B14 | unit-specific patch retrieval弱/负 | patch direct path必须重新证明，不能预设有效 |

### 2.3 失败路线的共同病因

过去若干架构路线都试图先规定一个漂亮的future-side factorization，再要求history适配它。它们常获得代数性质
或conditional geometry收益，却损失A6自由函数类。新的设计顺序必须反转：

1. 先验证$M$相对$g$是否存在conditional headroom；
2. 保留A6 global path作为可包含的预测函数；
3. 只为被D14证明的增量信息增加direct path；
4. 用matched controls判断收益来自信息，还是来自更多参数/非线性；
5. 最后才讨论basis、attention、fusion或encoder更新。

## 3. 外部调研与novelty边界

检索日期：2026-07-15。
来源：external primary/official sources；Zotero未用作discovery completeness证据，是否已收录未逐篇核验。

### 3.1 直接相关工作

| Work | 已覆盖内容 | 对当前claim的约束 |
| --- | --- | --- |
| CATS, NeurIPS 2024 | horizon-dependent future queries直接cross-attend historical patches，query间独立并共享参数 | `future query -> patch attention`不能单独claim创新 |
| BasisFormer, NeurIPS 2023 | learned history/future bases、cross-attention系数匹配与future basis consolidation | learned basis与history-basis matching不能单独claim |
| MQTransformer, ICLR 2022 | forecast-context dependent encoder-decoder attention与feedback-aware decoding | context-dependent history retrieval已有直接压力 |
| TimePerceiver, 2025 preprint | target timestamp queries从latent bottleneck检索，用于generalized temporal targets | continuous/learnable target query不是新贡献 |
| Memory Guided Transformer, VLDB 2025 | patch-wise memory与global attention结合local/global information | generic local/global memory融合不是新贡献 |
| DeepGLO, 2019 | global model与local model组合 | “global + local”二分叙事早已存在 |
| TimeCapsule, KDD 2025 | compressed predictive representation与内部forecast supervision | compression-aware representation本身不是新贡献 |
| MTS Information Bottleneck / CIB-MTSF | forecasting中的IB/CIB与压缩、inter/intra-series information | information bottleneck术语不能作为新理论 |

Primary URLs：

- CATS：https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf
- CATS code：https://github.com/dongbeank/cats
- BasisFormer：https://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html
- MQTransformer：https://openreview.net/forum?id=rxF4IN3R2ml
- TimePerceiver：https://arxiv.org/abs/2512.22550
- Memory Guided Transformer：https://www.vldb.org/pvldb/vol18/p239-cheng.pdf
- DeepGLO：https://arxiv.org/abs/1905.03806
- TimeCapsule：https://doi.org/10.1145/3711896.3737157
- MTS Information Bottleneck：https://pmc.ncbi.nlm.nih.gov/articles/PMC10217396/
- CIB-MTSF：https://www.ijcai.org/proceedings/2025/627

### 3.2 完整novelty chain

[Decision] 当前可探索的新颖性单位不是某个primitive，而是完整链条：

$$
\text{fixed-past unified prefixes}
\rightarrow
\text{single global compression boundary}
\rightarrow
\text{nested dual-memory contract}
\rightarrow
\text{projective target-specific access}
\rightarrow
\text{conditional predictive-gain accounting}.
$$

只有同时满足以下条件才可能形成SCI-level contribution：

1. D14证明$M$在给定$g$后仍有跨dataset conditional headroom；
2. architecture保留A6 global path，不因结构美观牺牲function class；
3. target-specific access对任意$H$保持exact prefix equality；
4. patch-path gain超越CATS-like query、generic nonlinear与capacity controls；
5. CPGA超越普通deep supervision、orthogonality penalty与loss reweighting；
6. 两项在matched end-to-end joint training下各有主效应。

### 3.3 本轮排除的替代方案

1. `future-query cross-attention only`：CATS/TimePerceiver已覆盖，最多作primitive/control；
2. `global-local dual branch only`：DeepGLO、Memory Guided Transformer等已有直接先例；
3. `information bottleneck regularization`：相关forecasting工作已覆盖，且不能回答target-specific lost evidence；
4. `balanced interval basis as sole decoder`：D4/D8/JAPO evidence不足；可作future geometry scaffold；
5. `explicit H embedding/router`：连续性shortcut与horizon-specific feature风险高，仍禁止；
6. `prefix risk weighting / measure loss`：内部路线已被D11/D12和历史实验削弱；
7. `generic orthogonality loss`：MMSE orthogonality是经典性质，不能独立claim。

## 4. Contribution 1：CADMO

`CADMO`（Compression-Aware Dual-Memory Operator）同时保留两种history representation：

1. `global state` $g=C(M)$：紧凑、稳定、负责whole-trajectory coherence；
2. `patch memory` $M$：不经global compaction，按future coordinate提供target-specific evidence。

### 4.1 Provisional tensor path

$$
X[B,720,C]\rightarrow M[B,C,P,D],
$$

$$
g=C(M)[B,C,K],\qquad K=256,
$$

$$
Q_T=[q_1,\ldots,q_T]\in\mathbb R^{T\times D_q},
$$

$$
U=\operatorname{PatchRead}(Q_T,M)
\in\mathbb R^{B\times C\times T\times D_v},
$$

$$
\hat Y_{1:T}=\operatorname{Fuse}(Q_T,g,U)
\in\mathbb R^{B\times C\times T}.
$$

请求$H$时只计算/返回$q_{1:H}$。任一$q_\tau$不得读取其他future queries的state，所以：

$$
\hat Y_{1:H_1}(X)=\left.\hat Y_{1:H_2}(X)\right|_{1:H_1},
\qquad H_1<H_2.
$$

### 4.2 为什么不是“A6 + residual patch”

论文对象是两个nested information levels的forecast fusion，不是把任意residual module粘在A6后面：

- global route回答“压缩状态足以支持的预测”；
- full-memory route回答“给定该global prediction后，哪些target仍需要原始patch evidence”；
- fusion必须有A6 containment arm，也必须有patch-disabled arm；
- patch route是否被使用由CPGA的conditional gain定义，而不是由“更大网络”自动解释。

### 4.3 必须冻结的invariants

1. exact target-prefix equality；
2. requested $H$不进入learned path；
3. patch-disabled时可恢复A6 function；
4. 不使用dense full-affine bypass；
5. 每个target存在到$M$的direct gradient path；
6. 同dataset profile、objective、epoch/checkpoint/evaluation protocol匹配；
7. params/FLOPs只报告，不作为否决理由，但必须有capacity-matched controls。

### 4.4 Encoder是否要改

[Decision] 当前不先改Encoder。A6已经产生$M[B,C,P,D]$，CADMO首先把它视为patch cache。只有D14显示：

- raw/history features有增量predictive information；
- 但A6 $M$相对$g$没有；

才允许回滚Step 4，设计最小encoder interface repair。否则同步改Encoder和decoder会丧失归因。

### 4.5 Basis是否要保留

[Decision] A6 learned basis保留为global branch与containment control；不再作为整篇论文的理论中心。D3-D6的
geometry/locality evidence可用于设计target query/support descriptor，但不能强制patch branch经过同一basis。
这避免basis限制CADMO，又保留A6强global trajectory prior。

## 5. Contribution 2：CPGA

`CPGA`（Conditional Predictive-Gain Accounting）训练模型回答：full patch memory相对global state到底贡献了
多少可泛化的predictive gain？

令：

$$
\hat Y_g=f_g(g),
\qquad
\hat Y_f=f(g,M),
\qquad
\Delta=\hat Y_f-\hat Y_g,
\qquad
e_f=Y-\hat Y_f.
$$

因为$g=C(M)$，两者是同一fixed past下的nested representation family。若两个branch分别逼近相应conditional
mean，则MSE projection有：

$$
\mathbb E\|Y-\mathbb E[Y\mid g]\|^2
-
\mathbb E\|Y-\mathbb E[Y\mid g,M]\|^2
=
\mathbb E\|\mathbb E[Y\mid g,M]-\mathbb E[Y\mid g]\|^2\ge0.
$$

同时：

$$
\mathbb E[e_f\odot\Delta]=0.
$$

### 5.1 Provisional objective

$$
\mathcal L
=\mathcal L_{full}(Y,\hat Y_f)
+\lambda_g\mathcal L_{global}(Y,\hat Y_g)
+\lambda_a\mathcal L_{account}(e_f,\Delta).
$$

`global loss`防止joint training故意削弱$g$，从而虚构patch-path gain；`account loss`在future-distance bins上约束
normalized moment，而不是惩罚$\|\Delta\|$本身。

### 5.2 它与普通auxiliary loss的区别

CPGA不是简单给global branch加deep supervision。其paper claim必须同时包含：

1. $g$与$M$的nested representation contract；
2. global-only与full-memory forecasts的explicit paired outputs；
3. patch-induced change的conditional gain decomposition；
4. moment/calibration diagnostics；
5. 对generic deep supervision、stop-gradient、random cache与capacity controls的主效应。

若最终只有`global auxiliary loss`有效，而accounting term无独立收益，CPGA不能成为Contribution 2。

### 5.3 最大理论风险

1. $g$与$M$都由joint encoder学习，模型可能改变两者的信息分工；
2. finite neural networks不是真实conditional means，orthogonality只是一种训练proxy；
3. batch moment可能低方差、易被scale操纵；
4. L1训练与MSE理论不完全一致；
5. full-memory route的收益可能完全由capacity解释。

因此CPGA在CADMO point-loss model通过前不实现。先有有效dual-memory operator，再诊断其accounting gap，最后
才进入Step 4-6 training-strategy设计。

## 6. 两项contribution如何无缝融入论文叙事

### 6.1 论文主线

传统unified multi-horizon forecasting通常把整段history压为单一forecast state，再从该state生成所有future
targets。这样有利于global coherence，但默认该state对每个target都是充分统计量。

本论文提出：

1. 先用D14检验这个“global state sufficiency”假设；
2. 若不充分，CADMO保留global state，同时给每个target直接访问完整patch evidence的路径；
3. CPGA进一步要求这条高分辨率路径的每次改变都能由conditional predictive gain解释；
4. 最终得到一个既统一、又不把所有target锁进同一压缩状态的fixed-past forecaster。

### 6.2 不是两个拼接创新点

- CADMO定义“信息以什么层级进入预测”；
- CPGA定义“额外信息路径什么时候值得改变预测”；
- CADMO没有CPGA，容易成为普通dual-branch attention；
- CPGA没有CADMO，没有明确的nested information carriers，只剩generic loss；
- joint story的核心对象是`global-only vs full-memory forecast pair`。

### 6.3 Provisional title

> Beyond Global Compression: Information-Accounted Dual-Memory Operators for Unified Multi-Horizon Forecasting

该标题仍为working title；只有D14与后续E2E gates通过后才能冻结。

## 7. D14 Conditional Patch-Memory Headroom Audit

D14只回答problem existence，不训练CADMO/CPGA。

### 7.1 核心问题

> 给定A6 compressed state $g$与global prediction，完整patch memory $M$是否仍包含跨dataset、跨seed、可在
> validation泛化的target-specific residual information？

### 7.2 数据与split

- 5 datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- 3 A6 checkpoints：seeds 2021/2022/2023；
- natural profiles固定；
- train batches只用于fit probe；
- chronological validation用于唯一final gate；
- test=false；
- frozen A6只作conditional information diagnostic，不作architecture effectiveness comparison。

### 7.3 两级probe

#### D14-A：linear incremental sufficiency

比较：

1. `global-only`：$g\rightarrow r$；
2. `full-memory affine`：$\operatorname{vec}(M)\rightarrow r$；
3. `global + incremental memory`：先partial-out $g$，再用$M$预测剩余residual；
4. `capacity-matched random features`；
5. `sample-wise patch shuffle`；
6. `target-shift control`。

它复核D2但直接以`conditional gain beyond g`定义统计，不把full-affine capacity当机制。

#### D14-B：structured target-memory interaction

用小型frozen-representation query probe比较：

1. `coefficient MLP`；
2. `flat-memory MLP`；
3. `ordered patch-query readout`；
4. `per-sample permuted patch-query`；
5. `random query / target-shifted query`；
6. `parameter-matched no-memory readout`。

D14-B只判断target-specific ordered interaction是否比generic nonlinearity/capacity更有解释力。即使指标好，也
不能直接升级为paper method；CATS-like cross-attention必须在后续正式设计中作为baseline/control。

### 7.4 Gate

总体pass需同时满足：

1. 至少3/5 datasets的validation conditional MSE gain为正；
2. 每个pass dataset至少2/3 seeds同方向；
3. ordered patch path优于coefficient-only、random-feature、per-sample permutation与target-shift controls；
4. five-dataset macro MSE gain至少`0.5%`；
5. 任一dataset不得出现超过`5%`的严重退化；
6. chronological split、fit-only preprocessing、shape/hash/finite invariants全部通过。

[Decision Boundary]

- A/B均fail：关闭当前“A6 patch memory超越global compression”主线，rollback Step 2；
- A fail、B pass：只支持structured nonlinear interaction，不支持generic information-bypass claim；
- A pass、B fail：存在容量型增量信息，但target-specific patch mechanism未被支持；
- A/B pass：只授权CADMO进入formal Step 4-6；不授权CPGA与remote method training；
- numerical/overfit pathology：标记invalid/design fault，不作方向级否决。

### 7.5 Frozen fairness boundary

D14的positive/negative只针对冻结A6 representation中的conditional accessibility。它不能：

- 宣称CADMO end-to-end优于A6；
- 因probe失败而否定所有patch-level encoder；
- 因probe成功而省略matched end-to-end training；
- 用A6 co-adaptation gap判断new decoder effectiveness。

真正paper-core gate必须是same initialization class、matched objective/optimization/checkpoint/evaluation的
end-to-end joint training。

## 8. 分阶段研究计划

### Phase A：Problem verification（当前）

- `current_step`：Step 2-3；
- 执行D14-A/B；
- 结果决定是否保留CADMO问题；
- method=false，remote=false，test=false。

### Phase B：CADMO Step 4-6

仅D14通过后：

1. source-informed implementation audit：CATS、BasisFormer、MQTransformer、TimePerceiver、MGT；
2. 数学证明A6 containment与exact prefix equality；
3. 冻结最小fusion形式，不做architecture sweep；
4. 预注册A6、global-only、CATS-like、dual-memory、capacity/random/permutation controls；
5. 通过narrative gate后才Step 7A。

### Phase C：CADMO E2E effectiveness

1. local shapes/prefix/gradient/containment tests；
2. five datasets × seed2021 validation-only screen；
3. pass后再补seed2022/2023；
4. 只有macro与dataset consistency gate通过，CADMO才占用Contribution 1。

### Phase D：CPGA problem and theory gate

在point-loss CADMO上测量：

- patch contribution energy；
- full-error × patch-change moment；
- distance-bin calibration；
- harmful patch contribution fraction；
- global branch sabotage/collapse。

若不存在稳定accounting gap，CPGA关闭，重新寻找Contribution 2；不能因理论优美强行实现。

### Phase E：CPGA E2E and joint attribution

执行：

$$
\text{CADMO on/off}\times\text{CPGA on/off}
$$

的`2x2` factorial，并加入global deep supervision、stop-gradient与generic orthogonality controls。两项均需独立
主效应；interaction只作joint story加分，不能替代主效应。

### Phase F：full paper matrix

- 5 datasets × 3 seeds × dense horizon evaluation；
- TimeAlign/A6、CATS-like、BasisFormer-style与current SOTA native baselines；
- second backbone/generalization gate；
- MSE/MAE、prefix consistency、distance-bin、attention/information diagnostics；
- params/FLOPs/training stability；
- 所有test只在protocol冻结后读取。

## 9. 自我反驳与风险评估

### 9.1 最强反对意见

1. A6的$g=256$可能已经是充分的；D2/D12负证据使这一风险很高；
2. patch direct path可能只是CATS的变体；
3. global/local融合与information bottleneck都不是新概念；
4. CPGA可能只是在经典orthogonality上加auxiliary loss；
5. 额外memory path会带来更强capacity，难以归因；
6. ETT/Weather benchmark未必需要target-specific event retrieval。

### 9.2 为什么仍值得先做D14

1. 它直接回答用户提出的“融合后`[B,C,R]`是否压缩patch信息”疑问；
2. 它复用现有A6 checkpoints，成本低，且先于架构实现；
3. 失败可以迅速关闭路线，避免再次堆模型；
4. 通过则为decoder与training principle提供同一问题证据；
5. 它比继续寻找new basis/router更贴近真实信息路径。

### 9.3 当前信心

| Judgment | Confidence |
| --- | --- |
| fixed-past主线比revision surface更符合当前论文任务 | high |
| A6存在值得审计的global compression boundary | high |
| $M$相对$g$必然有可用增量信息 | low-to-medium |
| CADMO完整链具有潜在contribution novelty | medium |
| CPGA可形成独立training contribution | medium-low，依赖CADMO与后续diagnostic |
| 当前已拥有可写入论文的两个创新点 | low；目前只是两个provisional candidates |

## 10. 最终决策

[Decision]

1. 当前论文主线正式回到fixed-past unified multi-horizon generation；
2. revision surface转移到`New-idea.md`，状态`deferred_next_paper`；
3. 当前provisional pair为`CADMO + CPGA`；
4. active cursor=`SC-D14 Conditional Patch-Memory Headroom Audit`，Step 2-3；
5. 只授权D14 protocol/analyzer implementation；
6. D14前不得实现CADMO、CPGA、remote method training或读取test；
7. CADMO与CPGA必须串行推进：先证明信息headroom与operator主效应，再审计training contribution；
8. 若D14 fail，明确回滚Step 2，不继续给patch direct path叠加basis、MoE或Encoder innovation。
