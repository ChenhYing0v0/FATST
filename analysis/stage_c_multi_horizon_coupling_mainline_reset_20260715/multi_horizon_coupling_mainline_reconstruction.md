# StageC Multi-Horizon Coupling Mainline Reconstruction

## 0. Decision Snapshot

| Field | Decision |
| --- | --- |
| `current_step` | Step 2-3：multi-horizon problem reconstruction and existence gate design |
| `problem` | 一个unified decoder是否应该为所有future targets固定同一种output coupling granularity？ |
| `existence_evidence` | forecasting-strategy literature提供强问题先验；D6提供间接内部crossing；直接cross-dataset evidence仍缺失 |
| `idea` | provisional `PCSD` + `CCRL` |
| `theory_check` | exact projectivity可构造；deterministic MSE下coupling只能claim finite-sample/capacity inductive bias，不能claim Bayes necessity |
| `design` | D14-A coupling-spectrum headroom + D14-B cross-fitted regret predictability；5 datasets × 3 seeds；validation-only |
| `narrative_gate` | problem chain明显强于CADMO/CPGA；method novelty仍需D14与formal Step 4-6共同验证 |
| `effectiveness_gate` | not applicable；当前不授权paper-core method training |
| `artifacts` | 本报告、`Papers/multi-horizon-output-coupling-audit.md`、新D14 protocol |
| `decision` | CADMO/CPGA退出active slots；ordered patch memory降为auxiliary probe；只授权D14 diagnostic design/implementation |

## 1. Why The Previous Mainline Was Rejected

[Accepted Critique] `ordered patch memory`回答的是“history representation以什么接口进入decoder”，不是
“multi-horizon forecasting独有的核心矛盾”。即使D14证明patch memory相对global coefficient仍有增量信息，
最多说明A6存在history compression/interface bottleneck；同一论证也可用于single-horizon forecasting，无法自然
解释为什么论文必须以unified multi-horizon为中心。

因此以下旧链条退出active paper mainline：

$$
\text{global compression}\rightarrow\text{dual memory}\rightarrow\text{conditional gain accounting}.
$$

具体决定：

1. `SC1-CADMO`：`rejected_by_narrative_scope`；不是机制已被实验否定，而是核心问题与论文目标不对齐；
2. `SC2-CPGA`：`rejected_with_parent_route`；脱离CADMO后只剩generic deep supervision/orthogonality；
3. 原patch-memory D14：改为`D14-P auxiliary_interface_probe`，不占active cursor，也不决定paper mainline；
4. ordered patch是否保留，只能在未来具体decoder需要history interface时作小型ablation。

## 2. Reconstruct The Scientific Problem From Multi-Horizon Forecasting Itself

### 2.1 Fixed-past unified task

给定同一段past $X$，模型定义最大研究域$\mathcal T=\{1,\ldots,T\}$上的forecast function：

$$
\hat Y_{1:T}=F_\theta(X),\qquad
\hat Y_{1:H}=\mathcal R_HF_\theta(X),\quad H\le T,
$$

其中$\mathcal R_H$只是prefix restriction。requested $H$不进入learned semantic path，因此同一target
$\tau$的预测不因用户请求了$H=96$还是$H=720$而改变。

A6已经满足该projective contract：

$$
X[B,720,C]\to M[B,C,P,D]\to g[B,C,256]\to
B_{1:T}[T,256]g\to\hat Y[B,C,T].
$$

因此“one model”“arbitrary prefix”“exact prefix equality”都是carrier facts，不是新贡献。

### 2.2 The overlooked choice: how future targets share a forecasting function

任何multi-horizon decoder都隐式选择了一种future-output coupling strategy：

| Family | Output coupling interpretation | Main benefit | Main risk |
| --- | --- | --- | --- |
| Direct / independent query | 每个future target拥有较独立的readout | flexibility；无rollout error | sharing不足；finite-sample variance高 |
| AR / recursive | target通过previous predictions顺序耦合 | 强trajectory dependency | error accumulation；串行；prefix state依赖 |
| MIMO / global basis | 整个future作为一个multi-output task | global sharing/coherence；并行 | 固定global sharing可能产生bias/rigidity |
| DIRMO / block MIMO | block内联合、block间相对独立 | 折中flexibility与sharing | block size需选择；通常对整个任务固定 |
| Future-query decoder | target-specific history retrieval | target flexibility与parameter sharing | query间常为全独立或固定self-attention |

[Core Reframing]

> Unified multi-horizon forecasting不应只统一“输出长度”，还应统一“future targets应以多大范围共享
> predictive structure”。现有decoder通常把coupling granularity固定为point、global、recursive chain或一个
> 预先选择的block size；但最优sharing scope可能随future distance与history instance变化。

这条问题链直接服务multi-horizon：如果只有一个target，就不存在future-output coupling granularity。

### 2.3 Why this is not a stochastic-dependency claim

[Theory Boundary] 对deterministic point forecast与separable squared loss，Bayes predictor为
$f_\tau^*(X)=\mathbb E[Y_\tau\mid X]$。在无限数据、无限capacity下，不需要显式建模
$Y_{\tau_1},Y_{\tau_2}$的联合分布也能达到各坐标Bayes MSE。

因此本项目不能声称：

- independent output在统计上必然错误；
- future covariance本身会自动改善point MSE；
- joint trajectory modeling具有population-risk必然优势。

可辩护的理论对象是有限样本与有限capacity下的shared estimation：

$$
\operatorname{Risk}(s)
=\operatorname{ApproximationBias}(s)
+\operatorname{EstimationVariance}(s)
+\operatorname{OptimizationError}(s),
$$

其中coupling scale $s$决定future tasks共享多少参数、latent function与regularization。Direct/MIMO/DIRMO的
经典结果与2025 Stratify都把该问题描述为bias-variance-flexibility trade-off；本项目研究的是：能否在一个
projective neural decoder内部学习该trade-off，而非为每个dataset/horizon外部搜索一种strategy。

## 3. Internal Evidence Audit

### 3.1 What supports the new question

1. [Fact] A6是single global low-rank/MIMO-like decoder：所有targets共享同一$g[B,C,256]$与full-domain
   basis；它代表coupling spectrum的global endpoint。
2. [Strong Evidence, indirect] D6在未参与选择的validation window复现support crossing：local b144在short
   horizons相对global DCT MSE `+1.1964%`，在long horizons `-1.2675%`；12/15 primary units crossing，
   short-positive 4/5 datasets，long-negative 5/5 datasets。
3. [Strong Evidence] D8/JAPO说明完全替换A6 global function class风险很高；新operator必须contain A6 global
   arm，而不是再次用rigid structured head覆盖它。
4. [Strong Evidence] B13/PMFO的current recurrent transition没有稳定主效应；因此首版不采用AR-style
   predicted-output feedback，优先研究parallel coupling。

### 3.2 What does not support it yet

D6比较的是future basis support，不是Direct/MIMO/DIRMO strategy。因此它只能提出假设：不同future distance
可能偏好不同sharing scope；不能证明output coupling granularity真的存在可利用crossing。

D9-D10未找到稳定history-scale到future-support映射；JAPO router长期接近uniform。因此“history能预测最佳
coupling scale”目前完全未被证明。该未知量正是D14-B必须检验的problem evidence。

### 3.3 Excluded shortcuts

- D11 strict directional gradient conflict为0/5，不能用gradient surgery或conflict-aware loss作SC2；
- D12 predictable rank allocation为1/5，不能回到CAPE/forecast-frame learning；
- generic full-affine与dense nonlinear capacity在D2不稳定，不能把更宽head解释为coupling mechanism；
- static interval/random mask supervision总体为负，不能把horizon sampling本身包装为training contribution；
- ordered patch memory尚未证明，但即使证明也只影响history interface，不决定coupling mainline。

## 4. External Prior-Art Audit And Novelty Boundary

搜索日期：2026-07-15。默认使用external primary-source search；Zotero仅为seed，不作为完整性或时效性证据。
本轮覆盖正式proceedings、OpenReview、arXiv official pages与official project pages。

### 4.1 Occupied spaces

1. Direct/MIMO/DIRMO已明确把output size/block size视为flexibility与dependency的折中；
2. Stratify（2025）进一步统一RecMO、DirMO、DirRecMO、RectifyMO，指出strategy optimum依赖domain与function
   class，并仍需ad-hoc strategy selection；
3. CATS将每个future horizon作为independent query，并跨horizon共享decoder parameters；
4. MQTransformer与TimePerceiver已覆盖context/target timestamp query；
5. Implicit Forecaster已从“逐点预测缺少global view”出发，通过constituent waves生成完整trajectory；
6. MQF2已在probabilistic forecasting中直接建模跨future-time dependency；
7. dynamic ensemble、meta-learning model selection、multi-output ensemble与TimeRouter已覆盖根据history/forecast
   evidence选择或加权多个forecasting experts；
8. cross-validation、out-of-fold scoring、regret minimization本身均为通用工具。

### 4.2 What cannot be claimed

- 首次研究Direct/MIMO trade-off；
- 首次使用block-wise multi-output forecasting；
- 首次用future queries；
- 首次对不同forecast experts做routing；
- 首次用cross-validation或regret监督router；
- 首次用multi-scale decoder、local/global mixture或wave basis；
- 首次捕获future dependency。

### 4.3 Remaining complete-chain novelty opportunity

[Provisional Novelty]

> 在一个fixed-past、exact-prefix neural decoder内，把point-to-global forecasting strategies表示为同一
> projective coupling spectrum；再以train-only counterfactual out-of-fold risk学习sample × target-region的
> coupling policy，从而不依赖requested-$H$、per-horizon model或external strategy search。

该完整链与classic DIRMO不同：DIRMO为一个任务选择固定block size并训练多模型；本项目在一个模型内保留
multiple nested coupling scopes并学习组合。它与generic MoE不同：experts不是任意architecture，而是同一
future target set上的可解释sharing scopes。它与CATS不同：CATS固定independent future queries；本项目显式
表示independent-to-global continuum。它与Implicit Forecaster不同：后者以waves提供global decoding；本项目
研究sharing scope的conditional selection。

[Risk] 这仍是complete-chain novelty，不是primitive novelty。投稿前必须citation-chain DIRMO/Stratify、dynamic
multi-output ensembles与2026 adaptive decoder工作；若发现within-model adaptive coupling spectrum的直接前例，
必须进一步收窄或关闭。

## 5. Provisional Contribution 1: PCSD

### 5.1 Name and role

`PCSD`：Projective Coupling-Spectrum Decoder。

它不是先决定requested horizon，再选择一个horizon-specific head；而是在固定future domain上定义多个output
sharing scopes：

$$
\mathcal S=\{1,b_1,b_2,\ldots,T\}.
$$

$s=1$接近Direct/independent endpoint，$s=T$是A6 global MIMO-like endpoint，中间$s$为parallel block-MIMO
scopes。所有scopes预测同一future coordinates。

### 5.2 Provisional operator

令$Z=E_\theta(X)$为shared history representation，$D_s$为scale-$s$的block-shared forecast operator：

$$
\hat y_\tau^{(s)}=D_s\left(Z,e_\tau,\mathcal B_s(\tau)\right),
$$

其中$\mathcal B_s(\tau)$只由固定future coordinate与nested partition确定。最终预测为：

$$
\hat y_\tau=\sum_{s\in\mathcal S}\alpha_s(X,\tau)\hat y_\tau^{(s)},
\qquad \sum_s\alpha_s(X,\tau)=1.
$$

requested $H$不进入$D_s$或$\alpha_s$：

$$
F_H(X)=\mathcal R_HF_T(X).
$$

### 5.3 Mandatory theory contracts

1. `projectivity`：同一$\tau$输出不依赖requested $H$；
2. `A6 containment`：存在参数设置使$\alpha_T=1$并恢复A6 global readout；
3. `parallel generation`：首版不读取previous predicted values；
4. `matched sharing`：中间scopes使用shared operators，不能靠每block独立大网络增加capacity；
5. `no horizon semantics`：允许target coordinate $\tau$，禁止requested horizon ID/embedding；
6. `no efficiency overclaim`：若所有arms均计算，就只claimaccuracy/inductive bias，不claimsparse efficiency。

### 5.4 Why it may be valuable

- relative to A6：保留global function，同时允许short/local或instance-specific flexibility；
- relative to CATS：不把所有future queries固定为independent；
- relative to DIRMO：不为整个dataset/horizon选一个固定block size；
- relative to AR：保留parallel prediction并避免rollout error accumulation；
- relative togeneric multiscale basis：scale表示的是future-output parameter sharing scope，不是frequency或wavelet
  component。

### 5.5 Major risks

- function-class gain可能仍由capacity解释；
- learned mixture可能退化为constant/equal mixture；
- contiguous blocks未必对应真实task relatedness；
- target-distance-only policy可能已足够，使instance adaptivity不成立；
- A6 global arm可能因carrier co-adaptation在frozen probe中被不公平优待；
- nested partition与local/global operator容易被审稿人视为DIRMO + MoE，必须由D14和matched controls证明必要性。

## 6. Provisional Contribution 2: CCRL

### 6.1 Name and problem

`CCRL`：Cross-fitted Coupling-Regret Learning。

普通mixture loss只观察最终mixture error：

$$
\ell\left(\sum_s\alpha_s\hat Y^{(s)},Y\right),
$$

它不能直接告诉router“哪个coupling scope在该sample/target region上更合适”。共享encoder与joint optimization还
可能让experts co-adapt，导致uniform routing或一个arm吞并其余arms；JAPO已经出现过这种现象。

### 6.2 Counterfactual target construction

在train split内做chronological cross-fitting。对每个held-out training sample $i$、target bin $b$和scale $s$，
得到没有用该sample拟合的counterfactual loss：

$$
L^{\mathrm{cf}}_{i,b,s}
=\ell_b\left(\hat Y^{(-k,s)}_i,Y_i\right).
$$

定义regret与soft target：

$$
R^{\mathrm{cf}}_{i,b,s}
=L^{\mathrm{cf}}_{i,b,s}-\min_j L^{\mathrm{cf}}_{i,b,j},
\qquad
q_{i,b,s}\propto\exp\left(-R^{\mathrm{cf}}_{i,b,s}/\tau_r\right).
$$

policy只读取inference可用的$X_i$ summary与target coordinate/bin，学习$q$：

$$
\alpha(X_i,b)=G_\phi(h(X_i),e_b).
$$

### 6.3 Provisional training objective

若D14-B支持regret predictability，formal Step 6才允许冻结objective。当前仅保留候选形式：

$$
\mathcal L=
\mathcal L_{mix}
+\lambda_{arm}\sum_s\mathcal L_s
+\lambda_{cf}\operatorname{CE}\left(\alpha,\operatorname{sg}(q^{cf})\right).
$$

- $\mathcal L_{mix}$训练最终forecast；
- per-arm loss防止arms失去独立forecast能力；
- counterfactual supervision给policy明确的coupling evidence；
- `stop-gradient`防止模型通过篡改regret labels降低loss。

### 6.4 Novelty boundary

cross-fitting、expert regret、meta-learning与forecast combination都不是新贡献。CCRL只有在以下完整task-specific
chain成立时才可能成为Contribution 2：

1. experts对应同一neural decoder内部的output coupling scopes，而非外部异构models；
2. regret定义在sample × target region上，而非dataset-level model rank；
3. policy不读取requested $H$，保持exact-prefix unified inference；
4. CCRL相对ordinary mixture loss、equal mixture、in-sample pseudo-label、target-only policy有独立主效应；
5. `PCSD on/off × CCRL on/off` factorial显示非冗余主效应与合理interaction。

[Novelty Confidence] `medium-low before D14 and formal Step 4 audit`。这比CPGA更紧密服务multi-horizon主线，
但generic ensemble/meta-learning overlap很强；不能提前冻结为最终Contribution 2。

## 7. Joint Paper Story

### 7.1 One-sentence thesis

> A unified multi-horizon forecaster should not hard-code one output-sharing strategy: it should represent a
> projective spectrum from target-wise to global coupling and learn, from counterfactual forecasting risk, how
> much future targets should share predictive structure.

### 7.2 Narrative chain

1. multi-horizon forecasting不是$H$个互不相关的single-step tasks，也不是必须由一个global trajectory head完成；
2. Direct、AR、MIMO、DIRMO与future-query decoders的本质差异之一，是future targets共享predictive function的
   scope；
3. 经典方法把该scope固定或通过external search选择，无法成为真正的one-model unified strategy；
4. PCSD在一个exact-prefix decoder内表示point-to-global coupling spectrum；
5. CCRL用train-only counterfactual risk为policy提供可识别监督，而不是期待ordinary mixture loss自动产生
   specialization；
6. 共同目标不是“更复杂的decoder”，而是取消per-horizon/per-dataset forecasting-strategy selection。

### 7.3 Working title

`Beyond a Fixed Forecasting Strategy: Coupling-Adaptive Decoding for Unified Multi-Horizon Forecasting`

备选：`One Forecast, Many Coupling Scales: Projective Coupling Learning for Multi-Horizon Forecasting`。

## 8. D14 Direct Problem Gate

旧patch-memory D14不再执行为mainline gate。新的D14为：

`D14 Output-Coupling Granularity and Regret-Predictability Audit`。

### 8.1 D14-A: Does coupling granularity matter?

使用两个common carriers，避免把结论绑定到A6 co-adaptation：

1. train-only normalized raw-history feature carrier；
2. frozen A6 representation sensitivity carrier。

在同一carrier上建立parameter/optimization-matched coupling heads：point-like、small block、medium block、
large block、global。首选analytically auditable blockwise reduced-rank regression或同预算shared-head family；普通
multi-output ridge若loss完全separable，会与逐target ridge等价，不能充当有效coupling diagnostic。

必须比较：

- best fixed coupling scale；
- per-distance-bin oracle；
- per-sample × bin oracle；
- contiguous vs random/permuted partitions；
- same-parameter generic capacity control；
- raw carrier与A6 carrier结论一致性。

D14-A总体pass需要至少3/5 datasets出现稳定scale crossing，并且oracle相对best fixed scale的macro MSE
headroom至少`0.5%`；至少2/3 seeds/folds同方向；random partition不能解释主要收益。

### 8.2 D14-B: Is the useful coupling choice predictable before seeing labels?

只使用D14-A的train OOF predictions构造regret labels。比较：

1. global constant policy；
2. equal mixture；
3. target-coordinate-only policy；
4. history-only policy；
5. history + target-coordinate policy；
6. in-sample pseudo-label control；
7. permuted-regret label control。

CCRL problem pass需要history + target在至少3/5 datasets上超过best fixed与target-only，并在validation实现
macro MSE至少`0.3%`的可兑现gain；classification accuracy不能替代forecast gain。

### 8.3 Decision matrix

| D14-A | D14-B | Decision |
| --- | --- | --- |
| fail | any | 固定coupling是未证实瓶颈；PCSD/CCRL均关闭，rollback Step 2 |
| pass | fail | PCSD可返回Step 4；CCRL关闭，Contribution 2重新设计 |
| pass | target-only pass | 只支持distance-conditioned fixed policy；instance-adaptive claim关闭 |
| pass | history+target pass | PCSD/CCRL jointly返回formal Step 4-6；method training仍未授权 |
| invalid | any | 修复diagnostic；不得方向级否决 |

### 8.4 Frozen fairness boundary

D14不是paper-core effectiveness comparison。frozen A6 carrier是co-adapted representation，global arm可能天然
受益；因此：

- primary problem gate使用neutral raw-history carrier；
- A6 carrier只作sensitivity；
- positive只证明conditional headroom；
- negative若两carriers不一致，只能标记`diagnostic_invalid_for_direction_rejection`；
- final method必须same initialization class、matched E2E joint training、5 datasets × 3 seeds。

## 9. Staged Research Plan

### Phase A — D14 Step 2-3

1. source-informed定义matched coupling head family；
2. local synthetic test验证scale、partition与parameter accounting；
3. 5 datasets × 3 seeds validation-only D14-A；
4. D14-A pass后才运行D14-B；
5. test=false，new paper method=false。

### Phase B — Formal Step 4-6

仅D14对应gate通过后：

1. 完整citation chaining与official implementation audit；
2. 证明PCSD projectivity、A6 containment与parameter-sharing contract；
3. 审计CCRL与dynamic ensemble/meta-learning的equivalence boundary；
4. 冻结method、controls、failure attribution与kill gates；
5. narrative gate失败则不实现。

### Phase C — Step 7A/7B screen

1. local shape/projectivity/gradient/parameter tests；
2. 先ETTm1、ETTh2、Weather三dataset screen；
3. screen pass后扩展ETTh1、ETTm2；
4. all comparisons same-run E2E，A6 global arm、fixed-scale arms、equal mixture、ordinary router、capacity
   controls齐全。

### Phase D — Step 9-10 full evidence

1. 5 datasets × 3 seeds × frozen horizons；
2. overall MSE/MAE + short/mid/long + per-sample oracle utilization；
3. coupling weights、expert disagreement、regret calibration、router collapse；
4. Direct/CATS-like、A6/MIMO-like、fixed DIRMO scales、Implicit Forecaster/global decoder等baselines；
5. `PCSD on/off × CCRL on/off` factorial；
6. second-backbone generality在两项主效应成立后再执行。

## 10. Final Judgment

[Strong Judgment] 用户对CADMO/CPGA的否定是正确的。ordered patch memory可以影响decoder效果，但不能解释
multi-horizon forecasting的独特科学矛盾，因此不应占据论文主线。

[Provisional Positive] output coupling granularity比history interface更接近multi-horizon任务本质，也拥有清晰的
Direct–MIMO–DIRMO理论与实验参照。PCSD + CCRL形成architecture + training principle闭环，论文叙事明显更
完整：一个model不仅生成所有horizons，还学习future targets应如何共享forecasting function。

[Self-Critique] 当前内部证据只对future support crossing提供间接支持；CCRL又面临dynamic ensemble/meta-learning
强prior-art压力。因此两项仍是`proposed_step2_3`，不是已经获得的论文贡献。D14若不能证明coupling-scale crossing
与out-of-fold regret predictability，应立即关闭，不把DIRMO + MoE的组合强行写成创新。

[Next Action] 只推进新D14 source-informed diagnostic implementation；原patch-memory probe保持
`auxiliary_not_scheduled`。
