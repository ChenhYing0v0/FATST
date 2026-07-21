# ISCF Step5 Common–Private Scope Interaction Theory and Narrative Gate

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | ISCF Step5 theory complete；advance Step6 concrete control/design audit only |
| `problem` | 已确认的pre-synthesis scope response dependence能否形成超越linear reparameterization与generic extra depth的task-specific operator？ |
| `existence_evidence` | Step4 D1.1在disjoint validation histories上15/15超过direction-null与matched random-init；common/private response median=`0.2803/0.7197`；4/5 datasets topology稳定 |
| `idea` | 在future-output coupling modes进入scope-specific synthesis前，以set mean和scope deviation构造zero-initialized common–private multiplicative interaction |
| `theory_check` | fixed linear scope mixing可被ISCF独立affine mode maps精确吸收；multiplicative common–private path改变有限网络function class，但不改变Bayes information set |
| `design` | working candidate `ISCF-v1-CPSI`；输入/输出`[B,C,S,D,K]`；shared bottleneck；permutation-equivariant；zero interaction精确包含ISCF-v0 |
| `narrative_gate` | `conditional_pass_to_step6_as_task_coupled_common_private_interaction` |
| `effectiveness_gate` | not applicable；no implementation/training/test |
| `artifacts` | 本报告与`configs/stage_c_iscf_v1_cpsi_step5.json`；Step4 D1.1 artifacts只作existence evidence |
| `decision` | `step5_theory_pass_step6_control_design_next`；active_method=none；implementation/remote/test false |

## 2. Reader path

本报告依次回答：

1. 为什么`5×5 mixing matrix`或common-mode linear sharing不是新function class；
2. 什么最小nonlinear interaction同时对应Step4的common/private evidence；
3. operator在真实ISCF tensor path中的shape、morphism和parameter cost；
4. 哪些controls才能把peer relation与generic capacity、linear sharing及late placement分开；
5. 在最新prior-art overlap下，论文叙事还能claim什么、不能claim什么；
6. Step6之前仍有哪些未解决项，因此为什么本轮不实现、不训练。

## 3. Existing ISCF-v0 tensor contract

代码审计基于`SIFFCouplingFieldReadout`的independent contract，不改变既有checkpoint identity：

| Stage | Tensor | Shape | Operation |
| --- | --- | --- | --- |
| encoder output | $h$ | `[B,C,R]` | fixed-past history representation |
| independent affine modes | $M$ | `[B,C,S,D,K]` | `einsum("bcr,sdrk->bcsdk", h, mode_weight) + mode_bias`；ISCF中$S=Q=5$且`scale_basis=I_5` |
| scope pooling | $Z_s$ | `[B,C,G_s,K]` | $Z_s=M_s^\top P_s$；$P_s$为scope-specific pooled future-coordinate descriptors |
| scope synthesis | $A_s$ | `[B,C,T]` | identity path plus `GELU(Z_s)` nonlinear path，使用shared temporal synthesis rows |
| late fusion | $\hat Y$ | `[B,C,T]` | arms `[B,C,S,T]`与direct policy `[B,C,T,S]`逐点加权 |

因此Step5 intervention point被限定在$M$产生之后、任一`_scope_forecast`之前。它不是对final forecast做residual
correction，也不读取requested horizon、target label、future context或oracle arm error。

## 4. Linear reparameterization theorem

### 4.1 Statement

对任意fixed linear cross-scope operator

$$
\widetilde M_s=\sum_{j=1}^{S}A_{sj}M_j,
$$

若每个ISCF mode本来就是hidden的独立affine map

$$
M_j=W_jh+b_j,
$$

则存在新的独立参数

$$
\widetilde W_s=\sum_j A_{sj}W_j,\qquad
\widetilde b_s=\sum_j A_{sj}b_j,
$$

使得

$$
\widetilde M_s=\widetilde W_sh+\widetilde b_s.
$$

这里$A_{sj}$可以是scalar、对$D/K$维的fixed linear map，或其low-rank factorization；只要operator在$M$上是
fixed linear并且不引入新的sample information，结论不变。

### 4.2 Consequence

[Fact] Cross-Stitch式$5\times5$ activation mixing、linear common/private decomposition、fixed graph diffusion及
linear peer mean都不扩展ISCF-v0的mode function class。它们最多改变parameterization、initialization和optimization
bias。结合既有output low-rank gate `0/15`，不能把这类linear mixing包装成新的paper mechanism。

因此Step5拒绝以下direct candidates：

- `learned_scope_matrix @ M`；
- `M_s + alpha * mean(M)`；
- 仅将五个modes concat后接一个linear projection；
- 恢复ordered log-scale matrix或固定pairwise topology。

这些形式仍应保留为matched linear controls，因为它们能判断optimization bias是否已解释candidate gain。

## 5. Why a plain peer MLP is still insufficient

一个直接的Deep Sets-style方案是

$$
M'_s=M_s+F\left(\frac{1}{S-1}\sum_{j\neq s}M_j\right).
$$

它确实引入nonlinearity，但peer mean仍是$h$的另一组affine projection。若只与ISCF-v0比较，收益可能来自“增加一个
projection和一层MLP”，而不是scope relation本身。self-only MLP虽然可作capacity control，却把linear/private path与
nonlinear path强制共享同一个projection，仍不是完全对称的归因。

故plain peer MLP没有通过Step5 narrative gate；它只作为可能的generic set-mixing control，不进入working method。

## 6. Working candidate: ISCF-v1-CPSI

`CPSI`是临时research ID，表示`Common–Private Scope Interaction`，不是最终论文名。令
$L=DK$，将$M$最后两维展平为

$$
X\in\mathbb{R}^{B\times C\times S\times L}.
$$

### 6.1 Common/private decomposition

$$
\mu=\frac{1}{S}\sum_{j=1}^{S}X_j
\in\mathbb{R}^{B\times C\times 1\times L},
$$

$$
\delta_s=X_s-\mu
\in\mathbb{R}^{B\times C\times S\times L}.
$$

$\mu$是scope-set invariant common state，$\delta_s$是zero-sum private deviation。这里的“common/private”是
operator定义，不把Step4 response-energy比例当作latent truth，也不冻结任何dataset-specific比例。

### 6.2 Multiplicative interaction

使用三组跨scope共享、无bias的linear maps：

$$
c=\operatorname{GELU}(W_c\mu),\qquad
p_s=\operatorname{GELU}(W_p\delta_s),
$$

$$
g_s=c\odot p_s,\qquad
m_s=W_og_s,
$$

其中$W_c,W_p\in\mathbb{R}^{r\times L}$，$W_o\in\mathbb{R}^{L\times r}$，所以
$c,p_s,g_s$分别为`[B,C,1,r]`、`[B,C,S,r]`、`[B,C,S,r]`，$m_s$为
`[B,C,S,L]`。最终native mode update为

$$
X'_s=X_s+m_s,
$$

再reshape为`[B,C,S,D,K]`并进入原有五个`_scope_forecast`。论文表述应为“scope interaction生成新的native
pre-synthesis modes”，不得表述成post-hoc residual corrector或adapter。

### 6.3 Why the product is necessary

- 若所有scopes相同，则$\delta_s=0$、$p_s=0$、$m_s=0$；generic common shift不会被误称为relation；
- 若common state为零，则$c=0$、$m_s=0$；纯private extra MLP也不是candidate的工作路径；
- 只有common与private两类状态同时存在时才产生interaction，直接对应D1.1的“stable shared response + majority
  private response”证据；
- $c\odot p_s$对$h$形成非线性乘性交互，不能由第4节的single affine reparameterization精确吸收。

[Strong Evidence] 这证明candidate具有超越linear mixing的finite-network function path；它不证明该path会改善forecast。

## 7. Equivariance, containment and optimization feasibility

### 7.1 Non-ordered equivariance

对任意scope permutation $\pi$，若modes及其关联的scope synthesis metadata共同置换：

- $\mu(\pi X)=\mu(X)$；
- $\delta(\pi X)=\pi\delta(X)$；
- shared $W_c,W_p,W_o$不依赖slot identity；
- 因而$m(\pi X)=\pi m(X)$且$X'(\pi X)=\pi X'(X)$。

所以operator是scope-permutation-equivariant，不读取scale数值、scale order或requested H。它也不假设ETTh2上未确认的
universal pairwise graph。

### 7.2 Exact ISCF-v0 containment

初始化

$$
W_o=0
$$

即可对任意input精确得到$m_s=0$与$X'=X$。这是active function-preserving path，不是random-initialized weight copy。
$W_c,W_p$使用标准fan-in initialization；首个optimization step中$W_o$可获得非零gradient，随后gradient才传播到
$W_c,W_p$。Step7A必须显式验证这一two-stage gradient opening，不能只验证parameter存在。

### 7.3 Parameter budget

无bias时新增参数为

$$
N_{\text{CPSI}}=3Lr=3DKr.
$$

以Step5 theory probe rank $r=32$计算：

| Dataset profile | $D$ | $K$ | $L=DK$ | Added params | vs active ISCF-v0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 4 | 109 | 436 | 41,856 | 1.1641% |
| ETTh2 | 4 | 116 | 464 | 44,544 | 2.2090% |
| ETTm1 | 4 | 116 | 464 | 44,544 | 2.2399% |
| ETTm2 | 4 | 106 | 424 | 40,704 | 0.5983% |
| Weather | 4 | 116 | 464 | 44,544 | 2.2090% |

$r=32$只用于证明small finite-capacity feasibility，不是observed-result-driven hyperparameter。Step6必须冻结单一全局rank
rule及其parameter matching；不得按dataset/test挑rank。

## 8. Bayes and task boundary

[Fact] CPSI不改变$\sigma(X_{\text{past}})$、requested H、target coordinate或loss。fixed past与pointwise MSE下，同一
future coordinate的Bayes conditional mean仍不依赖requested horizon。

[Hypothesis] CPSI可能有用的唯一合法解释是：在固定finite-capacity ISCF中，五个future-output coupling scopes已经形成
common/private local responses；让common state在scope-specific nonlinear synthesis之前调制private deviation，可能比五个
完全独立heads加late scalar mixture更有效地共享有限参数与组织nonlinear computation。

因此未来正结果只能支持finite-capacity inductive bias，不得claim新的future information、Bayes-optimal H dependence或
oracle routing。若matched self-only control解释收益，failure attribution必须是`capacity_control_explains`。

## 9. Primary-source and implementation audit

检索日期：`2026-07-21`。主题包括set equivariance、multi-task feature sharing、forecast expert mixing、2025–2026
multi-scale/expert interaction。primary sources来自NeurIPS paper/official repository、PMLR、CVPR、arXiv official
record与official code。以下文献均由external search发现或复核；本轮没有以Zotero coverage判断novelty，FSA subset中的
presence未核验，记为`unknown`。

| Prior | Mechanism / implementation fact | Adopted | Rejected / boundary |
| --- | --- | --- | --- |
| [Deep Sets](https://arxiv.org/abs/1703.06114) / [official repo](https://github.com/manzilzaheer/DeepSets) | permutation-invariant set functions以shared element transform与sum aggregation为核心；official repo为paper code | mean-based invariant common state、shared maps | generic set pooling不是novelty；不输出单一set summary |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) / [official code](https://github.com/juho-lee/set_transformer/blob/master/modules.py) | official `SAB`调用`MAB(X,X)`；Q/K/V projections、scaled dot-product attention、residual与row-wise nonlinear output | 作为pairwise expressive upper control | D1.1未证明attention或pairwise graph必要；$S=5$下不先引入attention |
| [Cross-Stitch](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf) | learned linear combinations在task streams间共享activation；initialization/optimization会影响结果 | matched linear sharing control | 第4节证明其在ISCF affine modes上不形成新function class |
| [MoLE](https://proceedings.mlr.press/v238/ni24a.html) | multiple forecasting experts由router加权输出并end-to-end训练 | output-mixture prior/control | ISCF已有late output mixture；不新增router，不claim expert mixing |
| [DMSC v5](https://arxiv.org/abs/2508.02753) | dynamic multi-scale decomposition、coarse-to-fine gated guidance与adaptive scale-routing MoE | 证明“multi-scale coordination”已是直接邻近prior | 不claim scale coordination、hierarchical guidance或adaptive routing；CPSI必须限定为future-output coupling modes的non-ordered common/private path |
| [TimeExpert](https://arxiv.org/abs/2509.23145) | attention-level local temporal experts与global expert共同组织history context | 证明pre-output expert interaction也并非空白 | 不claim“interaction occurs before output”本身；其history-token routing与本项目scope-mode interaction不同 |
| [MSCGrapher](https://proceedings.mlr.press/v286/yang25c.html) | adaptive graph correlation across scales用于inter-series dependency | dynamic correlation control boundary | 不引入variable graph，不把scope relation等同inter-series correlation |

Source-derived implementation decision：若Step6最终通过，组件应在本仓库本地实现，复用ISCF tensor/config contract；不复制
upstream module。Set Transformer只进入formal control的前提是Step6能给出pairwise necessity，否则保持deferred。

## 10. Mandatory controls and failure attribution

Step6必须冻结下列arms；缺一项不得进入production implementation：

| Arm | Same params? | Tests | Failure attribution |
| --- | --- | --- | --- |
| `ISCF-v0` | no added params | carrier baseline | candidate无增益时只说明new path无效 |
| `CPSI-SELF` | exact `3Lr` | 两个nonlinear factors都只读取$X_s$，排除generic multiplicative depth/capacity | 若解释CPSI gain：`capacity_control_explains` |
| `CPSI-LINEAR` | exact `3Lr` | 用$W_o(W_c\mu+W_p\delta_s)$替代product/GELU，排除linear sharing/optimization bias | 若解释gain：linear parameterization sufficient，paper mechanism fail |
| `CPSI-COMMON` | Step6待做exact matching | 只允许common path，检查private interaction必要性 | 若相当：common/private necessity fail |
| `POST-SYNTH INTERACTION` | Step6必须给出诚实matching rule | 将interaction放到scope synthesis后、final fusion前，检验placement specificity | negative只能在control数值健康时支持pre-synthesis placement |
| ordered SIFF-v2 / Q1 | frozen historical controls | 排除ordered field与common-only旧解释 | 不得复活closed claims |

此外：

- same-init-class end-to-end joint training是未来effectiveness gate；frozen replacement只可debug；
- validation只做four-horizon checkpoint selection与低成本implementation iteration；
- official test matrix必须在Step6后另行预注册并经用户授权；当前不访问test；
- 不加入router、second loss、entropy/orthogonality regularizer、requested-H embedding或oracle teacher。

预注册failure mapping：

1. CPSI≈ISCF且controls健康：`hypothesis_false_for_exact_cpsi_operator`，回Step4/5；不否定ISCF carrier；
2. SELF≈CPSI且二者提升：`capacity_control_explains`，CPSI narrative fail；
3. LINEAR≈CPSI且二者提升：linear sharing/optimization bias sufficient，回Step4；
4. POST-SYNTH≈CPSI：pre-synthesis necessity unsupported，收窄或关闭placement claim；
5. divergence、>100% degradation、zero gradient、validation/test reversal：按
   `optimization_or_numeric_pathology`或`design_fault_suspected`处理，不作方向级拒绝；
6. positive effectiveness但internal product path collapse：只能记`performance_partial_pass`。

## 11. Narrative gate and self-critique

### 11.1 Allowed contribution boundary

```text
single-model varied-horizon forecasting
-> future-output coupling scopes as native forecast operators
-> externally controlled evidence of stable common + private pre-synthesis response
-> linear sharing is provably reparameterizable
-> zero-initialized, non-ordered common–private multiplicative interaction
   before scope-specific nonlinear synthesis
-> matched self/linear/common/placement attribution
```

这是一条完整的problem–evidence–necessity–operator–control链，而不是“首次使用Deep Sets/feature mixing/MoE”。

### 11.2 Strongest counterargument

[Speculative] 即使CPSI超过SELF，它仍可能只是利用其他learned affine projections of the same hidden state，审稿人可能把它
视为generic cross-head feature sharing。DMSC已覆盖multi-scale gated coordination，Deep Sets/Set Transformer已覆盖set
interaction，因此component-level novelty有限。论文成立必须同时满足：

1. Step6给出无结构漏洞的COMMON与POST-SYNTH controls；
2. end-to-end official-test effectiveness超过ISCF-v0与A6_FULL；
3. SELF与LINEAR不能解释收益；
4. internal common–private product path实际非退化；
5. 贡献用future-output coupling problem限定，而不是泛称multi-scale interaction。

若Step6不能解决placement/capacity attribution，本candidate应在实现前关闭。不能因当前急于论文落地而跳过该风险。

### 11.3 Decision

```text
narrative_gate = conditional_pass_to_step6_as_task_coupled_common_private_interaction
decision = step5_theory_pass_step6_control_design_next
```

本pass只确认：linear shortcut已被排除，CPSI有明确的nonlinear function path、shape contract、equivariance、exact
containment与可审计claim boundary。它不是full method gate，更不是performance evidence。

## 12. Step6 handoff and authorization

Step6只允许完成：

1. 冻结`CPSI-SELF/CPSI-LINEAR/CPSI-COMMON/POST-SYNTH`的exact formulas、shapes、parameter gaps与initialization；
2. 判断POST-SYNTH control能否在不引入任意fixed projection的情况下诚实匹配；若不能，降低placement claim或关闭candidate；
3. 冻结global bottleneck-rank rule、same-init pairing、objective、checkpoint selector、internal product-health metrics；
4. 给出Step7A local checks和future formal matrix，但不实现、不启动。

当前授权：

```text
active_method = none
current_step = ISCF Step6 concrete design/control audit
method_implementation = false
remote_training = false
formal_test = false
router_or_second_loss = false
```
