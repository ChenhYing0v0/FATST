# StageC PCSD-CF Native Architecture Step 4-6 Audit

## Status

| Field | Value |
| --- | --- |
| `current_step` | `SC1-PCSD-CF` Step 4-6 complete；Step7A local implementation next |
| `problem` | fixed-past unified decoder需要在同一计算图中表达point/block/global future-output sharing scopes |
| `existence_evidence` | D14-A1 three-seed dual-carrier 5/5 stable crossing；strict oracle 7.1107%/9.1259% |
| `idea` | 一个shared coupling field经不同future-coordinate pooling operators产生全部scope arms |
| `theory_check` | exact A6 containment、full-domain projectivity与scale-topology separation可构造 |
| `narrative_gate` | conditional pass for local implementation |
| `effectiveness_gate` | pending；必须超过A6、equal/static policy与capacity-matched control |
| `remote_training` | false |
| `test_access` | false |
| `rollback` | local invariant fail -> Step5/6；carrier/arm skill fail -> Step4；direct credit fail only -> SC2 Step2-4 |

## 1. What We Are Redesigning

D14-A用五个独立、matched、end-to-end `GroupedMLPReadout`模型证明了coupling scope互补，但它们只是
problem probes，不是PCSD。旧D14-B1随后试图用独立cross-fit models生成risk labels，再监督未来的联合PCSD。
这会形成三处不一致：

1. teacher是独立scale models，student是共享representation的multi-arm decoder；
2. OOF risk只覆盖部分training samples，且随最终arms更新而stale；
3. 额外fold × scale训练不属于最终推理图，复杂度与generic routing novelty不匹配。

[Decision] `SC2-CCRL`退出paper-core并降为`diagnostic_only_not_scheduled`。研究循环返回PCSD Step4-6，先定义
native single-decoder tensor contract。新的training contribution只有在direct end-to-end control暴露可归因的
credit-assignment failure后才允许提出。

## 2. External Source Audit

检索日期为2026-07-16。范围包括multi-step strategy、future-query/coordinate decoder、operator learning、
local/global operator mixture与end-to-end MoE。external primary sources优先；Zotero只作seed，本轮不以其收录
状态判断novelty。

| Source | Covered Primitive | Boundary For PCSD-CF |
| --- | --- | --- |
| [Stratify, 2025](https://link.springer.com/article/10.1007/s10618-025-01135-1) | 以strategy parameters统一Direct/MIMO/DIRMO/recursive families并研究bias/variance | 固定或外部搜索的multi-output strategy continuum不是创新 |
| [CATS cross-attention, NeurIPS 2024](https://arxiv.org/abs/2405.16877) | horizon-dependent future queries与parameter sharing | future-coordinate query本身不是创新；requested-H semantics继续禁止 |
| [DeepONet, Nature MI 2021](https://doi.org/10.1038/s42256-021-00302-5) | branch编码input function、trunk编码output coordinate并以内积合成operator | branch/trunk、coordinate basis与separable synthesis不是创新 |
| [PoU-MoE DeepONet](https://arxiv.org/abs/2405.11907) | spatial partition-of-unity、local experts与operator mixture | local/global trunk mixture与spatial locality不是component novelty |
| [TimeFuse, ICML 2025](https://proceedings.mlr.press/v267/liu25cm.html) | sample-level features驱动heterogeneous forecast fusion | direct adaptive fusion必须是control，不能作为Contribution 2 |
| [Soft MoE, ICLR 2024](https://openreview.net/forum?id=jxpsAj7ltE) | fully differentiable soft assignment | soft routing本身不是创新 |
| [SMEAR, TMLR](https://openreview.net/forum?id=7I199lc54z) | parameter-space soft expert merging与standard-gradient training | parameter/expert merging不是创新捷径 |
| [sMCL, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/20d135f0f28185b84a4cf7aa51f29500-Abstract.html) | oracle-loss驱动并行expert specialization | online best-arm/oracle supervision不能替代新training contribution |

[Novelty Boundary] PCSD-CF不claim multi-scale、basis、coordinate query、pooling、MoE或soft fusion primitive。
保留的complete chain是：

`fixed-past unified multi-horizon problem -> one projective decoder parameter field -> scope pooling changes
future-output state sharing -> simultaneous point/block/global operators -> exact A6 subspace -> requested-H-free
direct allocation`。

## 3. A6 Carrier Contract

对A6-natural encoder：

$$
M\in\mathbb R^{B\times C\times P\times D_e}
\rightarrow z=\operatorname{flatten}(M)\in\mathbb R^{B\times C\times R}.
$$

A6-LBF使用：

$$
c=zW_c+b_c\in\mathbb R^{B\times C\times K},\qquad
\hat y=Bc+b_y,
$$

其中$K=256$、$B\in\mathbb R^{T\times K}$、$T=720$。PCSD-CF必须以构造性映射包含这个function class，
不能把随机初始化复制称为preserved learned capacity，也不使用frozen replacement。

## 4. Proposed Native Operator: PCSD Coupling Field

### 4.1 One shared parameter field

固定$D_q=4$维future-coordinate field：

$$
Q=[q_0,q_1,q_2,q_3]\in\mathbb R^{T\times D_q},
$$

其中$q_0(\tau)=1$，其余为zero-mean low-order DCT coordinates。它们是standard descriptors，不作为novelty。

decoder只保存一套mode maps：

$$
U\in\mathbb R^{D_q\times R\times K},\qquad
u_b\in\mathbb R^{D_q\times K},\qquad
V\in\mathbb R^{T\times 2K}.
$$

history首先只计算一次：

$$
Z_{d,k}=\sum_r z_rU_{d,r,k}+u_{b,d,k},\qquad
Z\in\mathbb R^{B\times C\times D_q\times K}.
$$

### 4.2 Scale means pooling the field, not selecting another model

对scope $s\in\{1,48,144,360,720\}$，$P_s$把full future domain划为$G_s=T/s$个groups并对$Q$取均值：

$$
\bar Q_s=P_sQ\in\mathbb R^{G_s\times D_q}.
$$

每个group的predictive state为：

$$
A^{(s)}_{g,k}=\sum_d\bar Q_{s,g,d}Z_{d,k}.
$$

因此：

- $s=1$：每个future target拥有自己的coordinate-conditioned state；
- $1<s<T$：同一group共享一个state；
- $s=T$：所有targets共享一个global state。

这五个arms没有独立Encoder、独立完整Decoder或独立parameter bank；它们只是同一field在五个pooling operators
下的evaluation。

### 4.3 Direct synthesis and nonlinear containment

使用feature lift $\psi(a)=[a,\operatorname{GELU}(a)]$，并由target row $V_\tau$合成：

$$
\hat y^{(s)}_\tau
=V_\tau^\top\psi\!\left(A^{(s)}_{g_s(\tau)}\right)+b_\tau.
$$

这是一个direct operator，不是`A6 + residual`。identity feature用于构造A6 subspace，GELU feature允许有限
capacity下不同sharing scopes产生非线性差异。

### 4.4 Direct end-to-end policy control

policy只读取$z$的32维projection与natural target coordinate，不读取requested $H$、dataset ID或future truth：

$$
p_s(X,\tau)=\operatorname{softmax}_s g(z,q_\tau),\qquad
\hat y_\tau=\sum_sp_s(X,\tau)\hat y^{(s)}_\tau.
$$

首个method control只优化actual full-domain pointwise L1，不加入risk、oracle、load-balance、diversity或
counterfactual auxiliary loss。这样才能先判断native PCSD是否自然可训练，以及Contribution 2的optimization
problem是否真实存在。

## 5. Theory Feasibility

### 5.1 Exact A6 containment

zero-mean coordinates满足：

$$
\frac1T\sum_\tau Q_\tau=[1,0,0,0].
$$

给定任意A6参数$(W_c,b_c,B,b_y)$，设置：

$$
U_0=W_c,\quad u_{b,0}=b_c,\quad U_{1:}=u_{b,1:}=0,\quad
V_{:,1:K}=B,\quad V_{:,K:}=0,\quad b=b_y.
$$

则任意scope都退化为同一个A6 function；特别地global arm精确等于A6。float64构造检查的maximum gap为
`0.0`。这证明function-class containment，不声称训练后capacity preservation。

### 5.2 Projectivity

所有arms与policy先产生$F_T(X)$，requested horizon只执行restriction：

$$
F_H(X)=\mathcal R_HF_T(X)=F_T(X)[:H].
$$

$H$不进入$Q$、$P_s$、$U$、$V$或policy，因此同一fixed past的prefix在不同requested horizons下相同。

### 5.3 Scale separation

当至少一个zero-mean mode $U_d\ne0$时，不同$P_sQ$一般不同；point/block/global scopes因此得到不同group
states。数值预审计中，$s=1/48/144/360/720$的descriptor group counts为`720/15/5/2/1`，global的
nonconstant RMS为`5.15e-17`，而其余scopes非零。Step7A仍必须检查random parameters下arm disagreement与
Jacobian sharing，而不能仅凭nominal scale通过。

### 5.4 Capacity boundary

$D_q=4,K=256$时coupling-field core（mode maps、mode bias、identity/GELU synthesis与temporal bias，暂不计
small policy）相对A6-LBF decoder约为：

| Profile state width $R$ | Ratio |
| ---: | ---: |
| 768 | 3.0291 |
| 1536 | 3.3590 |
| 3072 | 3.6184 |

参数差异不用于dataset profile选择或直接拒绝candidate，但performance attribution必须包含matched dense
nonlinear control和parameter/FLOP table。若generic matched head复制收益，只能记为
`capacity_control_explains`。

## 6. Frozen Step 7 Design

### 6.1 Step7A local-only invariants

1. five profiles下`memory [B,C,P,D_e] -> z [B,C,R] -> arms [B,C,5,720] -> forecast [B,H,C]`；
2. all dense horizons与arbitrary prefixes exact crop equality；
3. arbitrary A6 weight mapping的float64/float32 containment；
4. $s=1$ point state、intermediate group state与$s=720$ global state的Jacobian-sharing topology；
5. random parameter arm separation、equal-logit initialization与finite gradients；
6. canonical/random partitions只改变`P_s`，不改变parameter count；
7. effective config中requested-H feature、frozen replacement、warm-start与test均为false；
8. parameter/DoF/FLOP and activation-memory accounting。

### 6.2 Step7B seed-2021 screen arms

| Arm | Role |
| --- | --- |
| `A6_LBF_E2E` | accepted same-run carrier control |
| `PCSD_CF_M0` | exact A6 morphism control，非candidate |
| `PCSD_CF_FIXED_{1,48,144,360,720}` | same native field with fixed scope |
| `PCSD_CF_EQUAL` | uniform-scope mixture |
| `PCSD_CF_STATIC_TARGET` | target-coordinate-only policy |
| `PCSD_CF_DIRECT` | history × target direct end-to-end policy primary |
| `PCSD_CF_RANDOM_PARTITION` | contiguity specificity control |
| `DENSE_NONLINEAR_MATCHED` | generic capacity control |

所有arms使用A6-natural five profiles、full-H720 pointwise L1、from-scratch joint training、best-validation H720
checkpoint；test=false。先运行Step7A，remote仍需单独授权。

### 6.3 Effectiveness and next-problem gates

`PCSD_CF_DIRECT`作为Contribution-1 method candidate必须：

1. 至少3/5 datasets超过A6，five-dataset macro MSE gain `>=0.3%`；
2. 至少3/5超过equal与static-target，macro gain各`>=0.2%`；
3. 不被`DENSE_NONLINEAR_MATCHED`复制，且没有单dataset、parameter pathology或checkpoint reversal解释；
4. same-run arms保持skill与nontrivial separation，policy不是无差别uniform或单arm collapse；
5. seed2021通过后才允许seeds2022/2023 confirmation。

只有在以下条件同时成立时，才建立未来Contribution 2的credit-assignment problem：

- fixed/equal PCSD arms有skill且存在same-run oracle/marginal headroom；
- direct policy没有利用这些headroom；
- A6 containment、numeric path、capacity control与arm under-training均不能解释失败。

此时才允许`SC2-ICC`（working hypothesis：same-forward interventional coupling credit）进入Step2-4。当前不冻结
名称、loss或claim。

## 7. Failure Attribution

- `hypothesis_false`：native skilled arms在matched stable E2E下没有crossing或adaptive headroom；返回PCSD Step2；
- `intervention_point_wrong`：pooling未改变group predictive state/Jacobian；返回Step4 redesign；
- `readout_or_head_design_wrong`：M0 containment通过但fixed arms普遍弱于A6，或field rank/activation underfit；
  只拒绝PCSD-CF-v1；
- `optimization_or_numeric_pathology`：non-finite、collapse、>100% degradation或validation reversal；返回Step6；
- `capacity_control_explains`：dense matched或random partition复制gain；不通过Contribution 1；
- `credit_assignment_problem_supported`：arms有skill/headroom而direct policy misallocates；PCSD保留并授权SC2 Step2-4。

一次direct-policy失败不能自动证明新training contribution有价值；它必须先通过最后一类归因。

## 8. Narrative Gate And Decision

[Narrative Gate: Conditional Pass] PCSD-CF把D14-A独立模型发现的coupling spectrum转写成一个共享parameter
field上的scope pooling operator，tensor path、A6 subspace与projectivity均可解释。其问题、约束和实现链条仍与
fixed-past unified multi-horizon generation直接相连。

[Self-Critique] DCT coordinate field、pooling、DeepONet-style synthesis与soft policy都不是新primitive；contiguous
partition也只有4/5 dataset evidence。若random partition或dense matched解释收益，贡献链条即失败。独立models的
D14-A crossing也不保证shared field arms仍有skill，因此Step7B是hard gate，不得用oracle evidence替代。

[Decision] `authorize_pcsd_cf_step7a_local_only`。D14-B1取消，CCRL降级；remote、test、SC2 implementation均为
false。下一步只实现PCSD-CF module与local invariants，并同步code-facing explanation。
