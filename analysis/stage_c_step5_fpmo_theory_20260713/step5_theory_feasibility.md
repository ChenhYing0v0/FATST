# SC1-FPMO Step 5 Theory Feasibility

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-FPMO` |
| `current_step` | Step 5 complete |
| `algebra_gate` | pass：9个长度、53个prefix cases，max gap `5.329e-14` |
| `embedding_gate` | pass：存在无dense bypass的exact A6 morphism |
| `restriction_gate` | pass：只计算support与prefix相交的atoms即可exact恢复prefix |
| `irregular_length_gate` | pass：`T=1,2,3,5,7,16,96,720,721`均成立 |
| `scale_native_gate` | partial：独立scale maps可包含A6，但在T720退化为full-affine capacity class |
| `efficiency_claim` | fail/withdrawn：native restriction不等于比A6更快 |
| `Step5_decision` | `partial_pass_step6_design_only` |
| `method_implementation` | `false`；Step 6 narrative/control gate前不得实现或训练 |
| `rollback` | Step 6无法隔离structure与capacity -> Step 2/4，不进入Step 7 |

## 1. Question And Source Boundary

Step 5需要同时验证四件事：

1. FPMO是否能在不保留一条dense A6 output branch的前提下精确包含A6？
2. requested $H$是否只改变active output domain，且native computation等于full prediction的restriction？
3. 构造是否适用于任意$T$，而不是依赖720的质因数？
4. 引入scale-native history maps后，方法是否仍有独立、可归因的architecture content？

本轮使用[Unbalanced Haar](https://doi.org/10.1198/016214507000000860)的arbitrary-interval orthonormal
construction作为数学来源之一。该工作已明确提出support break可不位于中点的orthonormal Haar-like basis；
本项目只采用其interval-normalization思想，并固定使用data-independent balanced midpoint，不采用其
data-adaptive basis selection。结合[The Lifting Scheme](https://doi.org/10.1137/S0036141095289051)，这些
来源证明perfect-reconstruction interval transform是已有工具，不是FPMO的novelty。

## 2. Universal Interval Transform

### 2.1 Definition

令future domain为$\Omega_T=\{0,\ldots,T-1\}$。对任意interval
$I=[s,e)$，若$|I|>1$，在$m=s+\lfloor |I|/2\rfloor$分为$I_L=[s,m)$与
$I_R=[m,e)$；记$n_L=|I_L|$、$n_R=|I_R|$、$n=n_L+n_R$。

root scaling atom为

$$
\phi_T(t)=\frac{1}{\sqrt T}\mathbf 1_{[0,T)}(t).
$$

每个internal interval的detail atom为

$$
\psi_I(t)=
\sqrt{\frac{n_R}{n_Ln}}\mathbf 1_{I_L}(t)
-\sqrt{\frac{n_L}{n_Rn}}\mathbf 1_{I_R}(t).
$$

按breadth-first depth排列$phi_T$与全部$psi_I$，得到analysis matrix
$Q_T\in\mathbb R^{T\times T}$。每个row的support由其interval确定，与dataset、requested horizon及
benchmark horizons无关。

### 2.2 Lemma: orthonormality and perfect reconstruction

对任意detail atom：

$$
\|\psi_I\|_2^2
=n_L\frac{n_R}{n_Ln}+n_R\frac{n_L}{n_Rn}=1,
$$

且其interval sum为0，所以与parent scaling正交。两个detail supports若disjoint则内积为0；若nested，
较大atom在较小support上为常数，而较小atom的sum为0，因此仍正交。共有一个scaling atom与$T-1$个
internal-node details，故形成$\mathbb R^T$的orthonormal basis：

$$
Q_TQ_T^\top=Q_T^\top Q_T=I_T.
$$

这给出对任意正整数$T$的perfect reconstruction，不需要padding到$2^k$，也不需要
`90/30/10/5/1` factorization。

## 3. Exact A6 Embedding

A6 normalized operator写为

$$
c=Ah+a,\qquad y=Bc+b,
$$

其中$h\in\mathbb R^R$、$c\in\mathbb R^K$、$B\in\mathbb R^{T\times K}$。定义

$$
\bar B=Q_TB,\qquad \bar b=Q_Tb,
$$

并令FPMO coefficients与synthesis为

$$
\alpha=\bar Bc+\bar b,\qquad \widehat y=Q_T^\top\alpha.
$$

于是

$$
\widehat y=Q_T^\top Q_T(Bc+b)=Bc+b=y.
$$

[Fact] 这是无dense bypass的parameter morphism：model只保存/学习$ar B,\bar b$并经tree synthesis输出；
$B,b$不需要作为并行prediction branch存在。映射$(B,b)\leftrightarrow(\bar B,\bar b)$是bijective，故
`FPMO-M0`与A6有完全相同的function class和parameter count。

[Decision] `FPMO-M0`通过capacity-preservation gate，但只能是`control_only`。可逆坐标变化本身没有新增
predictive function，不能单独成为Contribution 1。

## 4. Native Prefix Restriction

对requested horizon $H\le T$定义active atom set

$$
\mathcal A_H=\{j:\operatorname{supp}(Q_{T,j:})\cap[0,H)\ne\varnothing\}.
$$

若$j\notin\mathcal A_H$，则对任意$t<H$都有$Q_{T,jt}=0$，所以

$$
R_HQ_T^\top\alpha
=Q_{T,[0,H),\mathcal A_H}^\top\alpha_{\mathcal A_H}.
$$

因此inactive coefficient不参与prefix，可以完全不生成。balanced interval tree中，一个prefix可分解为
有限个complete left subtrees加一条boundary path，所以active atoms为$O(H+\log T)$；脚本同时审计了
保守上界`2H+ceil(log2(T))`，并对上述9个长度的全部1,571个$H$通过bound check。

T720实测active coefficient counts：

| H | Active atoms | Active/H | Inactive atoms |
| ---: | ---: | ---: | ---: |
| 1 | 10 | 10.000 | 710 |
| 48 | 55 | 1.146 | 665 |
| 96 | 103 | 1.073 | 617 |
| 192 | 199 | 1.036 | 521 |
| 336 | 339 | 1.009 | 381 |
| 720 | 720 | 1.000 | 0 |

[Boundary] 该结果证明native restriction，不证明inference speedup。A6本身已直接计算`basis[:H] @ coeff`，
代价为$O(KR+HK)$；FPMO还需要tree/scale latent computation。论文不得使用“sublinear forecasting”或
“一定快于A6”的claim。

## 5. Scale-Native Extension

### 5.1 Direct-scale factorization

按tree depth把coefficient indices分组$J_\ell$，组大小$n_\ell$。A6到该组的effective affine map为

$$
G_\ell=(Q_TB)_{J_\ell,:}A,
\qquad
g_\ell=(Q_TB)_{J_\ell,:}a+(Q_Tb)_{J_\ell}.
$$

由于$\operatorname{rank}(G_\ell)\le\min(n_\ell,K)$，取
$k_\ell=\min(n_\ell,K)$，总存在factorization

$$
G_\ell=D_\ell A_\ell,
\quad
A_\ell\in\mathbb R^{k_\ell\times R},
\quad
D_\ell\in\mathbb R^{n_\ell\times k_\ell}.
$$

于是scale-native model可写为

$$
z_\ell=A_\ell h,
\qquad
\alpha_{J_\ell}=D_\ell z_\ell+g_\ell,
\qquad
y=Q_T^\top\alpha.
$$

每个$A_\ell$直接读取ordered flattened memory $h=\operatorname{vec}(M)$，不再通过parent-to-child
transition。对A6 effective map逐组SVD即可构造exact embedding；from-scratch模型也不要求先存在A6
checkpoint，因为$k_\ell$是architecture capacity上界。

### 5.2 Function-space result

独立$A_\ell$解除A6“所有future rows共享同一个K-dimensional history row space”的约束，因此function
class严格包含A6。但T720、$K=256$时，group sizes为：

```text
[1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 208]
```

每组均满足$n_\ell\le K$，所以$k_\ell=n_\ell$，`FPMO-DS`可表示任意
$720\times768$ affine map；它在function-class层面等价于full dense affine head，而不再只是rank-256
A6的保守扩展。

| Readout | Parameters | Relative to A6 | Function class at T720 |
| --- | ---: | ---: | --- |
| A6 rank-256 | 381,904 | 1.000 | rank≤256 affine |
| direct atom affine | 553,680 | 1.450 | full affine |
| independent scale factor | 684,326 | 1.792 | full affine with redundant factorization |

[Strong Evidence] `FPMO-DS`数学可行，但任何收益都可能由full-affine capacity解释。用户此前明确要求params
差异不作为超参数选择标准；这里也不以参数多否定方案。问题在于**mechanism attribution**：必须用同函数类
dense control证明结构本身有效。

## 6. Capacity-Containment No-Go Result

考虑没有shared latent path、每个scale拥有独立$k_\ell$维history state的direct-scale architecture。若要
包含所有A6 rank-$K$ operators，则对每个组必须满足

$$
k_\ell\ge\min(n_\ell,K),
$$

因为存在A6参数使该block达到最大rank。故必要条件为

$$
\sum_\ell k_\ell
\ge\sum_\ell\min(n_\ell,K).
$$

T720时右侧为720，而A6 budget $K=256$。因此以下三项不能同时满足：

1. 包含全部A6 function family；
2. 每个scale拥有独立、无shared path的history map；
3. 总scale latent budget仍为256。

可选代价只有：

- 共享A6 latent：capacity保留，但退化为`FPMO-M0`坐标变换；
- 独立scale maps：获得scale-native能力，但扩展到full-affine并产生capacity confound；
- shared latent + scale adapters：更省参数，但成为A6-plus-complement/residual-style路线，当前不作为paper core。

[Decision] 这是Step 5最关键的self-critique。不能把“exact containment、独立scale states、同capacity”同时
写进论文method claim。

## 7. Optimization Equivalence Boundary

`FPMO-DA`令$\alpha=Gh+g$，$y=Q_T^\top\alpha$。由于$Q_T$正交，它与direct dense
$Wh+q$通过$W=Q_T^\top G$一一对应。在full-horizon L2、matched transformed initialization、plain SGD与
isotropic weight decay下，两者只是isometric reparameterization，training dynamics可对应；Adam等
coordinate-wise optimizer可能打破该等价，但这属于optimization geometry，不是新增function class。

因此：

- `FPMO-DA`必须是matched capacity/optimizer control；
- 如果FPMO只在Adam下优于dense，论文claim必须收紧为optimization-coordinate effect；
- 如果`FPMO-DS`不能超过`FPMO-DA`，scale-native factorization没有独立支持。

## 8. Candidate Decisions

| Candidate | Status | Decision |
| --- | --- | --- |
| `FPMO-M0` | `control_only` | exact A6 morphism；验证function preservation与prefix algebra |
| `FPMO-DA` | `control_only` | full-affine atom head；隔离orthogonal-coordinate与capacity effect |
| `FPMO-DS` | `partial_pass` | exact containment + scale-native maps成立；capacity attribution待Step 6 |
| `FPMO-SA` | `rejected_by_narrative_gate` | shared A6 latent + scale adapters属于当前不采用的residual-style core |

[Decision] Step 5是`partial_pass_step6_design_only`，不是`narrative_ready`。允许Step 6设计tensor contract、
capacity controls和falsification；不允许Step 7 implementation/training。

## 9. Step 6 Required Gate

1. 冻结`FPMO-M0 / FPMO-DA / FPMO-DS / A6`四臂的function-class关系和parameter/FLOP报告；
2. 给出FPMO-DS不依赖horizon/query/router的实际tensor path；
3. 证明prefix inference不构造inactive atom coefficients，并诚实报告全部scale-latent overhead；
4. 预注册kill rule：`FPMO-DS <= FPMO-DA`则scale-native architecture claim失败；
5. 预注册capacity rule：只超过A6、未超过full-affine control不能成为Contribution 1；
6. MIPR继续held；不得用training objective帮助FPMO先过architecture gate。

若Step 6无法提出不被dense-equivalence解释的机制边界，FPMO应回滚Step 2/4，而不是进入实现。

## 10. Verification And Artifacts

`scripts/check_stage_c_fpmo_step5_theory.py`生成：

- `transform_checks.csv`：每个$T$的orthogonality、embedding、factorization和prefix max gap；
- `prefix_checks.csv`：每个$(T,H)$的active/inactive atoms与restriction gap；
- `scale_group_checks.csv`：每个depth group的atom count、actual rank与containment rank cap；
- `parameter_budget.json`：T720 function-space/parameter/no-go quantities；
- `theory_gate.json`：Step 5 machine-readable decision。

全部53个algebraic prefix cases与1,571个active-atom bound cases通过；float64最大误差
`5.3290705182007514e-14`，低于预注册tolerance `1e-10`。这只验证algebra，不构成forecast
effectiveness evidence。

## 11. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 5 complete；Step 6 design gate next |
| `problem` | exact containment与scale-native independence之间存在capacity trilemma |
| `existence_evidence` | Step4 operator audit + Step5 constructive algebra/no-go analysis |
| `idea` | arbitrary-length orthogonal interval morphism + direct-scale history factors |
| `theory_check` | embedding/restriction/irregular-T pass；efficiency claim withdrawn；capacity confound open |
| `design` | M0/DA controls与DS candidate分离；尚无model implementation |
| `narrative_gate` | partial；Step 6必须隔离dense-equivalence与capacity |
| `effectiveness_gate` | not started |
| `artifacts` | 3 CSVs、2 JSONs、candidate matrix、本报告、verification script |
| `decision` | `partial_pass_step6_design_only`；training unauthorized |
