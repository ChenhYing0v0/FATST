# SC1-PLGO Step 5 Theory Feasibility

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-PLGO` (Projective Local-Global Operator) |
| `current_step` | Step 5 complete |
| `construction` | Restricted-Global Nested Basis (`RGNB`) |
| `stable_reconstruction_gate` | pass：12个$(T,r_g)$ cases，max algebraic gap `2.141e-13` |
| `global_local_gate` | pass：root保持global DCT subspace，details为interval-local orthogonal complements |
| `prefix_restriction_gate` | pass：101个selected prefixes + 3,731个all-$H$ bound cases |
| `A6_morphism_gate` | pass：square RGNB可exact morph A6 readout |
| `function_novelty_gate` | fail：`PLGO-ONB-M0`与A6是isometric reparameterization |
| `overcomplete_union_gate` | fail as method：frame bounds $[1,2]$，但coefficient kernel维数为$r_g$ |
| `independent_group_gate` | fail as method：T720 exact containment要求rank caps sum=720，退化为full affine |
| `efficiency_claim` | withdrawn：support pruning成立，但generator-level speedup尚未成立 |
| `Step5_decision` | `partial_pass_step6_design_only` |
| `method_implementation` | `false`；只允许Step 6设计，不允许训练 |
| `rollback` | Step 6无法形成非dense-equivalent coefficient mechanism -> Step 4 redesign |

## 1. What We Tested And Why

D6已证明同一frozen-memory readout中存在稳定的support-scale crossing：local b144在short prefixes更好，
global DCT在long domain更好。Step 4据此提出PLGO，但只给出了叙事边界。Step 5需要回答：

1. global smooth subspace与interval-local supports能否组成stable、identifiable synthesis？
2. 同一组coefficients能否在任意$H$上通过domain restriction精确给出prefix？
3. 能否exact包含A6，而不靠dense bypass？
4. 如果能，是否真的产生新的predictive function，而不只是换坐标？
5. naive global/local union与independent support maps是否被overcompleteness或capacity解释？

本轮只做float64 algebra、function-class与selective-execution audit，不读取dataset、不训练model。

## 2. Restricted-Global Nested Basis

令最大future domain为$\Omega_T=\{0,\ldots,T-1\}$，global prototype matrix
$G\in\mathbb R^{T\times r_g}$取前$r_g$个orthonormal DCT-II modes。对balanced interval tree中的任意
$I=[s,e)$，定义scaling space

$$
V_I=\operatorname{span}(R_IG),\qquad
\dim V_I=\min(r_g,|I|).
$$

若$I$的children为$I_L,I_R$，则restriction关系给出

$$
V_I\subseteq V_{I_L}\oplus V_{I_R}.
$$

定义该interval的local detail space为

$$
W_I=(V_{I_L}\oplus V_{I_R})\ominus V_I.
$$

因此root scaling $V_{\Omega_T}$与全部$W_I$构成$T$维orthonormal basis $Q_T$：

$$
\mathbb R^T=V_{\Omega_T}\oplus\bigoplus_I W_I,
\qquad Q_T^\top Q_T=I_T.
$$

这个construction不是“把DCT和Haar拼起来”。global modes只在root出现一次；每个local detail是children
scaling union相对parent的orthogonal complement，所以basis保持square、complete与identifiable。
$r_g=1$时退化为balanced Haar；$r_g>1$时每个detail block消除$r_g$个global prototype moments。

### 2.1 Numeric pathology found and repaired

[Fact] 直接对$R_IG$做QR并不稳定。因为

$$
\cos(kx)=T_k(\cos x),
$$

低频DCT modes限制到短interval后形成高度ill-conditioned polynomial coordinates。T720、$r_g=16$时，
raw restricted matrices的最大condition number为`2.712e17`，131个nodes超过`1e12`；第一次constructive
run的max reconstruction gap达到`2.180e1`，该实现不能用于方向否决或通过。

修复不是numerical tolerance放宽。令$u=\cos x$，在每个interval内把$u$ affine rescale到
$v\in[-1,1]$，再用$T_0(v),\ldots,T_{r_g-1}(v)$表示同一个polynomial span。affine variable change不改变
degree-$<r_g$ polynomial space，因此不改变$V_I$，但改善coordinate conditioning。stable local chart的
最大condition number降为`1.784e3`，无node超过`1e12`。

[Decision] `stable local chart`是RGNB的必要数值实现条件，而不是可选优化。direct restricted-DCT QR被标记为
`optimization_or_numeric_pathology`，不能作为PLGO方向失败证据。

## 3. Constructive Algebra Results

测试cases为：

```text
(1,1), (2,2), (3,3), (5,4), (7,4), (16,4), (96,8),
(720,1), (720,4), (720,8), (720,16), (721,16)
```

其中tuple为$(T,r_g)$，同时覆盖non-dyadic、benchmark length、相邻prime-like length与不同global rank。

| Property | Max gap | Gate |
| --- | ---: | --- |
| orthonormality | `3.361e-15` | pass |
| nested inclusion | `2.658e-15` | pass |
| root DCT projector equality | `1.429e-15` | pass |
| detail global-prototype moment | `1.090e-15` | pass |
| support leakage | `0` | pass |
| exact A6 morphism | `2.132e-13` | pass |
| selected-prefix reconstruction | `2.141e-13` | pass |

全部值低于预注册tolerance `1e-10`。这证明RGNB algebra可行，不证明forecast effectiveness。

## 4. Exact A6 Morphism And Its No-Go Boundary

A6 future readout写成

$$
c=Ah+a,\qquad y=Bc+b,
$$

其中$B\in\mathbb R^{T\times K}$。定义RGNB coefficients

$$
\alpha=Q_T^\top Bc+Q_T^\top b,
\qquad \widehat y=Q_T\alpha.
$$

则

$$
\widehat y=Q_TQ_T^\top(Bc+b)=y.
$$

[Fact] 该morphism不需要并行dense output branch，且$B\leftrightarrow Q_T^\top B$为bijective；parameter
count与rank-$K$ function family均不变。

[Decision] `PLGO-ONB-M0`只能是mandatory control。任意fixed square orthonormal synthesis配合unrestricted
coefficient map，都只是A6的isometric coordinate transform。**RGNB的理论优美性本身不能成为
Contribution 1。**

## 5. Native Prefix Restriction

定义与prefix相交的active atoms：

$$
\mathcal A_H=\{j:\operatorname{supp}(q_j)\cap[0,H)\neq\varnothing\}.
$$

不在$\mathcal A_H$中的local atom在$[0,H)$上严格为零，因此

$$
R_HQ_T\alpha=Q_{T,[0,H),\mathcal A_H}\alpha_{\mathcal A_H}.
$$

同一份full-domain coefficients被用于所有$H$，$H$不进入learned path。所有3,731个$(T,H)$均满足保守界

$$
|\mathcal A_H|\le
\min\{T,H+r_g(\lceil\log_2T\rceil+1)\}.
$$

T720、$r_g=16$的selected counts为：

| $H$ | Active | Inactive | Active/$H$ |
| ---: | ---: | ---: | ---: |
| 1 | 102 | 618 | 102.000 |
| 48 | 131 | 589 | 2.729 |
| 96 | 176 | 544 | 1.833 |
| 144 | 205 | 515 | 1.424 |
| 192 | 266 | 454 | 1.385 |
| 336 | 369 | 351 | 1.098 |
| 512 | 549 | 171 | 1.072 |
| 720 | 720 | 0 | 1.000 |

[Boundary] 这证明support-selective synthesis，不证明end-to-end speedup。尤其$H=1$仍有102个boundary-path
atoms；若coefficient generator先生成全部720 coefficients，synthesis pruning几乎没有意义。A6本身只计算
`basis[:H] @ coeff`。在generator-level FLOPs和wall time实测前，PLGO不得claim效率优势。

[Boundary] 本轮证明的是固定$T_{max}$ future function的exact prefix restriction，不是“独立构造$Q_H$与
$Q_{T_{max}}$具有相同coefficients”的cross-length projective-family theorem；后者未建立，也不应写入论文。

## 6. Naive Global-Local Union Control

令$L\in\mathbb R^{T\times T}$为square local orthonormal basis，直接拼接
$S=[G,L]\in\mathbb R^{T\times(T+r_g)}$。则

$$
SS^\top=I+GG^\top,
$$

所以frame bounds严格为$[1,2]$，看似稳定。但对任意$a\in\mathbb R^{r_g}$：

$$
S\begin{bmatrix}a\\-L^\top Ga\end{bmatrix}=0.
$$

因此coefficient kernel维数至少$r_g$。T720、$r_g=16$的constructive audit得到：frame bounds
`1.000000-2.000000`、reconstruction gap `2.442e-15`、kernel gap `1.597e-16`，但kernel dimension=16且
global/local coherence约1。

[Decision] `PLGO-FRAME`稳定但non-identifiable，只能作overcomplete capacity control；不能作为paper method。

## 7. Independent Support-Group No-Go

将RGNB atoms按root/depth分组。T720、$r_g=16$的group sizes为

```text
[16, 16, 32, 64, 128, 256, 208]
```

若每组拥有独立history latent，且要包含全部A6 rank-$K$ operators，则第$l$组至少需要

$$
k_l\ge\min(n_l,K).
$$

当$K=256$时，rank caps之和为720，大于A6 shared latent budget 256；并且每组$n_l\le256$，所以独立
group architecture可表示任意$720\times768$ affine map，等价full affine。

[Strong Evidence] 这不是因为params更多而否决；params不参与profile选择。否决原因是mechanism
attribution：任何收益都可被full-affine function-class expansion解释。

因此以下三项不可同时获得：

1. exact包含全部A6 rank-256 family；
2. 每个support scale拥有独立history map；
3. 总latent budget仍为256且不退化full affine。

## 8. Candidate Matrix And Step 5 Decision

| Candidate | Algebra | Function boundary | Status |
| --- | --- | --- | --- |
| `PLGO-ONB-M0` | stable、exact prefix、exact A6 morph | 与A6同function class | `control_only` |
| `PLGO-FRAME` | stable frame | overcomplete、16维kernel | `control_only` |
| `PLGO-INDEPENDENT-GROUP` | stable、可包含A6 | T720退化full affine | `rejected_as_core` |
| `PLGO-ATOM-CONDITIONED-GENERATOR` | 可复用RGNB synthesis | containment、capacity与novelty尚未证明 | `provisional_step6_question` |

[Decision] Step 5为`partial_pass_step6_design_only`：

- [Fact] PLGO存在stable、square、global-local、prefix-restrictable数学骨架；
- [Fact] 该骨架没有自动产生新的forecast function；
- [Fact] 两条最直接的扩展分别被overcomplete confound与full-affine confound阻断；
- [Hypothesis] 只有让一个shared generator以atom descriptor生成active coefficients，才可能把local/global
  geometry变成真实inductive bias，同时不读取$H$；
- [Speculative] 该generator是否能优于matched dense/random-descriptor controls仍完全未知。

method implementation和remote training继续`false`。Step 6只能设计tensor contract、function controls、
parameter/FLOP matching与falsification；不能直接编码后用性能反推叙事。

## 9. Failure Attribution

本轮没有否定PLGO问题方向，但否定了三种具体claim：

1. direct restricted-DCT QR出现`optimization_or_numeric_pathology`；stable Chebyshev coordinate redesign后
   algebra通过，因此不能据初次失败拒绝方向；
2. `PLGO-ONB-M0`由`capacity_control_explains`/function equivalence解释，只能是control；
3. naive frame为`readout_or_head_design_wrong`：stable reconstruction不等于coefficient identifiability；
4. independent groups由`capacity_control_explains`阻断，不能作为local-global机制证据。

仍未测试的是shared atom-conditioned coefficient generation。若Step 6无法给出非dense-equivalent且可归因的
contract，failure是candidate-level narrative failure，rollback Step 4；不能通过Encoder、MoE或SC2 loss叠加
来挽救。

## 10. Step 6 Required Gate

1. 明确`memory [B,C,P,D] -> atom coefficients [B,C,N_H] -> output [B,C,H]`的tensor path；
2. atom descriptor只能包含support/scale/global-local geometry，不包含requested $H$或benchmark ID；
3. 同时冻结`A6 / ONB-M0 / matched dense / random descriptor / parameter-matched generator` controls；
4. 证明candidate不是full affine、不是A6 residual、不是overcomplete coefficient expansion；
5. 说明是否要求exact A6 containment；若放弃，必须给出capacity-preserving matched control与明确风险；
6. 只有generator原生跳过inactive atoms时才报告selective FLOPs；否则撤销efficiency claim；
7. Step 6 narrative gate不通过则回Step 4，禁止进入Step 7。

## 11. Artifacts And 11-Step Record

脚本`scripts/check_stage_c_plgo_step5_theory.py`生成：

- `basis_checks.csv`：12个basis cases的orthogonality、nesting、subspace、support与morphism gaps；
- `prefix_checks.csv`：101个selected prefix的active counts、bound与reconstruction gap；
- `active_bound_checks.csv`：每个case的all-$H$ bound结果；
- `conditioning_checks.csv`：raw/stable local coordinates的condition numbers；
- `frame_control_checks.csv`：overcomplete union的frame、kernel与identifiability；
- `function_class_budget.json`：T720 group rank caps、parameter ratios与full-affine no-go；
- `candidate_matrix.csv`：candidate/control boundary；
- `theory_gate.json`：machine-readable Step 5 decision。

| Field | Record |
| --- | --- |
| `current_step` | Step 5 complete；Step 6 design gate next |
| `problem` | local-prefix support与global-domain coherence需要同一horizon-agnostic future operator |
| `existence_evidence` | D6 disjoint-window crossing + Step 4 source audit |
| `idea` | RGNB stable co-synthesis + future shared coefficient mechanism |
| `theory_check` | synthesis/prefix/A6 morph pass；function novelty与capacity attribution未过 |
| `design` | RGNB冻结为数学scaffold；actual generator未冻结 |
| `narrative_gate` | partial；只进入Step 6 |
| `effectiveness_gate` | not started |
| `artifacts` | 6 CSVs、2 JSONs、protocol、code explanation、本报告 |
| `decision` | `partial_pass_step6_design_only`；training unauthorized |
