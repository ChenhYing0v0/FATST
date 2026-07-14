# SC1-JAPO Step 5 Theory Feasibility

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-JAPO`（Joint Atom-History Projective Operator） |
| `candidate_status` | `proposed`；theory已通过，完整Step 4-6 narrative gate尚未完成 |
| `current_step` | Step 5 complete；只授权 Step 6 method/control design |
| `a6_containment` | pass；4个$T$ cases最大误差`1.137e-13` |
| `exact_projectivity` | pass；22个prefix cases最大误差`1.172e-13` |
| `fixed_paf_increment` | pass at function-class level；存在JAPO函数不属于fixed affine PAF |
| `geometry_only_experts` | theory no-go再次成立；collapse误差`8.882e-15` |
| `continuity` | pass under dense expert-softmax contract；禁止hard top-k与atom-axis normalization |
| `optimization` | exact containment不等于可用initialization；identical experts构成严格symmetry trap |
| `controls` | uniform/history-only/atom-only/PERM/RANDOM same-bank controls已冻结 |
| `narrative_gate` | `conditional_pass_complete_contract_only`；generic MoE/nonlinear head不作claim |
| `implementation` | `false`；Step 6前不写method module、不启动remote training |
| `rollback` | Step 6无法形成可归因最小design -> Step 4；problem contract失效 -> Step 2/3 |

## 1. What We Planned To Test

Step 4只提出了一个provisional operator：在RGNB future atoms上，由history与atom geometry共同决定expert
mixture；requested $H$只选择active atoms。本轮回答五个阻断性问题：

1. JAPO是否在无dense bypass时仍严格包含任意A6 affine readout？
2. 同一个atom在不同requested prefix中是否获得完全相同的coefficient？
3. history-dependent joint gate是否真的超出fixed PAF，而非又一个可吸收的重参数化？
4. 连续性、初始化和specialization是否存在理论或数值病理？
5. 什么matched controls才能把joint mechanism与额外expert capacity分开？

本轮脚本只执行float64 algebra、autograd和metric sanity checks；没有读取dataset、validation或test，也没有
训练模型。

## 2. Candidate Tensor Contract

沿用A6 Encoder memory：

$$
M\in\mathbb R^{B\times C\times P\times D},\qquad
h=\operatorname{vec}(M)\in\mathbb R^{B\times C\times R},\quad R=PD.
$$

JAPO使用$E$个free RGNB coefficient maps：

$$
z_e=A_eh+a_e,\qquad
r_{j,e}=V_{e,j:}z_e+c_{j,e}.
$$

history context与fixed atom descriptor形成joint gate：

$$
s=G(h),\qquad \phi_j=\Phi(d_j),
$$

$$
\pi_{j,e}(h,d_j)=
\operatorname{softmax}_e\left(s^TR_e\phi_j+b_e\right),
$$

$$
\alpha_j=\sum_{e=1}^{E}\pi_{j,e}r_{j,e},\qquad
\widehat y_H=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H}.
$$

冻结的contract是：

- softmax只跨expert axis，不跨atoms；
- $d_j$只含support center、width、depth、global/local type，不含requested $H$；
- 首轮使用dense softmax，禁止hard top-k；
- 不使用atom-to-history-patch cross-attention；$G$可读取bijective flattened memory；
- primary candidate固定$E=2$，不做dataset-specific expert-count sweep。

## 3. A6 Containment Theorem

令任意A6 affine readout为

$$
y=B(Ah+a)+b.
$$

RGNB synthesis $Q\in\mathbb R^{T\times T}$为orthonormal matrix。定义

$$
V=Q^TB,\qquad c=Q^Tb.
$$

把每个JAPO expert设置为同一个coefficient map：

$$
r_{j,e}=[V(Ah+a)+c]_j\quad\forall e.
$$

因为$\sum_e\pi_{j,e}=1$，对任意有限router参数都有

$$
\alpha_j=\sum_e\pi_{j,e}r_{j,e}=[V(Ah+a)+c]_j,
$$

进而

$$
Q\alpha=QQ^T[B(Ah+a)+b]=y.
$$

[Fact] 该构造在$T\in\{16,31,96,720\}$、不同RGNB global ranks上最大误差为`1.137e-13`。
因此JAPO function class严格包含A6，而不需要`A6 output + residual`或parallel dense bypass。

[Important Boundary] 这是**存在性/function-class theorem**，不是初始化处方。把所有experts实际初始化成相同
A6 map会产生第7节的symmetry trap。

## 4. Exact Prefix Projectivity

对任意atom $j$，$r_{j,e}$和$\pi_{j,e}$只依赖$(h,d_j)$，不依赖active atom set，也不依赖requested $H$。
因此对$H_1<H_2$与$j\in\mathcal A_{H_1}$：

$$
\alpha_j^{(H_1)}=\alpha_j^{(H_2)}.
$$

inactive atoms的support不与$[0,H)$相交，所以

$$
Q_{[0,H),:}\alpha=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H}.
$$

[Fact] 22个selected prefixes中，active-only与full-atom evaluation的shared coefficient最大差为
`7.105e-15`，prefix output最大差为`1.172e-13`。paired atom-axis permutation最大差为`3.553e-14`。

[Decision] exact projectivity成立的必要实现不变量是：不得对active atoms做softmax/mean/global normalization；
否则改变$H$会改变shared atom coefficients。

## 5. Function-Class Increment And No-Go Boundary

### 5.1 Geometry-only mixture仍是fixed PAF

若gate只依赖geometry：

$$
\alpha_j=\sum_e\pi_e(d_j)\psi_e(d_j)^TA_eh,
$$

则固定operator row为

$$
W_j=\sum_e\pi_e(d_j)\psi_e(d_j)^TA_e,
$$

所以$\alpha_j=W_jh$。脚本对随机三expert构造的direct mixture与collapsed fixed map比较，最大差
`8.882e-15`。

[Decision] `JAPO-ATOM`只能作analytic control；它不能作为candidate，也不能用更多experts复活exact PAF。

### 5.2 Joint gate存在fixed affine PAF无法表示的函数

取scalar history $h$、descriptor $d=1$，两个experts为$r_1=h,r_2=-h$，router logits为$(h,-h)$。则

$$
f(h)=h\tanh(h).
$$

在$h\in\{-1,0,1\}$上：

$$
f(-1)=f(1)=\tanh(1),\qquad f(0)=0.
$$

任何affine function若两端相等，其slope必须为0，因此中心也必须等于两端，与上式矛盾。数值second
difference为`1.523188`，严格非零。

[Fact] JAPO的function class确实超出fixed affine PAF。

[Self-Critique] 这个反例只证明“存在更强函数”，不证明真实forecasting problem需要该函数，也不证明optimizer
会学到有用的joint routing。有效性仍必须由Step 7 artifacts决定。

## 6. Continuity Contract

只要$G$与$\Phi$连续，bilinear logits、dense softmax、affine expert maps与finite weighted sum都是连续函数。
在bounded input/parameter domain上，它们还是locally Lipschitz。requested $H$不进入这些函数，因此不存在
benchmark-horizon embedding或horizon-specific router边界。

随机float64 sanity check在$10^{-6}$ perturbation下得到finite local output ratios：history=`7.394`、descriptor=
`6.802`。这些数值只检查实现公式没有离散分支，不是global Lipschitz bound。

[Decision] Step 6首版继续禁止hard top-k、threshold dispatch、active-set normalization和$H$ embedding。未来若要
稀疏routing，必须作为新的design重新证明projectivity与continuity。

## 7. Optimization And Symmetry Audit

当所有expert outputs相同$r_e=r$时，mixture对router logit $\ell_e$的导数为

$$
\frac{\partial\alpha}{\partial\ell_e}=\pi_e(r_e-\alpha)=0.
$$

若router同时为uniform，experts还得到相同gradient，于是整个系统停留在对称子空间。autograd audit得到：

- identical-expert router gradient max=`0`；
- two-expert gradient pair gap=`0`；
- 打破expert symmetry后router gradient L2=`0.260915`。

2026 external audit也发现upcycled identical experts存在expert symmetry与early-specialization问题；该来源通过
external search发现，未依赖Zotero completeness：[Cluster-Aware Upcycling](https://arxiv.org/abs/2604.13508)。
其abstract与metadata已核对，但本项目的结论主要来自上述可直接证明的gradient identity，而不是外推其vision
experiments。

另一项2025/2026 OpenReview工作报告，常见load balancing可能导致expert overlap和过度uniform routing；网页
full text本轮受challenge阻断，因此只作为低置信度风险提示，不作为design依据：
[Advancing Expert Specialization for Better MoE](https://openreview.net/forum?id=iydmH9boLb)。

[Decision]

1. A6 containment只作为function-class guarantee；首版不得复制同一trained/random expert到所有branches；
2. Step 6采用from-scratch、independently initialized experts，不做warm-start/frozen replacement；
3. 首轮不加入load-balancing、orthogonality或specialization auxiliary loss；否则decoder与training contribution混淆；
4. 只记录diagnostics，不用entropy好看替代prediction与mechanism gate。

## 8. Mandatory Controls

| Arm | Same expert bank | Gate input | Purpose |
| --- | --- | --- | --- |
| `A6-LBF-natural` | no | none | accepted carrier/reference |
| `JAPO-JOINT-GEO` | yes | history + canonical geometry | complete candidate |
| `JAPO-UNIFORM` | yes | none | capacity/expert ensemble control |
| `JAPO-HISTORY` | yes | history only | sample conditioning without geometry |
| `JAPO-ATOM` | yes | geometry only | fixed-operator analytic control |
| `JAPO-JOINT-PERM` | yes | history + permuted geometry | canonical geometry necessity |
| `JAPO-JOINT-RANDOM` | yes | history + moment-matched random geometry | descriptor semantics control |

params差异继续只报告，不参与candidate选择。但`JAPO-JOINT-GEO`必须超过same-bank controls，才能排除extra
expert capacity、generic sample routing和arbitrary descriptor noise三个替代解释。

## 9. Specialization Statistics

每项定义已落到`metric_definitions.csv`：

- `normalized_gate_entropy`：$-\sum_e\pi_e\log\pi_e/\log E$的均值；只反映confidence；
- `minimum_mean_expert_usage`：各expert平均soft probability的最小值；检测soft starvation；
- `expert_output_disagreement`：expert coefficient outputs跨expert的population variance；为0时router无功能作用；
- `history_gate_sensitivity`：同atoms、不同histories的mean absolute gate difference；
- `geometry_gate_sensitivity`：同histories、不同atoms的mean absolute gate difference；
- `joint_interaction_residual`：两个histories与两个atoms的two-way probability contrast；
- `routing_effect_vs_uniform`：joint与uniform mixture在同expert outputs上的mean absolute coefficient gap。

脚本生成的synthetic non-degenerate state让全部统计为finite/nonzero，只证明统计实现可用。真实gate要求：

1. prediction相对same-bank controls通过；
2. history与geometry sensitivity均非退化；
3. expert disagreement与routing effect非零；
4. PERM/RANDOM不能解释canonical geometry收益。

不预注册任意entropy“越低越好”或usage“越均匀越好”的阈值，因为specialization与balance不是同义词。

## 10. Narrative Gate

| Criterion | Result |
| --- | --- |
| problem targets D8 failure | pass：从fixed separable map转为sample-conditional atom operator |
| A6 containment | pass：无dense bypass exact embedding |
| exact projectivity | pass：$H$只改变active set |
| strict function increment | pass：constructive non-affine witness |
| continuity | pass under frozen dense-softmax contract |
| geometry attribution | design-ready；需same-bank controls验证 |
| optimization feasibility | conditional：symmetry risk已定位，独立初始化可避开但未训练验证 |
| component novelty | fail if单独claim MoE/nonlinear decoder/geometry gate |
| complete multi-horizon contract | conditional pass；effectiveness未开始 |

[Decision] Step 5通过的对象是完整contract，不是其中任一generic primitive。JAPO的正式candidate status仍为
`proposed`，research cursor为`theory_pass / step6_design_pending`。这允许设计最小module和validation-only
protocol，不允许直接编码或远程训练。

## 11. Failure Attribution

- `hypothesis_false`: **not supported**；joint gate function class确实超出fixed PAF；
- `intervention_point_wrong`: **not shown**；coefficient operator正是D8诊断的失败点；
- `readout_or_head_design_wrong`: **risk remains**；更强class可能仍不匹配数据；
- `optimization_or_numeric_pathology`: **specific symmetry pathology found and bounded**；禁止identical init；
- `capacity_control_explains`: **major unresolved alternative**；由same-bank controls阻断。

当前不能说JAPO有效；能说它没有被algebra否决，并且已经知道什么结果才能证明完整mechanism。

## 12. Step 6 Required Design Decisions

Step 6只设计，不实现，必须冻结：

1. 最小$E=2$ expert maps的具体factorization与tensor shapes；
2. independent initialization与gate logit scale，避免symmetry/saturation；
3. canonical descriptor normalization，且证明不含requested $H$；
4. 七arms如何共享expert bank、initialization distribution与training budget；
5. validation-only first screen、primary comparisons与kill gates；
6. code-level prefix/projectivity/gradient/symmetry invariants；
7. 与A6的公平from-scratch E2E protocol；禁止frozen encoder replacement；
8. 若七arms成本过高，只能用预注册的两阶段筛选，不得删掉capacity/history/geometry attribution链。

Step 6若无法让`JOINT > UNIFORM/HISTORY/ATOM/PERM/RANDOM`形成清晰可解释的decision rule，则回Step 4；不得
因为JAPO理论更强就直接训练。

## 13. Artifact Map

- `scripts/check_stage_c_japo_step5_theory.py`：algebra、autograd与metric contract；
- `containment_checks.csv`：四个$T$的A6 embedding与atom-axis equivariance；
- `prefix_projectivity_checks.csv`：22个prefix的shared-coefficient/output equality；
- `geometry_only_collapse.json`：geometry-only fixed-operator no-go；
- `joint_noncollapse_witness.json`：strict function-class witness；
- `continuity_and_diagnostics.json`：continuity与statistic sanity checks；
- `initialization_symmetry.json`：identical-expert gradient trap；
- `control_matrix.csv`：Step 6 mandatory attribution arms；
- `metric_definitions.csv`：所有新统计的source、computation与meaning；
- `theory_gate.json`：machine-readable decision。

## 14. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 5 complete；Step 6 next |
| `problem` | fixed separable PAF保留geometry但丢失A6 operator freedom |
| `existence_evidence` | D6 support interaction；D8 geometry-positive/operator-negative |
| `idea` | history与atom geometry joint routing over free RGNB expert maps |
| `theory_check` | containment/projectivity/non-collapse/continuity pass；symmetry risk bounded |
| `design` | tensor contract与mandatory controls frozen；module/config未设计 |
| `narrative_gate` | conditional pass for complete contract only |
| `effectiveness_gate` | not started |
| `artifacts` | checker + 4 CSV + 5 JSON + this report |
| `decision` | `theory_pass_step6_design_only`；implementation/training false |
