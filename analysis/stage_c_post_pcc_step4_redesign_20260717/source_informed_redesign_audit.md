# Post-PCC Step4 Source-Informed Redesign Audit

## Current Position

| Field | Value |
| --- | --- |
| `current_step` | 11-step Step4 source-informed redesign complete；Step5 theory feasibility next |
| `problem` | arm skill recovery与coupling-scope specialization在current shared field/same-label objective中冲突 |
| `existence_evidence` | PCC 25/25 arm recovery，但minimum diversity retention仅20.57%-41.13% |
| `active_exact_candidates` | PCSD-CF-v1 rejected；PCC-v1-TI validation-screen fail |
| `provisional_redesign` | scale-indexed forecast field + measure-constrained competitive credit |
| `narrative_gate` | not ready；需Step5/6证明non-absorbability、projectivity与prior-art boundary |
| `implementation/remote/test` | false / false / false |

## External Search Record

- search date：2026-07-17；
- source policy：external-first，只使用primary paper/project pages；Zotero coverage未用于novelty判断；
- query scope：expert homogenization/specialization、forecasting expert loss、structure-guided time-series MoE、balanced
  assignment/Sinkhorn routing、heterogeneous experts；
- all key items below were discovered or refreshed by external search；未把“Zotero中缺失”当作novelty evidence。

## What Existing Work Blocks

1. forecasting中的direct expert loss已由[Expert Loss Integration](https://arxiv.org/abs/2605.10330)明确覆盖；其论文还用
   masking把experts绑定到不同data subsets。我们的`EQUAL_SKILL`必须作为prior/control，不能成为Contribution 2；
2. [AME-TS](https://arxiv.org/abs/2605.25166)已指出standard MoE specialization弱可辨识，并用forecastability、
   seasonality、trend与sparsity构造training-only structural prior；“结构先验引导router”本身不新；
3. [MoHETS](https://arxiv.org/abs/2601.21866)已用convolution/Fourier heterogeneous experts提供不同time-series
   inductive biases；“不同类型experts”本身不新；
4. [Advancing Expert Specialization](https://openreview.net/forum?id=iydmH9boLb)已经用orthogonality与routing variance
   对抗expert overlap；[GatePro](https://openreview.net/forum?id=X4U5ZUB6bY)也直接针对相似expert co-activation。
   因此直接添加output decorrelation、orthogonality或variance loss不仅叙事弱，也可能制造与forecast risk无关的差异；
5. [BASE](https://proceedings.mlr.press/v139/lewis21a.html)已把token-expert routing写成balanced assignment；
   [Selective Sinkhorn Routing](https://arxiv.org/abs/2511.08972)进一步覆盖entropy-regularized OT balancing。
   Sinkhorn/OT或equal marginal本身不能成为novelty。

## Code-Theory Diagnosis

current PCSD-CF先从history得到一份共享

$$
M(h)\in\mathbb R^{D_q\times K},
$$

然后每个scope $s$只通过pooled future coordinates $\bar d_{s,g}$读取同一$M(h)$，并共享
`identity_synthesis/nonlinear_synthesis/temporal_bias`。因此scope不是独立expert parameters，而是同一field的不同
pooling view。

在L1 proper loss下，如果所有arms都对同一$Y_t$接受大范围direct supervision，population optimum都趋向同一
conditional median。PCC的harmonic transport又把多个$H\ge t$的capability平均回$t$，进一步平滑scope差异。这解释了：

- `EQUAL_SKILL`获得大部分performance gain；
- full PCC把arms训练到fixed-run skill附近；
- 但arm outputs只保留plain separation的20%-41%；
- shared-gradient cosines多数显著为正；
- router虽然balanced，却难以利用已经缩小的function differences。

[Strong Evidence] 该冲突不是简单调$lambda$、floor或temperature就能可靠解决；training objective无法凭空创建
architecture中缺失的scope-conditioned history degrees。

## Provisional Redesign Pair

### A. Architecture: scale-indexed forecast field

不再让所有scopes共享唯一$M(h)$，而把internal coupling scale

$$
z_s=\frac{\log s}{\log T}
$$

作为**decoder内部结构坐标**，生成平滑的scope-conditioned modes：

$$
M_s(h)=\sum_{q=1}^{Q}\phi_q(z_s)M_q(h).
$$

然后仍用scope-specific pooled future coordinates读取$M_s(h)$并生成完整$T=720$ forecast；requested horizon继续只做
prefix crop，不进入$M_s$、router或decoder computation。

该方向的价值不在“多几个experts”，而在：

1. $Q=1$必须精确退化为current PCSD field；
2. $Q>1$提供可辨识、连续共享的coupling-scale history operators，而不是five independent models；
3. scale是output dependency granularity，不是benchmark/requested horizon；
4. full-domain generation与projectivity保持；
5. 必须通过matched parameter/rank controls证明不是简单扩宽。

[Risk] 若$M_s$ factorization能被一个更宽generic field吸收，或只等价于standard heterogeneous experts，则narrative
gate失败。Step5必须先做function-class与containment proof。

### B. Training: measure-constrained competitive credit

current floor在每个$t$给所有arms持续same-label credit，防starvation但促同质化。provisional替代是：先从dense-prefix
risk得到detached target-coordinate capability cost，再在projective target measure $\omega_t$下分配**有限skill mass**：

$$
A^*=\arg\min_{A\ge0}
\mathrm{KL}(A\|\omega\,c)
\quad\text{s.t.}\quad
\sum_s A_{ts}=\omega_t,
\quad
\sum_t A_{ts}=\rho_s.
$$

$\rho_s$不是compute load，而是每个coupling scope在一个batch中获得的minimum/target skill budget。与uniform per-target
floor不同，它可以保证每个arm总体不饿死，同时允许不同target coordinates形成竞争性specialization。skill loss与router
teacher都由同一个$A^*$得到，inference graph不变。

该候选不能宣称OT novelty。允许的完整claim只可能是
`dense prefix-risk measure -> natural-target capability -> coupling-scope skill-mass constraints -> one-stage projective decoder
training`。mandatory controls包括BASE/SSR-style generic balanced assignment、equal skill、pointwise capability、current PCC
与generic diversity regularization。

[Risk] equal $\rho_s$可能强迫无用scope；capability-derived $\rho_s$可能重新collapse；mini-batch Sinkhorn可能引入
moving-target/numeric pathology。Step5必须先证明feasible marginals、conservation、stable gradients与nontrivial allocation。

## Why The Pair Fits The Paper Mainline

论文仍研究`fixed-past unified multi-horizon generation`：同一past生成一个完整future function，不输入requested $H$。

- Contribution 1候选回答：如何让一个decoder在连续coupling scale上拥有可辨识、可共享、projective的forecast
  operators；
- Contribution 2候选回答：如何把all-prefix forecast measure转化为有限、竞争且不饿死arms的training credit；
- 两者共同处理PCC实验暴露的同一矛盾：没有architecture identifiability，training只会同质化；没有credit budget，
  identifiable arms又可能重新starve。

这比“再加diversity loss”更贴近multi-horizon核心，也保留decoder + training strategy的两项贡献结构。

## Step5 Entry Gate

下一步只做theory feasibility，不实现production method：

1. 证明$Q=1$ exact containment、任意$H$ prefix projectivity、$Q>1$ non-absorbability或明确其边界；
2. 给出parameter-matched factorization，禁止靠active parameter增量解释；
3. 证明allocation row/column marginals、projective measure conservation与finite Sinkhorn path；
4. 构造synthetic crossed case，要求competitive allocation同时优于uniform floor的homogenization与unbalanced hard
   assignment的starvation；
5. 冻结`old/new architecture × equal/new credit`的$2\times2$ factorial，分别验证两项贡献和interaction；
6. Step5任一核心证明失败则回Step4，不进入Step7A或remote。

decision=`step4_redesign_pair_provisional_step5_theory_only`。
