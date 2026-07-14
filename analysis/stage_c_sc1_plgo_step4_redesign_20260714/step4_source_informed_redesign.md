# SC1-PLGO Step 4 Source-Informed Intervention/Readout Redesign Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate_id` | `SC1-JAPO`（provisional）：Joint Atom-History Projective Operator |
| `current_step` | Step 4 complete；仅授权进入 Step 5 theory feasibility |
| `exact_PAF` | 关闭；不做 width、epoch 或 optimizer sweep 复活 |
| `flatten_boundary` | `memory [B,C,P,D] -> h [B,C,PD]`是 bijection，不构成信息压缩 |
| `patch_retrieval` | 不授权；缺少 history-patch 与 future-atom 的 canonical correspondence，且 B14 为负 |
| `geometry_only_experts` | theory no-go：固定总 rank 时可吸收到单一 PAF；扩 rank 时由 capacity control 解释 |
| `retained` | RGNB、local/global support geometry、atomwise restriction、domain-only requested $H$、from-scratch E2E fairness |
| `new_mechanism` | history 与 atom geometry 共同决定 coefficient operator；不得只由 geometry 或 $H$ routing |
| `method_implementation` | `false`；Step 5 theorem 与 Step 6 controls/narrative 未通过前不编码、不训练 |
| `rollback` | Step 5 无法同时满足 A6 containment、joint non-collapse 与 exact projectivity -> Step 2/3 problem reformulation |

## 1. What We Planned To Test

D8 已经稳定否定 exact shared-latent PAF，但保留了两个正向事实：

1. canonical RGNB geometry 相对 PERM/RANDOM controls 在两种 width 下均为 5/5 datasets 正向；
2. 同一 PAF 相对 same-run A6 平均退化 `28.10%`，near-budget width 只回收 `0.58%`。

因此 Step 4 需要区分四个问题：

1. 失败来自 flatten、$R\to256$ bottleneck，还是 descriptor-generated separable readout？
2. 直接恢复 patch-level tensor 是否能修复，还是会复活未被支持的 atom-to-patch retrieval？
3. geometry-conditioned multi-branch/MoE 是否真的扩展 function class？
4. 在 2026 prior art 下，什么边界仍可构成 multi-horizon unified forecasting 的 decoder contribution？

本轮只做 source、algebra、tensor-path 与 control audit；不读取 test，不训练模型。

## 2. D8 Failure Boundary

### 2.1 Shape is not the compression

A6 Encoder 输出

$$
M\in\mathbb R^{B\times C\times P\times D},\qquad
h=\operatorname{vec}(M)\in\mathbb R^{B\times C\times R},\quad R=PD.
$$

只要 reshape 顺序固定，`vec` 有精确逆映射，所以 patch identity 没有在 `[P,D] -> [R]` 时丢失。
真正的共享瓶颈发生在后续：

$$
z=Ah\in\mathbb R^K,\qquad K=256.
$$

A6 使用自由 temporal table $B\in\mathbb R^{T\times K}$：

$$
\widehat y=Bz.
$$

PAF 则使用 descriptor trunk $\Psi_\theta(D)\in\mathbb R^{T\times K}$ 与 RGNB synthesis $Q_T$：

$$
\widehat y=Q_T\Psi_\theta(D)z.
$$

[Strong Evidence] D8 的 matched width 没有修复 gap，说明问题不是简单“参数太少”；更直接的差异是自由
temporal rows 被强制落入一个由共享 descriptor network 生成的受限、难优化 family。

### 2.2 Why patch-level retrieval is not the immediate answer

保留 $M[B,C,P,D]$ 只改变 tensor organization，不自动定义 future atom $j$ 应读取哪个 history patch $p$。
history patch 坐标属于观察域，RGNB atom support 属于未来域；二者没有无需学习即可成立的 index alignment。

若直接令 atom query 对 history patches 做 cross-attention，则新增的是

$$
\text{future atom }d_j\rightarrow\text{history patch retrieval},
$$

而不是恢复被 flatten 丢失的信息。B14 对该需求只有 `1/6` settings、`0/3` datasets 通过；OFormer、GNOT、
BasisFormer 与 TimePerceiver 也已覆盖 query-to-input attention primitive。

[Decision] patch tensor仍可保留给 diagnostics 或 shared history context，但 Step 4 不授权 atom-specific
patch attention。若未来重新提出，必须先获得新的 cross-domain alignment evidence。

## 3. Fresh External Source Audit

- search date: 2026-07-14；
- queries/topics: DeepONet branch-trunk error、nonlinear operator decoder、operator MoE、geometry-conditioned
  routing、time-series MoE、step-specific multi-step representations、patch/query decoder；
- source policy: external primary sources与official code优先；Zotero未用于completeness判断；
- coverage boundary: SRP official anonymous code入口可定位但网页工具无法稳定展开；只使用OpenReview PDF与
  submission metadata，不据此评价未核对的实现细节。

| Work | Verified mechanism | Pressure on StageC |
| --- | --- | --- |
| [DeepONet error decomposition (2026)](https://arxiv.org/abs/2602.21910) | 大 inner dimension 下 branch/coefficient error可主导；shared branch相对stacked branch改善small-mode generalization，但存在detrimental inter-mode coupling | 不能把“更多独立branches”默认写成修复；必须同时控制共享带来的generalization与coupling |
| [GNOT](https://openreview.net/forum?id=JomvpMQ6NF) + [official code](https://github.com/thu-ml/GNOT/blob/master/models/mmgpt.py) | output geometry通过gate混合expert MLP，同时用cross-attention读取输入functions | geometry-gated experts与query-input interaction均不是新primitive |
| [OFormer](https://arxiv.org/abs/2205.13671) + [official code](https://github.com/BaratiLab/OFormer) | arbitrary output queries通过cross-attention读取输入function samples | atom-to-patch cross-attention的novelty与problem evidence均不足 |
| [MIONet](https://arxiv.org/abs/2202.06137) | 多branch与trunk通过tensor product学习multiple-input operators | multi-branch/tensor-product本身不是贡献边界 |
| [MoNO](https://arxiv.org/abs/2404.09101) | mixture of neural operators提供distributed approximation theorem | operator mixture/MoE不是新机制 |
| [Mixture of neural operator experts](https://arxiv.org/abs/2502.04562) | spatially conditioned experts用于boundary/domain decomposition | spatial/geometry-conditioned operator experts已有直接先例 |
| [NOMAD](https://arxiv.org/abs/2206.03551) | nonlinear latent-location decoder替代DeepONet线性内积 | joint nonlinear decoder不能单独作为novelty |
| [SRP/SRP++](https://openreview.net/forum?id=BiMmCbxKOS) | 用step-specific low-rank adaptation缓解step-invariant representation bottleneck | “不同未来步需要不同representation”已被直接提出；本项目不得claim首次，也不能退回per-step/per-horizon adapters |
| [AME-TS](https://arxiv.org/abs/2605.25166) | series structural descriptors形成expert prior并指导token routing | interpretable structure-guided time-series MoE已拥挤 |
| [MoHETS](https://arxiv.org/abs/2601.21866) | route history patches到heterogeneous experts，并用convolutional patch decoder支持arbitrary horizons | patch routing、heterogeneous experts与arbitrary-horizon decoder的组合已有强压力 |

[Decision] JAPO 的可辩护边界不能是 nonlinear decoder、MoE、geometry gate、patch routing、step-specific
representation 中任何单项。它只能来自完整 contract：

> 在 RGNB future atoms 上，由同一 history sample 与 atom support geometry 联合决定 coefficient operator；
> requested $H$只选择active atoms，任何共享atom在不同prefix下保持同一coefficient。

## 4. Geometry-Only Multi-Branch No-Go

考虑最自然的 geometry-only expert proposal：

$$
r_{j,e}=\psi_e(d_j)^TA_eh,\qquad
\alpha_j=\sum_{e=1}^{E}\pi_e(d_j)r_{j,e}.
$$

定义

$$
\widetilde\psi(d_j)=
\begin{bmatrix}
\pi_1(d_j)\psi_1(d_j)\\
\vdots\\
\pi_E(d_j)\psi_E(d_j)
\end{bmatrix},\qquad
\widetilde A=
\begin{bmatrix}
A_1\\ \vdots\\ A_E
\end{bmatrix}.
$$

则

$$
\alpha_j=\widetilde\psi(d_j)^T\widetilde Ah.
$$

这仍是单一 PAF：

- 若固定总 latent rank，multi-branch 不扩展 function class；
- 若每个expert都保留完整rank，扩展只来自更高rank/更多capacity；
- geometry-only routing不产生sample-specific operator，仍是固定linear history-to-atom map。

[Strong Evidence] 因此“给不同scale/atom配不同branch”不能直接进入Step 5；D2的depth grouping负证据也不需要
被重新解释成MoE正证据。

## 5. Provisional Candidate: SC1-JAPO

### 5.1 Core idea

`Joint Atom-History Projective Operator (JAPO)`只在routing同时依赖history与atom时成立。令：

- $M\in\mathbb R^{B\times C\times P\times D}$；$h=\operatorname{vec}(M)\in\mathbb R^R$；
- expert latent $z_e=A_eh\in\mathbb R^K$；
- free RGNB atom table $V_e\in\mathbb R^{T\times K}$；
- expert coefficient $r_{j,e}=V_{e,j:}z_e$；
- shared history context $s=G(h)$ 与 fixed atom geometry $\phi_j=\Phi(d_j)$；
- joint gate

$$
\pi_{j,e}(h,d_j)=\operatorname{softmax}_e\left(s^TR_e\phi_j+c_e\right).
$$

最终

$$
\alpha_j=\sum_{e=1}^{E}\pi_{j,e}(h,d_j)r_{j,e},
$$

$$
\widehat y_H=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H},\qquad
\mathcal A_H=\{j:\operatorname{supp}(q_j)\cap[0,H)\ne\varnothing\}.
$$

### 5.2 Why this addresses D8 without explicit horizon conditioning

1. gate依赖$(h,d_j)$，不能再吸收到一个fixed descriptor table；operator随sample改变；
2. requested $H$不进入$G,\Phi,R_e,A_e,V_e$，只构造active set；
3. 每个atom独立于其他active atoms计算，没有active-set softmax，因此prefix subset不改变shared atoms；
4. free $V_e$避免再次强迫完整temporal table由descriptor MLP记忆；
5. 对任意A6 factorization，可把所有experts设置为同一A6-equivalent RGNB map；任意convex gate后仍精确复现A6，
   因而理论上无需dense bypass即可包含A6 function。

[Hypothesis] JAPO把D8暴露的“固定separable operator”改成history-conditioned atom operator，同时保留RGNB已验证
geometry。但目前还没有证据证明joint routing比任意generic nonlinear head更好，所以只进入theory gate。

### 5.3 Patch interface decision

首个theory contract继续允许$G$读取bijective $h$，不引入atom-specific patch retrieval。代码实现若进入
Step 7，应保留原始$M$以生成diagnostics，并允许一个**对所有atoms共享**的ordered-patch context作为control；
不得让每个atom单独attention到patches，除非新的Step2/3证据先通过。

## 6. Candidate Queue Audit

| Candidate | Decision | Reason |
| --- | --- | --- |
| free RGNB M0 | `control_only` | exact A6 reparameterization；geometry不形成可识别mechanism |
| free A6 + geometry residual/correction | `rejected_core` | dense path可单独解释performance，且违反当前非residual边界 |
| geometry-only branch mixture | `rejected_by_theory` | 可代数吸收到更宽PAF；固定rank无新class，扩rank有capacity confound |
| nonlinear $F(h,d_j)$ MLP | `deferred/control` | 可解除separability，但NOMAD/SRP overlap强，geometry必要性难识别 |
| atom-to-patch cross-attention | `rejected_current_problem` | B14负证据 + OFormer/GNOT/BasisFormer/TimePerceiver prior art |
| `SC1-JAPO` joint gate | `theory_pending` | 唯一同时保留projectivity、A6 containment并真正打破fixed separability的候选 |

## 7. Narrative Gate

| Criterion | Result |
| --- | --- |
| clear problem motivation | pass：D8给出5-dataset E2E geometry-positive/operator-negative证据 |
| mechanism targets diagnosed failure | pass：joint gate直接解除fixed history-atom separability |
| tensor/gradient path | provisional pass：`h -> expert maps`与`(h,d_j) -> gate -> alpha_j -> RGNB` |
| novelty after latest sources | conditional：generic primitives拥挤，只保留task-specific complete contract |
| continuity/projectivity | provisional：atomwise、无$H$；需Step5 formal proof |
| capacity attribution | pending：free expert bank、uniform/history-only/atom-only controls必须冻结 |
| patch-level justification | fail for retrieval；shared context only |

[Decision] Step 4完成，`SC1-JAPO = source_informed_candidate / theory_pending`。这不是`narrative_ready`，
不授权Step 6 implementation design或Step 7 training。

## 8. Step 5 Required Proofs And Kill Gates

1. **A6 containment**：构造有限参数映射，使任意A6在无dense bypass下由JAPO精确表示；
2. **projectivity**：对任意$H_1<H_2$与shared active atom，coefficient完全相同；requested $H$不进入learned path；
3. **non-collapse**：证明history-dependent joint gate不能吸收到fixed PAF；atom-only gate必须作为analytic no-op/control；
4. **continuity**：$h$或$d_j$微小变化不能产生离散horizon/router跳变；首轮禁止hard top-k；
5. **capacity controls**：uniform gate、history-only gate、atom-only gate、PERM/RANDOM geometry与same expert bank均为
   mandatory；params差异只报告，不作候选选择；
6. **specialization identifiability**：定义gate entropy、expert disagreement、geometry dependence与history dependence；
   若joint gate退化为constant/history-only，则geometry mechanism失败；
7. **optimization boundary**：检查identical-expert symmetry、gate collapse与expert starvation；不能用额外loss堆叠掩盖
   decoder本身的failure；
8. **patch boundary**：先不使用atom-to-patch attention；若Step5证明必须使用，回Step2/3补problem evidence，而不是
   直接进入实现。

Step 5任一核心gate失败，则rollback Step 2/3。通过后Step 6才设计最小E、initialization、controls与
validation-only screen；SC2-MIPR继续held。

## 9. Failure Attribution

- `hypothesis_false`: **false for PLGO geometry**；D6/D7/D8 matched controls仍支持support geometry；
- `intervention_point_wrong`: **supported for exact PAF**；geometry只参数化fixed trunk，未改变sample-specific operator；
- `readout_or_head_design_wrong`: **strongly supported**；PAF相对A6五dataset大幅失败；
- `optimization_or_numeric_pathology`: **bounded**；epoch cap存在但不足解释gap；
- `capacity_control_explains`: **not main cause for D8**，但将是任何expert successor的主要替代解释。

本轮没有证明JAPO有效，只证明了什么候选值得继续，以及三个不应继续的shortcut：patch retrieval、geometry-only
experts、dense/residual bypass。

## 10. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 4 complete；Step 5 next |
| `problem` | fixed descriptor-generated separable readout无法同时保留A6 operator freedom与RGNB geometry |
| `existence_evidence` | D6 support crossing；D8 GEO vs controls +14.33%且vs A6 -28.10% |
| `idea` | joint history-atom routing over free RGNB expert coefficient maps |
| `theory_check` | geometry-only expert no-go完成；A6 containment/projectivity/non-collapse pending |
| `design` | principles and equations only；no module/config frozen |
| `narrative_gate` | source-level conditional；not narrative-ready |
| `effectiveness_gate` | not started |
| `artifacts` | D8 report、external source matrix、本Step4 report |
| `decision` | `SC1-JAPO theory_pending`；implementation/training false；rollback Step2/3 if Step5 fails |
