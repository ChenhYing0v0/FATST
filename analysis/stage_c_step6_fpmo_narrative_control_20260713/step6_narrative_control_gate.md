# SC1-FPMO Step 6 Narrative / Control Gate

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-FPMO-DS` |
| `current_step` | Step 6 complete |
| `linear_mechanism_gate` | fail：与`FPMO-DA`拥有相同full-affine function class |
| `scale_attribution_gate` | fail：同一分解对任意orthogonal coordinates与任意row grouping均成立 |
| `optimization_novelty_gate` | fail：剩余差异是已有deep linear factorization范畴的implicit bias |
| `nonlinear_rescue_gate` | fail as current candidate：需要新增未论证的nonlinear mechanism，且exact A6 containment不再自动成立 |
| `prefix_execution_gate` | algebra pass；efficiency claim fail：DS仍需构造全部720维scale latents |
| `narrative_gate` | `rejected_by_narrative_gate` |
| `effectiveness_gate` | not started；不得用performance事后提升为paper core |
| `rollback` | Step 2/3：先诊断rank、nonlinearity与scale alignment，再提出新Step 4 idea |
| `method_implementation` | `false`；M0/DA/DS均不进入Step 7 |

## 1. What We Test And Why

Step 5已经证明FPMO的arbitrary-length transform、A6 embedding与prefix restriction在代数上成立，但留下
一个paper-level问题：`FPMO-DS`是否真的引入了**不可由full-affine capacity或optimization coordinates
解释**的scale-native forecast mechanism？

Step 6不看test performance，也不实现model。它先回答：

1. DS相对DA是否改变可表达的forecast function；
2. 所谓scale structure是否依赖balanced interval basis，而不是任意坐标分组都成立；
3. 如果差异只来自factorized optimization，是否足以构成SCI architecture contribution；
4. nonlinear extension能否作为当前DS的自然延伸，而不是另开一个未经Step 4-5审计的新方法；
5. requested prefix是否真的减少DS的主要计算，而不只是少写出inactive coefficients。

## 2. Source-Informed Boundary

- [Implicit Regularization in Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html)
  已研究full-dimensional linear factorization在不改变可表示矩阵集合时产生的optimization bias。
- [Implicit Regularization in Deep Matrix Factorization](https://proceedings.neurips.cc/paper/2019/hash/c0c783b5fc0d7d808f1d14a6e9c8280d-Abstract.html)
  表明增加linear factorization depth可改变对low-effective-rank solutions的偏好，而且这种bias未必能由
  简单norm完整描述。
- [N-HiTS](https://arxiv.org/abs/2201.12886)已经以hierarchical interpolation与multi-rate sampling进行
  multiscale forecast synthesis。
- [PRISM](https://arxiv.org/abs/2512.24898)进一步采用tree partition、scale-specific features与hierarchical
  aggregation。因此“按scale分组后分别用一个MLP”本身已有很强prior-art压力。

[Strong Evidence] 这些来源支持“factorization可能改变训练轨迹”，但不支持“这种差异是新的future-scale
operator”。尤其是上述implicit-regularization结论依赖特定loss、initialization与optimizer条件；当前
TimeAlign joint training使用Adam类optimizer与pointwise L1，不能直接宣称nuclear-norm或low-rank定理成立。

## 3. Linear DS Is A Redundant Full-Affine Parameterization

对每个scale group $J_l$，Step 5的DS为

$$
z_l=A_lh,\qquad \alpha_{J_l}=D_lz_l+g_l,
$$

其中T720下$k_l=n_l=|J_l|$。定义$G_l=D_lA_l$，则

$$
\alpha_{J_l}=G_lh+g_l.
$$

由于可以取$D_l=I$、$A_l=G_l$，任意$G_l\in\mathbb R^{n_l\times R}$都可表示；反向又显然有
$D_lA_l\in\mathbb R^{n_l\times R}$。因此

$$
\mathcal F_{DS}=\mathcal F_{DA}=\{h\mapsto Gh+g\}.
$$

更强地，对任意可逆$S_l$，$(D_lS_l,S_l^{-1}A_l)$给出同一个$G_l$。DS增加的是non-identifiable
factorization，而不是新的forecast mapping。

[Fact] 该结论不依赖balanced Haar的语义。把$Q_T$替换成任意orthogonal matrix、把rows按相同sizes任意
分组，只要$k_l=n_l$，仍得到full affine class。因此当前linear DS没有可由function space定义的
“scale-native”边界。

## 4. Why Optimization Bias Is Not Enough

[Hypothesis] DS的两层linear factors可能在实际optimizer下形成不同于DA的implicit bias，并可能改善
validation performance。

但它不能通过本轮narrative gate，原因是：

1. 该机制属于已有deep linear/matrix factorization理论范畴；
2. 它对真实multiresolution basis并不专有，random orthogonal coordinates也可产生同类bias；
3. 当前Adam + L1 + jointly trained Encoder不满足可直接移植现有定理的条件；
4. 如果只在Adam或某个initialization下优于DA，最诚实的claim是optimizer-coordinate effect，而不是
   unified multi-horizon forecast operator；
5. 参数量不是否决理由，但同function class下额外参数必须被视为optimization/capacity control，而非
   mechanism evidence。

[Self-critique] 这不意味着factorized head一定无效；它可能是好用的engineering choice。否决的是把当前
linear DS作为论文Contribution 1，而不是否认它可能获得更低的MSE。

## 5. Nonlinear Rescue Audit

候选补救是加入$\rho_l$：

$$
\alpha_{J_l}=D_l\rho_l(A_lh)+g_l.
$$

它确实会使function class超出full affine，但不能被视为当前DS的无成本延伸：

- 普通GELU/SiLU block不再自动exact包含所有A6 affine operators；
- 通过identity bypass、learned interpolation或`x + f(x)`恢复containment，会重新进入generic
  Network Morphism / residual-style边界；
- 通过paired activations或扩大width精确表示identity，会引入新的capacity与activation construction；
- 即使实现成功，`per-scale MLP`仍必须同时击败matched dense MLP、random-orthogonal grouped MLP与
  random row-group MLP，才能说明收益来自future-scale alignment；
- N-HiTS与PRISM已使generic hierarchical/scale-specific nonlinear head具有较高novelty pressure。

[Decision] nonlinear version不是`FPMO-DS`的Step 7 implementation detail，而是一个需要重新通过
Step 2-5的问题与方法。它不能用于事后挽救本轮narrative gate。

## 6. Prefix Tensor Path And Honest Compute Boundary

对requested horizon $H$，synthesis只需要active atoms：

$$
\widehat y_{:H}=Q_{[0,H),\mathcal A_H}^{\top}\alpha_{\mathcal A_H}.
$$

M0或DA可以直接选择active rows生成$\alpha_{\mathcal A_H}$。但DS先为每个scale生成
$z_l=A_lh\in\mathbb R^{n_l}$；由于$D_l$是dense，某个active coefficient通常依赖整个$z_l$。T720下
$\sum_l n_l=720$，因此即使不写出inactive coefficients，仍需计算全部720维scale latents。

[Fact] DS保留exact prefix algebra，但没有比DA更强的domain-only execution，也没有证据比A6更快。
若把$D_lA_l$预先合并以按active rows计算，则推理时正好退化为DA。

## 7. Frozen Control Contract

| Arm | Function class | Purpose | Promotion rule |
| --- | --- | --- | --- |
| A6 | rank$\le256$ affine | accepted carrier | reference only |
| M0 | exactly A6 in $Q_T$ coordinates | morph/prefix equality | control only |
| Dense-FA | full affine in time coordinates | capacity reference | control only |
| DA | full affine in $Q_T$ coordinates | orthogonal-coordinate control | control only |
| DS-L | factorized full affine by depth groups | optimization-factorization probe | diagnostic only |
| DS-R | same factors under random orthogonal/group assignments | scale attribution control | diagnostic only |

若未来只看到`DS-L > A6`，结论为`capacity_control_explains`；若`DS-L > DA`但不超过`DS-R`，结论为
`optimization_parameterization_explains`；只有真实scale grouping稳定优于DA、Dense-FA与random controls，
才值得回到Step 4提出新的mechanism。但这也不能自动复活当前linear DS，因为其function-space claim仍为空。

## 8. Narrative Gate Decision

| Required property | Result | Evidence |
| --- | --- | --- |
| clear problem motivation | partial | D1支持future structure，但未证明scale-specific nonlinearity |
| mechanism novelty | fail | linear DS是已有full-affine factorization类别 |
| explainable tensor path | pass | history-to-scale-to-atom path清楚 |
| defensible contribution boundary | fail | mechanism不依赖真实scale coordinates |
| capacity attribution | fail | DS与DA同class且更过参数化 |
| prefix/domain claim | partial | algebra成立，但主要latent compute不随H裁剪 |

[Decision] `SC1-FPMO-DS = rejected_by_narrative_gate`。M0、DA与DS-L保留为theory/diagnostic controls，
不进入Step 7。SC2-MIPR继续held，不能用objective收益帮助一个未过architecture narrative gate的候选。

## 9. Failure Attribution

- `hypothesis_false`：**尚未成立**。D1的nested structure与frozen-memory evidence仍在；本轮没有证明
  scale-aligned nonlinear signal不存在。
- `intervention_point_wrong`：可能。只在最终linear readout中分组可能不足以形成可识别的scale mechanism。
- `readout_or_head_design_wrong`：对当前DS成立；其linear factors在composition后与DA相同。
- `optimization_or_numeric_pathology`：未测试，无训练artifact，不可归因。
- `capacity_control_explains`：理论上是必须控制的替代解释；尚无performance result。

因此这是`candidate_design_level_rejection`，不是对multiscale future operator方向的direction-level rejection。
rollback到Step 2/3，而不是继续堆叠nonlinearity、Encoder、MoE或MIPR。

## 10. Next Problem Diagnostic: SC1-D2

下一轮先验证一个更窄的问题：**A6 frozen ordered memory到future function之间，是否存在超出rank expansion
与generic nonlinearity的真实scale-aligned conditional structure？**

按同一frozen memory与validation-only协议比较：

1. rank-256 affine（A6-equivalent）；
2. full affine；
3. parameter-matched dense nonlinear head；
4. $Q_T$ true-scale grouped nonlinear head；
5. random-orthogonal与random-group nonlinear controls。

读法必须固定：`2 > 1`只支持rank/capacity问题；`3 > 2`只支持generic nonlinearity；只有`4`同时稳定
优于`3`和`5`，才支持scale alignment。D2是`diagnostic_only`，不会因为metric好看自动成为Contribution 1。
五dataset的正式threshold、random-control数量与artifact contract将在D2 protocol中预注册；在此之前不实现
paper-core model。

## 11. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 6 complete；rollback Step 2/3 |
| `problem` | linear scale factorization无法隔离真实scale mechanism |
| `existence_evidence` | D1 structure/memory evidence保留；scale-aligned nonlinearity未验证 |
| `idea` | FPMO-DS linear direct-scale factors |
| `theory_check` | exact embedding/restriction pass；DS=DA full affine；prefix compute claim weak |
| `design` | A6/M0/Dense-FA/DA/DS-L/DS-R controls冻结为diagnostic contract |
| `narrative_gate` | fail；mechanism novelty与contribution boundary不足 |
| `effectiveness_gate` | not started |
| `artifacts` | 本报告、candidate/control/D2 CSV、machine-readable JSON |
| `decision` | `rejected_by_narrative_gate`；rollback Step 2/3；training unauthorized |
