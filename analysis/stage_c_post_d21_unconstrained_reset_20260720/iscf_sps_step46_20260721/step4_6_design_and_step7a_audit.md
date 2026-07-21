# ISCF-SPS Step 4–7A：Scope-Projected Synthesis

## 1. 研究决策

| Field | Record |
| --- | --- |
| `current_step` | Step 4–6 gate通过；Step7A local implementation/audit |
| `problem` | ISCF的independent history maps有效，但shared unrestricted target synthesis使不同scope arms仍可学习相似完整forecast，scope geometry没有持续约束forward role与backward learning signal |
| `existence_evidence` | ISCF vs Q1-WIDE test MSE `+0.8496%`；oracle headroom median `8.5813%`；fusion仅9/15超过best fixed arm；scope360/720占120个bin winners的86.67%；canonical未超过random partition |
| `idea` | 在既有arm进入direct fusion前施加scope-native orthogonal local-DCT projector，同时约束arm forecast与回传error spectrum |
| `theory_check` | 不输入requested H；不新增信息；single pointwise MSE不变；specialization来自function/gradient subspace而不是auxiliary diversity loss |
| `design` | 保留ISCF五个independent maps、scope groups、policy与full-T crop；新增0 trainable parameters；identity/global/random matched controls |
| `narrative_gate` | `conditional_pass_as_scope_utilization_architecture` |
| `effectiveness_gate` | pending；test disabled；先做完整validation screen |
| `decision` | candidate=`SC-ISCF-SPS-v0`；只授权local Step7A；remote training/test仍为false |

用户将ISCF multi-scope architecture固定为本项目的architecture design prior。SAC negative不再作为停止条件，但仍作为
design evidence保留：它提示当前实现没有充分兑现temporal scope semantics，而不是允许在论文中删除该negative result。

## 2. 为什么当前ISCF可能没有充分利用scope

Encoder输出`hidden [B,C,R]`，五个independent maps生成

$$
M\in\mathbb{R}^{B\times C\times S\times D\times K}.
$$

对scope $s$，canonical groups为`indices_s [G_s,s]`，pooled coordinates为`[G_s,D]`，并生成共享group state

$$
Z_s\in\mathbb{R}^{B\times C\times G_s\times K}.
$$

但既有`_scope_forecast`随后用所有scopes共享的、逐target自由参数化的
`identity_synthesis [T,K]`和`nonlinear_synthesis [T,K]`解码。即使一个group内共享$Z_{s,g}$，不同target仍有完全不同的
synthesis rows。因此scope只约束group coefficient state，没有继续约束输出函数或其gradient spectrum；coarse arm仍可
借助逐target synthesis学习generalist full forecast。

这与现有evidence一致：independent maps超过Q1-WIDE，说明private parameterization有效；但canonical partition没有超过
random，且learned fusion没有稳定兑现oracle headroom。新的问题不是“scope architecture是否存在”，而是：

> 如何让scope extent成为每个arm可表达forecast与接收training gradient的原生边界，从而诱导可归因的functional specialization？

## 3. Scope-Projected Synthesis（SPS）

### 3.1 Local orthogonal projector

ISCF先原样产生raw arm $a_s\in\mathbb{R}^{B\times C\times T}$。对每个长度$s$的group，构造orthonormal DCT-II
basis $C_s\in\mathbb{R}^{s\times r_s}$，其中

$$
r_s=\min\left(s,\max\left(1,\operatorname{round}(Ks/T)\right)\right).
$$

projected group为

$$
\tilde a_{s,g}=C_sC_s^\top a_{s,g}.
$$

再按原indices scatter为`projected_arm [B,C,T]`，五arms堆叠为`[B,C,S,T]`，交给完全不变的direct policy
`weights [B,C,T,S]`：

$$
\hat y_t=\sum_s w_{t,s}\tilde a_{s,t}.
$$

requested horizon只在完整$T=720$计算后crop。

以现有dataset ranks $K\in\{106,109,116\}$为例，scope1/48/144/360/720的projected degrees分别约为
`720 / 105–120 / 105–115 / 106–116 / 106–116`。中大scopes获得近似相同的full-domain interpolation budget，
但以“更多local groups、每组更少modes”或“更少groups、每组更多modes”的不同方式分配。

### 3.2 为什么它会改变feature learning

$P_s=C_sC_s^\top$为symmetric idempotent projector。对任意forecast loss $L$，

$$
\frac{\partial L}{\partial a_s}
=P_s^\top\frac{\partial L}{\partial\tilde a_s}
=P_s\frac{\partial L}{\partial\tilde a_s}.
$$

因此SPS不仅事后平滑outputs；它在joint training中把每个scope-specific `mode_weight[s]`的gradient限制到对应local
resolution subspace。不同arms从同一target上接收不同结构的error signal，形成architecture-induced specialization。
该机制不需要orthogonality/diversity auxiliary loss，也不需要额外router。

### 3.3 Containment与controls

- `identity`：每个group保留full DCT rank，数值上精确等于ISCF-v0，是同一production code path的exact control；
- `global`：所有arms统一使用global DCT rank $K$，隔离generic output smoothing；
- `random partition`：参数、rank和projection相同，只改变target-to-group binding；
- `scope`：candidate，projection与每个arm的native group extent对齐；
- `ISCF-v0 parent`：历史/重训parent effectiveness reference。

SPS不增加trainable parameters。由于projected-out synthesis directions不参与candidate forward，报告必须同时给出stored
parameters、nominal active parameters与projected functional degrees，不得只用storage count声称capacity exact matched。

## 4. Primary-source boundary（2026-07-21）

检索范围：expert homogenization/specialization、multi-resolution output synthesis、forecast component specialization、
multi-scale expert fusion。仅使用arXiv、OpenReview、PMLR、AAAI和official proceedings；Zotero coverage未用于判断novelty。

1. [NHITS, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25854)已经用multi-rate input pooling、hierarchical
   interpolation与sequential additive blocks诱导不同frequency/scale specialization。故“hierarchical interpolation”或
   “不同forecast components”本身不是novelty；NHITS是SPS最重要的closest-prior control。
2. [N-BEATS, ICLR 2020](https://openreview.net/pdf?id=r1ecqn4YwB)已覆盖basis expansion与additive forecast blocks；
   SPS不能claim首次output basis decomposition。
3. [TimeMixer, ICLR 2024](https://arxiv.org/abs/2405.14616)已覆盖multi-scale input decomposition与Future-
   Multipredictor-Mixing；generic multi-scale predictors/fusion不新。
4. [FreqMoE, AISTATS 2025](https://proceedings.mlr.press/v258/liu25i.html)以frequency decomposition、specialized
   experts、gating和residual refinement做forecast；frequency specialists与output gating不新。
5. [Advancing Expert Specialization, NeurIPS 2025](https://arxiv.org/abs/2505.22323)与
   [Expert Divergence Learning, ICLR 2026](https://openreview.net/pdf?id=wrqYMYazm0)说明expert homogenization是可测的
   optimization problem，但两者依赖auxiliary specialization/routing objectives，不能作为本项目预设第二loss的依据。

可保留的完整贡献边界是：

```text
one fixed-past varied-horizon decoder
-> parallel future-output coupling scopes with independent history maps
-> scope-native orthogonal synthesis/gradient subspaces
-> target-wise full-domain composition
-> parent / identity / global / random matched attribution
```

SPS与NHITS的关键区别必须在论文中明确：NHITS是sequential residual stacks + multi-rate history pooling + additive
interpolation forecasts；SPS是shared encoder上的parallel output-coupling groups，以scope projector同时定义forward function
与backward gradient，再由target-wise policy形成同一个full-domain forecast。该区别只构成`conditional contribution-level
novelty`，最终仍需matched evidence。

## 5. Step7A local gates

1. candidate/identity/global输出shape均为`[B,720,C]`且finite；
2. 每个local DCT basis满足$C_s^\top C_s=I$，$P_s^2=P_s$；
3. `identity`与parent在same seed/parameters/input下exact equal；
4. candidate、identity、global及canonical/random parameter initialization hash一致；
5. five scope mode maps均收到finite nonzero gradients；
6. canonical/random只改变binding，rank/params不变且outputs不同；
7. production CLI完成one-batch synthetic smoke；
8. code-facing explanation记录shape、gradient path与code-theory boundary。

只有全部通过才允许冻结Step7B validation matrix。Step7B建议先做candidate/global/random三arms × five datasets ×
seed2021，并重训parent identity control；validation按四standard horizons mean MSE选checkpoint。candidate至少需要相对parent
`+0.3%` MSE、3/5 datasets、3/4 horizons，并超过global control，同时显示scope-conditioned spectral/gradient separation，
才值得申请formal test。

## 6. Failure attribution与授权

- local invariant失败：`design_fault_suspected`，回Step5/6修复；
- candidate与identity相同：projection未生效，`intervention_point_wrong`；
- candidate不超过global：`generic_smoothing_explains`；
- performance改善但specialization diagnostics不成立：`performance_partial_pass_without_mechanism_attribution`；
- validation material negative：关闭exact SPS-v0，回Step4；不得追加loss/router或按dataset选择rank；
- frozen replacement不用于方向级拒绝，formal gate必须E2E joint training。

当前授权：local implementation=true；remote training=false；formal test=false；modern baselines=false。

## 7. Step7A actual result

`conda r2026-fsa`下的production/local audit全部通过：

| Check | Result |
| --- | ---: |
| identity vs parent max absolute gap | `8.3447e-7` |
| full-domain prefix gap | `0` |
| max basis orthonormal error | `3.2196e-15` |
| max projector idempotence error | `1.5266e-16` |
| scope mode-map gradient norms | `[0.10884, 0.01261, 0.01264, 0.00868, 0.00635]` |
| one-target error impulse support | `[1,48,144,360,720]`，与scope widths精确一致 |
| scope vs global max absolute gap | `1.05729` |
| scope vs random max absolute gap | `0.25685` |
| production model shapes | full `[1,720,2]`；prefix `[1,96,2]` |
| production CLI | readout/projection/rank/four-H validation/final-val parse pass |

首轮float32 basis未满足严格$10^{-10}$ invariant，定位为construction precision而非model/path failure。basis buffer改为
float64，使用时才转换至arm dtype；修改后orthonormal/idempotence均在$10^{-15}$量级，forecast path保持float32。

Decision=`iscf_sps_step7a_contract_pass`。该结果只证明实现与理论合同成立，不是performance或paper-core pass。
machine artifact为`step7a_summary.json`。
