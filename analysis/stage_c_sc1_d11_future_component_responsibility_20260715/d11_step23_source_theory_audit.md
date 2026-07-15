# SC1-D11 Future-Component Responsibility: Step 2/3 Source And Theory Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | Contribution 1 Step 2/3 diagnostic design |
| `problem` | short/long prefix risks是否在shared forecast path上产生可定位的future-component directional conflict？ |
| `role` | `diagnostic_only`；不是decoder、loss或optimizer candidate |
| `existence_evidence` | D6 support crossing；D1 horizon-measure gradient separation；B9/B13 low-alignment；B14 exact retrieval-demand mismatch fail |
| `theory_check` | output-gradient可由任意complete orthogonal future decomposition精确、可加地归因 |
| `design_decision` | `theory_pass_diagnostic_protocol_frozen` |
| `method_authorization` | false；test=false；forecast training/update=false；SC2=false |

## 1. Why D11 Is Needed

D6已经证明fixed local/global future supports在short/long horizons上出现crossing，但D9/D10否定了稳定的
history-scale→future-depth alignment。剩余问题不再是“历史尺度如何路由”，而是：

> 同一个shared forecast path在服务不同prefix measures时，是否必须同时承受方向相反的future-component
> update；如果有，冲突发生在coeff、decoder basis还是encoder？

旧B9/B13只报告non-overlap future units的low cosine。低正cosine表示heterogeneity，不等于PCGrad意义下的
directional conflict；B14又没有支持error-conditioned retrieval demand与现有sensitivity之间的矛盾。因此D11
必须把`negative direction`、`magnitude imbalance`、`coordinate artifact`和`generic transformed loss`分开。

## 2. External Primary-Source Audit

检索日期：2026-07-15。Zotero仅作seed，本次关键工作均由external primary sources发现或复核。

| Source | Primary evidence | Boundary for D11 |
| --- | --- | --- |
| [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | conflicting gradients以负inner product为核心；同时指出curvature与gradient magnitude差异 | D11不能把low positive cosine写成conflict；必须报告negative rate与norm ratio |
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | 通过调节task loss权重平衡gradient magnitudes | magnitude imbalance必须作为simple balancing control，不等于architecture problem |
| [CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | average objective与worst local task improvement之间的conflict-aware optimization | 方向冲突若成立，仍需与generic gradient manipulation区分 |
| [Nash-MTL, ICML 2022](https://proceedings.mlr.press/v162/navon22a.html) | shared model中的task gradients可通过multi-objective bargaining组合 | generic MTL optimizer不是本项目的decoder novelty |
| [FreDF, ICLR 2025](https://openreview.net/forum?id=4A9IdSa1ul) / [code](https://github.com/Master-PLC/FreDF) | frequency-domain loss处理future-label correlation；官方实现可与temporal loss融合并调权 | “加frequency loss”已被覆盖，不能作为新贡献 |
| [Time-o1, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0cd62dea69635f4c5b569848267fe5a8-Abstract-Conference.html) / [code](https://github.com/Master-PLC/Time-o1) | PCA transformed label alignment同时针对label autocorrelation与forecast-step task overload | “变换future components并按importance监督”已是直接prior art |
| [DBLoss, arXiv 2025](https://arxiv.org/abs/2510.23672) / [code](https://github.com/decisionintelligence/DBLoss) | 对prediction/label做EMA trend-seasonal decomposition并分别计算loss | component loss本身重叠；该来源仍是preprint，证据等级低于正式会议 |
| [Hybrid Loss, withdrawn ICLR 2025](https://openreview.net/forum?id=Y89o3LAEHX) | global与component errors联合并动态调权 | 只作为claim-overlap警示，不作为已确立方法证据 |

[Decision] D11若通过，可形成的边界不是component loss或gradient surgery，而是：

`unified prefix measure -> exact future-component gradient responsibility -> intervention-point diagnosis -> decoder/objective co-design contract`。

其中前两项必须由本地证据成立；后两项仍需另行Step 4-6，不能由D11直接授权。

## 3. Exact Gradient-Responsibility Identity

令full-domain prediction为$\hat y_\theta\in\mathbb R^T$，某个prefix measure $\mu$的loss对output的gradient为

$$
v_\mu = \nabla_{\hat y} L_\mu.
$$

令$U=[U_1,\ldots,U_G]$为complete orthogonal future basis的group partition，$P_g=U_gU_g^\top$。由于
$\sum_gP_g=I$，定义group responsibility

$$
r_{\mu,g}^{(z)} = J_z^\top P_g v_\mu,
$$

其中$z$可以是A6 shared `coeff`或任一shared parameter block，$J_z=\partial\hat y/\partial z$。则严格有

$$
\sum_g r_{\mu,g}^{(z)}=J_z^\top v_\mu=\nabla_zL_\mu.
$$

这不是把$\|P_ge\|^2$冒充prefix loss。prefix mask通常不与$P_g$ commute，component energies不能简单相加；
D11分解的是**output gradient**，因此对MSE和L1都保持exact additivity。

## 4. What Counts As Conflict

1. `directional conflict`：$g_s^\top g_l<0$；这是first-order意义下沿一个regime下降会使另一个上升；
2. `low alignment`：cosine接近0但非负，只说明不同，不称为conflict；
3. `magnitude imbalance`：norm ratio大，但unit-normalized cosine非负；优先指向GradNorm/raw weighting control；
4. `component cancellation`：同一regime内$r_{\mu,g}$之间存在负inner products，导致
   $\|\sum_gr_g\|/\sum_g\|r_g\|$降低；只有canonical geometry超越DCT/random controls时，才可能支持
   future-support-specific architecture problem；
5. `coordinate artifact`：任意random orthogonal grouping同样产生的cancellation或responsibility shift，不支持
   RGNB/decoder claim。

## 5. Frozen Diagnostic Contract

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- checkpoints：A6 natural baseline seeds 2021/2022/2023；
- splits：fixed train batches与official validation batches各4；validation为primary，train只作sign replication；
- no test、no fitting、no forecast parameter update；
- short measure：equal mass over prefixes `{48,96,144}`；
- long measure：equal mass over prefixes `{336,512,720}`；`192/288`不参与primary gate；
- loss：MSE primary，L1作为当前A6 training-objective replication control；
- future decompositions：RGNB、DCT、3个fixed random orthogonal controls；全部group sizes为
  `[16,16,32,64,128,256,208]`；
- intervention tensors：component responsibility primary在`coeff [B,C,256]`；total short/long gradients另审计
  `encoder_params/coeff_params/basis_params/all_params`；
- A6 learned basis只做orthonormalized span/complement reachability audit，不伪造七层scale groups。

## 6. Frozen Decision Tree

### Gate A: strict total directional conflict

dataset support要求validation上`coeff`或`coeff_params/encoder_params`至少一条shared path满足：

- mean cosine `<0`且negative batch fraction `>=0.25`；
- 3 seeds中至少2个同方向；
- train split negative fraction `>=0.20`；
- L1 control至少保持negative sign，不要求相同magnitude。

至少3/5 datasets支持才称`directional_conflict_supported`。

### Gate B: support-specific component conflict

即使total gradients相容，component cancellation仍可能被sum掩盖。dataset support要求validation RGNB同时满足：

- short/long responsibility distribution JS `>=0.05`；
- within-regime negative pair fraction或same-component cross-regime negative fraction `>=0.20`；
- cancellation比3个random controls median至少高`0.05`，或JS至少高`0.02`；
- DCT不得完全解释：RGNB至少在上述两项之一比DCT高`0.02`；
- 2/3 seeds与train sign replication通过。

至少3/5 datasets支持才称`support_specific_component_conflict_supported`。

### Alternative outcomes

- directional/component pressure存在但RGNB不超过DCT/random：
  `transform_generic_pressure_sc2_only`；SC1不返回Step4，SC2只允许重新做novelty audit；
- norm ratio `>=2`但strict conflict与component gate失败：
  `magnitude_imbalance_only_simple_balancing_control`；
- 其余：`future_component_conflict_not_supported_rollback_step2`。

任何positive outcome只返回Step4；不得直接实现PCGrad、component loss、new decoder或SC2。

## 7. Failure Attribution Boundary

- gradient additivity、basis orthogonality、prefix-weight sum、full-forward reconstruction任一失败：
  `diagnostic_invalid_for_direction_rejection`；
- 仅MSE成立而L1不复现：`evaluation_risk_specific_only`；
- 仅A6 learned coordinate或单dataset成立：`coordinate_or_dataset_specific`；
- random controls解释：`coordinate_partition_explains`；
- stable negative结果只关闭“future-component directional conflict作为当前paper problem”，不否定RGNB
  projectivity、D6 support crossing或所有training strategy。

## 8. Self-Critique

- checkpoint-local gradients是local first-order evidence，不证明长期optimization trajectory；
- validation gradients不用于参数更新，但能确认signal不是train memorization；
- RGNB/DCT/random比较的是group projector family，不证明某个future basis必然是最佳decoder；
- 阈值用于阻止把普遍低cosine包装成paper problem；若结果只略低于gate，不应事后放宽。
