# SC-D22-HFA D22-A/B：Bayes 任务边界与 finite-capacity horizon frontier 审计

## 1. 当前节点与结论

| Field | Content |
| --- | --- |
| `current_step` | SC-D22-HFA Step 2；D22-A/B complete |
| `problem` | fixed past、deterministic point forecast下，requested horizon或lead-time是否要求新的horizon-dependent operator？ |
| `existence_evidence` | D18 specialists、A6_MEASURE/A6_FULL、H1..720 dense official-test curves、checkpoint与3-seed A6 stability artifacts |
| `idea` | 先分离Bayes-level task identity、finite-capacity/optimization tradeoff与target-coordinate information access |
| `theory_check` | pure request $H$在pointwise MSE下不改变同一coordinate的Bayes mean；finite-model tradeoff必须由matched empirical frontier证明 |
| `design` | 不训练新模型；复算D18 full-test dense curves并对照measure、split、checkpoint与seed-noise proxy |
| `narrative_gate` | finite-capacity horizon frontier未通过；generic H embedding/router不成立 |
| `effectiveness_gate` | not applicable；本轮是problem-existence audit |
| `artifacts` | 本报告及D18/A6既有CSV/JSON；远端D18 dense metrics只读复核 |
| `decision` | `finite_capacity_frontier_not_supported / d22_c_design_only_conditionally_justified` |

[Decision] 当前证据不支持把finite-capacity horizon frontier升为Stage C的paper problem。唯一稳定的局部信号是
H96 specialist在seed2021的5/5 datasets均优于`A6_MEASURE`；但H192、H336不成立，任何specialist都不能在
standard-horizon向量上Pareto-dominate shared control，且measure control已在所有lead-time bins稳定解释主要收益。

D22-A/B不能回答另一种不同的问题：future coordinate是否需要对raw history进行target-specific evidence access。
D14-A1的dual-carrier、three-seed oracle/crossing仍提供合理但非充分的headroom，因此本报告只冻结D22-C的
`diagnostic_only`设计；不实现、不训练、不访问新的test结果。ordered patch memory只作为诊断载体。

## 2. D22-A：Bayes 与 task boundary

### 2.1 Pure-request horizon theorem

令$X$为fixed past，$Y_t$为第$t$个future coordinate，$H\ge t$为requested horizon。若$H$只表示用户请求，
且不改变information set、样本分布或utility，则

$$
Y_t\perp H\mid X.
$$

在pointwise squared loss下，任意Bayes-optimal predictor满足

$$
f_t^*(X,H)
=\arg\min_a\mathbb E[(Y_t-a)^2\mid X,H]
=\mathbb E[Y_t\mid X,H]
=\mathbb E[Y_t\mid X].
$$

因此，取消exact projectivity只扩张finite-model function class；它不让pure request $H$成为新的predictive
information。若训练数据中$P(X,Y\mid H)$不同，模型学到的可能是task-selection或distribution-shift signal，不能
直接称为requested-horizon semantics。

### 2.2 哪些变化会真正越过该边界

| Case | Bayes target是否可能变化 | 本项目当前角色 |
| --- | --- | --- |
| 只改变requested $H$，同一$X$、$Y_t$与pointwise MSE | 否 | 禁止直接实现H embedding/router |
| 改变loss中各coordinate权重 | conditional mean不变；finite optimization可变 | A6_MEASURE类protocol/control |
| 加入known-future covariates $Z_{1:H}$ | 可以，因为information set改变 | 显式task pivot |
| 使用trajectory/coherence/decision loss | 可以，因为Bayes action/risk改变 | 显式task pivot |
| 改变sampling rate、resolution或compute budget | target mean未必变，但deployment-optimal computation可变 | 显式compute contract |
| 预测联合分布而非逐点mean | joint target与dependence结构改变 | probabilistic task pivot |
| forecast origin前移并获得新observations | 可以，因为$X$改变 | rolling-origin/revision task，不是pure $H$ |

### 2.3 外部 primary-source audit

检索日期：`2026-07-20`。检索范围为`varied/dynamic horizon forecasting`、`target timestamp query`、
`context-dependent multi-horizon attention`、`functional/continuous decoder`、`forecast stability/decision risk`和
`training-horizon learnability`。来源优先使用会议proceedings、arXiv full text、OpenReview论文页与official code
link；本轮未以Zotero覆盖率作结论。下表均为external discovery/verification，Zotero presence未查询。

| Primary source | 直接覆盖的边界 | 对D22的约束 |
| --- | --- | --- |
| [ElasTST, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html) | varied-horizon invariance、structured masks、multi-scale patches、horizon reweighting | invariance、multi-patch与harmonic-style measure不能单独claim；A6_MEASURE是mandatory control |
| [ProbTS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/55f2a27b1ac39dbfdd0fc83742dc87d7-Abstract-Datasets_and_Benchmarks_Track.html) | point/distributional forecasting across diverse horizons的benchmark边界 | horizon差异可反映task family；不能从benchmark异质性推出同一coordinate的H-semantics |
| [MQTransformer](https://arxiv.org/abs/2009.14799) | future-context-dependent decoder-encoder attention；query包含forecast-time context | generic future-query/history attention已有直接prior，且known-future context越过pure-request边界 |
| [TimePerceiver](https://arxiv.org/abs/2512.22550) | target-timestamp query从input latents选择性读取信息；任意/不连续target segments | target-coordinate query与ordered-memory retrieval均不是独立novelty；D22-C必须证明task-specific necessity |
| [FlowState](https://openreview.net/forum?id=R50AT6nAsM) | functional basis、continuous time、sampling-rate invariance与dynamic horizons | flexible horizon/continuous decoder本身已有强prior；compute/resolution属于不同contract |
| [TAT](https://arxiv.org/abs/2507.10349) | holiday/promotion等a-priori known future context驱动multi-horizon alignment | 是$Z_{1:H}$改变information set的正例，不支持pure-request H |
| [Temporal horizons in forecasting, TMLR 2025](https://openreview.net/forum?id=BeudQIxT1R) | autoregressive dynamical forecasting中training horizon与loss-landscape learnability tradeoff | 支持finite optimization frontier可存在，但其rollout setting不能直接外推到fixed-past direct multi-output A6 |
| [Beyond Accuracy, 2026](https://arxiv.org/abs/2601.10863) | probabilistic multi-horizon accuracy、rolling-origin coherence与decision weights | 是nonseparable risk/新information task pivot，不是当前pointwise fixed-past MSE的反例 |

[Strong Evidence] 到`2026-07-20`的primary-source refresh没有发现能够推翻pure-request theorem的工作。最新工作
反而把有效自由度明确放在target timestamps、known-future context、continuous sampling contract、probabilistic
joint forecast或rolling-origin coherence上。

[Coverage Gap] TimePerceiver与FlowState仍是2025公开版本；本轮没有把所有2026 workshop submissions视为已确认
prior，也没有完成modern varied-horizon methods的本地native reproduction。因此本结论只冻结problem boundary，
不声称完成最终paper novelty review或SOTA baseline comparison。

## 3. D22-B：artifact与统计量

### 3.1 数据来源与角色

1. `analysis/stage_c_post_ccsf_step24_reset_20260719/d18_step9/validation_test_cells.csv`：15个own-H
   validation/test cells；
2. D18 remote canonical root中每个arm的`test_audit_metrics_by_target_horizon.csv`：H1..720 official-test
   prefix MSE/MAE curves；本轮只读复算，不生成新预测、不改变checkpoint；
3. `checkpoint_summary.csv`与`protocol_invariants.csv`：epoch、best checkpoint、hash、finite与prefix contract；
4. `analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_metrics_by_seed.csv`：Weather、ETTm1、
   ETTh2三dataset的A6_FULL三seed稳定性，只作noise sensitivity proxy；
5. `A6_FULL`是弱统一control；`A6_MEASURE`是同architecture、same seed/profile/initialization class的primary
   shared control；`SPEC96/192/336`只改变training support与validation selector。

### 3.2 统计量定义

对arm $a$、dataset $d$和prefix horizon $h$，定义相对`A6_MEASURE`的prefix gain：

$$
G_{a,d}(h)=100\left(1-\frac{\operatorname{MSE}_{a,d}(1{:}h)}
{\operatorname{MSE}_{\text{measure},d}(1{:}h)}\right).
$$

正值表示arm更好。dataset macro是五个$G_{a,d}(h)$的算术平均；不按dataset scale或sample count加权。

由dense prefix curve恢复coordinate bin $[u,v]$的平均MSE：

$$
E_{a,d}[u{:}v]=\frac{vM_{a,d}(v)-(u-1)M_{a,d}(u-1)}{v-u+1},
$$

其中$M(h)$是prefix MSE，$M(0)=0$。`bin gain`以同样的相对公式比较$E$。这不是新test access，而是同一
full-test squared-error sum的代数重分组。

`standard-horizon Pareto dominance`要求同一specialist在$\{96,192,336,720\}$上对`A6_MEASURE`全部
$G\ge0$且至少一项$G>0$。`dense-prefix AUC gain`先在每个dataset对H1..720 prefix MSE求均值，再计算相对gain；
由于specialist tail未受监督，该量只揭示tradeoff，不作为own-H effectiveness gate。

`seed CV proxy`为A6_FULL三seed MSE的sample standard deviation除以mean。它不是paired arm-difference的置信区间，
只能判断D18单seed residual是否与已知训练波动同量级。

### 3.3 Own-H frontier复核

| Specialist | Own-H test MSE macro gain | Positive datasets | Own-H coordinate-bin结果 |
| --- | ---: | ---: | --- |
| `SPEC96` | **+1.2748%** | **5/5** | H1–48 `+1.6949%`、H49–96 `+0.9698%`，均5/5 |
| `SPEC192` | -0.1386% | 1/5 | H97–192 `-0.1639%`，3/5 |
| `SPEC336` | -0.6385% | 1/5 | H193–336 `-0.3643%`，1/5 |

[Strong Evidence] H96存在局部、跨dataset一致的short-prefix optimization signal；这比旧256-row probe的
`4/10`正区间更完整，因为这里使用full official-test rows。但它只出现在1/3训练horizons，不能支持一个稳定的
multi-horizon frontier。

### 3.4 Cross-horizon与Pareto复核

| Arm | G(96) | G(192) | G(336) | G(720) | Pareto-dominant datasets | Dense-prefix AUC gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SPEC96` | +1.2748% | -42.8689% | -50.5911% | -49.6851% | 0/5 | -43.2792% |
| `SPEC192` | -0.1313% | -0.1386% | -30.7081% | -43.1947% | 0/5 | -27.9908% |
| `SPEC336` | -1.1514% | -0.8911% | -0.6385% | -32.2274% | 0/5 | -13.8652% |

specialist在own-H之后的巨大退化是未监督tail的预期结果，不应被写成architecture incapacity；但它说明D18没有
发现一个可同时保持shared-model coverage的Pareto alternative。post-hoc在每个standard cell选择三个specialists中
最优者，也只有H96为`+1.2748%/5 of 5`；H192、H336、H720分别为`-0.1386%/1 of 5`、
`-0.6385%/1 of 5`、`-32.2274%/0 of 5`。该test oracle只作上界，不能选择candidate。

### 3.5 Measure control解释

`A6_MEASURE`相对`A6_FULL`的full-test coordinate-bin结果为：

| Lead-time bin | Macro MSE gain | Positive datasets |
| --- | ---: | ---: |
| H1–48 | +3.5635% | 5/5 |
| H49–96 | +1.3962% | 5/5 |
| H97–192 | +1.0395% | 5/5 |
| H193–336 | +1.4239% | 5/5 |
| H337–720 | +2.4895% | 5/5 |

对应prefix H48/H96/H192/H336/H720也全部5/5正向，macro分别为
`+3.5635% / +2.3152% / +1.5776% / +1.5011% / +2.1109%`。因此measure benefit不是某个单一horizon或tail造成，而是覆盖整个future domain的稳定
training-control effect。

### 3.6 Split、checkpoint与seed sensitivity

- specialist-vs-measure validation/test sign agreement仅`9/15`；validation本身为`+0.4205%`、7/15 cells，
  test为`+0.1659%`、7/15，二者都未形成完整frontier；
- 15/15 specialists训练7–20 epochs，0/15 best epoch卡预算边界；25/25 checkpoints、numeric与prefix invariants
  通过，故没有`optimization_or_numeric_pathology`证据；
- 在有三seed A6_FULL reference的9个cells中，D18 residual的绝对值只有5/9超过A6 seed CV；H96为3/3略高于
  CV，但这不是paired confidence interval，也不能替代specialist多seed确认；
- D18同时改变training support与validation selector，所以它检验的是“practical own-H optimization frontier”，
  不能把残差严格分解为representation capacity、loss weighting或checkpoint selection的单一因果效应。

## 4. 四层证据与 failure attribution

### 4.1 Paper-facing effectiveness

本轮没有method candidate。作为problem-existence evidence，own-H macro只有`+0.1659%`、7/15 cells；
stable multi-horizon frontier不成立。official test是primary decision surface；dense curves只作lead-time diagnosis。

### 4.2 Matched attribution

`A6_MEASURE > A6_FULL`在15/15 D18 own-H cells及全部五个dense bins稳定成立，解释了specialist相对弱control的
大部分表面收益。specialist与measure同architecture/params/initialization class，但objective和checkpoint selector按
问题定义不同，因此只可归因为horizon-specific optimization package，不可归因为H semantics。

### 4.3 Internal health

intervention实际改变prediction、prefix gradients与checkpoint；无collapse、non-finite或budget truncation。
H96 local signal真实存在于当前seed，但没有扩展为H192/H336或Pareto dominance。

### 4.4 Failure attribution

1. `hypothesis_false`：对“当前A6 natural carrier上存在稳定、跨horizon finite-capacity frontier”的精确命题是
   primary attribution；
2. `capacity_control_explains`：对相对A6_FULL的表面specialist优势成立，A6_MEASURE是主要解释；
3. `intervention_point_wrong`：不作为D18 failure主因，loss support确实改变；
4. `readout_or_head_design_wrong`：不支持，specialists与control共享readout；
5. `optimization_or_numeric_pathology`：不支持；
6. `unresolved_local_short_horizon_tradeoff`：H96 5/5 positive保留为局部线索，但不足以授权H-conditioned method、
   seeds或soft-projectivity rescue。

最终决定：`finite_capacity_frontier_not_supported`。这里的`not_supported`指paper-level existence evidence不足，
不是数学上证明任意finite model都不存在tradeoff。

## 5. D22-C conditional design freeze（不实现）

### 5.1 为什么仍允许设计

D22-A/B已经回答finite-capacity specialization，但没有直接比较“future coordinate × raw-history evidence access”。
D14-A1的neutral/A6 dual-carrier、three-seed scope crossing与strict oracle headroom仍说明不同future regions可能需要
不同history evidence；D21只否定split-stable nonlinear route-risk interaction，D17只关闭frozen post-hoc correction。
因此证据足以设计一次不同问题的小诊断，但不足以授权method。

### 5.2 Primary question

> 在fixed past、same information set与matched capacity下，target coordinate与ordered raw-history patches的joint
> access，是否在validation-fit到official-test transfer中稳定超过global compression、pooling、order/query
> shuffles与generic coordinate-conditioned control？

论文主语若未来成立，应是`lead-time-conditioned evidence operator`；ordered patch memory只是当前最小可审计载体。

### 5.3 Frozen arm roles

| Arm | Role | Required boundary |
| --- | --- | --- |
| `GLOBAL_COMPRESSED` | fixed-dimensional global raw-history state | primary baseline；same trainable parameter budget |
| `POOLED_MEMORY` | patch values/stats可用但去除ordered token access | 分离memory amount与ordered access |
| `ORDERED_TARGET_ACCESS` | target-coordinate query读取ordered raw-history patches | diagnostic intervention；不是method candidate |
| `ORDER_SHUFFLED` | 固定随机patch permutation，保留值分布与capacity | order specificity control |
| `TARGET_SHUFFLED_QUERY` | train-only固定置换target coordinates，保持query marginal | target-coordinate specificity control |
| `GENERIC_MATCHED` | flattened/MLP coordinate-conditioned raw-history control | 隔离attention/retrieval primitive与generic capacity |
| `A6_SENSITIVITY_*` | 在frozen A6 representation上对称复刻关键arms | conditional sensitivity only；不得方向级拒绝E2E路线 |

neutral/raw-history六臂是primary。A6 sensitivity只有在primary protocol有效后才解释representation dependence；
任何frozen replacement gap都不得作为paper-core effectiveness或方向级kill gate。

### 5.4 Split、统计与gate

- train split拟合probe/operator；validation只选择一个全局shared regularization/width并冻结checkpoint；official test
  在完整5 datasets × `{96,192,336,720}`矩阵上一次性评估，标记`test_informed`；
- primary statistic为`ORDERED_TARGET_ACCESS`相对每个control的20-cell MSE gain；同时报告MAE、dataset/horizon
  macro、coordinate bins、prediction diversity与parameter gap；
- problem gate要求ordered arm同时超过`GLOBAL_COMPRESSED`、`POOLED_MEMORY`、`ORDER_SHUFFLED`、
  `TARGET_SHUFFLED_QUERY`和`GENERIC_MATCHED`：每项macro MSE至少`+0.3%`，其中对global与generic至少
  `+0.5%`；关键两项至少11/20 cells、3/5 datasets、3/4 horizons正向，MAE非负；
- validation与test关键comparison必须同号，且不得出现non-finite、>100% cell degradation、parameter gap>1%或
  shuffled control构造泄漏；
- gate通过只返回Step4 source-informed design，不把diagnostic arm升级为method；失败且protocol/numeric有效则
  停止当前deterministic-MSE fixed-past architecture search；pathology则只标记
  `diagnostic_invalid_for_direction_rejection`并回Step3修复一次。

### 5.5 Authorization

`D22-C design_only / implementation_authorized=false / remote_training_authorized=false /
official_test_access_authorized=false`。

在另行完成static protocol、data leakage、matched capacity、validation selector与test metadata gate前，不得实现
production model、启动remote training或恢复CTD。Contribution 2继续保持open，不预设loss/router。

## 6. 最终研究决定

1. D22-A完成：pure request H不构成同一coordinate的新Bayes information；
2. D22-B完成：`finite_capacity_frontier_not_supported`，H96仅保留局部线索；
3. 不实现H embedding、horizon router、soft-projectivity或第二loss；
4. D22-C只冻结diagnostic design，等待独立prelaunch authorization；
5. 若D22-C有效失败，rollback到Step 2并停止当前deterministic-MSE fixed-past architecture search；随后只能收束
   现有projective unified paper，或显式pivot到known-future context、decision/coherence risk、compute/resolution、
   probabilistic joint target之一。
