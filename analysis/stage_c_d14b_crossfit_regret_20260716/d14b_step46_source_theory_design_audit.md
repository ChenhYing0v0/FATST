# StageC D14-B1 Source, Theory And Design Audit

## Status

| Field | Value |
| --- | --- |
| `current_step` | superseded before Step7A；retained as historical diagnostic design |
| `role` | `diagnostic_only_crossfit_coupling_risk_predictability` |
| `narrative_gate` | historical conditional pass；later consistency audit retired paper-core route |
| `remote_training` | false |
| `paper_method` | false |
| `test_access` | false |
| `rollback` | completed：CCRL retired；PCSD returned to native architecture Step4-6 |

> [Supersession 2026-07-16] 后续审计确认独立fold experts、partial OOF labels与最终joint PCSD之间存在
> teacher-student architecture mismatch、stale supervision和不属于最终推理图的工程开销。本设计未进入Step7A，
> 只保留为未来secondary control。active route见
> `analysis/stage_c_pcsd_native_reset_20260716/pcsd_cf_step46_source_theory_design_audit.md`。

## 1. What We Plan To Test

D14-A1已经证明point/block/global coupling scopes在五数据集、three seeds和两个carrier上存在稳定crossing，并且
sample × target-region oracle相对static bin policy仍有6.7948%/8.599% headroom。D14-B1只回答下一层问题：

> 不观察future truth时，仅根据inference-visible history与target coordinate，能否估计不同coupling scopes的
> conditional risk，并把该估计转化为超过fixed、target-only和ordinary direct-fusion policy的forecast gain？

它不是PCSD implementation，也不把oracle label当作可部署输入。

## 2. External Source Audit

本轮检索日期为2026-07-16，external primary sources优先。Zotero只作seed，不作为覆盖或novelty evidence。

| Source | What It Already Covers | Boundary For D14-B |
| --- | --- | --- |
| [FFORMA](https://www.sciencedirect.com/science/article/pii/S0169207019300895) | 由series features学习多个forecast methods的sample/series-level weights；指出随机选择风险与最终weighted-average forecast并非同一目标 | feature-based weighting、meta-learning和soft combination不是创新 |
| [TimeFuse, ICML 2025](https://proceedings.mlr.press/v267/liu25cm.html) | sample-level meta-features驱动heterogeneous forecasting models的adaptive fusion | history-conditioned sample-level fusion与direct fused loss是mandatory control |
| [TimeRouter, 2026 preprint](https://arxiv.org/abs/2606.11625) | context/CV/forecast features、oracle-best labels、nonlinear router、selective fallback与train OOF threshold selection | oracle expert labels、forecast snippets、nonlinear routing与fallback均不能claim；其preprint只作freshness pressure |
| [AME-TS, 2026 preprint](https://arxiv.org/abs/2605.25166) | 用forecastability/seasonality/trend/sparsity soft prior和KL alignment稳定MoE specialization | structural descriptor prior和KL-guided routing不能claim |
| [TimeExpert, 2025 preprint](https://arxiv.org/abs/2509.23145) | patch/timestamp local experts、shared global expert与query-dependent selection | local/global expert mixture与query routing本身不能claim；其作用点主要在history attention而非future-output coupling |
| [Learning to Defer, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html) | cost-sensitive expert selection与consistent surrogate | regret/cost-sensitive reduction不是创新 |
| [Calibrated Learning to Defer, ICML 2022](https://proceedings.mlr.press/v162/verma22c.html) | one-vs-all calibrated expert correctness | calibrated OvA/hard expert correctness不是创新 |
| [Time-series prediction and online learning, COLT 2016](https://proceedings.mlr.press/v49/kuznetsov16.html) | nonstationary time-series的regret/model-selection theory | 不claim首次time-series regret learning或model selection guarantee |

[Official Code Audit] TimeFuse官方实现的`ModelFusor`是single linear layer + softmax，并直接在weighted model
predictions上支持MSE/MAE/Huber/MixLoss；其meta-features覆盖statistics、ACF/stationarity、rate-of-change、AR与
spectrum。D14-B采用“inference-visible descriptors + direct fused-loss control”，但不复制其模块，也不采用
validation训练fusor/test评估的split协议，因为本项目official validation必须保持decision-only。TimeRouter官方仓库
冻结四个TSFMs，使用XGBoost 3.1.3、margin/diversity gate和CV-inverse fallback。D14-B把hard-oracle与
forecast-feature router作为control，不复制fallback；本地环境没有XGBoost，因此使用已安装的sklearn
`HistGradientBoostingRegressor`作nonlinear family sensitivity，而不是引入新依赖。

[Decision] generic “history predicts best expert”已高度拥挤，甚至不足以单独支撑Contribution 2。CCRL只有在完整
链条成立时保留conditional novelty：`one projective decoder内部的coupling-scope arms -> chronological
cross-fitted sample × target-region risk -> auxiliary risk distillation -> matched direct-fusion/hard-oracle controls ->
no requested-H semantics`。

## 3. Theory Feasibility

### 3.1 Conditional risk is estimable from out-of-fold errors

令coupling arm $s$的full-domain prediction为$f_s(x,\tau)$，conditional mean为$\mu(x,\tau)$。当$f_s$没有用
当前OOF target训练时：

$$
\mathbb E[(Y_\tau-f_s)^2\mid x]
=\operatorname{Var}(Y_\tau\mid x)+(\mu-f_s)^2.
$$

因此任意两arm的expected loss difference会消去不可约noise：

$$
\mathbb E[\ell_s-\ell_r\mid x]
=(\mu-f_s)^2-(\mu-f_r)^2.
$$

这为history-conditioned relative-risk regression提供了可识别目标。若expert已用同一样本target训练，上式的
OOF独立边界被破坏，in-sample error会产生co-adaptation/self-evaluation bias；所以cross-fitting不是装饰性步骤。

### 3.2 Realized best arm is a noisy label

单个sample上的$\arg\min_s\ell_s$受future noise影响。D14-B1不把hard winner当作primary truth，而预测centered
relative risk：

$$
r_{i,b,s}=\ell_{i,b,s}-\frac1{|\mathcal S|}\sum_j\ell_{i,b,j}.
$$

centering保持所有pairwise risk differences，并避免把共同sample difficulty误写成某一arm的competence。hard
oracle classification只作为TimeRouter/learning-to-defer-style control。

### 3.3 Regret weights are not automatically optimal mixture weights

对weights $p_s\ge0,\sum_sp_s=1$及$f_p=\sum_sp_sf_s$，squared loss满足：

$$
\sum_sp_s(Y-f_s)^2-(Y-f_p)^2
=\sum_sp_s(f_s-f_p)^2\ge0.
$$

所以weighted expert risk是mixture loss的上界，但两者相差prediction-diversity/cancellation项。直接把
$\operatorname{softmax}(-r)$称为optimal fusion是错误的。最终可行objective必须以actual fused forecast loss为主，
cross-fitted risk只作auxiliary supervision：

$$
\mathcal L_{hybrid}
=\mathcal L_{forecast}(f_p,Y)
+\lambda\,\operatorname{SmoothL1}(\hat r,r^{cf}).
$$

因此D14-B同时运行matched `DIRECT_FUSION`与`CCRL_HYBRID`；只有hybrid的增量才能归因给regret supervision。

### 3.4 Projectivity and horizon boundary

每个arm先生成$T=720$ full-domain forecast，policy只依赖$x$与natural target coordinate $u=\tau/T$：

$$
F_T(x)_\tau=\sum_sp_s(x,u)f_s(x,\tau),\qquad F_H=\mathcal R_HF_T.
$$

requested $H$不进入expert、risk estimator或policy，因此crop前后的prefix严格相同。target coordinate描述future
position，不是benchmark horizon ID；dataset ID和manual horizon rule同样禁止。

## 4. Leakage-Free Artifact Construction

### 4.1 Outer chronological cross-fitting

沿用D12已验证的purged forward contract：两个outer folds的OOF ranges为train windows的`[0.6,0.8)`与
`[0.8,1.0)`，fit prefix分别止于OOF前，past+future raw coverage之间冻结`purge_windows=1439`。每fold固定抽取
512个OOF windows；所有channel rows保持origin group，不做row-random split。

### 4.2 Checkpoint and official-validation separation

本次不能直接复用D14-A1的best-official-validation checkpoints作为primary evidence，因为同一validation随后还要
判断policy。每个outer fit prefix内部再保留purged tail作inner checkpoint selection。最终experts在完整train上按
outer folds的median inner-best epoch refit；official validation从未参与expert、policy、temperature或$\lambda$
选择，只作一次decision surface。

existing D14-A1 checkpoints只能作descriptive compatibility reference，不能进入primary D14-B gate。

## 5. Feature Contract

每个normalized history row构造36维train-independent descriptors：

- distribution 5维：skew、excess kurtosis、min、max、IQR；
- dynamics 5维：diff std、mean absolute diff、zero-crossing rate、turning-point rate、linear slope；
- autocorrelation 8维：lags `1,3,6,12,24,48,96,192`；
- spectrum 6维：spectral entropy、dominant frequency/power、low/mid/high band fractions；
- recent multiresolution 12维：windows `24,48,96,192`各自mean/std/slope。

target coordinate为11维：region center $u$、normalized width、$u^2$及$k=1..4$的sin/cos Fourier features。
这些features对所有datasets固定，不做per-dataset tuning。

为防止“简单regression head能力不足”造成假失败，primary使用matched two-layer 64-64 GELU risk MLP；另运行
`sklearn HistGradientBoostingRegressor`作为nonlinear family sensitivity。只有两类均失败，才允许把negative
解释为direction-level evidence；tree-only positive只触发policy redesign，不能通过paper method。

## 6. Frozen Arms

| Arm | Inputs / Objective | What It Tests |
| --- | --- | --- |
| `B0_OOF_BEST_FIXED` | train-OOF aggregate选择一个scale | static causal baseline |
| `B1_EQUAL_MIXTURE` | uniform weights | forecast-combination baseline |
| `B2_TARGET_ONLY_RISK` | target coordinate，risk loss | static distance policy |
| `B3_HISTORY_ONLY_RISK` | history descriptors，risk loss | sample policy但不区分future region |
| `B4_HISTORY_TARGET_RISK` | history + target，cross-fit risk loss | predictability problem primary |
| `B5_HISTORY_TARGET_DIRECT_FUSION` | matched policy，actual fused forecast loss | TimeFuse/ordinary fusion control |
| `B6_CCRL_HYBRID` | direct fusion + cross-fit centered-risk loss | Contribution-2-specific primary |
| `B7_HARD_ORACLE_HYBRID` | direct fusion + hard best-arm label | TimeRouter/L2D-style control |
| `B8_IN_SAMPLE_REGRET_HYBRID` | same hybrid但用in-sample expert errors | self-evaluation leakage control |
| `B9_PERMUTED_CROSSFIT_REGRET` | permute risk labels within target region | falsification control |
| `B10_RANDOM_HISTORY_HYBRID` | matched random history features | capacity/control signal |
| `B11_FORECAST_FEATURE_CCRL_CONTROL` | history/target + inference-visible forecast snippets | TimeRouter-style practical upper control；不计novelty |
| `B12_ORACLE_UPPER_BOUND` | validation future truth | descriptive only |

所有neural policy arms保留相同input slots与64-64 network；缺失feature block以zero mask代替，避免head capacity
变化解释B2/B3/B4差异。

## 7. Gates

### Gate B-P: predictability problem

`B4_HISTORY_TARGET_RISK`必须同时：

1. 至少3/5 datasets超过`B0`，five-dataset macro MSE gain `>=0.3%`；
2. 至少3/5超过`B2`，macro `>=0.2%`，证明history有target-only之外的增量；
3. 至少3/5超过`B3`，macro `>=0.1%`，证明target-region interaction有增量；
4. seed confirmation时至少2/3同方向；
5. primary MLP fail但tree positive只记`readout_or_head_design_wrong`，不作direction pass或fail。

### Gate B-C: CCRL contribution-specific value

`B6_CCRL_HYBRID`必须：

1. 至少3/5 datasets超过matched `B5_DIRECT_FUSION`，macro MSE gain `>=0.2%`；
2. 至少3/5超过`B7_HARD_ORACLE_HYBRID`与`B8_IN_SAMPLE_REGRET_HYBRID`；
3. `B9/B10`不能复制收益；
4. 无policy collapse、non-finite、validation reversal或单一dataset解释；
5. three-seed confirmation至少2/3同方向。

Gate B-P通过而B-C失败，只证明adaptive fusion值得研究，不足以保留CCRL作为Contribution 2。TimeFuse和
TimeRouter已经使generic router/fusion claim失去novelty。

## 8. Staged Execution

1. Step7A只实现local synthetic/split/leakage/objective-identity checks；
2. Step7B先运行`neutral_raw, seed2021, five datasets`；
3. 只有B-P与B-C同时通过，才运行`a6_natural, seed2021`；
4. dual-carrier seed2021通过后，才授权seeds2022/2023 confirmation；
5. D14-B全部通过也只让PCSD/CCRL返回formal method Step4-6；不直接读test或实现paper method。

## 9. Failure Attribution Boundary

D14-B1可能阻断的是“conditional risk可预测”或“CCRL supervision有独立价值”，而不是D14-A已经确认的
coupling crossing本身。失败必须按以下边界解释：

- `hypothesis_false`：在split/leakage/invariant全部通过、MLP与tree两类readout都稳定训练的前提下，B-P仍失败；
  这才构成关闭instance-adaptive predictability claim的证据，但不否定PCSD problem evidence；
- `intervention_point_wrong`：policy权重没有实际进入full-domain arm fusion，或carrier输出抹除了arm差异；只返回
  Step6修复tensor path，不能否定方向；
- `readout_or_head_design_wrong`：MLP失败而`HistGradientBoostingRegressor`通过，或出现明显underfit；只说明
  policy family不合适，结果标记为`diagnostic_invalid_for_direction_rejection`；
- `optimization_or_numeric_pathology`：non-finite、collapse、validation reversal或异常大幅退化；只拒绝当前
  implementation/protocol，并返回Step6；
- `capacity_control_explains`：B5 direct fusion、B9 permuted或B10 random-history复制B6收益；这会关闭CCRL的
  独立training claim，但不能把generic fusion gain归因给cross-fitted risk。

因此，只有matched stable diagnostic同时排除后四类解释后，B-P negative才允许记为`hypothesis_false`；B-C
negative最多关闭SC2-CCRL，PCSD仍返回Step4重新寻找与multi-horizon sharing直接相关的mechanism。

## 10. Narrative Gate And Decision

[Narrative Gate: Conditional Pass For Diagnostic] 问题与multi-horizon主线直接相连：同一decoder内部future targets
如何选择sharing scope。tensor path和supervision source可解释，且D14-A已确认headroom。Step7A diagnostic值得做。

[Novelty Risk: High] generic sample-level fusion、oracle routing、structural-prior KL与local/global experts均已有强
prior art。CCRL只能作为与PCSD不可分割的training principle，且必须在matched direct-fusion/hard-oracle controls
上形成独立增量。否则关闭SC2，不用改名保留。

[Historical Decision, Superseded] 本报告曾给出`authorize_d14b1_step7a_local_only`，但后续training-consistency
audit已在implementation前撤销该授权。当前D14-B1 local/remote/paper method/test均false；active decision为
`authorize_pcsd_cf_step7a_local_only`。
