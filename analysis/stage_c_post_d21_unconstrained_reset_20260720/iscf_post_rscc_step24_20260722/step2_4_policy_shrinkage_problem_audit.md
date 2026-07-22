# ISCF Post-RSCC Step2/4 Policy Shrinkage Problem Audit

## 1. Executive decision

Decision=`policy_shrinkage_problem_unresolved_proceed_d0_diagnostic_only`。

RSCC-v1失败后的新证据不支持继续修复coalition loss，却暴露了一个更基础的问题：在同一ISCF architecture与
`equal_skill` reliability contract下，ARMERR和SHUFFLED两个语义不同、均不含正确scope binding的route losses，
不仅取得几乎相同的validation提升，而且产生functionally near-equivalent的near-uniform policies。working
hypothesis是learned direct policy具有超过当前finite validation evidence所能支持的conditional flexibility，joint
training中的broad-target route regularization可能通过policy shrinkage改善泛化。

该假设当前仍`unresolved`，因为EQUAL是historical checkpoint而非本轮contemporaneous retrain；可见公共收益也可能来自
joint-training co-adaptation或run drift。下一步只允许`SC-ISCF-PSA-D0 — Policy Shrinkage Attribution`：复用15个
existing EQUAL validation replays，对冻结policy作预定义convex shrinkage，不训练、不访问official test、不改变
checkpoint。D0只检验`inference_weight_overfit`，不作为paper method，也不能以negative result拒绝joint-training方向。

## 2. Long-stage record

| Field | Content |
| --- | --- |
| `current_step` | 11-step loop Step2/4；D0为diagnostic-only |
| `problem` | ISCF direct policy可能在有限数据下过度条件化，使fusion未充分兑现stable arm diversity |
| `existence_evidence` | ARMERR/SHUFFLED共同超过EQUAL约0.656% validation MSE，且function-level近似、policy near-uniform |
| `idea` | 先用frozen convex shrinkage隔离inference-time policy flexibility，再决定是否存在值得Step4设计的ISCF-specific问题 |
| `theory_check` | convex shrinkage降低policy自由度但不增加information；Bayes boundary不变，只检验finite-sample/function-class effect |
| `design` | 15 existing EQUAL replays；fixed alpha grid；source-sample-aligned 147/109；LODO alpha selection；uniform/marginal/temperature controls |
| `narrative_gate` | `diagnostic_only_pass`；generic balancing/shrinkage prior overlap过强，paper method gate未通过 |
| `effectiveness_gate` | not applicable；validation diagnostic不能建立paper-facing effectiveness |
| `artifacts` | RSCC Step9 outputs、existing SCC D0 replay NPZ、后续PSA-D0 analysis outputs |
| `decision` | `policy_shrinkage_problem_unresolved_proceed_d0_diagnostic_only`；若D0不支持，回Step2并申请contemporaneous EQUAL control，而非拒绝ISCF |

## 3. Code and artifact evidence

### 3.1 Objective common factor

`baselines/timealign_official/layers/PCC.py`中，EQUAL、ARMERR、RSCC和SHUFFLED均使用
`skill_kind="equal"`，因此共享相同的fused loss与uniform arm-reliability loss。差别只在route target：ARMERR为
`pointwise`，RSCC为`coalition`，SHUFFLED为`coalition_shuffled`；total loss为

$$
\mathcal L = \mathcal L_{\mathrm{fused}} + \mathcal L_{\mathrm{equal\ skill}}
+ \lambda_{\mathrm{route}}\mathcal L_{\mathrm{route}}.
$$

RSCC Step9的matched comparisons为：

| Comparison | Validation MSE gain | Validation MAE gain |
| --- | ---: | ---: |
| RSCC vs EQUAL | `+0.5189%` | `+0.3972%` |
| ARMERR vs EQUAL | `+0.6577%` | `+0.4476%` |
| SHUFFLED vs EQUAL | `+0.6557%` | `+0.4544%` |
| ARMERR vs SHUFFLED | `+0.0020%` | `-0.0068%` |

正确coalition binding没有解释公共收益；该route已关闭，禁止继续做loss weight、seed、fallback、temperature或router
rescue。

### 3.2 Function-level audit

复用five dataset、seed2021 validation probes，直接比较相同source rows上的arrays。ARMERR与SHUFFLED的fused
prediction relative L1分别为：Weather `0.00301`、ETTm1 `0.00138`、ETTh1 `0.00350`、ETTh2
`0.00150`、ETTm2 `0.00462`；policy mean L1分别为`0.00830/0.00527/0.00254/0.00294/0.00693`。
对应arm prediction relative L1仅约`0.0024--0.0096`。

作为contrast，RSCC与ARMERR的policy mean L1为`0.0178--0.1114`，fused relative L1为
`0.0076--0.0265`。因此ARMERR与SHUFFLED不是只在macro metric上偶然同分，而是学习到了相近的forecast
function。

最终训练日志进一步显示：

- ARMERR policy entropy约`0.986--0.996`；
- SHUFFLED policy entropy约`0.991--0.998`；
- RSCC policy entropy约`0.775--0.965`；
- 两个controls的route-target entropy并不相同，但最终policy都接近uniform。

这些事实支持“common shrinkage/balancing effect”作为problem clue；它们不证明具体的因果机制。

### 3.3 Critical confound

EQUAL来自historical FCC/SCC reference，三个新arms来自RSCC contemporaneous matrix。环境、four-horizon checkpoint
selector、from-scratch status和paired initialization hash均匹配，但EQUAL没有在同一launch中重新训练。因此三种解释必须
保持分离：

1. `H1 inference_weight_overfit`：仅把frozen EQUAL policy向uniform收缩即可改善held-out risk；
2. `H2 training_coadaptation`：route loss通过训练期共同改变arms与policy，post-hoc shrinkage不能复现；
3. `H3 contemporaneous_retraining_drift`：公共gain来自新一轮训练或环境中未记录的run-level drift。

D0只能检验H1。H1失败不等于H2/H3成立，也不能拒绝ISCF或joint-training regularization。

## 4. Bayes/task boundary

在fixed past $X$、pointwise MSE、且requested horizon $H$不携带额外信息时，同一future coordinate $Y_t$的Bayes
predictor仍为$\mathbb E[Y_t\mid X]$，不能因requested horizon改变。因此本路线不引入H embedding/router，也不声称
改变Bayes optimum。

令EQUAL的五臂forecast为$a_s(X,t)$，direct policy为$p_s(X,t)$。D0定义

$$
p_{\alpha,s}(X,t)=(1-\alpha)p_s(X,t)+\alpha/S,
\qquad
\hat y_\alpha(X,t)=\sum_{s=1}^{S}p_{\alpha,s}(X,t)a_s(X,t).
$$

$\alpha$不访问target、不增加feature，也不改变arms；它只沿learned policy到uniform policy的one-dimensional
function frontier降低conditional flexibility。若held-out risk在$\alpha>0$稳定下降，支持的是finite-sample policy
complexity problem，而不是新的information source或horizon-dependent Bayes rule。

## 5. Latest primary-source audit

Search date=`2026-07-22`。范围包括`time-series mixture routing balance`、`forecast combination weight shrinkage`、
`stacked forecasting varying weights`、`diversity time-varying weights`、`MoE balancing theory`。Zotero coverage未被用作
novelty completeness evidence；以下均为external primary sources。

| Primary source | Relevant boundary |
| --- | --- |
| [Qian et al., *Forecast Combination Puzzle*](https://arxiv.org/abs/1505.00475) | simple average可因weight estimation error超过复杂组合；必须保留uniform/simple-average control，generic shrinkage不是新问题 |
| [Hasson et al., ICML 2023](https://proceedings.mlr.press/v202/hasson23a.html) | stacked-generalization family可显式限制weights随item/time/quantile变化；“控制combination-weight complexity”已有直接理论prior |
| [DTVW, 2025](https://arxiv.org/abs/2508.07136) | 用forecast diversity构造predictive prior学习time-varying combination weights；直接覆盖generic diversity/disagreement-aware weighting |
| [GateTS, 2025](https://arxiv.org/abs/2508.17515) | 面向TS MoE的balanced routing与无需auxiliary balancing loss已有直接工作；generic entropy/load balancing不能作为贡献 |
| [$\phi$-Balancing, ICML 2026](https://openreview.net/forum?id=DZbzIOguz4) | population-level balancing已有principled objective，且指出minibatch assignment heuristics可偏；generic balance loss不能作为新机制 |
| [FAME, 2026](https://arxiv.org/abs/2606.08896) | forecastability fingerprint与validation-mined expert suitability已用于routing；data-driven suitability/router claim已有高度邻近prior |
| [Dense Backpropagation, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f69707de866eb0805683d3521756b73f-Abstract-Conference.html) | inactive-expert dense training signals已用于改善MoE optimization；“更密的expert learning signal”也非空白 |

Novelty consequence：simple entropy regularization、uniform KL、temperature scaling、diversity-aware weight、generic
stacking/shrinkage和load balancing均不得直接升级为Contribution 2。即使D0支持H1，也必须回Step4构造并审计完整的
`ISCF scope semantics -> finite-evidence policy complexity -> native constraint/training contract -> matched attribution ->
paper-facing gain`链，且需要明确区别于上述prior。

## 6. SC-ISCF-PSA-D0 frozen design

### 6.1 Role and data

- `role=diagnostic_only_validation_frozen_probe`；
- 数据：15个existing EQUAL validation replays，five datasets × seeds 2021/2022/2023；
- arrays：`probe_arms [256,5,720]`、`probe_direct_policy [256,720,5]`、
  `probe_targets [256,720]`；
- row split沿用corrected source-sample boundary：147 fit rows / 109 evaluation rows；不得切开multivariate channel group；
- official test access=0；checkpoint mutation=0；forecast training=0。

### 6.2 Statistics

对每个run与$\alpha$计算：

1. `evaluation_l1`与`evaluation_mse`；
2. 相对$\alpha=0$的`gain_l1_percent`与`gain_mse_percent`；
3. `policy_entropy_before/after`与mean policy L1 movement；
4. 每dataset、seed、horizon-bin的gain，用于stability而非选择；
5. baseline policy entropy与best frozen shrinkage gain的Spearman；
6. across-fold selected alpha与risk curve flatness。

### 6.3 Selection without dataset/test tuning

冻结grid=`[0.0,0.05,0.10,0.20,0.30,0.50,0.75,1.0]`。采用five-fold
leave-one-dataset-out：每fold只在其余四个datasets的147 fit rows、all three seeds上，以macro mean L1选择一个
$\alpha$；该值随后固定到held-out dataset的109 evaluation rows与all seeds。不得按seed、horizon或held-out result
选$\alpha$。

### 6.4 Controls

| Control | Purpose |
| --- | --- |
| `alpha=0 learned policy` | source EQUAL function |
| `alpha=1 uniform policy` | endpoint/simple average；确认是否只是完全放弃learned policy |
| `row-marginal policy prior` | 区分uniform shrinkage与保留global scope-frequency的shrinkage |
| `temperature-to-uniform` | 区分probability-space convex shrinkage与generic logit smoothing |

`row-marginal`与temperature只作diagnostic controls，不进入候选method。所有controls必须使用同一arms、rows与targets；
不得重新训练或按held-out dataset调参。

### 6.5 Frozen gate

`finite_policy_shrinkage_problem_supported`仅当LODO convex shrinkage同时满足：

- macro evaluation L1 gain $>0$且MSE gain $>0$；
- 至少4/5 held-out datasets两项均正；
- 至少12/15 runs L1与MSE均正；
- 5个fold中至少4个selected $\alpha>0$；
- selected alpha不是5/5 folds均为`1.0`，否则decision降为
  `uniform_fusion_endpoint_supported_not_conditional_shrinkage`；
- 所有arrays finite、matrix complete、split与no-test contracts通过。

若方向正但未满足稳定性，decision=`finite_policy_shrinkage_problem_unresolved`。若macro、dataset与run stability均不
支持，decision=`frozen_inference_shrinkage_not_supported`；该negative只拒绝H1，failure attribution为
`frozen_probe_negative_joint_training_unresolved`，不允许direction-level rejection。

## 7. Validation/test roles and rollback

- validation fit rows只选global alpha；validation evaluation rows验证finite-policy frontier；
- official test完全禁止，本诊断不能建立paper-facing performance；
- D0 positive只允许回Step4做source-informed narrative/design gate，不自动授权method；
- D0 negative时，不做alpha-grid、dataset-specific或seed rescue。下一合理动作是设计一个five-run seed2021
  contemporaneous no-route EQUAL retrain control来区分H2/H3；该动作涉及new remote training，必须另行冻结设计并获授权；
- frozen replacement的negative不得用于拒绝joint-training方向。

## 8. Self-critique

[Strong Evidence] 两个no-binding controls的公共gain与function-level near-equivalence是真实且一致的problem clue。

[Uncertain] 把该clue解释为“overconfident policy”仍可能过早：near-uniform policy也可能只是route loss减弱了不稳定的
policy-arm co-adaptation，或本轮新训练共同产生了arms drift。D0只隔离post-hoc function frontier，不能重建训练动力学。

[Narrative risk] 即使D0为正，generic shrinkage/balancing已有强prior。若不能提出ISCF-native、必要且可被matched controls
识别的scope-specific contract，本路线应停留在diagnostic/carrier改进，不应为了补足contribution数量而包装成paper core。
