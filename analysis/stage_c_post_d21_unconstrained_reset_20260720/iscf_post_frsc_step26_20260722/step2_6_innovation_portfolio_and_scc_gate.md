# ISCF Post-FRSC Step2–6 Innovation Portfolio and SCC Gate

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | 用户扩大research scope后重新执行Step2–6；candidate仍需D0 problem diagnostic |
| `problem` | ISCF已有independent future-output scope fields和可观arm complementarity，但`equal_skill`把所有arm直接拉向同一future target，existing direct fusion又没有稳定兑现互补性 |
| `existence_evidence` | ISCF-v0 vs A6_FULL test MSE/MAE=`+1.3584%/+0.9144%`；oracle headroom median=`8.5813%`；fusion只在9/15 runs超过best fixed arm；五个arm共享同一uniform target auxiliary |
| `idea` | primary working route=`SC-ISCF-SCC-v0`：保留ISCF inference architecture，以closed-form leave-one-scope-out coalition risk定义train-only scope credit，校准既有direct policy；arm只通过fused forecasting risk学习，不再以uniform individual target loss强迫同质化 |
| `theory_check` | 不改变fixed-past MSE的Bayes target；target只在training credit中出现，inference不输入requested-H/label；primitive counterfactual routing已有prior，novelty只能位于ISCF task-specific full chain |
| `design` | 先复用existing ISCF artifacts完成D0；若通过，再冻结SCC/parent/fused-only/arm-error/shuffled-credit matched validation matrix；formal test需另行预注册和授权 |
| `narrative_gate` | `conditional_pass_to_d0_only`；full paper-core gate尚未通过 |
| `effectiveness_gate` | not evaluated；本轮无implementation、training或new test access |
| `artifacts` | 本报告；ISCF function/SAC/FCC、CPSI、SPS、FRSC existing reports and tensors |
| `decision` | `scc_problem_diagnostic_proposed_active_method_none`；rollback point=Step2/4 |

## 2. 用户范围更正与不变量

[Fact] 用户明确要求：不再把研究限制为“不允许loss/training/architecture变化”，但仍固定ISCF为后续研究的
architecture base，不以旧的canonical/random negative否定ISCF。目标同时包含：连贯paper narrative、补充创新性与
official-test性能提升。

本轮据此撤销“FRSC失败后不得探索loss/training”的局部执行限制，但不撤销以下研究规范：

1. ISCF-v0 core仍固定：five independent scope fields、future-output coupling groups和joint E2E carrier身份保留；
2. 任何新增模块必须解释其与ISCF tensor/gradient path的必要耦合，不能只是generic plugin；
3. 不为凑contribution预选第二个loss/router；method贡献数由通过matched attribution的机制决定；
4. official test仍是formal effectiveness surface；validation只用于checkpoint、ordinary choice与continuation；
5. 先做problem/narrative/design gate，再实现、remote train或访问new formal test。

## 3. 已有证据如何定位新问题

### 3.1 ISCF不是需要重新寻找的弱baseline

[Strong Evidence] ISCF-v0是当前最接近paper delivery的architecture carrier：three-seed official-test相对A6_FULL的
MSE/MAE为`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds正向。该结果是test-informed的
performance evidence，不等于ISCF canonical scope claim已经成立。

### 3.2 当前瓶颈不是“arm完全相同”，而是credit assignment

existing 15-run function audit给出：

- oracle headroom median `8.5813%`，每run八个future bins中成为best的scope数median=3；
- fusion只在9/15 runs超过best fixed arm；Weather 0/3、ETTm1 1/3；
- policy entropy跨dataset差异很大：ETTh1约`.989`，Weather约`.325`；
- pairwise function topology在4/5 datasets跨seed稳定，但canonical order footprint弱。

[Inference] arm function diversity和conditional complementarity已经存在；继续增加generic diversity regularizer不是首要缺口。
更直接的问题是：现有policy是否按照“一个arm加入当前coalition后使fused risk降低多少”分配权重。

### 3.3 `equal_skill`存在可验证的objective conflict

当前training tensor contract为：

- `arm_forecasts`: $A\in\mathbb R^{B\times C\times T\times S}$，$S=5$；
- `policy`: $P\in\mathbb R^{B\times C\times T\times S}$；
- `fused_forecast`: $\hat Y\in\mathbb R^{B\times T\times C}$；
- `target`: $Y\in\mathbb R^{B\times T\times C}$。

`equal_skill`实际使用harmonic coordinate measure $\omega_t$和pointwise L1，优化

$$
\mathcal L_{equal}=\sum_t\omega_t|\hat y_t-y_t|
+\frac{1}{S}\sum_{s=1}^{S}\sum_t\omega_t|a_{s,t}-y_t|.
$$

在fixed past、pointwise L1 risk下，每个unconstrained arm的individual Bayes target都是同一个conditional median；若
改为pointwise MSE，则都是同一个conditional mean $\mathbb E[Y\mid X]$。因此第二项不是specialization objective；
它是anti-starvation/generalist objective。
它可能帮助optimization，却与“各scope承担不同协作角色”的叙事存在结构张力。

[Self-critique] 不应从该公式直接断言arm必然collapse：不同scope maps、finite rank、nonconvex optimization和policy
gradient仍可产生差异，existing artifacts也证明确有差异。可成立的结论仅是：uniform individual target loss没有提供
coalition-specific role signal，且该缺口尚未被matched fused-only control隔离。

## 4. 2024–2026 primary-source audit

检索日期：`2026-07-22`。query/topic scope：time-series MoE specialization loss、expert-loss integration、
counterfactual/leave-one-expert-out routing、Shapley expert contribution、structural prior、diversity/orthogonality、
segment/frequency experts。来源为arXiv、OpenReview、NeurIPS proceedings、PMLR primary pages。Zotero FSA subset
presence本轮未可靠核验，以下均按external discovery处理；这降低“库内缺失即novel”的任何推断。

| Primary work | Covered mechanism | Boundary for ISCF |
| --- | --- | --- |
| [Expert Loss Integration, arXiv 2026](https://arxiv.org/abs/2605.10330) | global loss + individual expert losses；masked calibration windows；partial online learning | “expert-specific loss促进specialization”不能claim |
| [Advancing Expert Specialization, NeurIPS 2025](https://openreview.net/forum?id=iydmH9boLb) | orthogonality loss + routing variance loss | generic diversity/orthogonality/decisive routing不新 |
| [AME-TS, arXiv 2026](https://arxiv.org/abs/2605.25166) | temporal-structure descriptors形成routing prior | seasonality/trend/forecastability anchor不新 |
| [MoHETS, arXiv 2026](https://arxiv.org/abs/2601.21866) | shared temporal expert + routed Fourier experts | heterogeneous temporal/frequency roles不新 |
| [Seg-MoE, arXiv 2026](https://arxiv.org/abs/2601.21641) | contiguous segment-level routing | temporal locality/segment routing不新 |
| [Shapley-MoE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c66a9db149261435664284a20b6f1d42-Abstract-Conference.html) | coalition marginal contribution用于expert pruning | Shapley/expert marginal value primitive不新 |
| [TIGER, arXiv 2026](https://arxiv.org/abs/2606.15765) | leave-one-expert-out causal contribution监督frozen heterogeneous-expert router | counterfactual routing loss本身有直接prior |
| [LEO, OpenReview 2026](https://openreview.net/forum?id=HaQCWrbP5Z) | leave-one-expert-out training与intrinsic cross-validation | expert removal/dropout training不新 |
| [Counterfactual Routing Analysis, arXiv 2026](https://arxiv.org/abs/2605.07260) | 对equal-compute alternative routes做counterfactual utility audit | counterfactual blind-spot diagnosis不新 |
| [RPATH, TMLR 2026](https://openreview.net/pdf?id=kwpDOqas2x) | time-series MoE post-hoc counterfactual routing explanation | counterfactual diagnostics不能作为method novelty |
| [MP-MoE, ICML 2026](https://openreview.net/forum?id=7FsbfQgti4) | diversity-aware dynamic ensemble pruning | diversity-aware subset selection不新 |

[Decision] SCC不能声称发明counterfactual expert contribution、causal routing、expert loss或diversity training。
可能保留的contribution-level边界仅是：

```text
future-output coupling scopes
-> dense jointly-trained scope forecasts
-> exact no-extra-decoder coalition risk from the ISCF fusion algebra
-> train-only contribution calibration of the existing policy
-> matched fused-only / standalone-error / shuffled-credit attribution
-> unified-horizon official-test improvement
```

该完整链与TIGER的frozen heterogeneous VFMs + task instructions、Shapley-MoE的post-training pruning、
Expert Loss Integration的individual errors不同；但在D0和matched E2E结果前只能称`plausibly differentiated`，不能称novel。

## 5. Candidate portfolio

| Priority | Route | Role | Decision |
| --- | --- | --- | --- |
| A | `SC-ISCF-SCC-v0 — Scope Coalition Credit` | primary diagnostic-gated candidate | continue to D0 |
| B | fused-only / equal-to-fused annealing / metric-aligned MSE | objective-conflict与base-loss controls | required controls；not standalone contribution |
| C | common/private or pre-synthesis interaction successor | architecture backup | deferred；exact CPSI failed materially |
| D | fixed spectral/projector/full-rank conditioning | architecture family | exact SPS/FRSC closed；no rescue |
| E | orthogonality, load balance, entropy, structural anchor, frequency experts | generic specialization shortcuts | deprioritized by direct prior and local evidence |

选择A不是因为它最复杂，而是它唯一同时对应三个已证实事实：ISCF carrier强、arms有coalition headroom、policy未稳定兑现
headroom。B必须存在，因为删除`equal_skill`本身可能解释全部收益；若SCC不超过B，credit mechanism不成立。

## 6. SCC mathematical contract

ISCF dense fusion在每个$(b,c,t)$处为

$$
\hat y=\sum_{s=1}^{S}p_sa_s,\qquad \sum_s p_s=1.
$$

对第$s$个scope做renormalized removal：

$$
\hat y_{-s}=\frac{\hat y-p_sa_s}{\max(1-p_s,\epsilon)}.
$$

v0首先使用与parent相同的pointwise L1 risk，定义

$$
\Delta_s=\operatorname{stopgrad}\left[|\hat y_{-s}-y|-|\hat y-y|\right].
$$

$\Delta_s>0$表示在当前coalition和当前coordinate中，移除arm $s$会增大risk。由其正部构造credit target
$q_s$；当所有$\Delta_s\le0$时回退到uniform或skip该coordinate，具体语义必须在Step6冻结。working objective为

$$
\mathcal L_{SCC}=\mathcal L_{fused}
+\lambda(\tau)\operatorname{KL}(q^{coalition}\|p).
$$

重要边界：

1. 不保留uniform individual arm-to-target loss；arms通过与parent同measure的fused L1与ISCF function class共同学习；
2. $q$和$y$只在training存在，inference graph、参数量、latency和requested-H输入不变；
3. $\hat y_{-s}$由已有$A,P,\hat y$闭式计算，不重复调用Encoder/decoder；
4. SCC只训练既有policy还是同时改变arm gradient，必须由D0决定；v0默认只校准policy，避免一次改变两个机制；
5. `SCC coefficient=0 + fused-only objective`精确退化为matched control，而不是ISCF equal-skill parent；
6. MSE-risk SCC或MSE base loss只作后续独立factor，不在v0同时改变，避免把metric alignment与coalition credit混淆。

## 7. D0 existing-artifact problem diagnostic

### 7.1 Role and source

优先复用ISCF-v0的15个existing FCC checkpoints及其remote-preserved
`probe_arms [256,5,720]`、`probe_direct_policy [256,720,5]`、`probe_targets [256,720]`。
这是`diagnostic_only_test_informed_reuse`：不重新选择dataset/horizon，不训练，不建立effectiveness。

如existing test probe不足以回答history predictability，只允许用同一frozen checkpoint补一个validation replay；frozen
replacement/replay不能方向级拒绝后续E2E SCC。

### 7.2 Statistics

| Statistic | Definition and meaning |
| --- | --- |
| `loo_gain` | primary为L1 $\Delta_s$，MSE版本只作sensitivity；一个scope在当前coalition中的signed risk contribution |
| `positive_contributor_count` | 每row/bin中$\Delta_s>0$的scope数；判断credit是否退化为single winner或全负 |
| `policy_credit_spearman` | existing $p_s$与normalized positive $\Delta_s$的scope-rank correlation；量化当前policy alignment |
| `standalone_credit_spearman` | $-\ell(a_s,y)$与$\Delta_s$的correlation；判断coalition credit是否只是arm-error改名 |
| `credit_best_match` | standalone-best与coalition-best scope一致率 |
| `counterfactual_oracle_headroom` | 用target-visible $q$重组后相对current fusion的risk gap；只作upper bound |
| `credit_seed_stability` | 同dataset三seed的bin-level credit topology Spearman |
| `credit_policy_gap_by_entropy` | 按existing policy entropy分层的alignment/headroom；区分uniform与collapsed policy failure |

controls必须包括：scope-label shuffled credit、uniform credit、standalone arm-error credit，以及existing Q1-WIDE/
RANDOM-PARTITION sensitivity（若对应probe完整）。shuffled只破坏scope binding而保持credit marginals。

### 7.3 D0 decision

`coalition_credit_problem_supported`要求同时满足：

1. 15 runs的median `counterfactual_oracle_headroom >= 1%`；
2. 至少12/15 runs的median `positive_contributor_count >= 2`；
3. coalition-best与standalone-best match不超过80%，或两类credit Spearman不超过`.8`，证明不是PCC/CCSF重复信号；
4. 至少4/5 datasets存在跨seed稳定的bin-level credit topology（median Spearman至少`.5`）；
5. shuffled control不能保留同等alignment/headroom。

若1–2失败：`hypothesis_false`，回Step2，不实现SCC。若3失败：`capacity_control_explains`/existing arm-error credit覆盖，
回Step4。若4–5失败：`unresolved`，最多补validation frozen diagnostic，不进入Step7。

## 8. Conditional Step6 experiment plan

只有D0通过，才允许冻结下列same-initialization、same-profile、from-scratch validation matrix：

| Arm | Objective | Purpose |
| --- | --- | --- |
| `ISCF-EQUAL` | fused + uniform individual arm loss | frozen parent |
| `ISCF-FUSED` | parent harmonic fused L1 only | removal-of-equal-skill control |
| `ISCF-ARMERR` | fused + standalone arm-error credit | closest local/prior control |
| `ISCF-SCC` | fused + coalition credit calibration | candidate |
| `ISCF-SCC-SHUFFLED` | same credit marginals, shuffled scope binding | mechanism control |

development surface为five datasets × seed2021 × H96/192/336/720 validation MSE/MAE；所有arms共享four-horizon
checkpoint selector。候选continuation要求：

- SCC vs ISCF-EQUAL macro MSE至少`+0.3%`、MAE严格正、至少3/5 datasets和3/4 horizons；
- SCC vs ISCF-FUSED、ISCF-ARMERR、SCC-SHUFFLED的MSE均至少`+0.1%`；
- no numeric pathology；policy-credit alignment提高、oracle headroom下降、至少3 scopes保持nonzero usage/gradient；
- 不允许按dataset/horizon选择coefficient、schedule或fallback。

这些是validation continuation gates，不是formal mechanism pass。通过后必须另行冻结three-seed official-test full matrix；
test主比较为SCC vs ISCF-EQUAL，matched attribution比较为SCC vs FUSED/ARMERR/SHUFFLED，并保留A6_FULL performance
reference。official test仍是`test_informed`，不得描述为untouched holdout。

## 9. Paper narrative if and only if gates pass

暂定叙事不是“我们又设计了一个loss”，而是：

1. **Problem**：future-output coupling scopes提供多个有互补性的forecast fields，但independent expert accuracy不是
   coalition utility；uniform expert supervision会把roles重新拉向同一conditional point target；
2. **Architecture base**：ISCF在future output domain建立independent coupling fields，作为可学习的协作成员；
3. **Coordination mechanism**：SCC用ISCF fusion algebra得到exact train-time coalition contribution，并校准existing
   policy，不增加inference path；
4. **Evidence**：matched fused-only、standalone-error和shuffled-credit controls证明收益来自coalition-aware coordination，
   official-test MSE/MAE证明performance viability。

只有第4项完整通过后，paper contributions才可写成“ISCF architecture + ISCF-native cooperative training”的连贯双层链。
在此之前，ISCF仍是fixed carrier，SCC只是proposed mechanism，不把diagnostic或loss primitive计作第二创新点。

## 10. Failure attribution and authorization

- `hypothesis_false`：D0不存在stable/nondegenerate coalition contribution，回Step2；
- `capacity_control_explains`：FUSED、ARMERR或SHUFFLED解释收益，关闭SCC exact claim，回Step4；
- `intervention_point_wrong`：D0有signal但policy-only E2E无实现收益，回Step5审计arm-gradient intervention；
- `optimization_or_numeric_pathology`：只拒绝exact normalization/schedule，不拒绝ISCF architecture；
- `readout_or_head_design_wrong`：policy无法从history预测train-only credit，SCC router-calibration path关闭，不据此否定
  broader ISCF training research。

当前授权状态：

```text
active_method = none
SCC_status = proposed_diagnostic_gated
method_implementation_authorized = false
remote_training_authorized = false
formal_test_authorized = false
modern_baseline_matrix_authorized = false
next_action = D0 existing-artifact coalition-credit diagnostic
```
