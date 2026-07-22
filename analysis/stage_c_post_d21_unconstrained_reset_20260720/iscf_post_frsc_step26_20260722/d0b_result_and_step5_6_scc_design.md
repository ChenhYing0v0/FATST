# ISCF-SCC D0B Result and Step5–6 Design

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | D0B Step9 complete；SCC returned to Step5–6 and narrative-ready for Step7A |
| `problem` | ISCF arms有coalition utility，但equal-skill不提供协作角色信号，current policy未稳定兑现headroom |
| `existence_evidence` | D0 oracle signal强；D0B target-free held-out gain与shuffle/standalone attribution完整通过 |
| `idea` | 用exact leave-one-scope-out L1 risk构造train-only coalition credit，校准existing direct policy |
| `theory_check` | 不改变Bayes target/requested-H boundary；target只进入training credit；inference graph/参数量不变 |
| `design` | SCC + EQUAL/FUSED/ARMERR/SHUFFLED matched controls；20 new seed2021 validation runs + 5 reused parent runs |
| `narrative_gate` | `pass_to_step7a_matched_validation_only` |
| `effectiveness_gate` | not evaluated；formal test仍false |
| `artifacts` | D0B corrected CSV/JSON；本Step5–6 contract |
| `decision` | `scc_v0_narrative_ready_step7a_authorized` |

## 2. Corrected D0B result

第一次153/103 split因切开multivariate channel group被标记为`diagnostic_protocol_fault_predecision`。唯一有效结果使用
147/109 source-sample-aligned split，所有预注册gate均通过：

| Statistic | Corrected result |
| --- | ---: |
| runs | 15/15 |
| median target-free predicted L1 gain | `1.372656%` |
| positive gain | 15/15 |
| three-metric binding above shuffle p95 | 14/15 |
| median gain over standalone-credit probe | `0.514328` percentage point |
| gain over standalone positive | 13/15 |
| all-three-seeds positive datasets | 5/5 |

dataset median predicted gains为ETTh1 `.6894%`、ETTh2 `10.3603%`、ETTm1 `.1723%`、ETTm2
`1.6819%`、Weather `1.3727%`。唯一shuffle binding failure为ETTm2 seed2023；standalone胜出的两个run为ETTm1
seed2022和ETTm2 seed2023。故evidence是strong但非universal。

[Decision] `coalition_credit_information_access_supported_return_step5_6`。D0固定标签seed topology的失败仍保留，但不再
阻塞dynamic credit candidate，因为D0B直接验证了within-run target-free held-out predictability与cross-seed gain sign。

## 3. Narrative gate

SCC primitive与leave-one-expert-out、counterfactual routing和Shapley prior重叠，不能claim首创。可辩护的完整链为：

```text
ISCF future-output coupling fields
-> dense joint scope forecasts with exact fusion algebra
-> individual-target supervision conflicts with cooperative roles
-> no-extra-forward leave-one-scope coalition risk
-> train-only calibration of the existing policy
-> unchanged inference graph
-> matched fused-only / standalone-error / shuffled-binding attribution
```

D0/D0B使problem、information access和task-specific coupling获得直接证据，因此narrative gate从
`conditional_pass_to_diagnostic_only`提升为`pass_to_step7a_matched_validation_only`。这仍不等于paper-core通过；只有完整
official-test effectiveness与matched attribution后才能写成第二项贡献。

## 4. Exact SCC-v0 objective

现有ISCF tensors为`arms A [B,C,720,5]`、`policy P [B,C,720,5]`、`fused [B,720,C]`和
`target [B,720,C]`。对每个scope：

$$
\hat y_{-s}=\frac{\hat y-p_sa_s}{\max(1-p_s,10^{-6})},\qquad
\Delta_s=\operatorname{stopgrad}(|\hat y_{-s}-y|-|\hat y-y|).
$$

令$q_s=[\Delta_s]_+/\sum_j[\Delta_j]_+$；若全部非正，回退uniform。SCC loss为

$$
\mathcal L_{SCC}=\mathcal L_{harmonic\text{-}L1}(\hat y,y)
+\lambda(\tau)\,\mathcal L_{harmonic\text{-}KL}(q,P).
$$

`lambda`在前25% optimizer updates从0线性升至`.1`，此后固定；不做coefficient grid。SCC不含uniform
individual-arm target loss。`q`完整stop-gradient，因此KL只校准policy；arms仍从fused loss joint E2E学习。inference不增加
feature、router、requested-H、参数或latency。

## 5. Matched controls

| Arm | Existing/new objective | Role |
| --- | --- | --- |
| `ISCF-EQUAL` | existing `equal_skill` | frozen parent；5 historical seed2021 checkpoints可在hash/contract一致后复用 |
| `ISCF-FUSED` | existing `measure_only` | 删除individual loss与credit的objective control |
| `ISCF-ARMERR` | existing `pointwise_route_only` | standalone target-error credit control |
| `ISCF-SCC` | new `scope_coalition_credit` | candidate |
| `ISCF-SCC-SHUFFLED` | new `scope_coalition_credit_shuffled` | 保持credit vector、随机破坏scope binding |

SHUFFLED使用dedicated seeded generator逐`[B,C,T]` coordinate产生independent scope permutation，不消费model/data全局
RNG。五arms保持same architecture、seed2021、data、harmonic measure、optimizer、four-horizon checkpoint selector和from-scratch
initialization class。新训练为4 arms × 5 datasets=20 runs；EQUAL parent只在initialization/protocol/hash audit通过时复用，否则
补训。

## 6. Validation gate

development surface固定为5 datasets × `{96,192,336,720}` validation MSE/MAE；checkpoint score为四horizon mean
validation MSE。不得按dataset/horizon调lambda、fallback或schedule。

SCC continuation要求同时满足：

1. vs EQUAL macro MSE gain至少`.3%`、MAE严格正、至少3/5 datasets和3/4 horizons正；
2. vs FUSED、ARMERR、SHUFFLED的macro MSE gain各至少`.1%`；
3. no numeric pathology；policy-credit alignment提高、oracle headroom下降、至少3 scopes保持nonzero usage/gradient；
4. 全20 new-run matrix完整，所有negative cells报告。

validation只决定是否申请formal three-seed test，不建立paper effectiveness。formal test当前未授权。

## 7. Failure attribution and authorization

- SCC不超过FUSED：`removal_of_equal_skill_explains`，回Step4；
- SCC不超过ARMERR：`standalone_credit_sufficient`，回Step4；
- SCC不超过SHUFFLED：`capacity_or_regularization_control_explains`，回Step4；
- diagnostics健康但performance弱：`intervention_point_wrong`，回Step5审计arm-gradient path；
- divergence/extreme denominator：`optimization_or_numeric_pathology`，只拒绝exact v0；
- positive validation未经过formal test：`performance_partial_pass`，不得promotion。

```text
active_method = SC-ISCF-SCC-v0_narrative_ready_preimplementation
step7a_implementation_authorized = true
remote_training_authorized = false_until_step7a_checks_and_launch_record
formal_test_authorized = false
modern_baseline_matrix_authorized = false
next_action = implement_exact_loss_and_step7a_contract_tests
```
