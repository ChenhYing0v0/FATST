# SC-D15-T1 PCSD-CF-v1 Official Test Audit

## Decision

| Field | Result |
| --- | --- |
| `current_step` | PCSD-CF-v1 Step9-10 completed；PCC Step6 next |
| `candidate_version` | `SC1-PCSD-CF-v1` frozen |
| `test_access_date` | 2026-07-16 |
| `user_authorization` | 2026-07-16 explicit；primary milestone effectiveness gate |
| `matrix` | 12 arms × 5 datasets × seed2021 = 60/60 |
| `checkpoint_selection` | historical best-validation H720 MSE |
| `checkpoint_retraining` | false；60/60 SHA-256 invariant pass |
| `primary_result` | DIRECT vs A6 test -1.3994%，1/5 |
| `method_decision` | `rejected_effectiveness_test` |
| `diagnostic_decision` | `test_fail_with_arm_headroom` |
| `PCC` | Step6 test-informed design authorized；implementation/remote false |

## Why The Audit Matters

validation负责选择checkpoint，但论文最终performance结论必须看official test。本audit没有重新训练，也没有用
test挑epoch或配置；它一次性评估预注册完整矩阵，用于判断validation failure是否只是split artifact，以及PCC的
credit-starvation前提是否仍存在。

## Primary Effectiveness Result

DIRECT相对各reference的dense-H1..720 MSE AUC gain如下；正数表示DIRECT更好。

| Reference | Validation | Test | Test wins |
| --- | ---: | ---: | ---: |
| A6 | -1.5833% | -1.3994% | 1/5 |
| EQUAL | -0.0294% | -0.4984% | 1/5 |
| STATIC | -0.6266% | -0.5304% | 2/5 |
| DENSE matched | +2.3492% | -0.8942% | 1/5 |
| RANDOM partition | +0.4499% | -0.1164% | 2/5 |

逐dataset的DIRECT-vs-A6 test gain：

| Dataset | Validation | Test |
| --- | ---: | ---: |
| ETTh1 | -0.7801% | +1.5338% |
| ETTh2 | -2.6817% | -0.7352% |
| ETTm1 | -2.3408% | -2.6867% |
| ETTm2 | -1.9043% | -3.8896% |
| Weather | -0.2098% | -1.2196% |

primary threshold要求至少3/5 wins且macro gain不低于+0.3%。实际只有1/5且macro为负，故PCSD-CF-v1明确失败。
validation→test在25个dataset × reference cells中有13个sign reversals，尤其`DIRECT > DENSE`从5/5变1/5；但
DIRECT-vs-A6在两split均整体失败，所以test没有挽救method结论。

## Mechanism Diagnostic

- same-run oracle headroom test macro为+2.0197%，3/5 datasets为正；
- 25/25 DIRECT scope arms低于对应independently-trained fixed scope，median degradation 90.6647%；
- policy未数值collapse，所有5 datasets存在arm separation与至少一个skilled arm；
- independently-trained best fixed scope只在1/5 datasets超过A6，不能把scope family本身视为已验证method。

[Fact] exact PCSD-CF-v1的plain fused training与policy没有形成可发表的test优势。

[Strong Evidence] joint model内部仍有未被fused output稳定利用的relative arm skill，且same-run arms遭遇严重credit
starvation；这使PCC的研究问题仍可检验。

[Uncertainty] oracle headroom是post-hoc upper bound，不证明history可预测正确arm，也不证明PCC能恢复independent
training能力。test上的dense/random reversal还提示split/seed sensitivity。

## Failure Attribution And Rollback

本次否定层级是`exact implementation/method`，不是coupling-scope problem本身。归因同时包含：

1. `capacity_control_explains`：DIRECT未超过dense matched，method specificity不成立；
2. `readout_or_head_design_wrong` / training interaction：25/25 arms显著under-trained；
3. 尚不能写成`hypothesis_false`：D14-A的coupling crossing和本次same-run oracle headroom仍存在。

因此按预注册map执行`test_fail_with_arm_headroom`，回到SC2-PCC Step6，而不是直接实现PCC：先冻结
`MEASURE_ONLY/EQUAL_SKILL/CAPABILITY_SKILL_ONLY/ROUTE_ONLY/full PCC`、generic balancing与dense controls，明确
gradient path、moving-target稳定性、训练成本和hard rollback gate。任何后续architecture/objective变化升级为新的
`test_informed` candidate version，不能再次使用PCSD-CF-v1 test做调参。

## Artifact Accounting

- `gate.json`：60-run formal gate与macro comparisons；
- `deep_dive_gate.json`：oracle、fixed-scope与25-pair arm attribution；
- `validation_test_comparison.csv`：逐reference/dataset split comparison；
- `audit_decision.json`：完整性、frozen hash、decision与PCC授权；
- `raw/**/test_audit_metrics_by_target_horizon.csv`：60个逐H official test metrics；
- `raw/**/test_audit_invariants.json`：60个checkpoint/test protocol invariants。
