# ISCF-RSCC-v1 Step9 Validation Result and Rollback

## 1. Executive decision

Decision=`rscc_v1_control_attribution_fail_close_exact_route`。

RSCC-v1在validation上相对EQUAL获得可见性能提升，但两个预注册matched controls都取得更大提升，并且RSCC没有改善
policy-to-coalition-credit alignment。因此本轮不是“RSCC没有效果”，而是“observed gain不能归因于exact coalition
scope binding”。failure attribution=`capacity_control_explains`；按预注册gate关闭exact SCC/RSCC route，不访问formal
test，不补seed、lambda、epsilon、fallback或router-width rescue。

ISCF-v0仍按用户要求保留fixed architecture base/carrier。`ISCF-EQUAL-ARMERR`可保留为strong validation
carrier/control，但validation-only结果不足以把它升级为paper method或正式替换paper-facing base。

## 2. Artifact/protocol audit

| Audit | Result |
| --- | --- |
| new/effective runs | 15/15 new；20/20 effective |
| standard-horizon cells | 80/80 validation cells |
| required artifacts | missing=0 |
| checkpoint hashes | 20/20 unique |
| paired initialization | all four arms paired within every dataset |
| objective contract | 20/20 match |
| trained invariants | 20/20 pass |
| official-test use | 0 runs；formal test unauthorized |
| supervisor completion | `2026-07-22T14:59:55+08:00` |

本轮只读取validation。checkpoint由mean validation MSE over H96/H192/H336/H720选择；没有用test选择epoch、
candidate或control。

## 3. Layer 1 — validation effectiveness

| Comparison | MSE gain | MAE gain | cells | datasets | horizons |
| --- | ---: | ---: | ---: | ---: | ---: |
| RSCC vs EQUAL | `+0.5189%` | `+0.3972%` | 15/20 | 5/5 | 4/4 |
| EQUAL-ARMERR vs EQUAL | `+0.6577%` | `+0.4476%` | 17/20 | 5/5 | 4/4 |
| RSCC-SHUFFLED vs EQUAL | `+0.6557%` | `+0.4544%` | 17/20 | 5/5 | 4/4 |

RSCC通过预注册primary gate：MSE超过`+0.3%`、MAE为正、5/5 datasets与4/4 horizons为正。其dataset
MSE gains为ETTh1 `+0.0432%`、ETTh2 `+0.0262%`、ETTm1 `+1.7590%`、ETTm2 `+0.3130%`、
Weather `+0.4531%`。其中ETTm1贡献最大，ETTh1/ETTh2仅为很小正值。

按horizon，RSCC相对EQUAL为H96 `+0.5941%`、H192 `+0.6574%`、H336 `+0.7037%`、H720
`+0.1204%`。这说明可靠性保留后，coalition objective package不再像SCC-v0那样破坏carrier；但尚不能说明
exact credit binding是收益来源。

该层是validation continuation evidence，不是paper-facing official-test effectiveness。

## 4. Layer 2 — matched mechanism attribution

| Required control | RSCC MSE gain | RSCC MAE gain | cells | datasets | horizons | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vs EQUAL-ARMERR | `-0.1414%` | `-0.0509%` | 3/20 | 1/5 | 1/4 | fail |
| vs RSCC-SHUFFLED | `-0.1394%` | `-0.0577%` | 3/20 | 1/5 | 1/4 | fail |

两个controls的预注册要求均为RSCC至少`+0.1%` MSE；实际均为negative。更关键的是，EQUAL-ARMERR与
RSCC-SHUFFLED彼此几乎完全相同：ARMERR vs SHUFFLED MSE=`+0.0020%`、MAE=`-0.0068%`。二者都比
EQUAL约高`+0.656%` MSE，且都比RSCC约高`+0.14%`。

这构成强matched attribution反证：

1. 保留arm reliability后，额外training regularization/package可改善validation carrier；
2. 将coalition credit与正确scope绑定没有贡献额外收益；打乱binding反而略优；
3. standalone-error control与shuffled control近乎相同，说明可见gain由非特异性objective/regularization effect解释，
   而不是exact leave-one-scope-out credit语义。

RSCC只在ETTm2相对两个controls取得dataset-level正值，并只在H96获得约`+0.02%--0.03%`的horizon-level正值；
这不足以越过任何预注册guard，也不得事后收紧为ETTm2/H96-specific method。

## 5. Layer 3 — internal mechanism health

健康项：

- 15/15 new runs数值finite；
- 每run五个scope gradient paths全部nonzero；minimum recorded norm=`0.0461`；
- RSCC五个datasets的policy usage均为5/5 scopes；
- RSCC median coalition oracle headroom=`+18.2940%`，保持为正，EQUAL为`+18.0775%`；
- RSCC training argmax alignment约`0.303--0.394`，说明credit loss确实进入policy path。

失败项：

- RSCC median policy-credit Spearman=`0.1539`，低于EQUAL的`0.2052`；
- alignment只在ETTh2与ETTm2高于EQUAL，2/5 datasets；
- RSCC policy entropy明显低于near-uniform controls，说明它确实使policy更sharp，但sharpness没有转成更低forecast risk；
- large positive oracle headroom依然没有被learned fused forecast兑现。

因此这不是dead gradient、collapsed usage或numeric pathology。exact coalition loss改变了policy，却没有按理论方向改善
held-out policy-credit alignment，且performance被shuffled/no-coalition controls超过。

## 6. Layer 4 — failure attribution

Primary failure=`capacity_control_explains`，这里按治理定义包含“no-mechanism matched control解释收益”，并不意味着
参数量发生变化：四臂architecture/parameter class匹配。更精确的研究语言是
`matched_no_binding_objective_control_explains`。

- 不是`optimization_or_numeric_pathology`：finite、gradients、usage与artifacts均健康；
- 不是SCC-v0的`intervention_point_wrong`重复：RSCC已成功保住arm reliability与positive headroom；
- 不能用primary positive覆盖attribution fail：shuffled/ARMERR取得更强且几乎相同的gain；
- 不能把D0B information-access pass直接升级为useful mechanism：可预测credit存在，不代表以KL匹配它能改善forecast。

## 7. Paper-story consequence and rollback

1. Exact claim `future-output coupling scopes -> leave-one-scope-out credit -> useful policy coordination`不成立；
2. SCC-v0与RSCC-v1共同关闭coalition-credit route；不再做同一loss的seed/weight/fallback/router rescue；
3. ISCF architecture保持fixed base，但当前仍没有active paper-core extension；
4. EQUAL-ARMERR只保留为strong carrier/control clue：它提示“reliability-preserving training regularization”可能有用，
   但其primitive已有prior且当前没有scope-specific attribution，不能直接包装成第二contribution；
5. rollback到11-step Step2/4：重新界定ISCF在fixed-past deterministic-MSE范围内尚未解决、且能被matched controls
   识别的问题；完成新的problem/narrative/design gate前，不实现新method、不remote train、不访问formal test或modern
   baselines。

## 8. Artifacts

- `rscc_step9_remote_analysis/decision.json`
- `rscc_step9_remote_analysis/run_audit.csv`
- `rscc_step9_remote_analysis/validation_metrics.csv`
- `rscc_step9_remote_analysis/comparison_cells.csv`
- `rscc_step9_remote_analysis/comparison_summary.csv`
- `rscc_step9_remote_analysis/internal_health.csv`
- `rscc_step9_remote_analysis/training_health.csv`
- `scripts/analyze_stage_c_iscf_rscc_step9.py`
