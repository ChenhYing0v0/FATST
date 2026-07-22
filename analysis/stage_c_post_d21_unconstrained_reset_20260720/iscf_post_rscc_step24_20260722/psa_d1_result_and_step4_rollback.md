# SC-ISCF-PSA-D1 Result and Step4 Rollback

## 1. Executive decision

Decision=`joint_training_route_regularization_supported_as_carrier_clue`。

Contemporaneous EQUAL在5/5 datasets上与historical EQUAL达到checkpoint SHA256、four-horizon validation metrics、
fused forecast、five arms与direct policy逐值完全相同。`new_equal_vs_historical` MSE/MAE=`0.0000%/0.0000%`，
recovery ratio=`0.0`；因此`H3 contemporaneous_retraining_drift`被排除。

ARMERR与SHUFFLED相对new EQUAL的validation MSE/MAE分别为`+0.6577%/+0.4476%`与
`+0.6557%/+0.4544%`，均17/20 cells、5/5 datasets、4/4 horizons为正。结合PSA-D0 post-hoc shrinkage
negative，当前证据支持：收益来自additional route constraint引起的joint-training arm-policy co-adaptation，而不是
checkpoint drift或inference-time smoothing。

该结论仍只是`carrier_clue`，不是paper method。ARMERR/SHUFFLED target semantics互不一致且彼此近乎同效，generic
load balancing/entropy/forecast-combination regularization已有直接prior；现阶段不能把任一control改名为创新。回Step4，
下一最小实验只应检验information-free uniform policy anchor是否足以复现公共gain。

## 2. Protocol completion

| Audit | Result |
| --- | --- |
| new training runs | 5/5 |
| effective runs | 20/20 |
| standard validation cells | 80/80 |
| H96/H192/H336/H720 metrics | complete |
| objectives | 20/20 expected=observed |
| initialization pairing | 5/5 datasets；four arms exact |
| artifacts/invariants | complete/pass |
| checkpoint diagnostic replay | 5/5；SHA nonmutation |
| numeric/gradient health | pass |
| official-test access | 0 |

Initial v0只完成Weather/ETTm1/ETTh1 training，原因是post-training evaluator缺少future-bin config后使workers退出；
ETTh2/ETTm2按原冻结matrix补跑。v0.1只修复evaluator contracts，未改变training command、objective、selector、gates或
已训练checkpoint。该fault在result前被标记并完整修复，不造成selective rerun。

## 3. H3 run-drift audit

| Evidence | Result |
| --- | --- |
| new vs historical macro MSE | `0.0000%` |
| new vs historical macro MAE | `0.0000%` |
| dataset/horizon/cell wins | `0/5`, `0/4`, `0/20`；全部tie |
| checkpoint SHA256 | 5/5 exact match |
| fused relative L1 | 5/5=`0.0` |
| policy mean L1 | 5/5=`0.0` |
| arms relative L1 | 5/5=`0.0` |
| recovery ratio | `0.0`，gate `>=0.75` fail |

同seed、同code path、同initialization与selector的EQUAL训练是逐bit reproducible。ARMERR/SHUFFLED公共gain不能由“最近
重训碰巧更好”解释。

## 4. H2 joint-training co-adaptation audit

### 4.1 Aggregate comparisons

| Comparison | MSE gain | MAE gain | Cells | Datasets | Horizons |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARMERR vs new EQUAL | `+0.6577%` | `+0.4476%` | 17/20 | 5/5 | 4/4 |
| SHUFFLED vs new EQUAL | `+0.6557%` | `+0.4544%` | 17/20 | 5/5 | 4/4 |

### 4.2 Dataset MSE gains

| Dataset | ARMERR | SHUFFLED |
| --- | ---: | ---: |
| Weather | `+0.5071%` | `+0.5265%` |
| ETTm1 | `+2.3385%` | `+2.3098%` |
| ETTh1 | `+0.1253%` | `+0.1797%` |
| ETTh2 | `+0.0857%` | `+0.0793%` |
| ETTm2 | `+0.2321%` | `+0.1836%` |

### 4.3 Horizon MSE gains

| Horizon | ARMERR | SHUFFLED |
| --- | ---: | ---: |
| H96 | `+0.5616%` | `+0.5663%` |
| H192 | `+0.7982%` | `+0.8054%` |
| H336 | `+0.9644%` | `+0.9538%` |
| H720 | `+0.3067%` | `+0.2975%` |

两controls的收益跨所有dataset/horizon aggregates为正，但主要margin由ETTm1与H192/H336贡献；ETTh1/ETTh2较小。
因此它是stable validation clue，不得表述为所有cells的大幅提升。

## 5. Training and internal health

New EQUAL 5 runs分别训练16/9/8/9/6 epochs；minimum scope gradient norms为
`0.0585/0.0494/0.0725/0.0638/0.1342`，全部finite/nonzero。Maximum route weight与weighted route loss均
严格为0，证明它没有意外route intervention。

已有RSCC health evidence显示ARMERR/SHUFFLED final policy entropy均接近1，并学习到functionally near-equivalent
forecasts；RSCC exact binding更sharp但性能更差。PSA-D0进一步表明把frozen EQUAL policy向uniform/marginal/temperature
post-hoc smoothing均macro negative。因此最符合全部证据的解释是：broad route constraint在训练过程中改变了policy
allocation，继而改变fused-loss流向arms/shared encoder的gradient path；收益依赖co-adaptation，不能在训练后替换weights
得到。

这仍是mechanistic inference，尚未由uniform-target matched intervention直接确认。

## 6. Four-layer interpretation

1. `paper_facing_effectiveness`：not evaluated；official test access=0。
2. `matched_mechanism_attribution`：H3被exact retrain否定；H2得到two independent no-binding controls与D0 negative共同支持。
3. `internal_mechanism_health`：new EQUAL exact reproducibility、all gradients/finite/route-zero通过；controls已有near-uniform
   policy与near-equivalent function evidence。
4. `failure_attribution`：`none_control_clue_only`。不是method pass；缺失的attribution层是“broad anchor本身”相对
   ARMERR/shuffled targets的matched intervention。

## 7. Narrative and novelty boundary

当前可支持的问题陈述是：

> 在ISCF dense multi-scope fusion中，policy不仅决定inference mixture，也在joint training中分配fused-loss gradients；
> unconstrained direct policy可能形成不利的arm-policy co-adaptation，而broad train-time policy constraints能改善carrier。

不能支持的claim：

- ARMERR credit语义有效；
- shuffled coalition语义有效；
- exact scope binding有效；
- generic entropy/load balancing是新贡献；
- validation single-seed clue已经达到paper effectiveness。

## 8. Rollback and next gate

回Step4，冻结下一最小diagnostic candidate=`SC-ISCF-UPA-D2 — Uniform Policy Anchor`：保留EQUAL fused + reliability
loss，只增加与ARMERR/SHUFFLED同weight/schedule的information-free uniform-target policy KL。它回答“target semantics
是否完全不必要，公共gain是否仅由broad joint-training anchor产生”。

D2必须是diagnostic/control-only；即使positive也不能直接成为paper method。只有D2支持problem后，才允许结合ISCF
scope semantics与primary sources重新做完整narrative/design gate。D2 implementation与new remote training当前未授权。

## 9. Artifacts

- `psa_d1_remote_analysis/decision.json`
- `psa_d1_remote_analysis/run_audit.csv`
- `psa_d1_remote_analysis/validation_metrics.csv`
- `psa_d1_remote_analysis/comparison_cells.csv`
- `psa_d1_remote_analysis/comparison_summary.csv`
- `psa_d1_remote_analysis/function_drift.csv`
- `psa_d1_remote_analysis/training_health.csv`
- remote output root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_psa_d1`
