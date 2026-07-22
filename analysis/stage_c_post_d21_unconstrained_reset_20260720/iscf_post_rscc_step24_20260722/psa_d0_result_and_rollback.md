# SC-ISCF-PSA-D0 Result and Rollback

## 1. Executive decision

Decision=`frozen_inference_shrinkage_not_supported`。

Primary convex-uniform shrinkage在15-run leave-one-dataset-out evaluation上的macro L1/MSE gain为
`-0.2431%/-0.1218%`，只有1/5 datasets、2/15 runs两项同时为正。4/5 folds虽然在other-dataset fit rows上选择了
nonzero alpha，但大多在held-out dataset反转；这拒绝`H1 inference_weight_overfit`，也明确禁止继续做alpha grid、
dataset、seed或future-position tuning。

该结果不是ISCF或joint-training regularization的方向级否定。frozen probe没有重建ARMERR/SHUFFLED route loss在训练期
引起的arm-policy co-adaptation，且historical EQUAL没有contemporaneously retrain。remaining hypotheses为
`H2 training_coadaptation`与`H3 contemporaneous_retraining_drift`；二者只能由新五dataset、seed2021、no-route EQUAL
retrain control区分。该control尚未授权，当前active method仍none，remote training与official test均false。

## 2. Protocol and artifact audit

| Audit | Result |
| --- | --- |
| source artifacts | 15/15 existing EQUAL validation replays |
| primary matrix | five held-out datasets × three seeds = 15 runs |
| split | 147 fit / 109 evaluation rows；source-sample aligned |
| selection | five-fold leave-one-dataset-out；fixed alpha grid |
| primary family | convex uniform |
| controls | convex scope-marginal；temperature-to-uniform |
| tensors | arms `[256,5,720]`；policy `[256,720,5]`；targets `[256,720]` |
| finite/simplex/scale checks | pass |
| forecast training | 0 |
| checkpoint mutation | 0 |
| official test access | 0 |

## 3. Primary result

| Metric | Result | Frozen gate | Pass |
| --- | ---: | ---: | --- |
| macro L1 gain | `-0.2431%` | `>0` | no |
| macro MSE gain | `-0.1218%` | `>0` | no |
| joint-positive datasets | `1/5` | `>=4/5` | no |
| joint-positive runs | `2/15` | `>=12/15` | no |
| folds selecting alpha > 0 | `4/5` | `>=4/5` | yes |
| matrix complete | `15/15` | complete | yes |

Dataset result：

| Held-out dataset | LODO alpha | L1 gain | MSE gain | Joint positive |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | `0.30` | `-0.0906%` | `-0.2564%` | no |
| ETTh2 | `0.00` | `0.0000%` | `0.0000%` | no |
| ETTm1 | `0.20` | `+0.0702%` | `+0.1297%` | yes |
| ETTm2 | `0.50` | `-0.1558%` | `-0.3186%` | no |
| Weather | `0.75` | `-1.0392%` | `-0.1639%` | no |

只有ETTm1获得很小正值。不得把结果收缩为ETTm1-specific方法：dataset tuning被预注册禁止，而且该幅度远小于后续
paper-core所需的`+0.3%`级别margin。

## 4. Selection reversal

Convex-uniform在source fit rows选择的alpha及held-out L1 gain为：

| Held-out | Source-fit L1 gain | Held-out L1 gain | Interpretation |
| --- | ---: | ---: | --- |
| ETTh1 | `+0.0915%` | `-0.0906%` | reversal |
| ETTh2 | `0.0000%` | `0.0000%` | source selects no change |
| ETTm1 | `+0.0340%` | `+0.0702%` | weak replication |
| ETTm2 | `+0.0876%` | `-0.1558%` | reversal |
| Weather | `+0.1746%` | `-1.0392%` | severe reversal |

这表明one-dimensional shrinkage本身足以在fit surface产生小正值，却缺少cross-dataset generalization。Weather尤其说明
aggressive shrinkage不能作为global safeguard；不能用source-fit positive解释held-out failure。

Baseline entropy与held-out L1 gain的Spearman仅`0.1643`。低entropy不是足以预测shrinkage收益的risk signal，因而也不
支持以policy entropy直接构造新loss/router。

## 5. Matched diagnostic controls

| Family | Macro L1 gain | Macro MSE gain | Joint-positive runs | Joint-positive datasets |
| --- | ---: | ---: | ---: | ---: |
| convex uniform | `-0.2431%` | `-0.1218%` | 2/15 | 1/5 |
| convex scope marginal | `-0.2570%` | `-0.1477%` | 3/15 | 1/5 |
| temperature | `-0.2615%` | `-0.2378%` | 4/15 | 2/5 |

保留source-dataset scope marginal没有修复结果；temperature smoothing也没有。三种不同趋近broad policy的
transformations一致为macro negative，因此D0 failure不是convex-uniform parameterization特有的design fault。

## 6. Future-position diagnostics

Primary convex-uniform在四个disjoint position segments上的macro L1/MSE为：

| Positions | L1 gain | MSE gain | Joint-positive runs |
| --- | ---: | ---: | ---: |
| `[0,96)` | `-0.0405%` | `+0.2226%` | 8/15 |
| `[96,192)` | `-1.1322%` | `-2.1197%` | 5/15 |
| `[192,336)` | `-0.3878%` | `-0.2062%` | 8/15 |
| `[336,720)` | `-0.1193%` | `+0.0192%` | 3/15 |

没有segment在L1/MSE上同时macro positive。temperature在`[0,96)`两项为正，但它是post-result diagnostic control，
overall family仍negative，且future-position-specific选择未预注册；不得据此新建short-horizon temperature method。

## 7. Four-layer interpretation

1. `paper_facing_effectiveness`：not evaluated；official test access=0。
2. `matched_mechanism_attribution`：three frozen shrinkage families一致negative；H1不受支持。
3. `internal_mechanism_health`：arrays、simplex、split与matrix均健康；4/5 source folds确实选择nonzero change，说明probe不是
   identity-only，但held-out reversal显著。
4. `failure_attribution`：`frozen_probe_negative_joint_training_unresolved`。不是numeric pathology；也不能归为
   `hypothesis_false` for joint training，因为frozen replacement没有训练期co-adaptation。

## 8. Research consequence

- 关闭“post-hoc uniform shrinkage / temperature smoothing”作为candidate的可能性；
- 不把ARMERR/SHUFFLED公共gain解释为简单的inference overconfidence correction；
- generic entropy、balance、temperature或uniform-KL route不通过narrative gate，也不因controls正向而升级；
- RSCC/SCC exact route继续closed；不得借D0结果恢复其loss rescue；
- ISCF-v0保持fixed architecture base/carrier，active method=none；
- 下一最小识别实验是`SC-ISCF-PSA-D1` contemporaneous EQUAL retrain control：five datasets × seed2021、validation only，
  完全匹配RSCC matrix的code/config/init/selector。若new EQUAL复现historical EQUAL，则H3弱化、H2获得支持；若new
  EQUAL接近ARMERR/SHUFFLED，则公共gain主要由run drift解释。

D1是control-only，不是paper method。其design/launch必须另行冻结；remote training尚未授权，formal test保持false。

## 9. Artifacts

- `psa_d0_remote_analysis/decision.json`
- `psa_d0_remote_analysis/dataset_summary.csv`
- `psa_d0_remote_analysis/selection_curves.csv`
- `psa_d0_remote_analysis/selected_run_metrics.csv`
- `psa_d0_remote_analysis/evaluation_curves.csv`
- `psa_d0_remote_analysis/position_bin_metrics.csv`
- `scripts/analyze_stage_c_iscf_psa_d0.py`
- `configs/stage_c_iscf_psa_d0.json`
