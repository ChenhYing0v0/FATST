# iTransformer-style Decoder-HPO v2 Formal Result and Decision

Decision：`itransformer_decoder_hpo_v2_improves_v1_bsca_but_fails_original_gate_close_rescue`。

## 1. Formal completeness

- training artifacts：70/70；
- unique training checkpoint hashes：70/70；
- new formal jobs/cells：70/70 checkpoints、280/280 standard cells；
- candidate pool：14 new profiles + v1 BSCA reference，共300/300 cells；
- official-test access date：2026-08-16；
- test role：`test-tuned-hyperparameter-selection-and-paper-benchmark`；
- selection unit：one dataset-level profile shared by H96/H192/H336/H720；
- all negative trials retained：true；
- checkpoint nonmutation：true；
- table mutation：false。

首次formal queue在16/70 complete时被旧scope invariant错误中止。预测、metrics、prefix consistency与numeric health均正常；仅`p12/p13`的configured scopes与旧global-default comparison不匹配。修复只将expected scopes改为checkpoint effective config，未改变模型、checkpoint、test loader或selection rule。一个failed checkpoint完成hash-guarded recovery smoke后，queue从17/70断点恢复并最终闭合。该事件归因为`formal_invariant_checker_protocol_defect`，不是model或optimization failure。

## 2. Frozen selected profiles

| Dataset | Selected profile | Mean MSE | Mean MAE | Decoder parameters | Best validation epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| Weather | `p03_lr2p00` | 0.251003 | 0.272896 | 370,829 | 4 |
| ETTm1 | `p11_policy64x128_lr0p50` | 0.386234 | 0.390865 | 102,873 | 27 |
| ETTm2 | `p03_lr2p00` | 0.286673 | 0.323587 | 91,961 | 10 |
| ETTh1 | `p08_coord2_capmatch_lr0p50` | 0.437724 | 0.437843 | 413,901 | 5 |
| ETTh2 | `p04_lr4p00` | 0.334841 | 0.374515 | 91,961 | 3 |

这些profiles严格由每个dataset的four-H mean test MSE选择，同一profile服务全部四H。不同dataset命中不同轴：Weather/ETTm2偏好2× decoder LR，ETTh2偏好4× decoder LR，ETTm1偏好更宽allocation MLP，ETTh1偏好2-dimensional future coordinate与capacity-matched wider rank。不存在一个统一decoder profile主导全部datasets。

## 3. Effectiveness result

| Dataset | Selected vs Original MSE gain | Selected vs Original MAE gain | Selected vs v1 BSCA MSE gain | Selected vs v1 BSCA MAE gain |
| --- | ---: | ---: | ---: | ---: |
| Weather | +0.282% | -0.176% | +0.599% | +0.015% |
| ETTm1 | +0.070% | -0.690% | +1.754% | +1.347% |
| ETTm2 | +0.117% | +0.176% | +2.154% | +2.196% |
| ETTh1 | -3.110% | -3.376% | +4.346% | +3.308% |
| ETTh2 | +0.996% | +0.923% | +0.677% | +1.021% |

Selected BSCA macro MSE/MAE为`0.339295/0.359941`，v1 Original Decoder为`0.337592/0.357262`。因此相对Original：

- macro MSE gain=`-0.505%`；
- macro MAE gain=`-0.750%`；
- dataset-mean MSE wins=`4/5`；
- MSE/MAE cell wins=`12/20`与`7/20`。

虽然4/5 datasets的mean MSE获胜，但预注册gate要求macro MSE与macro MAE同时改善，并至少赢3/5 dataset means。ETTh1的系统性退化超过其他四个datasets的小幅收益，故overall gate为FAIL。

相对旧v1 BSCA，selected HPO结果macro MSE/MAE改善`2.128%/1.719%`，五个datasets的MSE均改善；因此本轮HPO有效修复了部分optimization/capacity mismatch，但不足以超过native Original Decoder。

## 4. Search-space evidence

v1 reference在15-profile pool中的MSE排名分别为Weather 9、ETTm1 13、ETTm2 15、ETTh1 11、ETTh2 5，说明extended budget与decoder-specific tuning具有实际影响。尤其ETTm1 selected checkpoint的best validation epoch为27，验证v1的10-epoch预算不足。

14个new profiles的validation/test MSE Spearman rank correlation分别为：

- Weather：0.424；
- ETTm1：0.182；
- ETTm2：0.495；
- ETTh1：0.873；
- ETTh2：0.851。

Weather与ETTm1存在明显validation/test ranking mismatch；本项目按作者授权采用test-tuned dataset-level selection，因此结果有效但必须明确披露为test-tuned，不能描述为untouched-holdout evidence。

ETTh1的14-profile test MSE range达到10.514%，是decoder结构最敏感的数据集。`p08`已将v1 BSCA改善4.346%，但仍比Original差3.110%，说明该差距不是简单邻域LR或训练预算不足可以解释。

## 5. Four-layer decision

1. `paper_facing_effectiveness`：FAIL。Selected BSCA未改善macro MSE/MAE。
2. `matched_mechanism_attribution`：不启动。预注册规则只在performance gate通过后补selected-profile matched `+ISCF`；当前追加该矩阵不能恢复portability claim。
3. `internal_mechanism_health`：prefix gap为0、numeric health正常，HPO显著改善v1 BSCA；没有optimization/numeric pathology可否定结果。
4. `failure_attribution`：exact design=`readout_or_head_design_wrong_for_ETTh1_itransformer_representation_compatibility`；claim level=`hypothesis_false_for_general_cross_backbone_decoder_portability_on_current_evidence`。这不否定ISCF-BSCA在DLinear-style carrier上的正向结果，也不否定HPO对旧iTransformer replacement head的改善。

## 6. Rollback and paper consequence

按冻结rollback，iTransformer decoder rescue在此关闭：不继续追加test-informed邻域HPO、不启动extra seeds，也不补matched ISCF。Canonical Decoder-Transfer paper table保持现有DLinear/PatchTST完整matched table，不用该best-tuned iTransformer结果替换或选择性删除负向Transformer evidence。

论文中不得声称decoder可普遍迁移到不同backbones。可保留的最强表述是：ISCF-BSCA在DLinear-style carrier上显示正向transfer evidence，但PatchTST与test-tuned iTransformer-style carriers均未超过各自native decoder，因此cross-backbone portability尚未得到支持。

Canonical machine-readable evidence：

- `candidate_pool_300_cells.csv`：全部candidate cells；
- `candidate_profile_means.csv`：75个dataset-profile means；
- `selected_profiles.csv`：5个dataset-level selections；
- `selected_vs_original_40_cells.csv`：selected与Original完整comparison；
- `result_summary.json`：gate与provenance摘要；
- `formal_artifact_manifest.csv`：70个formal artifact hashes。
