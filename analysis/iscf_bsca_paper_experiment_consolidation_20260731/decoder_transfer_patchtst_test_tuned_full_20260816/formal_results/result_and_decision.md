# PatchTST Decoder-HPO 全量 Test-Tuned Formal Result

日期：2026-08-16  
Decision：`full_unique_audit_complete_mse_only_partial_gain_bidirectional_metric_gate_failed`。

## 1. 完整性与执行审计

- parent HPO training objects：50/50；unique checkpoint hashes：40/40；
- 已有v2.1 formal artifacts复用：5 checkpoints / 20 standard cells；
- 本轮新增formal：35/35 checkpoints / 140/140 standard cells；
- unique formal matrix：40/40 checkpoints / 160/160 cells；
- duplicate aliases扩展后：10 profiles × 5 datasets × four H=`200/200` cells；
- 加入v1 BSCA reference：220/220 candidate-pool cells；
- final selector：每dataset一个profile，按H96/H192/H336/H720 mean official-test MSE；
- per-H、per-seed、per-metric、per-cell selection：false；
- formal artifact manifest SHA256：`5daa6ec83b8df240cd053ec0940421b3bcf918fb6743cb08b943c1785c48ff2e`；
- checkpoint nonmutation、finite metrics、test invariant与exact-prefix checks：全部通过。

首次driver在三个新job读取test split后，因legacy PatchTST `effective_config.adapter`没有序列化`pcsd_scales`而在invariant构造处停止，未写出可选用的formal metrics。精确修复只在adapter字段缺失时读取同一run的`initialization_contract["pcsd_scales"]`；model、checkpoint、test loader、prediction与selection rule均不改变。修复后完整矩阵从头按hash guard恢复，并闭合全部artifact。

## 2. 冻结selector结果

| Dataset | Selected profile | MSE | MAE | MSE gain vs Original | MAE gain vs Original |
| --- | --- | ---: | ---: | ---: | ---: |
| Weather | `p07_wd1e3` | 0.228092 | 0.252658 | +0.251% | +0.298% |
| ETTm1 | `p10_rank1p50_lr0p50_wd1e4` | 0.349225 | 0.372842 | +2.564% | +0.624% |
| ETTm2 | `p01_lr0p25` | 0.258024 | 0.315045 | +0.371% | -0.410% |
| ETTh1 | `p01_lr0p25` | 0.398305 | 0.419428 | -0.972% | -0.738% |
| ETTh2 | `p01_lr0p25` | 0.315431 | 0.363626 | -1.553% | -0.857% |
| Macro | dataset-level selection | 0.309815 | 0.344720 | **+0.134%** | **-0.253%** |

相对Original Decoder，selected BSCA赢`3/5` dataset MSE means、`11/20` MSE cells与`9/20` MAE cells。MSE从v1/v2.1的轻微负向推进到`+0.134%`，但MAE仍退化`0.253%`，因此预注册的MSE/MAE双正向gate失败。

## 3. 完整search-space trade-off audit

该部分是post-hoc diagnostic，不替代冻结的MSE selector：

- Weather有5个profiles同时优于Original的mean MSE/MAE；
- ETTm1只有`p10_rank1p50_lr0p50_wd1e4`同时正向；
- ETTm2只有`p03_lr2p00`同时正向，但它不是mean-MSE winner；
- ETTh1与ETTh2在完整11-candidate pool中都没有MSE/MAE同时正向的profile；
- 枚举每dataset 11个candidates的`11^5=161,051`种组合，没有任何组合能让macro MSE与macro MAE同时优于Original；
- 若事后按MAE最小选择，macro MAE可改善`+0.137%`，但macro MSE退化`0.051%`。

[Strong Evidence] 这不是单一selector偶然选错造成的gate failure，而是冻结search space内的MSE--MAE trade-off。不能通过把ETTm2换成MAE友好profile或选择性删除ETTh1/ETTh2来构造双正向结论。

## 4. Four-layer decision

1. `paper_facing_effectiveness`：完整矩阵通过；MSE-only为轻微正向，但MSE/MAE联合gate失败。
2. `matched_mechanism_attribution`：本轮selected profiles没有逐profile matched +ISCF controls；即使MSE单指标为正，也不能归因于BSCA objective。
3. `internal_mechanism_health`：不由本轮test-tuned performance audit评估；它不能补救effectiveness gate。
4. `failure_attribution`：没有numeric、artifact或checkpoint pathology。Claim-level为`hypothesis_false_for_bidirectional_metric_patchtst_portability_within_frozen_search_space`；design-level继续为`readout_or_head_design_wrong_for_patchtst_representation_compatibility`。

## 5. 论文结论与后续

可诚实支持的最强结论是：dataset-level decoder HPO能消除PatchTST transfer在macro MSE上的小幅deficit，并在Weather、ETTm1、ETTm2三个datasets取得mean-MSE改善；但该收益很小、MAE不一致，且ETTh1/ETTh2没有双指标正向candidate。

当前证据不能支持“ISCF-BSCA decoder可稳定迁移到PatchTST”或general cross-backbone portability。Canonical Decoder-Transfer table保持v2.1不变；不补matched +ISCF、不追加PatchTST search/seeds，也不选择性写入MSE-only winner。DLinear-style的限定正向证据与PatchTST/iTransformer负向边界继续同时保留。下一实验cursor返回Figure 5 paper closure。

关键artifacts：

- `unique_checkpoint_160_cells.csv`：40 unique checkpoints的standard cells；
- `expanded_profile_200_cells.csv`：duplicate aliases展开后的完整HPO矩阵；
- `candidate_pool_220_cells.csv`：加入v1 BSCA reference的selector输入；
- `selected_profiles.csv`、`selected_20_cells.csv`：冻结MSE selector输出；
- `selected_dataset_gains.csv`：逐dataset相对Original的收益；
- `posthoc_tradeoff_summary.json`：完整组合trade-off diagnostic；
- `formal_artifact_manifest.csv`：formal artifact hashes与复用角色。
