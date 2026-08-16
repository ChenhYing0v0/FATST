# Decoder-Transfer 三数据集范围：结果与下一门控

日期：2026-08-16
Decision：`three_dataset_best_config_performance_pass_two_matched_iscf_controls_required`。

## 1. 作者指定范围与披露边界

正文Decoder-Transfer范围按作者要求缩减为Weather、ETTm1、ETTm2，仍使用H96/H192/H336/H720与MSE/MAE。ETTh1、ETTh2的既有完整负向结果不删除，继续保存在five-dataset formal audit与supplementary/limitation evidence中。

该范围是在观察five-dataset结果后确定，因此必须披露为`author_refined_posthoc_scope`，不能描述成预注册三数据集confirmatory experiment，也不能用它否认ETTh1/ETTh2上的迁移失败。

## 2. 当前严格matched结果

严格matched版直接把v2.1完整表限制到三个datasets，所有PatchTST-style `+ISCF`与`+ISCF-BSCA`仍共享同一dataset-level decoder profile。

| Backbone | Original | +ISCF | +ISCF-BSCA | BSCA gain vs Original |
| --- | ---: | ---: | ---: | ---: |
| DLinear-style | 0.304 / 0.333 | 0.285 / 0.322 | 0.289 / 0.322 | +4.915% / +3.276% |
| PatchTST-style | 0.282 / 0.314 | 0.283 / 0.317 | 0.280 / 0.315 | +0.605% / -0.379% |

[Fact] DLinear-style在三数据集范围内仍为双指标正向，但`+ISCF`的MSE略优于`+ISCF-BSCA`。PatchTST-style严格matched BSCA只在MSE上转正，MAE仍未超过Original。

## 3. PatchTST best-config结果

使用40个unique checkpoints完整formal audit后冻结的dataset-level mean-MSE winners：

| Dataset | Selected BSCA profile | MSE gain vs Original | MAE gain vs Original | MSE/MAE cell wins |
| --- | --- | ---: | ---: | ---: |
| Weather | `p07_wd1e3` | +0.251% | +0.298% | 2/4, 4/4 |
| ETTm1 | `p10_rank1p50_lr0p50_wd1e4` | +2.564% | +0.624% | 4/4, 4/4 |
| ETTm2 | `p01_lr0p25` | +0.371% | -0.410% | 3/4, 0/4 |
| Macro | — | **+1.268%** | **+0.192%** | **9/12, 8/12** |

在作者指定的三数据集范围内，best-config PatchTST-style相对Original满足macro MSE/MAE双正向，并赢3/3 dataset mean MSE。因此就“完整decoder能否在PatchTST carrier上取得正向performance”而言，现有HPO已经充分；继续扩大BSCA HPO会增加test-tuning程度和资源成本，而不会解决当前真正的evidence gap。

## 4. 为什么暂不能把best-config表视为完整matched attribution

- ETTm2 `p01_lr0p25`的matched `+ISCF` checkpoint与formal结果可复用；
- Weather winner `p07_wd1e3`尚无同profile `+ISCF` checkpoint；
- ETTm1 winner `p10_rank1p50_lr0p50_wd1e4`尚无同profile `+ISCF` checkpoint。

当前best-config表保留v2.1 `+ISCF` row作为背景control，但Weather/ETTm1并非与new BSCA winners严格profile-matched。因此该表可以支持Full decoder相对Original的test-tuned performance portability，不能用于BSCA objective相对ISCF的严格归因。

## 5. 下一最小实验，不是继续HPO

建议冻结两个matched `+ISCF` runs：

1. Weather × `p07_wd1e3` × seed2021；
2. ETTm1 × `p10_rank1p50_lr0p50_wd1e4` × seed2021。

两者均应from-scratch end-to-end joint training，复用各自BSCA winner的encoder、rank、base LR、readout LR multiplier、readout weight decay、batch、patience与initialization class，唯一方法差异为`pcc_objective_mode=measure_only`。Checkpoint仍由four-H validation mean MSE选择；2/2 unique hashes与matched initialization通过后，再请求2-checkpoint / 8-cell formal test。ETTm2不重复训练或访问test。

由于远程quota当前约199G/200G soft limit，启动前应先做只删除可重建临时/旧dense diagnostic artifacts的精确清理，不删除本轮formal evidence或selected checkpoints。

## 6. 当前artifact角色

- `table_decoder_transfer_three_dataset_matched.tex`：当前完全matched、可直接审计的三数据集表；
- `table_decoder_transfer_three_dataset_best_config.tex`：性能最强的paper candidate，等待两个matched `+ISCF` controls闭合；
- `strict_matched_72_cells.csv`与`best_config_candidate_72_cells.csv`：两版表的完整four-H输入；
- `result_summary.json`：收益、selected profiles和matched-control缺口；
- five-dataset negative evidence继续由`decoder_transfer_patchtst_v2p1_20260815`与`decoder_transfer_patchtst_test_tuned_full_20260816`保留。
