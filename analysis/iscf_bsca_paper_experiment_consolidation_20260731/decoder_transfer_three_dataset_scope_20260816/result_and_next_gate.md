# Decoder-Transfer 三数据集范围：结果与下一门控

日期：2026-08-16
Decision：`three_dataset_framework_portability_complete_no_additional_hpo_or_controls`。

## 1. 作者指定范围与披露边界

正文Decoder-Transfer范围按作者要求缩减为Weather、ETTm1、ETTm2，仍使用H96/H192/H336/H720与MSE/MAE。ETTh1、ETTh2的既有完整负向结果不删除，继续保存在five-dataset formal audit与supplementary/limitation evidence中。

该范围是在观察five-dataset结果后确定，因此必须披露为`author_refined_posthoc_scope`，不能描述成预注册三数据集confirmatory experiment，也不能用它否认ETTh1/ETTh2上的迁移失败。

## 2. 历史三臂diagnostic结果

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

## 4. 当前论文claim与canonical comparison

本节不承担ISCF与BSCA之间的component attribution；该任务由Core-Ablation单独负责。Decoder-Transfer只检验完整ISCF-BSCA framework接入不同encoder/backbone后，是否相对对应native Original Decoder保持正向performance。因此正文canonical comparison只保留：

- DLinear-style Original Decoder versus DLinear-style ISCF-BSCA；
- PatchTST-style Original Decoder versus PatchTST-style ISCF-BSCA。

`+ISCF` rows保留在historical diagnostic artifacts中，但不进入正文Table 5，也不构成missing evidence。现有结果支持的限定结论是：在Weather、ETTm1与ETTm2范围内，完整ISCF-BSCA framework在DLinear-style与PatchTST-style两类backbones上均取得macro MSE/MAE双正向结果。这支持evaluated-scope transferability和cross-backbone applicability，不支持对未测试backbones、datasets或domains使用universal/architecture-agnostic表述。

## 5. HPO与新增实验判断

无需继续PatchTST HPO，也无需补训Weather/ETTm1 `+ISCF` controls。原因如下：

1. 两个backbone的完整framework均已在三数据集macro MSE/MAE上超过对应Original Decoder；
2. PatchTST在3/3 dataset mean MSE上正向，当前主要claim已满足；
3. 继续HPO只会增加test-tuning程度，不改变framework-level evidence类型；
4. `+ISCF`补训只服务内部BSCA attribution，而该归因已明确不属于本节目标。

因此本部分关闭新增remote training与formal test需求，后续只需完成paper wording与limitations披露。

## 6. 当前artifact角色

- `table_decoder_transfer_three_dataset_framework.tex`：正文canonical framework-level comparison；
- `framework_portability_48_cells.csv`：正文表的完整four-H输入；
- `table_decoder_transfer_three_dataset_matched.tex`与`table_decoder_transfer_three_dataset_best_config.tex`：保留三臂historical diagnostic；
- `result_summary.json`：收益、selected profiles和framework-level claim contract；
- five-dataset negative evidence继续由`decoder_transfer_patchtst_v2p1_20260815`与`decoder_transfer_patchtst_test_tuned_full_20260816`保留。
