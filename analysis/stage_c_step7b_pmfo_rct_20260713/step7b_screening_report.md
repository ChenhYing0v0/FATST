# StageC Step 7B PMFO-RCT Screening Report

## Scope

三数据集、五arms、seed2021；训练保持frozen full-H720 pointwise L1，
best checkpoint由H720 validation MSE选择；test一次生成H720后聚合H1..720 MSE/MAE。
`dense_mse_auc`定义为720个prefix MSE的算术平均，对应uniform horizon measure。

## Run Summary

| dataset | arm | dense_mse_auc | dense_mae_auc | h48_mse | h96_mse | h192_mse | h336_mse | h720_mse | epochs_ran | invariant_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTm1 | a6 | 0.359529 | 0.377619 | 0.280985 | 0.300221 | 0.333654 | 0.369100 | 0.423494 | 8 | True |
| ETTm1 | dense_mlp_matched | 0.370298 | 0.384540 | 0.294732 | 0.315532 | 0.345366 | 0.378199 | 0.431377 | 7 | True |
| ETTm1 | pmfo_no_transition | 0.363843 | 0.376907 | 0.284833 | 0.305526 | 0.338941 | 0.373754 | 0.425450 | 9 | True |
| ETTm1 | pmfo_no_conservation | 0.370923 | 0.385500 | 0.289039 | 0.311610 | 0.345102 | 0.380339 | 0.434751 | 7 | True |
| ETTm1 | pmfo_rct | 0.367023 | 0.382658 | 0.287237 | 0.309632 | 0.342415 | 0.376387 | 0.428766 | 7 | True |
| ETTh2 | a6 | 0.324085 | 0.379975 | 0.212503 | 0.254268 | 0.297766 | 0.332463 | 0.417747 | 7 | True |
| ETTh2 | dense_mlp_matched | 0.331959 | 0.388613 | 0.243007 | 0.273280 | 0.310712 | 0.336106 | 0.416409 | 6 | True |
| ETTh2 | pmfo_no_transition | 0.326680 | 0.384991 | 0.248455 | 0.276808 | 0.306361 | 0.324295 | 0.414239 | 7 | True |
| ETTh2 | pmfo_no_conservation | 0.344358 | 0.396414 | 0.248248 | 0.282495 | 0.320736 | 0.347540 | 0.435845 | 6 | True |
| ETTh2 | pmfo_rct | 0.325635 | 0.382969 | 0.224081 | 0.261544 | 0.300616 | 0.327604 | 0.422308 | 7 | True |
| Weather | a6 | 0.228321 | 0.256568 | 0.116247 | 0.145481 | 0.186768 | 0.237133 | 0.308754 | 17 | True |
| Weather | dense_mlp_matched | 0.228530 | 0.258031 | 0.119274 | 0.147052 | 0.187673 | 0.236660 | 0.306918 | 8 | True |
| Weather | pmfo_no_transition | 0.231594 | 0.259319 | 0.120610 | 0.149723 | 0.190874 | 0.240280 | 0.310317 | 13 | True |
| Weather | pmfo_no_conservation | 0.231197 | 0.257107 | 0.117497 | 0.144518 | 0.187607 | 0.239179 | 0.314649 | 9 | True |
| Weather | pmfo_rct | 0.229974 | 0.256464 | 0.116088 | 0.145628 | 0.188329 | 0.238725 | 0.311744 | 12 | True |

## Gate

- decision: `rollback_step4`；
- complete: `15/15`；
- failure attribution: `readout_or_head_design_wrong`；
- macro PMFO vs A6: `-1.0955%`；
- worst dataset PMFO vs A6: `-2.0844%`；
- macro PMFO vs per-dataset best control: `-0.3953%`；

## Mechanism Diagnostics

| comparison | macro dense-MSE improvement | interpretation |
| --- | ---: | --- |
| PMFO vs A6 | -1.0955% | exact v1 effectiveness failed |
| PMFO vs dense matched | 0.7193% | weak structured-decoder signal, not gate-level evidence |
| PMFO vs no-transition | 0.0486% | recursive transition not independently supported |
| PMFO vs no-conservation | 2.3393% | conservation retained for redesign |

### PMFO versus A6 by horizon segment

| dataset | segment | pmfo_improvement_pct | pmfo_winning_horizons | segment_horizon_count |
| --- | --- | --- | --- | --- |
| ETTm1 | H1-48 | -3.971171 | 0 | 48 |
| ETTm1 | H49-96 | -2.793411 | 0 | 48 |
| ETTm1 | H97-192 | -2.870598 | 0 | 96 |
| ETTm1 | H193-336 | -2.188352 | 0 | 144 |
| ETTm1 | H337-720 | -1.699846 | 0 | 384 |
| ETTh2 | H1-48 | -9.870398 | 0 | 48 |
| ETTh2 | H49-96 | -3.849340 | 0 | 48 |
| ETTh2 | H97-192 | -1.909683 | 0 | 96 |
| ETTh2 | H193-336 | 0.527943 | 106 | 144 |
| ETTh2 | H337-720 | 0.288372 | 210 | 384 |
| Weather | H1-48 | 0.528475 | 48 | 48 |
| Weather | H49-96 | 0.153698 | 42 | 48 |
| Weather | H97-192 | -0.530757 | 0 | 96 |
| Weather | H193-336 | -0.751328 | 0 | 144 |
| Weather | H337-720 | -0.849318 | 0 | 384 |

[Fact] 15/15 runs与15/15 trained invariants通过，无numeric、prefix或protocol pathology。

[Strong Evidence] PMFO-RCT相对A6在三数据集dense-MSE AUC均退化；ETTm1的720个horizon无一胜出。

[Strong Evidence] conservative synthesis相对no-conservation在三数据集均改善；该组件不因v1整体失败而被否定。

[Strong Evidence] recursive transition相对no-transition的macro improvement接近零且跨数据集不一致，不能作为v1贡献。

[Inference] 当前失败更符合readout/function-class replacement问题，而非Encoder不足：Step 7B没有操纵Encoder，
因此不能对Encoder sufficiency作因果结论。

## Decision And Rollback

`PMFO-RCT v1`作为paper-core候选关闭，回滚Step 4。关闭范围仅限当前固定mixed-radix tree、
state transition与A6 readout整体替换的组合，不拒绝projective operator方向。下一轮先诊断
A6 function class、fixed tree partition和history-to-node interface，禁止叠加MIPR、Encoder或MoE掩盖失败。

[Boundary] 单seed只能形成`partial_pass`或rollback，不能形成effectiveness claim。
