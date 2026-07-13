# A6-LBF Natural Baseline Test Reference

## 结论

[Fact] `A6-LBF-natural-baseline` 已完成 Weather、ETTm1、ETTh2 上的 3 seeds × 8 horizons
测试，共 `72/72` 行，无缺失。profile 与 checkpoint 均在 test 前由 validation 冻结，
`selection_used_test=false`，因此该结果可作为 StageC 后续 mechanism/control 的固定 test reference。

[Decision] baseline reference 状态为 `frozen_test_reference_ready`。后续实验必须继续使用
contract hash `254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`，
不得根据本表重新选择 `patch_num`、`d_model`、`d_ff`、checkpoint 或 seed。

## Test MSE / MAE（3-seed mean）

| Dataset | H48 | H96 | H144 | H192 | H288 | H336 | H512 | H720 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weather | .1163/.1523 | .1453/.1878 | .1666/.2098 | .1869/.2291 | .2228/.2588 | .2377/.2699 | .2786/.2987 | .3101/.3194 |
| ETTm1 | .2837/.3371 | .3040/.3493 | .3255/.3614 | .3364/.3674 | .3599/.3795 | .3711/.3850 | .3990/.3990 | .4255/.4125 |
| ETTh2 | .2182/.3038 | .2579/.3314 | .2805/.3467 | .2997/.3603 | .3247/.3803 | .3320/.3870 | .3567/.4072 | .4143/.4441 |

每个单元格为 `MSE/MAE`。完整精度与逐 seed 数值见同目录 CSV。

## 稳定性

- Weather 的 MSE CV 为 `0.13%-0.38%`，稳定。
- ETTm1 的短 horizon 波动较高，H48 MSE CV 为 `2.74%`，随 horizon 增长下降。
- ETTh2 的 H48 MSE CV 为 `5.30%`，是唯一超过 5% 的 test 单元；H96 为 `2.97%`，其余均低于 `2.1%`。

[Strong Evidence] 这说明 ETTh2 短 horizon 的 seed uncertainty 必须在后续表格中保留，不能只报
单 seed。该波动是 post-freeze test variability，不是此前 ETTh2 training-validation 在 epoch 20
相对 best epoch 恶化 `31.63%-44.95%` 的同一现象。

## 执行记录

- commit: `0d3218edc7d6a5bc3f43edec3f5b5f2094ebbb38`
- server: `529_Lab-3090`
- environment: `moe`
- GPU mapping: Weather=`0`，ETTm1=`1`，ETTh2=`2`
- start: `2026-07-13T15:58:03+08:00`
- end: `2026-07-13T15:59:47+08:00`
- checkpoint role: validation-selected restored-best checkpoint；test 只做一次 post-freeze evaluation

## Artifacts

- `natural_baseline_test_metrics_by_seed.csv`: 72 行逐 seed/horizon 结果；
- `natural_baseline_test_metrics_aggregate.csv`: 24 行 mean/std/CV；
- `natural_baseline_test_summary.json`: completeness、contract 与 leakage audit；
- `raw/`: 远端同步的 dataset-level 原始表和 launcher log。
