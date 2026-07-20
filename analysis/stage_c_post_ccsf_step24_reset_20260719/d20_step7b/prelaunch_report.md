# D20 CST Step 7B Prelaunch Report

## Frozen question

在strong A6 coefficient operator中，$q=64$ compact Fourier summary是否同时：

1. 相对same-run from-scratch A6产生可复现的transfer gain；
2. 相对同维fixed random orthogonal summary产生frequency-specific gain。

## Frozen matrix

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- arms：`A6_MEASURE_RETRAIN`、`A6_CST_SPEC`、`A6_CST_RANDOM`；
- seed：2021；
- 15个from-scratch runs；
- official test：每个run报告H96/H192/H336/H720，共60 cells；
- checkpoint：validation四个标准horizon的mean MSE；
- 禁止test选择checkpoint以及dataset/horizon/cell-specific tuning。

这是`test_informed`的problem-existence diagnostic，formal-test access count为1。用户已授权推进到remote launch；
confirmation和paper method仍为false。

## Prelaunch checks

`prelaunch_gate.json`记录`10/10 pass`：

- Step6、Step7A、production model/trainer/evaluator/analyzer/runner hash一致；
- Step7A 9/9 dependency成立；
- 15-run与60-cell matrix完整且无重复；
- official-test authorization、checkpoint non-mutation与promotion boundary明确；
- remote shell syntax和15-job dry-run通过；
- evaluator覆盖D20 production contract，analyzer synthetic smoke通过；
- negative结果的rollback映射已冻结。

## Decision and rollback

Decision=`step7b_prelaunch_pass_step8_authorized`。若SPEC不超过A6，回Step2判断D19 skip evidence不可迁移；若SPEC
不超过RANDOM，回Step4并判定generic added path/capacity control explains；若internal path异常或出现numeric pathology，
回Step6重做诊断设计。即使两个主comparison均通过，也只能回Step4设计native non-residual operator，不能直接把
D20 concat-style diagnostic写成论文贡献。
