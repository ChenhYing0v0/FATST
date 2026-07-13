# StageC A6-LBF Natural Baseline Test Reference

## Purpose

该工具不训练、不选择profile，也不比较candidate。它只在natural profile已经validation-only冻结后，加载
seeds2021/2022/2023的`checkpoint.pt`，对test split执行一次dense-horizon评估，生成以后所有StageC
mechanism实验共用的reference table。

## Artifact Routing

- seed2021 medium-width checkpoint来自Phase A；
- seed2021 ETTm1 narrow-width checkpoint来自Phase B；
- seeds2022/2023来自Phase C stability confirmation；
- 每个checkpoint必须与frozen contract的`patch_num/d_model/d_ff/seed`一致。

`checkpoint.pt`是training runner恢复best-validation state后保存的模型。test不参与checkpoint、profile或
hyperparameter selection，输出显式标记`post_freeze_reference_only`。

## Tensor And Evaluation Flow

对每个run，读取原`effective_config.json`重建`TimeAlign.Model`，加载state dict，在同一test loader上输出
H48/96/144/192/288/336/512/720的MSE/MAE。每个dataset得到3 seeds × 8 horizons = 24 rows；全矩阵72 rows。

analyzer输出逐seed表与mean/sample-std/CV聚合表。后续candidate必须使用同一contract、seeds与test
aggregation，不允许根据该reference回调baseline配置。

## Code-Theory Consistency

理论目标是获得冻结research instrument的无泄漏test reference。代码通过只读既有checkpoint、验证contract
字段、固定best-validation selector实现该目标。它不证明baseline是per-horizon optimal，也不把test结果用于
模型选择；若artifact或contract mismatch，分析必须停止而不是自动寻找替代checkpoint。
