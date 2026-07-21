# SC1-SIFF-v3-TSAF Step8 Remote Authorization and Launch

## 1. Authorization

2026-07-21用户指令“继续推进远程实验”授权冻结的seed2021 TSAF Phase A：

- 25个new arms × datasets的from-scratch joint training；
- checkpoint只由validation H96/H192/H336/H720 mean MSE选择；
- 25/25训练完成后，对完整45-run/180-cell effective matrix执行一次formal test audit；
- test不得改checkpoint、选择arm/profile/horizon/cell或触发post-hoc tuning；
- seeds2022/2023 confirmation、paper-core promotion与任何matrix/gate修改仍未授权。

machine authorization gate为15/15 cases、10/10 categories。config SHA256为
`32ec229c66c5683a48793b5bbde3c4f6e99790e4f6e05e4076ec6284c351e748`。

## 2. Pre-launch repairs

在任何remote pull/training前发现并修复两个protocol wiring缺口：

1. generic checkpoint evaluator的matrix-size helper只读取`arms`，现同时接受TSAF使用的`effective_arms`；
2. TSAF config补充`coupling_scales`、equal-skill `training_contracts`、eight future bins与
   `expected_runs=45`，使formal evaluator能验证readout/training/matrix contract。

runner新增`FORMAL_TEST_ONLY=1`模式。normal mode只训练并生成validation artifacts；formal-test mode要求25/25
training artifacts先完整，再逐checkpoint执行test evaluator并比较test前后SHA256。两种模式不能由test labels选择
checkpoint。

## 3. Frozen execution order

1. focused commit并push；
2. remote repo保留三份历史dirty CSV，执行`git pull --ff-only`；
3. 重读GPU/process状态；
4. 25-job dry-run；
5. Weather-TSAF与ETTm2-independent两项2-train/2-eval-batch resource smoke；
6. smoke finite、无OOM且artifacts完整后启动三GPU dataset-major training；
7. 25/25 training完整前不访问formal test；
8. 25/25 test完整后运行冻结four-layer analyzer；
9. 完整报告negative cells与failure attribution，不做局部选择。

## 4. Launch record

等待commit/push、remote pull与resource smoke后填写。当前尚未执行remote training或official test。
