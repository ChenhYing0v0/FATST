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

### 4.1 Commit and remote state

- commit：`6cef063ecfa4cc12aaa3eb0e5e1bbbfcca42092b`；
- remote repo：`/home/yingch/projects/FATST`，`git pull --ff-only`成功；
- 三份历史dirty analysis CSV原样保留，未stash、清理或覆盖；
- config SHA256：`32ec229c66c5683a48793b5bbde3c4f6e99790e4f6e05e4076ec6284c351e748`；
- profile SHA256：`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`。

### 4.2 Resource smoke

2026-07-21 10:16+08:00，GPU0/1：

| Smoke | Train steps | Train loss | Validation mean MSE | Result |
| --- | ---: | ---: | ---: | --- |
| Weather `tsaf` | 2 | 1.748264 | 1.279264 | finite；no OOM；policy=`target-scale-field` |
| ETTm2 `siff_independent_target_only` | 2 | 0.994514 | 0.338739 | finite；no OOM；policy=`static-target` |

smoke只使用training/validation batch，`evaluation_split=none`，不构成performance evidence。

### 4.3 Training launch

- launch time：`2026-07-21T10:17:06+08:00`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_v3_tsaf_v1`；
- driver PID：`1705027`；runner PID：`1705029`；
- workers：GPU 0/1/2；dataset-major 25 jobs；
- first jobs：Weather `tsaf` / `siff_categorical_target_only` / `tsaf_permuted_scale`；
- initial active memory：1645/2004/1646 MiB，utilization 85/89/83%；
- initial status：training 0/25 complete，formal test 0/25；
- formal-test mode没有启动。

output-root supervisor PID=`1713424`，只等待training runner PID=`1705029`退出。它随后先检查
`training=25/25`：若不完整则exit 4且不访问test；若完整则执行`git pull --ff-only`、记录formal-test commit，启动
`FORMAL_TEST_ONLY=1`。test 25/25后才运行冻结analyzer；不完整则exit 5且不分析。supervisor日志为
`phase_a_supervisor.log`，脚本与日志均位于repo-external output root。

训练期间不得remote pull或修改config/gates。只有training 25/25且artifact completeness通过，才允许单独启动
`FORMAL_TEST_ONLY=1`。
