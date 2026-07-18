# CCSF tau0.25 formal Phase-A remote launch

## Launch provenance

- launch time：`2026-07-18T17:27:08+08:00`；
- remote host：`529_Lab-3090`；
- remote repo：`/home/yingch/projects/FATST`；
- commit：`604e1b8f8e63f6d00337cd1504e5628e4f5f90cf`；
- conda env：`moe`；
- GPUs：0/1/2，均为RTX 3090；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_ccsf_v1_tau25_phase_a`；
- driver log：`driver_seed2021.log`；
- driver process：`801928`；workers：`801951/801955/801960`。

## Preflight

launch前GPU0/1/2均约15 MiB used、24110 MiB free、utilization 0%。remote repo保留三个既有dirty analysis CSV，
它们与本阶段文件不重叠；`git pull --ff-only`成功到commit`604e1b8`，未修改这些用户文件。

remote formal prelaunch复核为15/15。Weather/CCSF_RELCAL三train-batch resource smoke通过：effective config记录
`max_train_batches=3`，training loss fields finite，output位于`_resource_smoke/ccsf_relcal_weather_seed2021`。

## Active matrix

runner冻结为50 runs：10 arms × 5 datasets × seed2021，预计200个official-test标准cells。training checkpoint只由
validation H96/H192/H336/H720 mean MSE选择；每个run训练结束后才由授权evaluator读取test，写入实际
`test_access_date`，并验证checkpoint hash前后不变。

首批三项均为Weather：

1. GPU0：`a6_measure`；
2. GPU1：`siff_v1_equal`；
3. GPU2：`siff_v1_relcal`。

启动后GPU0/1/2分别约433/2004/1966 MiB used，GPU utilization约20%/88%/88%，三个training workers均存活。

## Decision boundary

当前11-step状态是Step8 formal Phase-A running；Step9 effectiveness尚未开始。运行期间不pull、不改变arm、profile、
temperature、objective、checkpoint或test matrix，不进行高频值守。50/50完成后必须先同步完整artifacts并执行
four-layer analyzer；confirmation seeds2022/2023仍未授权。
