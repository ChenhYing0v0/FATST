# TimeAlign Main I Eight-Dataset Reproduction Prelaunch

## Scope

本轮只补齐TimeAlign，不启动其他baseline或新的ISCF-BSCA HPO。目标矩阵为8 datasets × H96/H192/H336/H720 × seed2021，共32个fixed-H independent models及32行MSE/MAE。

- ETTm2与Weather已有8个artifact-complete runs；其adapter、model、dataset、preset与checkpoint hashes均保持不变，因此标记为`reusable`。
- ETTh1、ETTh2、ETTm1、ECL、Solar与Exchange共24个runs为`new`。
- 前7个datasets使用TimeAlign official scripts中的dataset/horizon presets。
- upstream没有Exchange脚本；按用户既有授权采用ETTh1-derived bootstrap，角色固定为`source_informed_etth1_bootstrap_not_official`。

## Training and test boundary

所有runs使用`L=720`、`label_len=48`、official fixed-H model、official-last checkpoint、无early stopping。ETTh1 H96严格保留官方脚本中的1 epoch，其余runs为10 epochs；ECL使用官方batch size 16，其余为32。FATST adapter只把official test访问限制到训练结束后一次，并记录effective config、environment、checkpoint、metrics和logs。

新24 runs不保存prediction arrays。原因不是选择性结果处理，而是ECL/Solar完整test predictions可能触及remote 220 GiB hard quota；所有MSE/MAE、checkpoint、config、environment、training log、segment metrics、diagnostics与source/dataset hashes仍强制保留。既有8 runs的predictions继续保留。

## Resource and rollback gate

- preflight observed 3 × RTX 3090 idle，free memory约24.1 GiB/GPU；
- quota=`175/200/220 GiB`，已有TimeAlign 8 runs=`2.0 GiB`；
- 先执行24个new jobs的1-epoch/two-batch resource smoke，`final_evaluation_split=none`；
- 任一OOM、NaN/Inf、Traceback、dataset/source hash mismatch或required artifact缺失即停止full launch；
- full queue使用GPU0--2动态调度，先ECL、Solar和ETTm1，再ETT-hourly与Exchange；
- 32/32矩阵完整前不得形成paper-facing TimeAlign替换行。

Machine contract=`configs/timealign_official_main_i_reproduction.json`。Local checker、JSON parse、shell syntax和dry-run均通过。Decision=`timealign_main_i_32_run_matrix_frozen_resource_smoke_then_remote_launch`。
