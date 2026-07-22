# ISCF-BSCA-v1 three-seed confirmation remote launch

## 当前状态

`current_step=Step 8 confirmation training active`。2026-07-22 19:39:57 +08:00，冻结的seeds2022/2023
confirmation在`529_Lab-3090`启动。当前只执行10个BSCA training/validation runs；formal test仍由
`10/10 training complete` hard guard阻断。

## Launch provenance

- Git commit：`72e3356720acba45d9dcd81aa197b138a4e64b59`
- Config SHA256：`167d886d137aa098c06a42dbd8a4bcc2e8c61939f823400ab6c1c945adb365e0`
- Profile SHA256：`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`
- Output root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_bsca_v1_confirmation`
- Driver PID：`62015`（remote wrapper；外层SSH shell PID为`62012`）
- Scheduling：seed2022=`GPU 0 1`；seed2023=`GPU 2`
- Launch command：`OUTPUT_ROOT=... SEED2022_GPUS="0 1" SEED2023_GPUS="2" bash scripts/remote/run_stage_c_iscf_bsca_v1_confirmation.sh`

## Resource and smoke audit

启动前GPU 0/1/2均为RTX 3090，memory used=`18 MiB`、free=`24107 MiB`、utilization=`0%`，无active compute
process。Weather seed2022 resource smoke于19:39:47完成，真实GPU forward/backward与artifact写入通过。

启动后首批三个jobs为Weather seed2022、ETTm1 seed2022和Weather seed2023；epoch 1已产生finite loss。19:40:22
GPU 0/1/2占用约`1535/766/1536 MiB`，利用率约`63%/25%/66%`，无OOM或numeric pathology。

## Authorization boundary

- 允许：冻结的10-run confirmation training；10/10后同一protocol的一次formal test。
- 禁止：读取partial validation结果作选择，修改objective/lambda/profile/rank/selector，按dataset/horizon调参，增加新loss/router。
- Formal test：当前`0/10`，runner在10/10 training artifacts完成前必须失败关闭。

## decision

Decision=`confirmation_step8_training_active_formal_test_guarded`。下一步只做低频完整性/epoch进度检查；10/10后执行
single formal test与three-seed analyzer。
