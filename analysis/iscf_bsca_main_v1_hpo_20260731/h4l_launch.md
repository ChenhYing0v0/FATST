# ISCF-BSCA-MAIN-v1 H4L Launch Record

## Decision

H4L local/remote checker与48/48 full-matrix resource smoke均通过。正式48-job train/validation已于2026-08-04 10:09:00在GPU0/1/2 detached启动，test=0。

Decision=`H4L_48_job_train_validation_active_test_zero`。

## Frozen launch evidence

| Field | Value |
| --- | --- |
| commit | `c4c1e6bcf834ea189f1dace62a66c710b769aee4` |
| config hash | `3efb06bb9efeed0c51a44ddf74c007312eb6fb6260d64b6e65105e2b6ee84d81` |
| search-space hash | `90de181a070bdeba2e25e709a524a64c7c21fa9b9ccf7c71502da26778c1471d` |
| jobs | 48 = ETTm2 24 + Weather 24 |
| seed | 2021 |
| GPUs | 0, 1, 2；launch前均18 MiB、0% utilization |
| process | detached shell PID `3714210`；runner PID `3714214` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4l` |
| quota | resource smoke后约196/220 GiB |
| formal test | not authorized；0/48 |

Resource smoke于10:05:58--10:08:09完成：48/48 run directories与48/48 `metrics_by_target_horizon.csv`存在，failure-pattern hits=0；最大capacity、最大patch、短context及四个TimeAlign-inspired profiles均通过。Smoke只使用2个train batches、2个validation batches和1 epoch，没有official-test access。

正式训练首批为`Weather__h4l_d256_ff512_r256`、`ETTm2__h4l_d256_ff512_r128`与`Weather__h4l_d256_ff256_r128`。启动后约1分三项均处于epoch 1、iteration 1600--1700，GPU memory约0.7--1.6 GiB，无OOM/NaN/Inf。按冻结预算的conservative ETA为8--18小时；early stopping可能缩短。

## Completion gate

只有48/48 jobs均生成checkpoint、training log、four-H validation metrics、effective config、initialization contract与diagnostics，且numeric-health和provenance audit通过，才冻结checkpoint manifest。之后先汇报validation/artifact completeness并请求H4L formal-test授权；不得执行partial test、per-H selection、H4M或3-seed扩展。
