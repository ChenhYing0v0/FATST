# SC1-JAPO Seed-2022 Remote Launch Record

| Field | Value |
| --- | --- |
| `server` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `3d37440577c0d90beedee6e4746503de2af77b51` |
| `environment` | conda `moe` |
| `start` | `2026-07-15T11:37:11+08:00` |
| `supervisor_pid` | `3185307` |
| `worker_pids` | `3185323 / 3185327 / 3185332` |
| `GPU` | RTX 3090 GPUs `0 1 2` |
| `preflight` | each GPU used 15 MiB / 24576 MiB，utilization 0%，no compute process |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_japo_e2e` |
| `matrix` | 5 datasets × 7 arms × seed2022；35 jobs |
| `protocol` | unchanged from seed2021；from-scratch paired initialization；full-H720 pointwise L1；best-val；validation-only；test=false |
| `profile_hash` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |

启动后首批三个jobs为A6/Weather、A6/ETTm1、A6/ETTm2，均已进入epoch 3，GPU used memory约
`400–456 MiB`。按seed2021同矩阵约26分钟的wall time，初始预计在`12:04 +08:00`附近完成；最终以日志为准。
runner完成后自动对seed2021/2022执行冻结two-seed mean gate。
