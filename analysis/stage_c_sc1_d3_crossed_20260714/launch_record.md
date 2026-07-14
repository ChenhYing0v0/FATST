# SC1-D3 Remote Launch Record

| Field | Value |
| --- | --- |
| `date` | 2026-07-14 |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `b0c540420a6307c7c2aca91bcb8f11191e24563c` |
| `conda_env` | `moe` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d3_crossed` |
| `D2_input_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d2_formal5` |
| `start` | `2026-07-14T13:33:02+08:00` |
| `finish` | `2026-07-14T13:34:13+08:00` |
| `matrix` | 5 datasets × 3 checkpoints × 3 missing-cell structure seeds = 45 fits |

Launch前`nvidia-smi`显示GPU 0/1/2均为RTX 3090，各使用15 MiB、空闲24110 MiB、utilization 0%。
调度为GPU0 `Weather -> ETTh1`、GPU1 `ETTm1 -> ETTh2`、GPU2 `ETTm2`。remote dry-run的worker与
analyzer synthetic smoke均通过后启动正式runner。

所有dataset worker均正常输出`stage_c_sc1_d3_done ... fits=9`；remote analyzer完成后，由
`scripts/sync_stage_c_sc1_d3_crossed_results.sh`同步raw artifacts，并在local repo使用同一预注册config
独立重算gate。
