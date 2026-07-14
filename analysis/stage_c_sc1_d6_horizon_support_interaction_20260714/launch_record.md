# SC1-D6 Remote Launch Record

| Field | Value |
| --- | --- |
| `date` | 2026-07-14 |
| `remote_host` | `529_Lab-3090` |
| `commit` | `54edd347bcb0e9a097ef973194c6d33a30b592db` |
| `conda_env` | `moe` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d6_horizon_support_interaction` |
| `start` | `2026-07-14T14:42:31+08:00` |
| `finish` | `2026-07-14T14:47:39+08:00` |
| `matrix` | 5 datasets × 3 checkpoints × 3 grouping seeds × 5 bases = 225 fits |

Launch前GPU 0/1/2均为RTX 3090，各使用15 MiB、空闲24110 MiB、utilization 0%。调度为GPU0
`Weather -> ETTh1`、GPU1 `ETTm1`、GPU2 `ETTm2 -> ETTh2`。225/225完成，remote/local analyzer decision一致。
