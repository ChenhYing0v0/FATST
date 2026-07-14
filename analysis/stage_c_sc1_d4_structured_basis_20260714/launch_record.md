# SC1-D4 Remote Launch Record

| Field | Value |
| --- | --- |
| `date` | 2026-07-14 |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `ace864192bec45bf79e4ec51878e0559f603415e` |
| `conda_env` | `moe` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d4_structured_basis` |
| `start` | `2026-07-14T13:57:17+08:00` |
| `finish` | `2026-07-14T14:04:20+08:00` |
| `matrix` | 5 datasets × 3 checkpoints × 3 grouping seeds × 7 bases = 315 fits |

Launch前GPU 0/1/2均为RTX 3090，各使用15 MiB、空闲24110 MiB、utilization 0%。调度为GPU0
`Weather -> ETTh1`、GPU1 `ETTm1 -> ETTh2`、GPU2 `ETTm2`。remote worker/analyzer dry-run通过后正式启动。

五个dataset均输出63/63 fits；remote analyzer完成后同步raw artifacts，并在local repo独立重算decision。
