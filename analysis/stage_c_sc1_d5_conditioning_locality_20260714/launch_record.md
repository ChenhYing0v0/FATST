# SC1-D5 Remote Launch Record

| Field | Value |
| --- | --- |
| `date` | 2026-07-14 |
| `remote_host` | `529_Lab-3090` |
| `commit` | `983292974ee8b4b8c3dd22bb1d003ea9bdcf87df` |
| `conda_env` | `moe` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d5_conditioning_locality` |
| `start` | `2026-07-14T14:20:01+08:00` |
| `finish` | `2026-07-14T14:33:39+08:00` |
| `matrix` | 5 datasets × 3 checkpoints × 3 grouping seeds × 13 bases = 585 fits |

Launch前GPU 0/1/2均为RTX 3090，各使用15 MiB、空闲24110 MiB、utilization 0%。585/585 fits完成，remote
analyzer后由local analyzer独立重算。首轮调度使GPU2在ETTm2后空闲；后续wrapper已改为GPU2继续ETTh2，
不影响本轮artifact validity。
