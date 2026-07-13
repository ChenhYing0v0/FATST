# SC1-D2 Core3 Remote Launch Record

| Field | Value |
| --- | --- |
| `server` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `a3d69a40da92510c38ffef5330794aa06c03480e` |
| `environment` | conda `moe` |
| `python` | `3.12.13` |
| `torch` | `2.9.0+cu128` |
| `CUDA` | `12.8` |
| `GPU model` | NVIDIA GeForce RTX 3090 ×3 |
| `preflight memory` | GPU0/1/2 each `15 MiB / 24576 MiB`；no compute process |
| `assignment` | Weather→GPU0；ETTm1→GPU1；ETTh2→GPU2 |
| `command` | `bash scripts/remote/run_stage_c_sc1_d2_diagnostic.sh` |
| `output` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d2_core3_precheck` |
| `start` | `2026-07-13T23:38:26+08:00` |
| `finish` | `2026-07-13T23:40:10+08:00` |
| `matrix` | 3 datasets × 3 checkpoint seeds × 11 arms = 99/99 |
| `max observed memory` | initial active snapshot GPU0 `441 MiB`、GPU1/2 `414 MiB` |

remote dry-run先通过runner/analyzer synthetic smoke。正式运行只训练probe heads，forecast checkpoints
保持冻结；输出位于repo外。运行结束后GPU0/1/2均回到`15 MiB`。
