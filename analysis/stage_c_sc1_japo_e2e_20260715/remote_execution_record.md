# SC1-JAPO Seed-2021 Remote Execution Record

| Field | Value |
| --- | --- |
| `server` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `90e4164aaaf9fc31b12f7b4b37459106a5df580d` |
| `environment` | conda `moe` |
| `start` | `2026-07-15T10:48:04+08:00` |
| `finish` | `2026-07-15T11:14:31+08:00` |
| `GPU` | RTX 3090 GPUs `0 1 2` |
| `preflight` | each GPU used 15 MiB / 24576 MiB，utilization 0% |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_japo_e2e` |
| `matrix` | 5 datasets × 7 arms × seed2021；35/35 |
| `protocol` | from-scratch paired initialization；full-H720 pointwise L1；best-val；validation dense H1..720；test=false |
| `profile_hash` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |

远程raw artifacts与checkpoints继续保留在repo-external `output_root`。本地只保留独立重算后的聚合CSV、gate、
failure attribution与研究解释，不把checkpoint或逐run中间文件提交进repository。
