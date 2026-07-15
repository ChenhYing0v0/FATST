# SC1-D9-A Remote Execution Record

| Field | Value |
| --- | --- |
| date | 2026-07-15 |
| host | `529_Lab-3090` |
| repo | `/home/yingch/projects/FATST` |
| commit | `ea68be7645186573cd6f8087b4f94ae2f00b1acd` |
| environment | conda `moe` |
| device | CPU-only diagnostic；GPU state只作policy preflight |
| GPU preflight | GPUs 0/1/2各`15 MiB` used、`24110 MiB` free、0% utilization |
| command | `bash scripts/remote/run_stage_c_sc1_d9_history_support_operator.sh` |
| output | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d9_history_support_operator` |
| start | `2026-07-15T13:26:08+08:00` |
| finish | `2026-07-15T13:26:36+08:00` |
| matrix | 5 datasets × 3 frozen A6 seeds = 15/15 |
| test used | false |
| forecast training | false |
| decision | `operator_scale_hypothesis_not_supported` |

轻量结果已同步到`raw/`；remote baseline checkpoints未复制进repo。
