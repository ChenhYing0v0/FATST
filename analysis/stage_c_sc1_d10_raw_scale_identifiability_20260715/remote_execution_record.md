# SC1-D10 Remote Execution Record

| Field | Value |
| --- | --- |
| date | 2026-07-15 |
| host | `529_Lab-3090` |
| repo | `/home/yingch/projects/FATST` |
| commit | `8cc1a78366f22e0e00cc15e67e4386f1d94d3b91` |
| environment | conda `moe` |
| GPUs | 0/1/2，preflight均`15 MiB` used、`24110 MiB` free、0% utilization |
| command | `bash scripts/remote/run_stage_c_sc1_d10_raw_scale_identifiability.sh` |
| output | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d10_raw_scale_identifiability` |
| start | `2026-07-15T13:52:03+08:00` |
| finish | `2026-07-15T13:52:39+08:00` |
| scheduling | GPU0 Weather→ETTh1；GPU1 ETTm1→ETTh2；GPU2 ETTm2 |
| splits | train fit + temporal gap + train holdout + official validation |
| test used | false |
| forecast training | false |
| decision | `raw_aligned_scale_not_supported_rollback_step2` |

轻量raw artifacts已同步至本地ignored `raw/`；aggregate CSV/JSON与decision report在analysis root重新生成并纳入版本控制。
