# SC1-D7 Remote Launch Record

| Field | Value |
| --- | --- |
| `start_time` | 2026-07-14T16:58:44+08:00 |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `d51d7d425cdf225a4a8d8bd1148c717197b2e5d7` |
| `environment` | conda `moe` |
| `runner_pid` | `1790157` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d7_descriptor_sufficiency` |
| `matrix` | 5 datasets × 3 frozen checkpoints × 7 arms = 105 fits |
| `test_usage` | false |

GPU preflight：GPU0/1/2均为RTX 3090，启动前memory used=`15 MiB`、free=`24110 MiB`、utilization=`0%`。

Workload-aware schedule：

- GPU0：Weather；
- GPU1：ETTm1 -> ETTh2；
- GPU2：ETTm2 -> ETTh1。

Command：

```bash
GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_sc1_d7_descriptor_sufficiency.sh
```

启动后首轮观测GPU0/1/2 memory used约`441/438/456 MiB`，utilization约`17%/15%/26%`；三个首发dataset
workers均已存在。
