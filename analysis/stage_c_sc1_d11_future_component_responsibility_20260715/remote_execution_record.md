# SC1-D11 Remote Execution Record

## Accepted Run

| Field | Value |
| --- | --- |
| host | `529_Lab-3090` |
| repository | `/home/yingch/projects/FATST` |
| commit | `6c90b7be903ccd83630d5c04b02c874d5409908a` |
| environment | conda `moe` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d11_future_component_responsibility_v2` |
| start | `2026-07-15T14:27:10+08:00` |
| gate artifact mtime | `2026-07-15T14:27:44.402237780+08:00` |
| role | `diagnostic_only` |
| data boundary | train + official validation；test=false |

## GPU Preflight And Scheduling

启动前：

| GPU | Model | Used | Free | Utilization | Workload |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | RTX 3090 | 15 MiB | 24110 MiB | 0% | Weather |
| 1 | RTX 3090 | 15 MiB | 24110 MiB | 0% | ETTm1 -> ETTh2 |
| 2 | RTX 3090 | 15 MiB | 24110 MiB | 0% | ETTm2 -> ETTh1 |

五个workers均报告`total_rows=240`、`component_rows=240`；accepted analyzer decision为
`transform_generic_pressure_sc2_only`。

## Execution Correction

调用时误把runner的environment switch `DRY_RUN=1`写成了位置参数`--dry-run`，因此该命令实际直接启动了v2。
这不是runner bug。实际启动前已经完成：

1. zero-vector policy code commit/push；
2. remote `git pull --ff-only`并核对exact commit；
3. 三GPU `nvidia-smi` preflight；
4. worker/analyzer synthetic smoke、Python compile、JSON parse与shell syntax checks。

实际v2使用冻结config与独立output root，未覆盖invalid first run；故该调用错误不改变data、model、gate或artifact
有效性。

## Invalid First Run Boundary

初次output root为
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d11_future_component_responsibility`。其analyzer因zero
responsibility vector产生undefined cosine而停止。该run不得支持方向判断，也未被accepted v2覆盖。修复规则为：

- zero responsibility不计为conflict；
- cosine只对双方norm均大于`1e-12`的active pairs求均值；
- negative fraction保留全部pairs，zero dot为non-negative；
- 新增`short_zero_group_count`与`long_zero_group_count`。

冻结gate、dataset、checkpoint、batch、loss、basis与threshold均未改变。
