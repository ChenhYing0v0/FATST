# PCSD-CF Step7B Remote Launch Record

## Scope

- `candidate`: `SC1-PCSD-CF-v1`
- `current_step`: Step7B effectiveness screen / 11-step Step8 remote execution
- `host`: `529_Lab-3090`
- `remote_repo`: `/home/yingch/projects/FATST`
- `commit`: `b9693ecf0df31a12405b6578668a8e0bbc28807e`
- `environment`: `moe`
- `launched_at`: `2026-07-16T16:10:33+08:00`
- `runner_pid`: `1249337`（remote launch supervisor `1249335`）
- `gpu_ids`: `0,1,2`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b`
- `launcher_log`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b/launcher_seed2021.log`
- `matrix`: 12 arms × 5 datasets × seed2021 = 60 jobs
- `evaluation`: validation dense H1..720 full-crop；`test_used=false`

## Resource Audit And Smoke

启动前GPU 0/1/2均为空闲：每卡总显存24576 MiB、已用15 MiB、可用24110 MiB、utilization 0%。先在GPU 0
执行`PCSD_CF_DIRECT / Weather / batch32 / one-batch` resource smoke，约8秒完成，未出现OOM或numeric error；
smoke输出位于`_resource_smoke/pcsd_direct_weather_seed2021`。

## Launch Command

```bash
GPU_IDS="0 1 2" SEED=2021 scripts/remote/run_stage_c_pcsd_cf_step7b.sh
```

runner将effective config、profile/design hash与每个run的独立log写入output root。启动时profile hash为
`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`，design hash为
`d103468ba63acf2d2c3ea1031b11d39b81526a9eda1cbd4ae796443dc2324684`。

## Initial Progress Check

三个worker已进入预期位置：

| GPU | Initial job | Observed progress at 16:11:37 |
| ---: | --- | --- |
| 0 | `1/60 A6 / Weather` | epoch 4, iteration 1100 |
| 1 | `2/60 A6 / ETTm1` | epoch 4, iteration 1000 |
| 2 | `3/60 A6 / ETTh1` | completed at epoch 8 with best epoch 3；continued to `6/60 PCSD_M0 / Weather`, epoch 1 iteration 1100 |

首次检查为`completed=1/60`；runner、三个workers均存活。基于首批速度，粗略预计矩阵为几十分钟量级，实际
完成时间以Weather与PCSD full-arm runtime为准，不据该早期观测承诺精确ETA。

## Decision Boundary

当前decision=`step7b_remote_seed2021_running`。这只表示实现、资源与launch gate通过，不表示PCSD-CF
effectiveness通过。等待60/60 validation artifacts后执行Step9 analyzer，再按预注册effectiveness、mechanism、
capacity与failure-attribution gates进入Step10。SC2、test、seeds2022/2023继续held。
