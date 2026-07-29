# Introduction Evidence Visualization Pilot Remote Launch Record

## 1. Launch identity

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-VIZ-v1` |
| `commit` | `9cc2d24e2359310dea2c6fc1764303a2da5d2c65` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `remote_environment` | `moe` |
| `dataset` | Weather |
| `seed` | 2021 |
| `new_runs` | 9 |
| `split` | validation only |
| `test_accessed` | false |
| `formal_test_authorized` | false |
| `fallback_authorized` | false |
| `launch_time` | `2026-07-29T17:40:13+08:00` |
| `supervisor_pid` | `1413423` |

## 2. Frozen matrix

| Evidence | Runs | GPU scheduling |
| --- | --- | --- |
| Prefix disagreement | DLinear × Weather × H96/H192/H336/H720 × seed2021 | dynamic three-GPU queue |
| Sharing demand | Neutral × Weather × s1/s8/s32/s128/s720 × seed2021 | dynamic three-GPU queue |

远程输出根目录：

`/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1`

supervisor log：

`/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1/supervisor_Weather.log`

## 3. GPU preflight and resource smoke

启动前，GPU 0/1/2均为NVIDIA GeForce RTX 3090，显存状态均为
`18 MiB used / 24107 MiB free`，GPU utilization均为`0%`。GPU 0上的CUDA
resource smoke通过：

- five scales均能完成forward/backward；
- prediction与internal tensor shapes符合冻结contract；
- parameter counts相同；
- gradients为finite且nonzero。

## 4. Immediate bounded health check

在`2026-07-29T17:41:33+08:00`执行一次启动后有界检查：

- completed status：neutral `0/5`，DLinear `0/4`；
- active jobs：neutral `s=1,8,32`；
- GPU 0：`969 MiB used`，utilization `100%`；
- GPU 1：`916 MiB used`，utilization `71%`；
- GPU 2：`912 MiB used`，utilization `84%`；
- supervisor与三份job log均已创建；
- 未观察到即时import、CUDA或OOM错误。

这里的`0/9`只表示检查时尚无一个完整run结束，不代表任务未启动。后续不进行高频
polling；训练完成后再同步validation artifacts并运行冻结的两套analyzer。

## 5. Claim and decision boundary

本轮结果只能用于判断：

1. Weather上是否存在适合Introduction展示的、purposefully selected但非极端的
   prefix-disagreement example；
2. Weather/seed2021上matched neutral scales是否形成清晰的exploratory
   future-region risk crossover。

不得据此宣称cross-dataset prevalence、formal problem-existence pass或
out-of-sample adaptive benefit。same-validation region oracle仍只作descriptive
panel。若Weather图形清晰，则停止dataset search；若不清晰，必须先汇报，再请求
是否授权`ETTm1` fallback。

## 6. Current decision

`decision=initial_weather_9run_validation_pilot_running`

下一动作：等待远程训练完成；收到完成信号后，同步全部9个run及自动分析产物，
核验matrix completeness、alignment guards和figure selection disclosure，再决定
两项图是否达到Introduction illustrative evidence标准。
