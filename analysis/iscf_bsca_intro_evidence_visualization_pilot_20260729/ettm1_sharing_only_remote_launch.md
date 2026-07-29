# ETTm1 Sharing-Only Visualization Screen Remote Launch

## Launch identity

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-VIZ-v1` |
| `role` | illustrative visualization candidate search |
| `commit` | `465931fd2fb7ac41929dc00fad6e857fbd4b27e3` |
| `remote_host` | `529_Lab-3090` |
| `dataset` | ETTm1 |
| `seed` | 2021 |
| `run_mode` | sharing-only |
| `new_runs` | 5 |
| `sharing_extents` | 1, 8, 32, 128, 720 |
| `split` | validation only |
| `test_accessed` | false |
| `launch_time` | `2026-07-29T20:17:27+08:00` |
| `launcher_pid` | 1621997 |

## Frozen controls

- exact same `NeutralSharingExtentForecaster` parameterization；
- five variants均为111,312 parameters；
- uniform full-domain pointwise MSE；
- 12个60-step future regions；
- crossover margin=`0.5%`；
- descriptive headroom threshold=`0.5%`；
- 不重复DLinear prefix runs；
- ETTh2、formal test与full matrix未授权。

## GPU and health check

启动前GPU 0/1/2均为RTX 3090，分别只有18 MiB显存占用。CUDA resource smoke通过。
启动后一次有界检查确认：

| GPU | Active scale | Memory used | Utilization |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 565 MiB | 37% |
| 1 | 8 | 548 MiB | 37% |
| 2 | 32 | 548 MiB | 46% |

三份job log均已创建，未观察到即时import、CUDA或OOM错误。当前`0/5`表示尚无完整
run结束。

## Output and next action

Output root：

`/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1`

Supervisor log：

`/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1/supervisor_ETTm1_sharing.log`

Decision=`ettm1_sharing_visualization_screen_running`。完成5/5后同步validation
figures与summary，判断是否形成具有说服力的figure candidate；该判断不作为
ISCF-BSCA architecture effectiveness gate。
