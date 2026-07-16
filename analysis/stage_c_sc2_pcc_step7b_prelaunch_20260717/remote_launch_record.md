# SC2-PCC-v1-TI Step 7B Remote Launch Record

## Launch Summary

| Field | Value |
| --- | --- |
| `current_step` | 11-step Step8 remote execution running |
| `host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `282b96c49ec1230a2ccb703781ad53d10334f916` |
| `environment` | conda `moe` |
| `launch_time` | `2026-07-17T00:49:41+08:00` |
| `launcher_pid` | `1915464` |
| `runner_pid_at_startup` | `1915466` |
| `gpu_workers_at_startup` | GPU0 `1915485`；GPU1 `1915489`；GPU2 `1915494` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc2_pcc_step7b` |
| `launcher_log` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc2_pcc_step7b/launcher_seed2021.log` |
| `decision` | `step7b_remote_seed2021_running` |

## Pre-launch Audit

Remote `git pull --ff-only`已把代码更新到上述commit。remote worktree中原有两处与本阶段无关的modified CSV，
本次未修改、覆盖或清理它们。

启动前GPU状态：

| GPU | Model | Used MiB | Free MiB | Utilization |
| ---: | --- | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% |
| 1 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% |
| 2 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% |

Remote dry-run通过：`jobs=45`，config hash
`e547ed3c0da11d5c117967bf72da71999c913755ed9183703fbed86deb21cce7`，profile hash
`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`，`test=false`。

## Resource Smoke

GPU0完成`pcc_transport_full / Weather / seed2021`的one-train-batch、one-eval-batch、one-epoch smoke。产物包含
`checkpoint.pt`、`effective_config.json`、`environment.json`、`initialization_contract.json`、
`model_diagnostics.json`、`training_log.csv`与`smoke.log`。smoke结束后三张GPU均回到15 MiB占用，说明当前batch
size与执行路径可在RTX 3090上安全启动。该smoke没有执行完整dense validation，不构成performance evidence。

## Frozen Remote Matrix

- nine objective modes × five datasets × seed2021 = 45 new runs；
- dataset-major slow-first：`Weather -> ETTm1 -> ETTm2 -> ETTh1 -> ETTh2`；
- three workers固定使用GPU 0/1/2；
- checkpoint selection：best validation H720 MSE；
- evaluation：validation dense H1..720 full-crop；
- official test、confirmation seeds与conditional Phase B仍为`false`；
- A6、plain DIRECT、DENSE_MATCHED与five fixed-scope arms只读复用，不重训。

## Startup Health Check

启动15秒后的单次检查显示：

- `completed=0/45`，符合刚启动状态；
- GPU0/1/2分别开始job 1/2/3，均为Weather：`measure_only`、`equal_skill`、
  `pointwise_route_only`；
- 三张GPU显存约为1983/2024/2004 MiB，GPU2已观察到45%利用率；
- 三个training logs均已写入正确dataset、mode、output directory与`pcc_objective_mode`；
- launcher、runner与三个worker进程均存活。

因此本次只判定remote execution健康启动，不对PCC effectiveness作任何结论。按用户要求不继续长期值守；45/45
返回后再执行Step9 artifact audit、完整analyzer与Step10 effectiveness/failure-attribution decision。
