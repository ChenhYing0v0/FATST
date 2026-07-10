# Phase5 StageB C0 ETTm1 Encoder Control Launch Record

| Field | Value |
| --- | --- |
| `launch_time` | 2026-07-10 17:53:53 Asia/Shanghai |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `fb579f3d66e2cc46be40ca20f20a3dd70de08416` |
| `conda_env` | `moe` |
| `launcher_pid` | `3136247` |
| `gpu_ids` | `0 1 2` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate` |
| `launcher_log` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate/_logs/launcher.log` |
| `matrix` | ETTm1, six arms, seed 2021, 10 epochs, dual last/best-val evaluation |
| `status` | `remote_running_initial_process_check_passed` |

## GPU preflight

启动前 3 张 RTX 3090 均为空闲：

| GPU | Used MiB | Free MiB | Utilization |
| ---: | ---: | ---: | ---: |
| 0 | 15 | 24110 | 0% |
| 1 | 15 | 24110 | 0% |
| 2 | 15 | 24110 | 0% |

## Launch command

```bash
cd /home/yingch/projects/FATST
git pull --ff-only origin main
nohup env GPU_IDS="0 1 2" \
  OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate \
  CONDA_ENV=moe \
  bash scripts/remote/run_phase5_stage_b_c0_ettm1_carrier_protocol_gate.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate/_logs/launcher.log \
  2>&1 < /dev/null &
```

## Initial confirmation

Launcher 与首批三个 arms 均存活；GPU 0/1/2 分别出现对应 Python compute process，初始显存约
`280/278/278 MiB`。launcher log 已记录正确 commit、六臂矩阵和三个 `run_start`。

按用户要求，启动确认后不做长期值守。远程完成后由用户通知，再执行 sync 与预注册 analyzer。
