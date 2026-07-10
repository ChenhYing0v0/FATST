# Phase5 C1 Global-Anchored Multi-Patch Gate Launch Record

| Field | Value |
| --- | --- |
| `launch_time` | 2026-07-10 20:46:40 Asia/Shanghai |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `055044b81d5b9683ffde8c4e86f8ffda9e69f70c` |
| `conda_env` | `moe` |
| `launcher_pid` | `3349678` |
| `gpu_ids` | `0 1 2` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_c1_global_anchored_multipatch_gate` |
| `launcher_log` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_c1_global_anchored_multipatch_gate/_logs/launcher.log` |
| `matrix` | 3 datasets x A6 dual reference/P16-S8/P48-S24 x seed2021 = 9 runs |
| `status` | `remote_running_initial_process_check_passed` |

## GPU preflight

启动前 GPU 0/1/2 均为 RTX 3090，used memory `15 MiB`、free memory `24110 MiB`、utilization `0%`。

## Dropout policy

```text
token/input           0.0
attention weights     0.0
attention residual    0.1
FFN hidden            0.1
FFN residual          0.1
```

Legacy A6 reference保留各 dataset source-faithful legacy dropout。C1 不把 legacy ETTm1 `0.9` 传入
attention。

## Scale policy

- `gamp_p16s8`：89 valid local tokens；
- `gamp_p48s24`：29 valid local tokens；
- shared scale优先；dataset-specific scale只能由 validation minimum选择，不能由 test选择。

## Initial confirmation

Launcher与首批 Weather 三个 arms均存活，分别分配至 GPU 0/1/2；日志确认 commit、9-run matrix、dropout
policy与三个 `run_start` 正确。按既有约定，初始确认后不长期值守；远程完成后由用户通知再同步分析。
