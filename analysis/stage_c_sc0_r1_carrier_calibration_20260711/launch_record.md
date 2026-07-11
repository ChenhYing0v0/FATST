# StageC SC0-R1 Carrier Calibration Launch Record

## Launch

- `candidate`: `SC0-R1`
- `role`: validation-only standardized carrier control
- `launch_time`: `2026-07-11T16:22:49+08:00`
- `local/remote_commit`: `15b391b529d28e07fc934cdebe02aa5694d64dfe`
- `profile_hash`: `3ebd07d647cdd4b0e8ea36a53eea9451d21f438a79164f74b8f4e8095426f31a`
- `remote_host`: `529_Lab-3090`
- `remote_repo`: `/home/yingch/projects/FATST`
- `conda_env`: `moe`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_r1_carrier_calibration`
- `launcher_pid`: `602169`
- `GPUs`: `0,1,2`
- `matrix`: `3 datasets × 3 arms × 3 seeds = 27 runs`
- `order`: dataset-major `Weather -> ETTm1 -> ETTh2`，每批三卡并行

## Resource Preflight

| GPU | Model | Used MiB | Free MiB | Utilization | Compute process |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | RTX 3090 | 15 | 24110 | 0% | none |
| 1 | RTX 3090 | 15 | 24110 | 0% | none |
| 2 | RTX 3090 | 15 | 24110 | 0% | none |

Remote `/home` 可用约`962 GiB`。Remote worktree在pull前clean，随后fast-forward到launch commit；remote
SC0-R1 semantic gate与runner dry-run均通过。

## Frozen Training And Gate

- `max_epochs=20`, `patience=5`, `min_delta=0`, `restore-best`；
- full-720 L1 training与full-720 validation MSE selection；
- final evaluation split为validation；profile freeze前不读取test；
- pooled-mean与median-seed winner必须一致；
- selected arm至少赢2/3 seeds；
- pooled每dataset regret不超过3%；任一seed-dataset regret不超过5%。

## Launch Command

```bash
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_r1_carrier_calibration \
GPU_IDS="0 1 2" \
bash scripts/remote/run_stage_c_sc0_r1_carrier_calibration.sh
```

命令通过`nohup`后台启动，stdout/stderr写入`_launcher.log`。启动后确认三个Weather seed2021 arms各启动
一次，三个GPU均进入训练，无重复output directory。
