# SC-RETRO-FAIR-v1 Remote Launch

- `launch_time`: 2026-07-17T21:45:08+08:00
- `source_commit`: `d294aab26c4505e34a219ca8a4e25e4fcb9c9428`
- `remote_repo`: `/home/yingch/projects/FATST`
- `environment`: `moe`
- `GPUs`: 0, 1, 2
- `prelaunch_memory`: each RTX 3090 used 15 MiB, free 24110 MiB
- `resource_smoke`: Weather × `siff_pcc`，GPU0，1 train/eval batch，passed
- `matrix`: 14 arms × 5 datasets × seed2021 = 70 runs
- `checkpoint`: validation mean MSE over H96/H192/H336/H720
- `formal_evaluation`: official test H96/H192/H336/H720 MSE/MAE
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_fair_reaudit_v1`
- `runner_pid`: 3500074
- `launcher_pid`: 3500072
- `nohup_log`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_fair_reaudit_v1/nohup_seed2021.log`
- `initial_jobs`: Weather `a6_full` on GPU0；Weather `pcsd_direct` on GPU1；
  Weather `dense_measure` on GPU2
- `status`: running；no high-frequency monitoring
