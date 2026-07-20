# SC-D21-EVS Remote Launch Record

| Field | Value |
| --- | --- |
| `launch_time` | `2026-07-20T12:55:59+08:00` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `53661b1fbba84dd0ecd4082dc0f9f77102e657e6` |
| `remote_pid` | `93190` |
| `gpu_ids` | `0,1,2` |
| `gpu_prelaunch_memory_used` | `18 MiB, 18 MiB, 18 MiB` |
| `gpu_prelaunch_memory_free` | `24107 MiB, 24107 MiB, 24107 MiB` |
| `source_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a1_dual_carrier_grouped_mlp` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d21_evidence_validity_surface` |
| `matrix` | 2 carriers × 5 datasets × 5 canonical arms × val/test = 100 evaluations |
| `operation` | frozen checkpoint inference only；no forecasting training；no checkpoint mutation |

Remote command:

```bash
nohup bash scripts/remote/run_stage_c_d21_evs.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/stage_c_d21_evidence_validity_surface/launcher.log \
  2>&1 < /dev/null &
```

Prelaunch checks:

- remote `git pull --ff-only` completed while preserving three unrelated modified analysis CSV files;
- 85/85 D14 seed2021 checkpoint files were present; D21 reads the 50 canonical checkpoints;
- remote dry-run recovered 100/100 planned jobs and passed the D21 Step7A checker;
- first startup audit at `2026-07-20T12:56:21+08:00` reported 9/100 complete;
- completed anchor exports contained 4096 probe rows and 192 past-only features;
- non-anchor exports correctly reported zero descriptor features and only stored aligned losses.

Completion：`2026-07-20T13:00:06+08:00`，`100/100` jobs complete，`100/100` NPZ与`100/100`
invariant JSON。同步后Step9 decision为`close_exact_evs_problem_split_stability_failed_return_step2`；未改变problem
gate、policy controls、readout或threshold。
