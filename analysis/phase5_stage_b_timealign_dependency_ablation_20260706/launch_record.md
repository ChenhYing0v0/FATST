# Phase5 StageB TimeAlign Dependency Ablation Launch Record

`current_step`: StageB Step 8 remote training after B4 dependency audit.

## Launch Summary

| Field | Content |
| --- | --- |
| `launched_at` | `2026-07-06T16:43:57+08:00` |
| `remote_host` | `529_Lab-3090` / `star3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `731bb0e642004029faf6aeef27fce7ab90927907` |
| `conda_env` | `moe` |
| `dataset_root` | `/home/yingch/dataset` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_timealign_dependency_ablation` |
| `launcher_log` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_timealign_dependency_ablation/launcher.log` |
| `launcher_pid` | `99608` |
| `checkpoint_policy` | `official-last` |
| `seed` | `2021` |

## GPU Preflight

Before launch:

| GPU | Model | Total MiB | Used MiB | Free MiB | Util % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 1 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 2 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |

Initial health check after launch:

| GPU | Used MiB | Free MiB | Util % | Initial run |
| ---: | ---: | ---: | ---: | --- |
| 0 | 4431 | 19694 | 96 | `Weather/current_align_recon` |
| 1 | 444 | 23681 | 0 | `ETTm1/current_align_recon` |
| 2 | 848 | 23277 | 28 | `ETTh2/current_align_recon` |

## Matrix

| Arm | `w_recon` | `w_align` |
| --- | ---: | ---: |
| `current_align_recon` | `1.0` | `0.1` |
| `no_align_recon` | `1.0` | `0.0` |
| `align_no_recon` | `0.0` | `0.1` |
| `no_align_no_recon` | `0.0` | `0.0` |

Datasets:

- `Weather`
- `ETTm1`
- `ETTh2`

Total runs: `4 arms * 3 datasets = 12`.

## Command

```bash
cd /home/yingch/projects/FATST
OUT=/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_timealign_dependency_ablation
mkdir -p "$OUT"
nohup bash scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh > "$OUT/launcher.log" 2>&1 &
```

## Initial Status

[Fact] The first three runs started successfully:

- `Weather/current_align_recon` on GPU 0;
- `ETTm1/current_align_recon` on GPU 1;
- `ETTh2/current_align_recon` on GPU 2.

[Fact] Launcher log showed ETTh2 reaching epoch 5 during the first health check, with no early crash.

## Next Check

Poll with:

```bash
ssh 529_Lab-3090 'tail -n 120 /home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_timealign_dependency_ablation/launcher.log'
```

When all runs finish, sync output artifacts and run a result analyzer before making any B5 basis-aware alignment decision.
