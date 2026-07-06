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

## Completion Status

| Field | Content |
| --- | --- |
| `completed_at` | `2026-07-06` |
| `completed_runs` | `12/12` |
| `local_synced_artifacts` | `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/raw/` |
| `analysis_report` | `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/stage_b_dependency_ablation_report.md` |
| `decision` | `dependency_ablation_pass_for_head_contribution_but_not_for_b5` |

[Fact] All 12 runs finished. Local sync excludes large `checkpoint.pt` and `predictions_test.npz` files; the remote
output root remains the source for full raw artifacts.

[Result] `no_align_no_recon` changed mean MSE by only `+0.07%` vs `current_align_recon` and won `7/12` horizon
settings. `align_no_recon` was slightly better on mean MSE (`-0.04%`) but the effect is too small to support B5 as a
paper-core method.

## Scheduling Note

[Fact] The current runner assigned jobs by simple arm/dataset order and modulo GPU id. This allowed several long
Weather jobs to accumulate on GPU 0 during the matrix.

[Decision] Do not retroactively change this completed experiment. For future remote matrices, use workload-aware or
dataset-major scheduling so Weather/ETTm1 are distributed across available GPUs before filling shorter jobs.
