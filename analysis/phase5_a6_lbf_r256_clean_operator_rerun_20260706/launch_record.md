# Phase5 A6-LBF-r256 Clean Operator Rerun Launch Record

`current_step`: StageA clean-operator validation after B4/B6 diagnostics.

## Launch Summary

| Field | Content |
| --- | --- |
| `launched_at` | `2026-07-06T18:48:10+08:00` |
| `remote_host` | `529_Lab-3090` / `star3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `git_commit` | `9450e742e0fb1c5302996dbecf21b2024c8212fb` |
| `conda_env` | `moe` |
| `dataset_root` | `/home/yingch/dataset` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_a6_lbf_r256_clean_operator_rerun_20260706` |
| `launcher_log` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_a6_lbf_r256_clean_operator_rerun_20260706/launcher.log` |
| `launcher_pid` | `254410` |
| `checkpoint_policy` | `official-last` |
| `seed` | `2021` |

## GPU Preflight

Before launch:

| GPU | Model | Total MiB | Used MiB | Free MiB | Util % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 1 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |
| 2 | NVIDIA GeForce RTX 3090 | 24576 | 18 | 24107 | 0 |

Initial run assignment:

| Dataset | GPU | Purpose |
| --- | ---: | --- |
| Weather | 0 | slow dataset distributed alone |
| ETTm1 | 1 | medium dataset distributed alone |
| ETTh2 | 2 | shorter dataset distributed alone |

Initial health check after launch:

| GPU | Used MiB | Free MiB | Util % |
| ---: | ---: | ---: | ---: |
| 0 | 997 | 23128 | 64 |
| 1 | 412 | 23713 | 11 |
| 2 | 438 | 23687 | 17 |

## Command

```bash
cd /home/yingch/projects/FATST
OUT=/home/yingch/exp_outputs/r-2026-fatst/phase5_a6_lbf_r256_clean_operator_rerun_20260706
mkdir -p "$OUT"
OUTPUT_ROOT="$OUT" GPU_IDS="0 1 2" DATASETS="Weather ETTm1 ETTh2" \
  nohup bash scripts/remote/run_phase5_a6_lbf_r256_main.sh > "$OUT/launcher.log" 2>&1 &
```

## Rationale

[Fact] B4 dependency ablation showed pure no-align/no-recon A6-LBF remains competitive.

[Fact] The active code now removes future reconstruction/alignment modules entirely. This changes parameter
initialization order and should be validated as the paper's clean A6 evidence rather than relying only on historical
no-align/no-recon artifacts.

[Decision] This run is validation of the clean A6 carrier, not a new StageB method experiment.
