# Phase2-E2 R.3 Prediction Artifact Remote Launch

更新时间：2026-06-23 23:42 +08:00

[Fact] Phase2-E2 R.3 prediction artifact collection 已在 `529_Lab-3090` 启动。

- local commit at launch:
  `d6d1a6e`;
- checker-fix commit pulled after launch:
  `7eb13c2`;
- remote project:
  `/home/yingch/projects/FATST`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions`;
- run name:
  `PatchEncoderPrefixRiskWeighted`;
- conda env:
  `moe`;
- seed:
  `2021`;
- epochs:
  `100`;
- target horizons:
  `96,192,336,720`;
- datasets:
  `ETTm1 Weather ETTh2`;
- selected GPUs:
  `1`, `2`;
- PID:
  `121880`;
- prediction saving:
  `SAVE_PREDICTIONS=1`;
- heavy artifact retention:
  `KEEP_HEAVY_ARTIFACTS=1`.

GPU check before launch:

| GPU | Memory used | Memory free | Utilization |
| ---: | ---: | ---: | ---: |
| 0 | `6597 MiB` | `17528 MiB` | `54%` |
| 1 | `19 MiB` | `24107 MiB` | `0%` |
| 2 | `18 MiB` | `24107 MiB` | `0%` |

Launch command:

```bash
cd /home/yingch/projects/FATST
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions \
  bash scripts/remote/run_phase2_qdf_alignment_r3_predictions.sh
```

Initial progress check:

- `ETTm1`: running, `epoch=1/100`, `val_mean_mse=0.610148`, `eta_sec=5991.6`;
- `Weather`: running, epoch not yet printed;
- `ETTh2`: queued;
- predictions written so far:
  `0/12`.

[Next] After completion, run:

```bash
bash scripts/sync_phase2_qdf_alignment_r3_predictions.sh
```

Then inspect:

- `analysis/phase2_qdf_alignment_diagnostic_20260623/phase2_qdf_residual_alignment_losses.csv`;
- `analysis/phase2_qdf_alignment_diagnostic_20260623/phase2_qdf_residual_alignment_report.md`.
