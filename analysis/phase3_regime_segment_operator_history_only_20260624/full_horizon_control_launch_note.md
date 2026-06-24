# Phase3-C History-Only Full-Horizon Control Launch

更新时间：2026-06-24 12:05 CST

## Purpose

This control addresses the remaining horizon-set confound after `history_only_h96_h720` passed.

[Control] Run `PatchEncoderRegimeSegmentTargetOperator` with:

- `USE_WINDOW_POSITION=0`;
- `TARGET_HORIZONS=96,192,336,720`;
- same primary baseline horizon set as R.3.

## Command

Remote host:

- `529_Lab-3090`

Remote project:

- `/home/yingch/projects/FATST`

Git commit:

- `e1d5f93d4c0a5970a4c2dbcd9833c388a956399f`

Launch command:

```bash
cd /home/yingch/projects/FATST
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator_history_only_full
mkdir -p ${OUTPUT_ROOT}/_logs
GPU_IDS="1 2" \
DATASETS="ETTm1 Weather ETTh2" \
TARGET_HORIZONS="96,192,336,720" \
EPOCHS=100 \
RUN_NAME="PatchEncoderRegimeSegmentTargetOperatorHistoryOnlyFull" \
USE_WINDOW_POSITION=0 \
OUTPUT_ROOT=${OUTPUT_ROOT} \
nohup bash scripts/remote/run_phase3_regime_segment_operator_gate.sh \
  > ${OUTPUT_ROOT}/_logs/phase3_regime_segment_operator_history_only_full_outer.log 2>&1 &
```

PID:

- `240983`

## GPU Check

Before launch:

| GPU | Used MiB | Free MiB | Util % | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2649 | 21476 | 1 | avoided |
| 1 | 18 | 24107 | 0 | selected |
| 2 | 5476 | 18649 | 0 | selected with margin |

## Initial Progress

At 2026-06-24 12:04 CST:

| Dataset | Status | Epoch | GPU |
| --- | --- | --- | ---: |
| `ETTm1` | running | `2/100` | 1 |
| `Weather` | running | `unknown/100` | 2 |
| `ETTh2` | queued | `0/100` | pending |

[Decision] Full-horizon history-only control is running. This is the decisive control for whether Phase3-C can
move from positive diagnostic evidence to a mechanism-pass candidate.
