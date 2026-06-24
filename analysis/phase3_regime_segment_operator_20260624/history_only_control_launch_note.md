# Phase3-C History-Only Control Launch

更新时间：2026-06-24 11:16 CST

## Purpose

This control addresses the `window_index_norm` concern from the Phase3-C minimal gate.

[Concern] The positive `window_index_norm + h96,h720` result may encode split-local position rather than a robust
history-regime mechanism.

[Control] Run the same `PatchEncoderRegimeSegmentTargetOperator` with `USE_WINDOW_POSITION=0` and
`TARGET_HORIZONS=96,720`.

## Command

Remote host:

- `529_Lab-3090`

Remote project:

- `/home/yingch/projects/FATST`

Git commit:

- `1dcce5ce985647d49628e10317035b80307020a1`

Launch command:

```bash
cd /home/yingch/projects/FATST
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator_history_only
mkdir -p ${OUTPUT_ROOT}/_logs
GPU_IDS="1" \
DATASETS="ETTm1 Weather ETTh2" \
TARGET_HORIZONS="96,720" \
EPOCHS=100 \
RUN_NAME="PatchEncoderRegimeSegmentTargetOperatorHistoryOnly" \
USE_WINDOW_POSITION=0 \
OUTPUT_ROOT=${OUTPUT_ROOT} \
nohup bash scripts/remote/run_phase3_regime_segment_operator_gate.sh \
  > ${OUTPUT_ROOT}/_logs/phase3_regime_segment_operator_history_only_outer.log 2>&1 &
```

PID:

- `139925`

## GPU Check

Before launch:

| GPU | Used MiB | Free MiB | Util % | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2649 | 21476 | 0 | avoided |
| 1 | 18 | 24107 | 0 | selected |
| 2 | 10810 | 13315 | 0 | avoided |

## Initial Progress

At 2026-06-24 11:15 CST:

| Dataset | Status | Epoch | GPU |
| --- | --- | --- | ---: |
| `ETTm1` | running | `1/100` | 1 |
| `Weather` | queued | `0/100` | 1 |
| `ETTh2` | queued | `0/100` | 1 |

[Decision] History-only control is running normally. This is the first required control before any Phase3-C
mechanism claim can pass.
