# Phase3 Horizon-Set Interference Control Launch

更新时间：2026-06-24 12:50 CST

## Purpose

This control tests whether the earlier positive `h96,h720` Phase3-C result came from the operator or from removing
intermediate horizons `192/336` from mixed-horizon training pressure.

[Control] Run the R.3 carrier without the Phase3-C operator:

- model variant: `target_set`;
- step loss: `prefix_risk`;
- target horizons: `96,720`;
- run name: `PatchEncoderPrefixRiskWeightedH96H720`.

## Command

Remote host:

- `529_Lab-3090`

Remote project:

- `/home/yingch/projects/FATST`

Git commit:

- `77d2a78987fecd69d8f28a40aa052ccca68650cd`

Launch command:

```bash
cd /home/yingch/projects/FATST
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase3_horizon_set_interference
mkdir -p ${OUTPUT_ROOT}/_logs
GPU_IDS="1 2" \
DATASETS="ETTm1 Weather ETTh2" \
TARGET_HORIZONS="96,720" \
EPOCHS=100 \
RUN_NAME="PatchEncoderPrefixRiskWeightedH96H720" \
MODEL_VARIANT=target_set \
STEP_LOSS_WEIGHTING=prefix_risk \
STEP_LOSS_ALPHA=0.5 \
OUTPUT_ROOT=${OUTPUT_ROOT} \
nohup bash scripts/remote/run_phase1_target_set_decoder_gate.sh \
  > ${OUTPUT_ROOT}/_logs/prefix_risk_h96_h720_outer.log 2>&1 &
```

PID:

- `1056199`

## GPU Check

Before launch:

| GPU | Used MiB | Free MiB | Util % | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 4982 | 19143 | 1 | avoided |
| 1 | 18 | 24107 | 0 | selected |
| 2 | 18 | 24107 | 1 | selected |

## Initial Progress

At 2026-06-24 12:50 CST:

| Dataset | Status | Epoch | GPU |
| --- | --- | --- | ---: |
| `ETTm1` | running | `6/100` | 1 |
| `Weather` | running | `2/100` | 2 |
| `ETTh2` | queued | `0/100` | pending |

[Note] The generic Phase3 progress checker expects a Phase3 log subdirectory, while this control reuses the
Phase1 runner and writes logs under `_logs/phase1_target_set_decoder_gate/`. The run itself is active.
