# Phase3-C Regime/Segment Operator Remote Launch

更新时间：2026-06-24 10:48 CST

## Command

Remote host:

- `529_Lab-3090`

Remote project:

- `/home/yingch/projects/FATST`

Git commit:

- `bf9af01e31eb3b9ceaaa623448ee2ca5e45ae296`

Launch command:

```bash
cd /home/yingch/projects/FATST
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator
mkdir -p ${OUTPUT_ROOT}/_logs
GPU_IDS="1 2" \
DATASETS="ETTm1 Weather ETTh2" \
TARGET_HORIZONS="96,720" \
EPOCHS=100 \
OUTPUT_ROOT=${OUTPUT_ROOT} \
nohup bash scripts/remote/run_phase3_regime_segment_operator_gate.sh \
  > ${OUTPUT_ROOT}/_logs/phase3_regime_segment_operator_outer.log 2>&1 &
```

PID:

- `3736385`

## Environment

- conda env: `moe`
- output root: `/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator`
- run name: `PatchEncoderRegimeSegmentTargetOperator`
- model variant: `regime_segment_operator`
- target horizons: `96,720`
- datasets: `ETTm1 Weather ETTh2`
- step loss weighting: `prefix_risk`
- use window position: `1`
- heavy artifacts: runner default removes `checkpoint.pt` and `predictions_test.npz`

## GPU Check

Before launch:

| GPU | Used MiB | Free MiB | Util % | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 3964 | 20161 | 2 | avoided |
| 1 | 18 | 24107 | 0 | selected |
| 2 | 18 | 24107 | 0 | selected |

Initial running check:

| Time | Dataset | GPU | Status |
| --- | --- | ---: | --- |
| 2026-06-24 10:46 CST | `ETTm1` | 1 | running |
| 2026-06-24 10:46 CST | `Weather` | 2 | running |
| 2026-06-24 10:46 CST | `ETTh2` | pending | queued |

## Progress Check

At 2026-06-24 10:48 CST:

| Dataset | Status | Epoch | ETA |
| --- | --- | --- | --- |
| `ETTm1` | running | `3/100` | 2026-06-24 11:40 CST |
| `Weather` | running | `1/100` | 2026-06-24 13:09 CST |
| `ETTh2` | queued | `0/100` | after one GPU frees |

[Decision] Remote Phase3-C minimal gate is running normally. Next action is to wait for completion, then sync
`metrics_by_target_horizon.csv`, `metrics_by_segment.csv`, `prefix_consistency.csv`,
`regime_segment_operator_stats.csv`, and `regime_feature_stats.csv` for analysis against R.3.
