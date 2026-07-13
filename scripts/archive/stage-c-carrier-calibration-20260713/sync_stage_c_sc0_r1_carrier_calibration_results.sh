#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_r1_carrier_calibration}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc0_r1_carrier_calibration_$(date '+%Y%m%d')}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_mechanism_control_r1.json}"

echo "sync_stage_c_sc0_r1_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
mkdir -p "${ANALYSIS_ROOT}/raw"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}'"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'checkpoint_last.pt' \
  --exclude 'checkpoint_best_val.pt' \
  --exclude 'predictions_val.npz' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" "${ANALYSIS_ROOT}/raw/"
python scripts/analyze_stage_c_sc0_r1_carrier_calibration.py \
  --raw-root "${ANALYSIS_ROOT}/raw" --output-dir "${ANALYSIS_ROOT}" --config "${CONFIG_PATH}"
echo "sync_stage_c_sc0_r1_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
