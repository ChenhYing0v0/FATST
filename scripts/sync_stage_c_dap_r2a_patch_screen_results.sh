#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_dap_r2a_patch_screen_$(date '+%Y%m%d')}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_dataset_profile_calibration_r2.json}"
mkdir -p "${ANALYSIS_ROOT}/raw"
rsync -av --exclude 'checkpoint*.pt' --exclude 'predictions_val.npz' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${ANALYSIS_ROOT}/raw/"
python scripts/analyze_stage_c_dap_r2a_patch_screen.py \
  --raw-root "${ANALYSIS_ROOT}/raw" --output-dir "${ANALYSIS_ROOT}" --config "${CONFIG_PATH}"
echo "sync_stage_c_dap_r2a_done=$(date -Is)"
