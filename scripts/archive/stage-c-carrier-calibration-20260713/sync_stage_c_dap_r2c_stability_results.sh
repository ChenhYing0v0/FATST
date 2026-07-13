#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_C="${REMOTE_C:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
ANALYSIS_A="${ANALYSIS_A:-analysis/stage_c_dap_r2a_patch_screen_20260712}"
ANALYSIS_B="${ANALYSIS_B:-analysis/stage_c_dap_r2b_width_screen_20260712}"
ANALYSIS_C="${ANALYSIS_C:-analysis/stage_c_dap_r2c_stability_20260712}"

mkdir -p "${ANALYSIS_C}/raw"
rsync -av --exclude 'checkpoint*.pt' --exclude 'predictions_val.npz' \
  "${REMOTE_HOST}:${REMOTE_C}/" "${ANALYSIS_C}/raw/"
python scripts/analyze_stage_c_dap_r2c_stability.py \
  --phase-a-root "${ANALYSIS_A}/raw" --phase-b-root "${ANALYSIS_B}/raw" \
  --confirmation-root "${ANALYSIS_C}/raw" --r2b-summary "${ANALYSIS_B}/r2b_summary.json" \
  --output-dir "${ANALYSIS_C}" --config configs/stage_c_dataset_profile_calibration_r2.json
