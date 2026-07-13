#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_A="${REMOTE_A:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
REMOTE_B="${REMOTE_B:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
ANALYSIS_A="${ANALYSIS_A:-analysis/stage_c_dap_r2a_patch_screen_20260712}"
ANALYSIS_B="${ANALYSIS_B:-analysis/stage_c_dap_r2b_width_screen_20260712}"
mkdir -p "${ANALYSIS_B}/raw"
rsync -av --exclude 'checkpoint*.pt' --exclude 'predictions_val.npz' "${REMOTE_HOST}:${REMOTE_B}/" "${ANALYSIS_B}/raw/"
python scripts/analyze_stage_c_dap_r2b_width_screen.py \
  --phase-a-root "${ANALYSIS_A}/raw" --phase-b-root "${ANALYSIS_B}/raw" \
  --phase-a-summary "${ANALYSIS_A}/r2a_summary.json" --output-dir "${ANALYSIS_B}" \
  --config configs/stage_c_dataset_profile_calibration_r2.json
