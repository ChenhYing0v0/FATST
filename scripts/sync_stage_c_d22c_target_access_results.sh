#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d22c_target_access_v1_1}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_post_d21_unconstrained_reset_20260720/d22c_target_access_v1_1}"
CONFIG="${CONFIG:-configs/stage_c_d22c_target_access_diagnostic.json}"

mkdir -p "${ANALYSIS_ROOT}/raw"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${ANALYSIS_ROOT}/raw/"
python scripts/analyze_stage_c_d22c_target_access.py \
  --input-root "${ANALYSIS_ROOT}/raw" --config "${CONFIG}" \
  --output-dir "${ANALYSIS_ROOT}"
echo "stage_c_d22c_sync=complete analysis=${ANALYSIS_ROOT}"
