#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_step7b_pmfo_rct}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_step7b_pmfo_rct_20260713}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

mkdir -p "${ANALYSIS_ROOT}/raw"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_ROOT}'"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'checkpoint_last.pt' \
  --exclude 'checkpoint_best_val.pt' \
  --exclude 'predictions_*.npz' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "stage_c_step7b_sync=complete analysis=skipped"
  exit 0
fi

python scripts/analyze_stage_c_step7b_pmfo_rct.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --output-dir "${ANALYSIS_ROOT}"
echo "stage_c_step7b_sync=complete analysis=${ANALYSIS_ROOT}"
