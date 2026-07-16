#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_pcsd_cf_step7b_seed2021_20260716}"
SEED="${SEED:-2021}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

mkdir -p "${ANALYSIS_ROOT}/raw"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_ROOT}'"
rsync -av \
  --exclude 'checkpoint*.pt' \
  --exclude 'predictions*.npz' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "stage_c_pcsd_cf_step7b_sync=complete analysis=skipped"
  exit 0
fi

python scripts/analyze_stage_c_pcsd_cf_step7b.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --output-dir "${ANALYSIS_ROOT}" \
  --seed "${SEED}"
python scripts/analyze_stage_c_pcsd_cf_step7b_deep_dive.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --run-summary "${ANALYSIS_ROOT}/run_summary.csv" \
  --output-dir "${ANALYSIS_ROOT}" \
  --seed "${SEED}"
echo "stage_c_pcsd_cf_step7b_sync=complete analysis=${ANALYSIS_ROOT}"
