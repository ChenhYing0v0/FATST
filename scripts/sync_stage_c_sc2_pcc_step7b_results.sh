#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc2_pcc_step7b}"
REFERENCE_ROOT="${REFERENCE_ROOT:-analysis/stage_c_pcsd_cf_step7b_seed2021_20260716/raw}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc2_pcc_step7b_seed2021_20260717}"
SEED="${SEED:-2021}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"
CONDA_BIN="${CONDA_BIN:-/opt/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-r2026-fsa}"

mkdir -p "${ANALYSIS_ROOT}/raw"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_ROOT}'"
rsync -av \
  --exclude 'checkpoint*.pt' \
  --exclude 'predictions*.npz' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "stage_c_sc2_pcc_step7b_sync=complete analysis=skipped"
  exit 0
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc2_pcc_step7b.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${ANALYSIS_ROOT}" \
  --seed "${SEED}"
echo "stage_c_sc2_pcc_step7b_sync=complete analysis=${ANALYSIS_ROOT}"
