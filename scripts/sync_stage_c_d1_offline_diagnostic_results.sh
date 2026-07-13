#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d1_pmfo_pir_offline}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_d1_pmfo_pir_offline_20260713}"

mkdir -p "${ANALYSIS_ROOT}/raw"
rsync -av --exclude='*.pt' --exclude='*.npz' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${ANALYSIS_ROOT}/raw/"
python scripts/analyze_stage_c_d1_offline_diagnostic.py \
  --input-root "${ANALYSIS_ROOT}/raw" \
  --output-dir "${ANALYSIS_ROOT}"
