#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d4_structured_basis}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc1_d4_structured_basis_20260714}"
RAW_ROOT="${ANALYSIS_ROOT}/raw"
D4_CONFIG="${D4_CONFIG:-configs/stage_c_sc1_d4_structured_basis.json}"

mkdir -p "${RAW_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${RAW_ROOT}/"
python scripts/analyze_stage_c_sc1_d4_structured_basis.py \
  --input-root "${RAW_ROOT}" --d4-config "${D4_CONFIG}" \
  --output-dir "${ANALYSIS_ROOT}"
