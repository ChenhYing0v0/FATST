#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d3_crossed}"
D2_ANALYSIS_ROOT="${D2_ANALYSIS_ROOT:-analysis/stage_c_sc1_d2_formal5_20260714/raw}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc1_d3_crossed_20260714}"
RAW_ROOT="${ANALYSIS_ROOT}/raw"
D3_CONFIG="${D3_CONFIG:-configs/stage_c_sc1_d3_crossed_basis_group.json}"

mkdir -p "${RAW_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${RAW_ROOT}/"
python scripts/analyze_stage_c_sc1_d3_crossed_diagnostic.py \
  --d2-root "${D2_ANALYSIS_ROOT}" --d3-root "${RAW_ROOT}" \
  --d3-config "${D3_CONFIG}" --output-dir "${ANALYSIS_ROOT}"
