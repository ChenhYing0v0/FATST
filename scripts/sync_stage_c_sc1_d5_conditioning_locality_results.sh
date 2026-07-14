#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d5_conditioning_locality}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc1_d5_conditioning_locality_20260714}"
RAW_ROOT="${ANALYSIS_ROOT}/raw"
D5_CONFIG="${D5_CONFIG:-configs/stage_c_sc1_d5_conditioning_locality_frontier.json}"

mkdir -p "${RAW_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${RAW_ROOT}/"
python scripts/analyze_stage_c_sc1_d5_conditioning_locality.py \
  --input-root "${RAW_ROOT}" --d5-config "${D5_CONFIG}" \
  --output-dir "${ANALYSIS_ROOT}"
