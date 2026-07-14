#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d7_descriptor_sufficiency}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc1_d7_descriptor_sufficiency_20260714}"
RAW_ROOT="${ANALYSIS_ROOT}/raw"
D7_CONFIG="${D7_CONFIG:-configs/stage_c_sc1_d7_descriptor_sufficiency.json}"

mkdir -p "${RAW_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${RAW_ROOT}/"
python scripts/analyze_stage_c_sc1_d7_descriptor_sufficiency.py \
  --input-root "${RAW_ROOT}" --d7-config "${D7_CONFIG}" \
  --output-dir "${ANALYSIS_ROOT}"
