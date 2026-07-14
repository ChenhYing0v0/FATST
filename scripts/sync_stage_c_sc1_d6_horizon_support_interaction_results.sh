#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d6_horizon_support_interaction}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/stage_c_sc1_d6_horizon_support_interaction_20260714}"
RAW_ROOT="${ANALYSIS_ROOT}/raw"
D6_CONFIG="${D6_CONFIG:-configs/stage_c_sc1_d6_horizon_support_interaction.json}"

mkdir -p "${RAW_ROOT}"
rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${RAW_ROOT}/"
python scripts/analyze_stage_c_sc1_d6_horizon_support_interaction.py \
  --input-root "${RAW_ROOT}" --d6-config "${D6_CONFIG}" \
  --output-dir "${ANALYSIS_ROOT}"
