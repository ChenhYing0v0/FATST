#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_stage_b_c0_ettm1_carrier_protocol_gate_$(date '+%Y%m%d')}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

echo "sync_phase5_stage_b_c0_ettm1_carrier_protocol_gate_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "analysis_root=${ANALYSIS_ROOT}"
mkdir -p "${ANALYSIS_ROOT}/raw"

ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}'"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'checkpoint_last.pt' \
  --exclude 'checkpoint_best_val.pt' \
  --exclude 'predictions_test.npz' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" != "1" ]]; then
  python scripts/analyze_phase5_stage_b_c0_ettm1_carrier_protocol_gate.py \
    --raw-root "${ANALYSIS_ROOT}/raw" \
    --output-dir "${ANALYSIS_ROOT}"
fi

echo "sync_phase5_stage_b_c0_ettm1_carrier_protocol_gate_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
