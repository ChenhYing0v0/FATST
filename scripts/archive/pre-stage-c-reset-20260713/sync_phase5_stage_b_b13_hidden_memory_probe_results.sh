#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_future_unit_hidden_probe}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_stage_b_b13_future_unit_hidden_composition_$(date '+%Y%m%d')}"

echo "sync_phase5_stage_b_b13_hidden_memory_probe_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "analysis_root=${ANALYSIS_ROOT}"

mkdir -p "${ANALYSIS_ROOT}"
ssh "${REMOTE_HOST}" "test -s '${REMOTE_OUTPUT_ROOT}/b13_future_unit_composition_report.md'"
rsync -av "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" "${ANALYSIS_ROOT}/"

echo "sync_phase5_stage_b_b13_hidden_memory_probe_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
