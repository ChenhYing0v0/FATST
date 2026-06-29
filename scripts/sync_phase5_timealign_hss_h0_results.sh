#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h0_prefix_gate}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_timealign_hss_h0_prefix_gate_$(date '+%Y%m%d')}"
SEED="${SEED:-2021}"
FIXED_REFERENCE_CSV="${FIXED_REFERENCE_CSV:-analysis/phase5_timealign_official_gate_20260626/phase5_timealign_official_unified_gap.csv}"

echo "sync_phase5_timealign_hss_h0_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
mkdir -p "${ANALYSIS_ROOT}/raw"

rsync -av \
  --include='*/' \
  --include='metrics_by_target_horizon.csv' \
  --include='metrics_by_segment.csv' \
  --include='training_log.csv' \
  --include='effective_config.json' \
  --include='environment.json' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

python scripts/analyze_phase5_timealign_hss_h0_prefix_gate.py \
  --root "${ANALYSIS_ROOT}/raw" \
  --output-dir "${ANALYSIS_ROOT}" \
  --checkpoint-policy "${CHECKPOINT_POLICY}" \
  --seed "${SEED}" \
  --fixed-reference-csv "${FIXED_REFERENCE_CSV}"

echo "sync_phase5_timealign_hss_h0_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
