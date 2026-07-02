#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_timealign_hss_a4s_validation_prefix_signal_export_$(date '+%Y%m%d')}"
SEED="${SEED:-2021}"

REMOTE_PATH="${REMOTE_ROOT}/phase5_timealign_hss_a4s_validation_prefix_signal_export"

echo "sync_phase5_timealign_hss_a4s_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_path=${REMOTE_PATH}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"

mkdir -p "${ANALYSIS_ROOT}/raw"
rsync -av \
  --include='*/' \
  --include='validation_prefix_diagnostics.csv' \
  --include='validation_prefix_diagnostics_config.json' \
  --include='*.log' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${ANALYSIS_ROOT}/raw/"

python scripts/analyze_phase5_timealign_hss_a4s_validation_prefix_signals.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --a4-root analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701 \
  --output-dir "${ANALYSIS_ROOT}" \
  --checkpoint-policy "${CHECKPOINT_POLICY}" \
  --seed "${SEED}"

echo "sync_phase5_timealign_hss_a4s_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
