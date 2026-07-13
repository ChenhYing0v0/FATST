#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_official_gate}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_timealign_official_gate_$(date '+%Y%m%d')}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-1}"

echo "sync_phase5_timealign_official_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "skip_analysis=${SKIP_ANALYSIS}"

mkdir -p "${ANALYSIS_ROOT}/raw"

echo "checking_remote_artifacts=1"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}/${CHECKPOINT_POLICY}'"

echo "syncing_artifacts=1"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'predictions_test.npz' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/${CHECKPOINT_POLICY}/" \
  "${ANALYSIS_ROOT}/raw/${CHECKPOINT_POLICY}/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "skip_analysis=1"
  echo "sync_phase5_timealign_official_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  exit 0
fi

python scripts/analyze_phase5_timealign_official_gate.py \
  --raw-root "${ANALYSIS_ROOT}/raw/${CHECKPOINT_POLICY}" \
  --output-dir "${ANALYSIS_ROOT}" \
  --checkpoint-policy "${CHECKPOINT_POLICY}"

echo "sync_phase5_timealign_official_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
