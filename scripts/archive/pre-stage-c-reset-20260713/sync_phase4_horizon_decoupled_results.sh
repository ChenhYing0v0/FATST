#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_horizon_decoupled}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase4_horizon_decoupled_gate_$(date '+%Y%m%d')}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-1}"

echo "sync_phase4_horizon_decoupled_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "skip_analysis=${SKIP_ANALYSIS}"

mkdir -p "${ANALYSIS_ROOT}/raw"

echo "checking_remote_artifacts=1"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}'"

echo "syncing_artifacts=1"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'predictions_test.npz' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "skip_analysis=1"
  echo "sync_phase4_horizon_decoupled_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  exit 0
fi

echo "analysis_not_implemented=1"
echo "Write scripts/analyze_phase4_horizon_decoupled_gate.py after the remote artifact set returns." >&2
exit 2
