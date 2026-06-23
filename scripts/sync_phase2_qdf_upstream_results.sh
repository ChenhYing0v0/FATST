#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase2_qdf_upstream_gate_20260623}"
LOCAL_CONDA_ENV="${LOCAL_CONDA_ENV:-r2026-fsa}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

echo "sync_phase2_qdf_upstream_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "local_conda_env=${LOCAL_CONDA_ENV}"

mkdir -p "${ANALYSIS_ROOT}/raw"

echo "checking_remote_artifacts=1"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}'"

echo "syncing_qdf_upstream_artifacts=1"
rsync -av \
  --exclude 'checkpoint.pth' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/" \
  "${ANALYSIS_ROOT}/raw/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "skip_analysis=1"
  exit 0
fi

echo "running_analysis=1"
conda run -n "${LOCAL_CONDA_ENV}" python scripts/analyze_phase2_qdf_upstream_gate.py \
  --analysis-root "${ANALYSIS_ROOT}"

echo "decision_report=${ANALYSIS_ROOT}/phase2_qdf_upstream_decision_report.md"
echo "sync_phase2_qdf_upstream_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
