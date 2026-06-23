#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_error_process_decoder}"
RUN_NAME="${RUN_NAME:-PatchEncoderErrorProcessDecoder}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase2_error_process_decoder_gate_20260623}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-phase2_error_process_decoder}"
LOCAL_CONDA_ENV="${LOCAL_CONDA_ENV:-r2026-fsa}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

echo "sync_phase2_error_process_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_output_root=${REMOTE_OUTPUT_ROOT}"
echo "run_name=${RUN_NAME}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "local_conda_env=${LOCAL_CONDA_ENV}"

mkdir -p "${ANALYSIS_ROOT}/raw/${RUN_NAME}" "${ANALYSIS_ROOT}/raw/_logs"

echo "checking_remote_artifacts=1"
ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_ROOT}/${RUN_NAME}'"

echo "syncing_model_artifacts=1"
rsync -av \
  --exclude 'checkpoint.pt' \
  --exclude 'predictions_test.npz' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/${RUN_NAME}/" \
  "${ANALYSIS_ROOT}/raw/${RUN_NAME}/"

echo "syncing_logs=1"
rsync -av \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_ROOT}/_logs/" \
  "${ANALYSIS_ROOT}/raw/_logs/"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "skip_analysis=1"
  exit 0
fi

echo "running_analysis=1"
conda run -n "${LOCAL_CONDA_ENV}" python scripts/analyze_phase2_error_process_decoder_gate.py \
  --analysis-root "${ANALYSIS_ROOT}" \
  --model-name "${RUN_NAME}" \
  --output-prefix "${OUTPUT_PREFIX}"

echo "decision_report=${ANALYSIS_ROOT}/${OUTPUT_PREFIX}_decision_report.md"
echo "sync_phase2_error_process_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
