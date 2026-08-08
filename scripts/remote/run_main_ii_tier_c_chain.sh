#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU="${GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

[[ -n "${EXPECTED_COMMIT}" ]] || { echo "EXPECTED_COMMIT is required" >&2; exit 2; }
actual_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "code commit mismatch: expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
  exit 3
}

TRAINING_PID_FILE="${OUTPUT_ROOT}/formal_training.pid"
[[ -s "${TRAINING_PID_FILE}" ]] || { echo "missing training PID file" >&2; exit 4; }
training_pid="$(cat "${TRAINING_PID_FILE}")"
echo "tier_c_chain_start=$(date -Is) code_commit=${actual_commit} training_pid=${training_pid}"

while kill -0 "${training_pid}" 2>/dev/null; do
  complete="$(find "${OUTPUT_ROOT}/training" -mindepth 2 -maxdepth 2 -name DONE -type f 2>/dev/null | wc -l | tr -d ' ')"
  echo "tier_b_wait=$(date -Is) complete=${complete}/21"
  sleep "${POLL_SECONDS}"
done

complete="$(find "${OUTPUT_ROOT}/training" -mindepth 2 -maxdepth 2 -name DONE -type f 2>/dev/null | wc -l | tr -d ' ')"
[[ "${complete}" == "21" ]] || {
  echo "Tier B terminated without complete matrix: ${complete}/21" >&2
  exit 5
}
echo "tier_b_gate=pass time=$(date -Is) checkpoints=21"

{
  echo "tier_c_launch=$(date -Is)"
  echo "repo_commit=${actual_commit}"
  echo "gpu=${GPU}"
  quota -s 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
} >"${OUTPUT_ROOT}/tier_c_launch_record.txt"

REPO_ROOT="${REPO_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" GPU_IDS="${GPU}" \
  CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" \
  bash "${REPO_ROOT}/scripts/remote/run_main_ii_h720_training.sh" formal-test-new
echo "tier_c_new_checkpoints=pass time=$(date -Is) evaluations=21 raw_rows=84"

REPO_ROOT="${REPO_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" GPU="${GPU}" \
  CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" \
  bash "${REPO_ROOT}/scripts/remote/run_main_ii_reused_formal_tests.sh" run
echo "tier_c_reused_checkpoints=pass time=$(date -Is) new_evaluations=42 iscf_reuse=7 raw_rows=196"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/analyze_main_ii_h720_prefix_results.py" \
    --results-root "${OUTPUT_ROOT}" \
    --main-i-table "${REPO_ROOT}/analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_final_amd_simpletm_20260808/table_data_long.csv" \
    --output-dir "${OUTPUT_ROOT}/aggregate_audit"
echo "main_ii_tier_c_chain=pass time=$(date -Is) checkpoints=70 raw_rows=280 aggregate_cells=224"
