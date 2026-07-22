#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/stage_c_iscf_psa_d1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_psa_d1}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
PYTHON_BIN="${PYTHON_BIN:-/home/yingch/.conda/envs/moe/bin/python}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

test -s "${CONFIG}"
test "${#GPU_IDS[@]}" -ge 1

AUTHORIZED="$(${PYTHON_BIN} -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["validation_diagnostic_replay_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$(${PYTHON_BIN} -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"
if [[ "${AUTHORIZED}" != "true" || "${TEST_AUTHORIZED}" != "false" ]]; then
  echo "PSA-D1 validation diagnostic authorization boundary failed" >&2
  exit 3
fi

mapfile -t DATASETS < <("${PYTHON_BIN}" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["datasets"]))' "${CONFIG}")

run_dir() {
  echo "${OUTPUT_ROOT}/iscf_equal_contemporaneous/$1/h720_full/seed2021"
}

is_complete() {
  local directory
  directory="$(run_dir "$1")"
  [[ -s "${directory}/pcsd_validation_diagnostics.npz" \
    && -s "${directory}/trained_invariants.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  complete=0
  for dataset in "${DATASETS[@]}"; do
    if is_complete "${dataset}"; then complete=$((complete + 1)); fi
  done
  echo "psa_d1_diagnostics_status=$(date -Is) validation=${complete}/${#DATASETS[@]}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_diagnostic_logs"
mkdir -p "${LOG_ROOT}"

run_one() {
  local index="$1" gpu="$2" dataset directory log before after
  dataset="${DATASETS[${index}]}"
  directory="$(run_dir "${dataset}")"
  log="${LOG_ROOT}/${dataset}_seed2021.log"
  if is_complete "${dataset}"; then
    echo "skip_existing dataset=${dataset}"
    return 0
  fi
  test -s "${directory}/checkpoint.pt"
  test -s "${directory}/metrics_by_target_horizon.csv"
  before="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${directory}/checkpoint.pt")"
  echo "diagnostic_start=$(date -Is) job=$((index + 1))/${#DATASETS[@]} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${directory}" --design "${CONFIG}" \
      --test-audit-config "${CONFIG}" --evaluation-split val \
      --probe-rows 256 --device cuda >"${log}" 2>&1
  after="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${directory}/checkpoint.pt")"
  if [[ "${before}" != "${after}" ]]; then
    echo "checkpoint_mutated dataset=${dataset}" >&2
    return 1
  fi
  echo "diagnostic_done=$(date -Is) job=$((index + 1))/${#DATASETS[@]} dataset=${dataset} gpu=${gpu} checkpoint_sha256=${after}"
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#DATASETS[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" == "0" ]]
echo "psa_d1_validation_diagnostics_done=$(date -Is)"
