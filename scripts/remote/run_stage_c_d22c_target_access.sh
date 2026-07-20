#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d22c_target_access_v1_1}"
CONFIG="${CONFIG:-configs/stage_c_d22c_target_access_diagnostic.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 3 ]]; then
  echo "D22-C requires three GPU ids" >&2
  exit 2
fi
test -s "${CONFIG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_d22c_target_access.py --synthetic-smoke \
    --output-dir /tmp/fatst_d22c_remote_smoke
  echo "stage_c_d22c_dry_run=pass"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
{
  echo "stage_c_d22c_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config=${CONFIG}"
  echo "role=diagnostic_only_raw_history_primary"
  echo "splits=train_validation_official_test"
  echo "validation_role=checkpoint_selection_only"
  echo "test_role=one_shot_test_informed_complete_problem_gate"
  echo "method_training=false"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"

run_dataset() {
  local dataset="$1" gpu="$2"
  if [[ -s "${OUTPUT_ROOT}/${dataset}/metadata.json" ]]; then
    echo "skip_existing dataset=${dataset}"
    return
  fi
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python scripts/run_stage_c_d22c_target_access.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" \
    --config "${CONFIG}" --output-dir "${OUTPUT_ROOT}" --device cuda \
    >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1
  echo "worker_done=$(date -Is) dataset=${dataset} gpu=${gpu}"
}

worker() {
  local gpu="$1"
  shift
  local dataset
  for dataset in "$@"; do
    run_dataset "${dataset}" "${gpu}"
  done
}

worker "${GPU_IDS[0]}" Weather ETTh1 & pid0="$!"
worker "${GPU_IDS[1]}" ETTm1 ETTh2 & pid1="$!"
worker "${GPU_IDS[2]}" ETTm2 & pid2="$!"
status=0
for pid in "${pid0}" "${pid1}" "${pid2}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "stage_c_d22c_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_d22c_target_access.py \
  --input-root "${OUTPUT_ROOT}" --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/_analysis"
echo "stage_c_d22c_finished=$(date -Is)"
