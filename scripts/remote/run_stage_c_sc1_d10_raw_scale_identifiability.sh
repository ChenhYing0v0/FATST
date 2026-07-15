#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d10_raw_scale_identifiability}"
DESIGN="${DESIGN:-configs/stage_c_sc1_d10_raw_scale_identifiability.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 3 ]]; then
  echo "SC1-D10 requires three GPU ids" >&2
  exit 2
fi
test -s "${DESIGN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_sc1_d10_raw_scale_identifiability.py \
      --synthetic-smoke
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_sc1_d10_raw_scale_identifiability.py \
      --synthetic-smoke
  echo "stage_c_sc1_d10_dry_run=pass"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
{
  echo "stage_c_sc1_d10_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "role=diagnostic_only"
  echo "splits=train_fit_train_holdout_validation"
  echo "test_used=false"
  echo "forecast_training=false"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"

run_dataset() {
  local dataset="$1" gpu="$2"
  if [[ -s "${OUTPUT_ROOT}/${dataset}/matrix_cell_metrics.csv" \
    && -s "${OUTPUT_ROOT}/${dataset}/binary_cell_metrics.csv" \
    && -s "${OUTPUT_ROOT}/${dataset}/metadata.json" ]]; then
    echo "skip_existing dataset=${dataset}"
    return
  fi
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_sc1_d10_raw_scale_identifiability.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" \
      --design "${DESIGN}" --output-dir "${OUTPUT_ROOT}" --device cuda \
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
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" != "0" ]]; then
  echo "stage_c_sc1_d10_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc1_d10_raw_scale_identifiability.py \
    --input-root "${OUTPUT_ROOT}" --design "${DESIGN}" \
    --output-dir "${OUTPUT_ROOT}/_analysis"
echo "stage_c_sc1_d10_finished=$(date -Is)"
