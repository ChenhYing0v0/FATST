#!/usr/bin/env bash
set -euo pipefail

PHASE_A_ROOT="${PHASE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
PHASE_B_ROOT="${PHASE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
PHASE_C_ROOT="${PHASE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d1_pmfo_pir_offline_v2}"
CONTRACT="${CONTRACT:-configs/stage_c_mechanism_control_natural_dataset_profiles.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
TRAIN_BATCHES="${TRAIN_BATCHES:-8}"
VAL_BATCHES="${VAL_BATCHES:-4}"
GRADIENT_BATCHES="${GRADIENT_BATCHES:-2}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-0.01}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
DATASETS=(Weather ETTm1 ETTh2)

if [[ "${#GPU_IDS[@]}" -lt "${#DATASETS[@]}" ]]; then
  echo "need at least ${#DATASETS[@]} GPU ids" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  test -s "${CONTRACT}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_d1_offline_diagnostic.py --synthetic-smoke
  echo "stage_c_d1_dry_run=pass datasets=${#DATASETS[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "stage_c_d1_start=$(date -Is) commit=$(git rev-parse HEAD)"
pids=()
for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[${index}]}"
  gpu="${GPU_IDS[${index}]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_d1_offline_diagnostic.py \
      --phase-a-root "${PHASE_A_ROOT}" \
      --phase-b-root "${PHASE_B_ROOT}" \
      --phase-c-root "${PHASE_C_ROOT}" \
      --contract "${CONTRACT}" \
      --output-dir "${OUTPUT_ROOT}" \
      --dataset "${dataset}" \
      --device cuda \
      --train-batches "${TRAIN_BATCHES}" \
      --val-batches "${VAL_BATCHES}" \
      --gradient-batches "${GRADIENT_BATCHES}" \
      --ridge-lambda "${RIDGE_LAMBDA}" \
      >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1 &
  pids+=("$!")
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu} pid=$!"
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_d1_offline_diagnostic.py \
    --input-root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}"
echo "stage_c_d1_done=$(date -Is)"
