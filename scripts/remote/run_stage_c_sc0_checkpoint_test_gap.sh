#!/usr/bin/env bash
set -euo pipefail

SC0_ROOT="${SC0_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_checkpoint_test_gap}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
DATASETS=(Weather ETTm1 ETTh2)

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "stage_c_sc0_checkpoint_test_start=$(date -Is)"
echo "git_commit=$(git rev-parse HEAD)"
echo "sc0_root=${SC0_ROOT} output_root=${OUTPUT_ROOT}"

pids=()
for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  gpu="${GPU_IDS[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_sc0_checkpoint_test_gap.py \
      --raw-root "${SC0_ROOT}" --output-dir "${OUTPUT_ROOT}" \
      --dataset "${dataset}" --device cuda \
      >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1 &
  pids+=("$!")
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu} pid=$!"
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc0_checkpoint_test_gap.py \
    --input-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}"
echo "stage_c_sc0_checkpoint_test_done=$(date -Is)"
