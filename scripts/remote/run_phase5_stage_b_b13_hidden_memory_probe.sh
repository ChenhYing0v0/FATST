#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_future_unit_hidden_probe}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_probe_inputs/a6_clean}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_ID="${GPU_ID:-0}"
STATE_DIM="${STATE_DIM:-64}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-4096}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-1024}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-1024}"
UNIT_SIZES="${UNIT_SIZES:-180 240}"
SEEDS="${SEEDS:-2021 2022 2023}"

mkdir -p "${OUTPUT_ROOT}"

echo "phase5_stage_b_b13_hidden_memory_probe_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "dataset_root=${DATASET_ROOT}"
echo "gpu_id=${GPU_ID}"
echo "memory_source=hidden"
echo "unit_sizes=${UNIT_SIZES}"
echo "seeds=${SEEDS}"
echo "rows=${MAX_TRAIN_ROWS}/${MAX_VAL_ROWS}/${MAX_TEST_ROWS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

read -r -a UNIT_SIZE_ARGS <<< "${UNIT_SIZES}"
read -r -a SEED_ARGS <<< "${SEEDS}"

command=(
  python scripts/analyze_phase5_stage_b_b13_future_unit_composition_probe.py
  --analysis-root "${OUTPUT_ROOT}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --dataset-root "${DATASET_ROOT}"
  --memory-source hidden
  --unit-sizes "${UNIT_SIZE_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
  --state-dim "${STATE_DIM}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --extract-batch-size "${EXTRACT_BATCH_SIZE}"
  --max-train-rows "${MAX_TRAIN_ROWS}"
  --max-val-rows "${MAX_VAL_ROWS}"
  --max-test-rows "${MAX_TEST_ROWS}"
  --device cuda
)

if [[ -n "${PYTHON_BIN}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${command[@]:1}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" "${command[@]}"
fi

echo "phase5_stage_b_b13_hidden_memory_probe_done=$(date -Is)"
find "${OUTPUT_ROOT}" -maxdepth 1 -type f | sort
