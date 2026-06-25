#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_gradient_conflict_diagnostic}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_ID="${GPU_ID:-1}"
DATASETS="${DATASETS:-ETTh2,Weather}"
WARMUP_STEPS="${WARMUP_STEPS:-40}"
DIAGNOSTIC_BATCHES="${DIAGNOSTIC_BATCHES:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-2021}"

mkdir -p "${OUTPUT_ROOT}"

if [[ -f "/home/anaconda3/etc/profile.d/conda.sh" ]]; then
  # Non-interactive SSH shells on 529_Lab-3090 do not load the zsh conda hook.
  # shellcheck source=/dev/null
  . "/home/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "/home/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/home/anaconda3/bin/conda"
  elif [[ -x "/data/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/data/anaconda3/bin/conda"
  else
    echo "Unable to locate conda. Set CONDA_BIN=/path/to/conda." >&2
    exit 1
  fi
fi

echo "phase4_gradient_conflict_diagnostic_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_id=${GPU_ID}"
echo "datasets=${DATASETS}"
echo "warmup_steps=${WARMUP_STEPS}"
echo "diagnostic_batches=${DIAGNOSTIC_BATCHES}"
echo "batch_size=${BATCH_SIZE}"
echo "seed=${SEED}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_phase4_gradient_conflict_diagnostic.py \
  --dataset-root "${DATASET_ROOT}" \
  --datasets "${DATASETS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --diagnostic-batches "${DIAGNOSTIC_BATCHES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --output-root "${OUTPUT_ROOT}"

echo "phase4_gradient_conflict_diagnostic_done=$(date -Is)"
