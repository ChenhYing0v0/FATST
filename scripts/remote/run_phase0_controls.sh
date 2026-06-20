#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase0_controls}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "${LOG_ROOT}"

if [[ -f "/home/anaconda3/etc/profile.d/conda.sh" ]]; then
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

horizons=("96" "192" "336" "720")

echo "phase0_controls_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_id=${GPU_ID}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"

for horizon in "${horizons[@]}"; do
  run_dir="${OUTPUT_ROOT}/DLinearOfficialInit/ETTh2/h${horizon}/seed${SEED}"
  run_log="${LOG_ROOT}/DLinearOfficialInit_ETTh2_h${horizon}_seed${SEED}.log"
  if [[ -s "${run_dir}/metrics.json" && -s "${run_dir}/checkpoint.pt" ]]; then
    echo "skip_existing model=DLinearOfficialInit dataset=ETTh2 horizon=${horizon}"
  else
    echo "run_start=$(date -Is) model=DLinearOfficialInit dataset=ETTh2 horizon=${horizon}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python baselines/dlinear/train.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset ETTh2 \
      --pred-len "${horizon}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --init-mode pytorch_default \
      --run-name DLinearOfficialInit \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda 2>&1 | tee "${run_log}"
    echo "run_done=$(date -Is) model=DLinearOfficialInit dataset=ETTh2 horizon=${horizon}"
  fi
done

for horizon in "${horizons[@]}"; do
  run_dir="${OUTPUT_ROOT}/PatchEncoderFixedHeadOfficialETT/ETTh2/h${horizon}/seed${SEED}"
  run_log="${LOG_ROOT}/PatchEncoderFixedHeadOfficialETT_ETTh2_h${horizon}_seed${SEED}.log"
  if [[ -s "${run_dir}/metrics.json" && -s "${run_dir}/checkpoint.pt" ]]; then
    echo "skip_existing model=PatchEncoderFixedHeadOfficialETT dataset=ETTh2 horizon=${horizon}"
  else
    echo "run_start=$(date -Is) model=PatchEncoderFixedHeadOfficialETT dataset=ETTh2 horizon=${horizon}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python baselines/patch_encoder_fixed_head/train.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset ETTh2 \
      --pred-len "${horizon}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --patch-len 16 \
      --stride 8 \
      --d-model 16 \
      --n-heads 4 \
      --encoder-layers 3 \
      --d-ff 128 \
      --dropout 0.3 \
      --head-dropout 0.0 \
      --run-name PatchEncoderFixedHeadOfficialETT \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda 2>&1 | tee "${run_log}"
    echo "run_done=$(date -Is) model=PatchEncoderFixedHeadOfficialETT dataset=ETTh2 horizon=${horizon}"
  fi
done

echo "phase0_controls_done=$(date -Is)"
