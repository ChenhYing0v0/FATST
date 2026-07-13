#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase0_patch_seed_variance}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_ID="${GPU_ID:-1}"
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

datasets=("ETTh2" "ETTm1" "Weather")
horizons=("96" "720")
seeds=("2021" "2022" "2023")

echo "phase0_patch_seed_variance_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_id=${GPU_ID}"
echo "epochs=${EPOCHS}"

for dataset in "${datasets[@]}"; do
  for horizon in "${horizons[@]}"; do
    for seed in "${seeds[@]}"; do
      run_dir="${OUTPUT_ROOT}/PatchEncoderFixedHead/${dataset}/h${horizon}/seed${seed}"
      run_log="${LOG_ROOT}/PatchEncoderFixedHead_${dataset}_h${horizon}_seed${seed}.log"
      if [[ -s "${run_dir}/metrics.json" && -s "${run_dir}/checkpoint.pt" ]]; then
        echo "skip_existing model=PatchEncoderFixedHead dataset=${dataset} horizon=${horizon} seed=${seed}"
        continue
      fi

      echo "run_start=$(date -Is) model=PatchEncoderFixedHead dataset=${dataset} horizon=${horizon} seed=${seed}"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits

      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python baselines/patch_encoder_fixed_head/train.py \
        --dataset-root "${DATASET_ROOT}" \
        --dataset "${dataset}" \
        --pred-len "${horizon}" \
        --epochs "${EPOCHS}" \
        --seed "${seed}" \
        --run-name PatchEncoderFixedHead \
        --output-root "${OUTPUT_ROOT}" \
        --device cuda 2>&1 | tee "${run_log}"

      echo "run_done=$(date -Is) model=PatchEncoderFixedHead dataset=${dataset} horizon=${horizon} seed=${seed}"
    done
  done
done

echo "phase0_patch_seed_variance_done=$(date -Is)"
