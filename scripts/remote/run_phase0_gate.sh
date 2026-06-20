#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase0}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/phase0_gate}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_ID="${GPU_ID:-1}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "${LOG_ROOT}"

models=(
  "DLinear:baselines/dlinear/train.py"
  "PatchEncoderFixedHead:baselines/patch_encoder_fixed_head/train.py"
  "SegTSFTDenseFixedHead:baselines/segtsft_dense_fixed_head/train.py"
)
datasets=("ETTh2" "ETTm1" "Weather")
horizons=("96" "192" "336" "720")

echo "phase0_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_id=${GPU_ID}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"

for model_entry in "${models[@]}"; do
  model_name="${model_entry%%:*}"
  train_script="${model_entry#*:}"
  for dataset in "${datasets[@]}"; do
    for horizon in "${horizons[@]}"; do
      run_dir="${OUTPUT_ROOT}/${model_name}/${dataset}/h${horizon}/seed${SEED}"
      run_log="${LOG_ROOT}/${model_name}_${dataset}_h${horizon}_seed${SEED}.log"
      if [[ -s "${run_dir}/metrics.json" && -s "${run_dir}/checkpoint.pt" ]]; then
        echo "skip_existing model=${model_name} dataset=${dataset} horizon=${horizon}"
        continue
      fi

      echo "run_start=$(date -Is) model=${model_name} dataset=${dataset} horizon=${horizon}"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits

      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" \
        python "${train_script}" \
        --dataset-root "${DATASET_ROOT}" \
        --dataset "${dataset}" \
        --pred-len "${horizon}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --output-root "${OUTPUT_ROOT}" \
        --device cuda 2>&1 | tee "${run_log}"

      echo "run_done=$(date -Is) model=${model_name} dataset=${dataset} horizon=${horizon}"
    done
  done
done

echo "phase0_gate_done=$(date -Is)"
