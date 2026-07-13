#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase1_step_specific_state}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase1_step_specific_state_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"
KEEP_HEAVY_ARTIFACTS="${KEEP_HEAVY_ARTIFACTS:-0}"

mkdir -p "${LOG_ROOT}"

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

models=(
  "PatchEncoderFixedHead:baselines/patch_encoder_fixed_head/train.py:--run-name PatchEncoderFixedHead"
  "PatchEncoderFixedHeadAdapter:baselines/patch_encoder_fixed_head_adapter/train.py:--run-name PatchEncoderFixedHeadAdapter"
  "PatchEncoderStepSpecificStateAdapter:baselines/patch_encoder_step_specific_state_adapter/train.py:--run-name PatchEncoderStepSpecificStateAdapter"
)
datasets=("ETTh2" "ETTm1" "Weather")
horizons=("96" "192" "336" "720")
read -r -a gpu_ids <<< "${GPU_IDS}"

echo "phase1_step_specific_state_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "keep_heavy_artifacts=${KEEP_HEAVY_ARTIFACTS}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_one() {
  local model_name="$1"
  local train_script="$2"
  local extra_args="$3"
  local dataset="$4"
  local horizon="$5"
  local gpu_id="$6"
  local run_dir="${OUTPUT_ROOT}/${model_name}/${dataset}/h${horizon}/seed${SEED}"
  local run_log="${LOG_ROOT}/${model_name}_${dataset}_h${horizon}_seed${SEED}.log"

  if [[ -s "${run_dir}/metrics.json" ]]; then
    echo "skip_existing model=${model_name} dataset=${dataset} horizon=${horizon}"
    return 0
  fi

  echo "run_start=$(date -Is) model=${model_name} dataset=${dataset} horizon=${horizon} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${train_script}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset "${dataset}" \
    --pred-len "${horizon}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    ${extra_args} 2>&1 | tee "${run_log}"
  if [[ "${KEEP_HEAVY_ARTIFACTS}" != "1" ]]; then
    find "${run_dir}" -maxdepth 1 -type f \( -name "checkpoint.pt" -o -name "predictions_test.npz" \) -delete
  fi
  echo "run_done=$(date -Is) model=${model_name} dataset=${dataset} horizon=${horizon} gpu=${gpu_id}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"

for model_entry in "${models[@]}"; do
  IFS=":" read -r model_name train_script extra_args <<< "${model_entry}"
  for dataset in "${datasets[@]}"; do
    for horizon in "${horizons[@]}"; do
      gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
      run_one "${model_name}" "${train_script}" "${extra_args}" "${dataset}" "${horizon}" "${gpu_id}" &
      job_index=$((job_index + 1))
      if (( job_index % max_jobs == 0 )); then
        wait
      fi
    done
  done
done

wait
echo "phase1_step_specific_state_gate_done=$(date -Is)"
