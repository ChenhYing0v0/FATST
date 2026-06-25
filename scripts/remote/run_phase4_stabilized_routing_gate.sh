#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_stabilized_routing_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_stabilized_routing_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1 2}"
DATASETS="${DATASETS:-ETTh2 Weather}"
SEED="${SEED:-2021}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
SUPERVISION_PRED_LEN="${SUPERVISION_PRED_LEN:-720}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
FINETUNE_PATIENCE="${FINETUNE_PATIENCE:-5}"
FINETUNE_LR="${FINETUNE_LR:-0.001}"
SUPERVISION_CONDITION_TOP_RATIO="${SUPERVISION_CONDITION_TOP_RATIO:-0.25}"
SUPERVISION_AUX_WEIGHT="${SUPERVISION_AUX_WEIGHT:-0.1}"
SUPERVISION_RESIDUAL_PERIODS="${SUPERVISION_RESIDUAL_PERIODS:-24,48,96,168}"

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

IFS=" " read -r -a datasets <<< "${DATASETS}"
IFS=" " read -r -a gpu_ids <<< "${GPU_IDS}"
IFS="," read -r -a target_horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${target_horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

echo "phase4_stabilized_routing_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS}"
echo "datasets=${DATASETS}"
echo "target_horizons=${TARGET_HORIZONS}"
echo "pretrain_epochs=${PRETRAIN_EPOCHS}"
echo "finetune_epochs=${FINETUNE_EPOCHS}"
echo "finetune_lr=${FINETUNE_LR}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_train() {
  local dataset="$1"
  local gpu_id="$2"
  local run_name="$3"
  local strategy="$4"
  local epochs="$5"
  local learning_rate="$6"
  local patience="$7"
  shift 7
  local extra_args=("$@")
  local run_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${run_name}_${dataset}_seed${SEED}.log"

  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing run_name=${run_name} dataset=${dataset}"
    return 0
  fi

  echo "run_start=$(date -Is) run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/patch_encoder_target_set_decoder/train.py \
    --dataset-root "${DATASET_ROOT}" \
    --dataset "${dataset}" \
    --target-horizons "${TARGET_HORIZONS}" \
    --supervision-strategy "${strategy}" \
    --supervision-pred-len "${SUPERVISION_PRED_LEN}" \
    --step-loss-weighting uniform \
    --step-loss-alpha 0.5 \
    --epochs "${epochs}" \
    --patience "${patience}" \
    --learning-rate "${learning_rate}" \
    --seed "${SEED}" \
    --run-name "${run_name}" \
    --model-variant target_set \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${extra_args[@]}" 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id}"
}

run_dataset() {
  local dataset="$1"
  local gpu_id="$2"
  local pretrain_name="PatchEncoderFullTimeMSE720Pretrain"
  local finetune_name="PatchEncoderStabilizedDynamicResidualRouting"
  local pretrain_ckpt="${OUTPUT_ROOT}/${pretrain_name}/${dataset}/${horizon_label}/seed${SEED}/checkpoint.pt"

  run_train "${dataset}" "${gpu_id}" "${pretrain_name}" "full_time_mse" "${PRETRAIN_EPOCHS}" "0.0001" "10"

  if [[ ! -s "${pretrain_ckpt}" ]]; then
    echo "Missing pretrain checkpoint: ${pretrain_ckpt}" >&2
    exit 1
  fi

  run_train \
    "${dataset}" \
    "${gpu_id}" \
    "${finetune_name}" \
    "dynamic_residual_stability_routing" \
    "${FINETUNE_EPOCHS}" \
    "${FINETUNE_LR}" \
    "${FINETUNE_PATIENCE}" \
    --init-checkpoint "${pretrain_ckpt}" \
    --freeze-non-adapter \
    --supervision-condition-top-ratio "${SUPERVISION_CONDITION_TOP_RATIO}" \
    --supervision-aux-weight "${SUPERVISION_AUX_WEIGHT}" \
    --supervision-residual-periods "${SUPERVISION_RESIDUAL_PERIODS}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"
for dataset in "${datasets[@]}"; do
  gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
  run_dataset "${dataset}" "${gpu_id}" &
  job_index=$((job_index + 1))
  if (( job_index % max_jobs == 0 )); then
    wait
  fi
done

wait
echo "phase4_stabilized_routing_gate_done=$(date -Is)"
