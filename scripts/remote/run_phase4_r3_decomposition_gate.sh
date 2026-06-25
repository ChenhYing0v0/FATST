#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_r3_decomposition_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_r3_decomposition_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1 2}"
DATASETS="${DATASETS:-ETTh2 Weather}"
SEED="${SEED:-2021}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
SUPERVISION_PRED_LEN="${SUPERVISION_PRED_LEN:-720}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-full_time_mse horizon_mixed single_720_prefix_risk r3_prefix_risk}"

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
IFS=" " read -r -a strategies <<< "${SUPERVISION_STRATEGIES}"
IFS="," read -r -a target_horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${target_horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

run_name_for_strategy() {
  case "$1" in
    full_time_mse) echo "PatchEncoderFullTimeMSE720" ;;
    horizon_mixed) echo "PatchEncoderHorizonMixedUniform" ;;
    single_720_prefix_risk) echo "PatchEncoderSingle720PrefixRisk" ;;
    r3_prefix_risk) echo "PatchEncoderR3PrefixRisk" ;;
    *) echo "Unknown strategy: $1" >&2; exit 1 ;;
  esac
}

step_weighting_for_strategy() {
  case "$1" in
    full_time_mse|horizon_mixed) echo "uniform" ;;
    single_720_prefix_risk|r3_prefix_risk) echo "prefix_risk" ;;
    *) echo "Unknown strategy: $1" >&2; exit 1 ;;
  esac
}

echo "phase4_r3_decomposition_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS}"
echo "datasets=${DATASETS}"
echo "target_horizons=${TARGET_HORIZONS}"
echo "supervision_strategies=${SUPERVISION_STRATEGIES}"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "learning_rate=${LEARNING_RATE}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_train() {
  local dataset="$1"
  local gpu_id="$2"
  local strategy="$3"
  local run_name
  local step_weighting
  run_name="$(run_name_for_strategy "${strategy}")"
  step_weighting="$(step_weighting_for_strategy "${strategy}")"
  local run_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${run_name}_${dataset}_seed${SEED}.log"

  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing run_name=${run_name} strategy=${strategy} dataset=${dataset}"
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
    --step-loss-weighting "${step_weighting}" \
    --step-loss-alpha 0.5 \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --run-name "${run_name}" \
    --model-variant target_set \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"
for dataset in "${datasets[@]}"; do
  for strategy in "${strategies[@]}"; do
    gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
    run_train "${dataset}" "${gpu_id}" "${strategy}" &
    job_index=$((job_index + 1))
    if (( job_index % max_jobs == 0 )); then
      wait
    fi
  done
done

wait
echo "phase4_r3_decomposition_gate_done=$(date -Is)"
