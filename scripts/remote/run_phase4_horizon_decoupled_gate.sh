#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_horizon_decoupled}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_horizon_decoupled_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
SUPERVISION_PRED_LEN="${SUPERVISION_PRED_LEN:-720}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-full_time_mse r3_prefix_risk random_future_mask interval_supervision component_basis_top component_basis_balanced curriculum_units}"
DATASETS="${DATASETS:-ETTh2 ETTm1 Weather}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
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

run_name_for_strategy() {
  case "$1" in
    full_time_mse) echo "PatchEncoderFullTimeMSE720" ;;
    r3_prefix_risk) echo "PatchEncoderR3PrefixRisk" ;;
    random_future_mask) echo "PatchEncoderRandomFutureMask" ;;
    interval_supervision) echo "PatchEncoderIntervalSupervision" ;;
    conditioned_future_unit_scheduling) echo "PatchEncoderConditionedFutureUnitScheduling" ;;
    predictability_downweight) echo "PatchEncoderPredictabilityDownweight" ;;
    late_conflict_adapter_routing) echo "PatchEncoderLateConflictAdapterRouting" ;;
    dynamic_residual_stability_routing) echo "PatchEncoderDynamicResidualStabilityRouting" ;;
    component_basis_top) echo "PatchEncoderComponentTop" ;;
    component_basis_balanced) echo "PatchEncoderComponentBalanced" ;;
    curriculum_units) echo "PatchEncoderCurriculumUnits" ;;
    *) echo "PatchEncoder_${1}" ;;
  esac
}

step_loss_for_strategy() {
  case "$1" in
    r3_prefix_risk) echo "prefix_risk" ;;
    *) echo "uniform" ;;
  esac
}

read -r -a datasets <<< "${DATASETS}"
read -r -a strategies <<< "${SUPERVISION_STRATEGIES}"
read -r -a gpu_ids <<< "${GPU_IDS}"
read -r -a phase4_extra_args <<< "${PHASE4_EXTRA_ARGS:-}"
IFS="," read -r -a target_horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${target_horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

echo "phase4_horizon_decoupled_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "datasets=${DATASETS}"
echo "target_horizons=${TARGET_HORIZONS}"
echo "supervision_pred_len=${SUPERVISION_PRED_LEN}"
echo "supervision_strategies=${SUPERVISION_STRATEGIES}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_one() {
  local strategy="$1"
  local dataset="$2"
  local gpu_id="$3"
  local run_name
  local step_loss
  run_name="$(run_name_for_strategy "${strategy}")"
  step_loss="$(step_loss_for_strategy "${strategy}")"
  local run_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${run_name}_${dataset}_seed${SEED}.log"
  local extra_args=()

  if [[ -n "${STEPS_PER_EPOCH}" ]]; then
    extra_args+=(--steps-per-epoch "${STEPS_PER_EPOCH}")
  fi
  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    extra_args+=(--max-eval-batches "${MAX_EVAL_BATCHES}")
  fi

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
    --step-loss-weighting "${step_loss}" \
    --step-loss-alpha 0.5 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --run-name "${run_name}" \
    --model-variant target_set \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${phase4_extra_args[@]}" \
    "${extra_args[@]}" 2>&1 | tee "${run_log}"
  if [[ "${KEEP_HEAVY_ARTIFACTS}" != "1" ]]; then
    find "${run_dir}" -type f \( -name "checkpoint.pt" -o -name "predictions_test.npz" \) -delete
  fi
  echo "run_done=$(date -Is) run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"

for strategy in "${strategies[@]}"; do
  for dataset in "${datasets[@]}"; do
    gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
    run_one "${strategy}" "${dataset}" "${gpu_id}" &
    job_index=$((job_index + 1))
    if (( job_index % max_jobs == 0 )); then
      wait
    fi
  done
done

wait
echo "phase4_horizon_decoupled_gate_done=$(date -Is)"
