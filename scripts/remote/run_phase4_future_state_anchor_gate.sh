#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_future_state_anchor_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_future_state_anchor_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1 2}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
SUPERVISION_PRED_LEN="${SUPERVISION_PRED_LEN:-720}"
DATASETS="${DATASETS:-Weather ETTh2}"
ARMS="${ARMS:-F1-C0 F1-C1 F1-A0 F1-A1 F1-W0}"
JOB_ORDER="${JOB_ORDER:-dataset_major}"
FUTURE_TEACHER_LAYERS="${FUTURE_TEACHER_LAYERS:-1}"
FUTURE_TEACHER_HEADS="${FUTURE_TEACHER_HEADS:-8}"
FUTURE_TEACHER_D_FF="${FUTURE_TEACHER_D_FF:-256}"
FUTURE_ALIGN_WEIGHT="${FUTURE_ALIGN_WEIGHT:-0.01}"
FUTURE_RELATION_WEIGHT="${FUTURE_RELATION_WEIGHT:-0.0}"
FUTURE_RECON_WEIGHT="${FUTURE_RECON_WEIGHT:-0.001}"
FUTURE_RECON_NORMALIZATION="${FUTURE_RECON_NORMALIZATION:-target_energy}"
FUTURE_ALIGN_WEIGHTING="${FUTURE_ALIGN_WEIGHTING:-reconstruction_confidence}"
FUTURE_CONFIDENCE_TEMPERATURE="${FUTURE_CONFIDENCE_TEMPERATURE:-1.0}"
FUTURE_CONFIDENCE_FLOOR="${FUTURE_CONFIDENCE_FLOOR:-0.05}"
FUTURE_RECON_EPS="${FUTURE_RECON_EPS:-1e-6}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
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

run_name_for_arm() {
  case "$1" in
    F1-C0) echo "PatchEncoderFSAF1SinglePrefixBase" ;;
    F1-C1) echo "PatchEncoderFSAF1R3Base" ;;
    F1-A0) echo "PatchEncoderFSAF1SinglePrefixFutureAnchor" ;;
    F1-A1) echo "PatchEncoderFSAF1R3FutureAnchor" ;;
    F1-W0) echo "PatchEncoderFSAF1FullTimeFutureAnchor" ;;
    F2-A0) echo "PatchEncoderFSAF2SinglePrefixSelectiveAnchor" ;;
    F2-A1) echo "PatchEncoderFSAF2R3SelectiveAnchor" ;;
    *) echo "PatchEncoderFSAF1_${1}" ;;
  esac
}

strategy_for_arm() {
  case "$1" in
    F1-C0|F1-A0|F2-A0) echo "single_720_prefix_risk" ;;
    F1-C1|F1-A1|F2-A1) echo "r3_prefix_risk" ;;
    F1-W0) echo "full_time_mse" ;;
    *) echo "unknown" ;;
  esac
}

future_enabled_for_arm() {
  case "$1" in
    F1-A0|F1-A1|F1-W0|F2-A0|F2-A1) echo "1" ;;
    *) echo "0" ;;
  esac
}

step_loss_for_strategy() {
  case "$1" in
    single_720_prefix_risk|r3_prefix_risk) echo "prefix_risk" ;;
    *) echo "uniform" ;;
  esac
}

read -r -a datasets <<< "${DATASETS}"
read -r -a arms <<< "${ARMS}"
read -r -a gpu_ids <<< "${GPU_IDS}"
IFS="," read -r -a target_horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${target_horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

echo "phase4_future_state_anchor_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "learning_rate=${LEARNING_RATE}"
echo "datasets=${DATASETS}"
echo "arms=${ARMS}"
echo "job_order=${JOB_ORDER}"
echo "target_horizons=${TARGET_HORIZONS}"
echo "supervision_pred_len=${SUPERVISION_PRED_LEN}"
echo "future_teacher_layers=${FUTURE_TEACHER_LAYERS}"
echo "future_align_weight=${FUTURE_ALIGN_WEIGHT}"
echo "future_relation_weight=${FUTURE_RELATION_WEIGHT}"
echo "future_recon_weight=${FUTURE_RECON_WEIGHT}"
echo "future_recon_normalization=${FUTURE_RECON_NORMALIZATION}"
echo "future_align_weighting=${FUTURE_ALIGN_WEIGHTING}"
echo "future_confidence_floor=${FUTURE_CONFIDENCE_FLOOR}"
echo "steps_per_epoch=${STEPS_PER_EPOCH:-auto}"
echo "max_eval_batches=${MAX_EVAL_BATCHES:-all}"
echo "keep_heavy_artifacts=${KEEP_HEAVY_ARTIFACTS}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_one() {
  local arm="$1"
  local dataset="$2"
  local gpu_id="$3"
  local run_name
  local strategy
  local step_loss
  local future_enabled
  run_name="$(run_name_for_arm "${arm}")"
  strategy="$(strategy_for_arm "${arm}")"
  step_loss="$(step_loss_for_strategy "${strategy}")"
  future_enabled="$(future_enabled_for_arm "${arm}")"

  if [[ "${strategy}" == "unknown" ]]; then
    echo "unknown_arm=${arm}" >&2
    exit 1
  fi

  local run_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${run_name}_${dataset}_seed${SEED}.log"
  local extra_args=()
  local future_args=()

  if [[ -n "${STEPS_PER_EPOCH}" ]]; then
    extra_args+=(--steps-per-epoch "${STEPS_PER_EPOCH}")
  fi
  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    extra_args+=(--max-eval-batches "${MAX_EVAL_BATCHES}")
  fi

  if [[ "${future_enabled}" == "1" ]]; then
    future_args+=(
      --future-teacher-layers "${FUTURE_TEACHER_LAYERS}"
      --future-teacher-heads "${FUTURE_TEACHER_HEADS}"
      --future-teacher-d-ff "${FUTURE_TEACHER_D_FF}"
      --future-align-weight "${FUTURE_ALIGN_WEIGHT}"
      --future-relation-weight "${FUTURE_RELATION_WEIGHT}"
      --future-recon-weight "${FUTURE_RECON_WEIGHT}"
      --future-recon-normalization "${FUTURE_RECON_NORMALIZATION}"
      --future-align-weighting "${FUTURE_ALIGN_WEIGHTING}"
      --future-confidence-temperature "${FUTURE_CONFIDENCE_TEMPERATURE}"
      --future-confidence-floor "${FUTURE_CONFIDENCE_FLOOR}"
      --future-recon-eps "${FUTURE_RECON_EPS}"
    )
  fi

  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing arm=${arm} run_name=${run_name} dataset=${dataset}"
    return 0
  fi

  echo "run_start=$(date -Is) arm=${arm} run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id} future_enabled=${future_enabled}"
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
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --run-name "${run_name}" \
    --model-variant target_set \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${future_args[@]}" \
    "${extra_args[@]}" 2>&1 | tee "${run_log}"
  if [[ "${KEEP_HEAVY_ARTIFACTS}" != "1" ]]; then
    find "${run_dir}" -type f \( -name "checkpoint.pt" -o -name "predictions_test.npz" \) -delete
  fi
  echo "run_done=$(date -Is) arm=${arm} run_name=${run_name} strategy=${strategy} dataset=${dataset} gpu=${gpu_id}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"

launch_job() {
  local arm="$1"
  local dataset="$2"
  local gpu_id
  gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
  run_one "${arm}" "${dataset}" "${gpu_id}" &
  job_index=$((job_index + 1))
  if (( job_index % max_jobs == 0 )); then
    wait
  fi
}

case "${JOB_ORDER}" in
  dataset_major)
    for dataset in "${datasets[@]}"; do
      for arm in "${arms[@]}"; do
        launch_job "${arm}" "${dataset}"
      done
    done
    ;;
  arm_major)
    for arm in "${arms[@]}"; do
      for dataset in "${datasets[@]}"; do
        launch_job "${arm}" "${dataset}"
      done
    done
    ;;
  *)
    echo "unknown_job_order=${JOB_ORDER}" >&2
    exit 1
    ;;
esac

wait
echo "phase4_future_state_anchor_gate_done=$(date -Is)"
