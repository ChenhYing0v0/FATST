#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_future_state_alignment}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_future_state_alignment_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
RUN_NAME="${RUN_NAME:-PatchEncoderFutureStateAlignment}"
STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-prefix_risk}"
STEP_LOSS_ALPHA="${STEP_LOSS_ALPHA:-0.5}"
FUTURE_TEACHER_LAYERS="${FUTURE_TEACHER_LAYERS:-1}"
FUTURE_TEACHER_HEADS="${FUTURE_TEACHER_HEADS:-8}"
FUTURE_TEACHER_D_FF="${FUTURE_TEACHER_D_FF:-256}"
FUTURE_STATE_DIM="${FUTURE_STATE_DIM:-0}"
FUTURE_ALIGN_WEIGHT="${FUTURE_ALIGN_WEIGHT:-0.02}"
FUTURE_RELATION_WEIGHT="${FUTURE_RELATION_WEIGHT:-0.01}"
FUTURE_RECON_WEIGHT="${FUTURE_RECON_WEIGHT:-0.001}"
FUTURE_RECON_NORMALIZATION="${FUTURE_RECON_NORMALIZATION:-none}"
FUTURE_ALIGN_WEIGHTING="${FUTURE_ALIGN_WEIGHTING:-uniform}"
FUTURE_CONFIDENCE_TEMPERATURE="${FUTURE_CONFIDENCE_TEMPERATURE:-1.0}"
FUTURE_CONFIDENCE_FLOOR="${FUTURE_CONFIDENCE_FLOOR:-0.0}"
FUTURE_RECON_EPS="${FUTURE_RECON_EPS:-1e-6}"
DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
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

read -r -a datasets <<< "${DATASETS}"
read -r -a gpu_ids <<< "${GPU_IDS}"
IFS="," read -r -a target_horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${target_horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

echo "phase2_future_state_alignment_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "target_horizons=${TARGET_HORIZONS}"
echo "run_name=${RUN_NAME}"
echo "step_loss_weighting=${STEP_LOSS_WEIGHTING}"
echo "step_loss_alpha=${STEP_LOSS_ALPHA}"
echo "future_teacher_layers=${FUTURE_TEACHER_LAYERS}"
echo "future_teacher_heads=${FUTURE_TEACHER_HEADS}"
echo "future_teacher_d_ff=${FUTURE_TEACHER_D_FF}"
echo "future_state_dim=${FUTURE_STATE_DIM}"
echo "future_align_weight=${FUTURE_ALIGN_WEIGHT}"
echo "future_relation_weight=${FUTURE_RELATION_WEIGHT}"
echo "future_recon_weight=${FUTURE_RECON_WEIGHT}"
echo "future_recon_normalization=${FUTURE_RECON_NORMALIZATION}"
echo "future_align_weighting=${FUTURE_ALIGN_WEIGHTING}"
echo "future_confidence_temperature=${FUTURE_CONFIDENCE_TEMPERATURE}"
echo "future_confidence_floor=${FUTURE_CONFIDENCE_FLOOR}"
echo "future_recon_eps=${FUTURE_RECON_EPS}"
echo "datasets=${DATASETS}"
echo "steps_per_epoch=${STEPS_PER_EPOCH:-auto}"
echo "max_eval_batches=${MAX_EVAL_BATCHES:-all}"
echo "keep_heavy_artifacts=${KEEP_HEAVY_ARTIFACTS}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_one() {
  local dataset="$1"
  local gpu_id="$2"
  local run_dir="${OUTPUT_ROOT}/${RUN_NAME}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${RUN_NAME}_${dataset}_mixed_seed${SEED}.log"
  local extra_args=()

  if [[ -n "${STEPS_PER_EPOCH}" ]]; then
    extra_args+=(--steps-per-epoch "${STEPS_PER_EPOCH}")
  fi
  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    extra_args+=(--max-eval-batches "${MAX_EVAL_BATCHES}")
  fi
  if [[ "${FUTURE_STATE_DIM}" != "0" ]]; then
    extra_args+=(--future-state-dim "${FUTURE_STATE_DIM}")
  fi

  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing model=${RUN_NAME} dataset=${dataset}"
    return 0
  fi

  echo "run_start=$(date -Is) model=${RUN_NAME} dataset=${dataset} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/patch_encoder_target_set_decoder/train.py \
    --dataset-root "${DATASET_ROOT}" \
    --dataset "${dataset}" \
    --target-horizons "${TARGET_HORIZONS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --run-name "${RUN_NAME}" \
    --step-loss-weighting "${STEP_LOSS_WEIGHTING}" \
    --step-loss-alpha "${STEP_LOSS_ALPHA}" \
    --future-teacher-layers "${FUTURE_TEACHER_LAYERS}" \
    --future-teacher-heads "${FUTURE_TEACHER_HEADS}" \
    --future-teacher-d-ff "${FUTURE_TEACHER_D_FF}" \
    --future-align-weight "${FUTURE_ALIGN_WEIGHT}" \
    --future-relation-weight "${FUTURE_RELATION_WEIGHT}" \
    --future-recon-weight "${FUTURE_RECON_WEIGHT}" \
    --future-recon-normalization "${FUTURE_RECON_NORMALIZATION}" \
    --future-align-weighting "${FUTURE_ALIGN_WEIGHTING}" \
    --future-confidence-temperature "${FUTURE_CONFIDENCE_TEMPERATURE}" \
    --future-confidence-floor "${FUTURE_CONFIDENCE_FLOOR}" \
    --future-recon-eps "${FUTURE_RECON_EPS}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda \
    "${extra_args[@]}" 2>&1 | tee "${run_log}"
  if [[ "${KEEP_HEAVY_ARTIFACTS}" != "1" ]]; then
    find "${run_dir}" -type f \( -name "checkpoint.pt" -o -name "predictions_test.npz" \) -delete
  fi
  echo "run_done=$(date -Is) model=${RUN_NAME} dataset=${dataset} gpu=${gpu_id}"
}

job_index=0
max_jobs="${#gpu_ids[@]}"

for dataset in "${datasets[@]}"; do
  gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
  run_one "${dataset}" "${gpu_id}" &
  job_index=$((job_index + 1))
  if (( job_index % max_jobs == 0 )); then
    wait
  fi
done

wait
echo "phase2_future_state_alignment_gate_done=$(date -Is)"
