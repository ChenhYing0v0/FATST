#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_carrier_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase5_timealign_carrier_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1 2}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
DATASETS="${DATASETS:-Weather ETTh2}"
HORIZONS="${HORIZONS:-96 192 336 720}"
SEQ_LEN="${SEQ_LEN:-720}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "${LOG_ROOT}"

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

read -r -a gpu_ids <<< "${GPU_IDS}"
read -r -a datasets <<< "${DATASETS}"
read -r -a horizons <<< "${HORIZONS}"

echo "phase5_timealign_carrier_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "datasets=${DATASETS}"
echo "horizons=${HORIZONS}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size=${BATCH_SIZE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_eval_batches=${MAX_EVAL_BATCHES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

dataset_args() {
  local dataset="$1"
  if [[ "${dataset}" == "Weather" ]]; then
    echo "--d-model 128 --d-ff 256 --learning-rate 0.0001 --no-layer-norm --local-margin 0.5 --global-margin 0.0 --w-align 0.1"
  elif [[ "${dataset}" == "ETTh2" ]]; then
    echo "--d-model 32 --d-ff 32 --learning-rate 0.0005 --local-margin 0.0 --global-margin 0.0 --w-align 0.1"
  else
    echo "--d-model 32 --d-ff 32 --learning-rate 0.0005 --local-margin 0.0 --global-margin 0.0 --w-align 0.1"
  fi
}

run_job() {
  local dataset="$1"
  local mode="$2"
  local horizon="$3"
  local gpu="$4"
  local run_name pred_len target_horizons label
  if [[ "${mode}" == "fixed" ]]; then
    pred_len="${horizon}"
    target_horizons="${horizon}"
    run_name="TimeAlignCarrierFixedH${horizon}"
    label="h${horizon}"
  else
    pred_len="720"
    target_horizons="96,192,336,720"
    run_name="TimeAlignCarrierUnified720"
    label="mixed"
  fi
  local horizon_label
  horizon_label="mixed_$(echo "${target_horizons}" | tr ',' '\n' | sed 's/^/h/' | paste -sd_ -)"
  local run_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  local run_log="${LOG_ROOT}/${run_name}_${dataset}_${label}_seed${SEED}.log"
  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" && -s "${run_dir}/checkpoint.pt" ]]; then
    echo "skip_existing run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon}"
    return 0
  fi
  echo "run_start=$(date -Is) run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon} gpu=${gpu}"
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_carrier/train.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --seq-len "${SEQ_LEN}" \
      --pred-len "${pred_len}" \
      --target-horizons "${target_horizons}" \
      --patch-num 48 \
      --e-layers 2 \
      --dropout 0.1 \
      --w-recon 1.0 \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${SEED}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers "${NUM_WORKERS}" \
      --run-name "${run_name}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda \
      $(dataset_args "${dataset}") 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon} gpu=${gpu}"
}

pids=()
launch_count=0

wait_for_wave() {
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  pids=()
}

for dataset in "${datasets[@]}"; do
  for horizon in "${horizons[@]}"; do
    gpu="${gpu_ids[$((launch_count % ${#gpu_ids[@]}))]}"
    run_job "${dataset}" "fixed" "${horizon}" "${gpu}" &
    pids+=("$!")
    launch_count=$((launch_count + 1))
    if [[ "${#pids[@]}" -ge "${#gpu_ids[@]}" ]]; then
      wait_for_wave
    fi
  done
  gpu="${gpu_ids[$((launch_count % ${#gpu_ids[@]}))]}"
  run_job "${dataset}" "unified" "720" "${gpu}" &
  pids+=("$!")
  launch_count=$((launch_count + 1))
  if [[ "${#pids[@]}" -ge "${#gpu_ids[@]}" ]]; then
    wait_for_wave
  fi
done

wait_for_wave

echo "phase5_timealign_carrier_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}" -path "*/metrics_by_target_horizon.csv" -type f | sort
