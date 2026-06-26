#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_official_gate}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/${CHECKPOINT_POLICY}}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1 2}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
DATASETS="${DATASETS:-Weather ETTm2 ETTh2}"
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

if [[ "${CHECKPOINT_POLICY}" != "official-last" && "${CHECKPOINT_POLICY}" != "best-val" ]]; then
  echo "CHECKPOINT_POLICY must be official-last or best-val." >&2
  exit 1
fi

echo "phase5_timealign_official_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
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

run_job() {
  local dataset="$1"
  local mode="$2"
  local horizon="$3"
  local gpu="$4"
  local pred_len target_horizons run_name label horizon_label run_dir run_log
  if [[ "${mode}" == "fixed" ]]; then
    pred_len="${horizon}"
    target_horizons="${horizon}"
    run_name="TimeAlignOfficialFixedH${horizon}_${CHECKPOINT_POLICY}"
    label="h${horizon}"
  else
    pred_len="720"
    target_horizons="96,192,336,720"
    run_name="TimeAlignOfficialUnified720_${CHECKPOINT_POLICY}"
    label="mixed"
  fi
  horizon_label="mixed_$(echo "${target_horizons}" | tr ',' '\n' | sed 's/^/h/' | paste -sd_ -)"
  run_dir="${OUTPUT_ROOT}/${CHECKPOINT_POLICY}/${run_name}/${dataset}/${horizon_label}/seed${SEED}"
  run_log="${LOG_ROOT}/${run_name}_${dataset}_${label}_seed${SEED}.log"
  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" && -s "${run_dir}/checkpoint.pt" ]]; then
    echo "skip_existing run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon}"
    return 0
  fi
  echo "run_start=$(date -Is) run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode "${mode}" \
      --seq-len "${SEQ_LEN}" \
      --pred-len "${pred_len}" \
      --target-horizons "${target_horizons}" \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${SEED}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers "${NUM_WORKERS}" \
      --run-name "${run_name}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --checkpoint-policy "${CHECKPOINT_POLICY}" 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) run_name=${run_name} dataset=${dataset} mode=${mode} horizon=${horizon} gpu=${gpu}"
}

pids=()
next_gpu=0

launch_job() {
  local dataset="$1"
  local mode="$2"
  local horizon="$3"
  local gpu="${gpu_ids[${next_gpu}]}"
  next_gpu=$(((next_gpu + 1) % ${#gpu_ids[@]}))
  run_job "${dataset}" "${mode}" "${horizon}" "${gpu}" &
  pids+=("$!")
}

compact_pids() {
  local alive=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      alive+=("${pid}")
    fi
  done
  pids=("${alive[@]}")
}

wait_for_slot() {
  while [[ "${#pids[@]}" -ge "${#gpu_ids[@]}" ]]; do
    wait -n
    compact_pids
  done
}

for dataset in "${datasets[@]}"; do
  for horizon in "${horizons[@]}"; do
    wait_for_slot
    launch_job "${dataset}" "fixed" "${horizon}"
  done
  wait_for_slot
  launch_job "${dataset}" "unified" "720"
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_timealign_official_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" -path "*/metrics_by_target_horizon.csv" -type f | sort
