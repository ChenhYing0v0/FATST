#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3e_target_conditioned_nested_gate}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
H1_CHECKPOINT_ROOT="${H1_CHECKPOINT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1_readout_gate/official-last/TimeAlignOfficialUnified720_H1_target_set_decoder_multiprefix_official-last}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/${CHECKPOINT_POLICY}}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
DATASETS="${DATASETS:-Weather ETTm1 ETTh2}"
ARMS="${ARMS:-target_conditioned_nested_warm target_conditioned_nested_scratch}"
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
read -r -a arms <<< "${ARMS}"

if [[ "${CHECKPOINT_POLICY}" != "official-last" && "${CHECKPOINT_POLICY}" != "best-val" ]]; then
  echo "CHECKPOINT_POLICY must be official-last or best-val." >&2
  exit 1
fi

arm_config() {
  local arm="$1"
  case "${arm}" in
    target_conditioned_nested_warm)
      echo "target-conditioned-nested-segment-decoder multi-prefix 1 32 32 warm"
      ;;
    target_conditioned_nested_scratch)
      echo "target-conditioned-nested-segment-decoder multi-prefix 1 32 32 scratch"
      ;;
    *)
      echo "Unknown A3E arm: ${arm}" >&2
      exit 1
      ;;
  esac
}

h1_checkpoint_path() {
  local dataset="$1"
  echo "${H1_CHECKPOINT_ROOT}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}/checkpoint.pt"
}

echo "phase5_timealign_hss_a3e_target_conditioned_nested_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "h1_checkpoint_root=${H1_CHECKPOINT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "datasets=${DATASETS}"
echo "arms=${ARMS}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size=${BATCH_SIZE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_eval_batches=${MAX_EVAL_BATCHES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_job() {
  local dataset="$1"
  local arm="$2"
  local gpu="$3"
  local readout_mode loss_mode prefix_samples continuous_min_prefix continuous_prefix_step init_mode
  read -r readout_mode loss_mode prefix_samples continuous_min_prefix continuous_prefix_step init_mode <<< "$(arm_config "${arm}")"
  local checkpoint_path run_name run_dir run_log
  checkpoint_path="$(h1_checkpoint_path "${dataset}")"
  if [[ "${init_mode}" == "warm" && ! -s "${checkpoint_path}" ]]; then
    echo "Missing H1 checkpoint: ${checkpoint_path}" >&2
    exit 1
  fi
  run_name="TimeAlignOfficialUnified720_A3E_${arm}_${CHECKPOINT_POLICY}"
  run_dir="${OUTPUT_ROOT}/${CHECKPOINT_POLICY}/${run_name}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}"
  run_log="${LOG_ROOT}/${run_name}_${dataset}_seed${SEED}.log"
  if [[ -s "${run_dir}/metrics_by_target_horizon.csv" && -s "${run_dir}/checkpoint.pt" ]]; then
    echo "skip_existing run_name=${run_name} dataset=${dataset} arm=${arm}"
    return 0
  fi
  echo "run_start=$(date -Is) run_name=${run_name} dataset=${dataset} arm=${arm} readout_mode=${readout_mode} loss_mode=${loss_mode} init_mode=${init_mode} checkpoint=${checkpoint_path} gpu=${gpu}"
  cmd=(
    python baselines/timealign_official/train_repo.py
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode unified \
      --seq-len "${SEQ_LEN}" \
      --pred-len 720 \
      --target-horizons 96,192,336,720 \
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
      --checkpoint-policy "${CHECKPOINT_POLICY}" \
      --readout-mode "${readout_mode}" \
      --pred-loss-mode "${loss_mode}" \
      --prefix-samples "${prefix_samples}" \
      --continuous-min-prefix "${continuous_min_prefix}" \
      --continuous-prefix-step "${continuous_prefix_step}"
  )
  if [[ "${init_mode}" == "warm" ]]; then
    cmd+=(--warm-start-checkpoint "${checkpoint_path}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    "${cmd[@]}" 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) run_name=${run_name} dataset=${dataset} arm=${arm} gpu=${gpu}"
}

pids=()
next_gpu=0

launch_job() {
  local dataset="$1"
  local arm="$2"
  local gpu="${gpu_ids[${next_gpu}]}"
  next_gpu=$(((next_gpu + 1) % ${#gpu_ids[@]}))
  run_job "${dataset}" "${arm}" "${gpu}" &
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
  for arm in "${arms[@]}"; do
    wait_for_slot
    launch_job "${dataset}" "${arm}"
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_timealign_hss_a3e_target_conditioned_nested_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" -path "*/metrics_by_target_horizon.csv" -type f | sort
