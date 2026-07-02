#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REMOTE_ROOT}/phase5_timealign_hss_a4s_validation_prefix_signal_export}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
H1_CHECKPOINT_ROOT="${H1_CHECKPOINT_ROOT:-${REMOTE_ROOT}/phase5_timealign_hss_h1_readout_gate/${CHECKPOINT_POLICY}/TimeAlignOfficialUnified720_H1_target_set_decoder_multiprefix_${CHECKPOINT_POLICY}}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
SEQ_LEN="${SEQ_LEN:-720}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DATASETS="${DATASETS:-Weather ETTm1 ETTh2}"
PATHS="${PATHS:-h1_target_set h1c_row_gated a2_nested a3c_warm_nested a3d_teacher_preserved a3e_target_conditioned_warm a3e_target_conditioned_scratch}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/${CHECKPOINT_POLICY}}"

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
read -r -a paths <<< "${PATHS}"

path_config() {
  local path_id="$1"
  case "${path_id}" in
    h1_target_set)
      echo "phase5_timealign_hss_h1_readout_gate TimeAlignOfficialUnified720_H1_target_set_decoder_multiprefix_${CHECKPOINT_POLICY} target-set-decoder"
      ;;
    h1c_row_gated)
      echo "phase5_timealign_hss_h1c_capacity_preserving_gate TimeAlignOfficialUnified720_H1C_row_gated_dense_head_multiprefix_${CHECKPOINT_POLICY} row-gated-dense-head"
      ;;
    a2_nested)
      echo "phase5_timealign_hss_a2_interface_gate TimeAlignOfficialUnified720_A2_nested_segment_decoder_multiprefix_${CHECKPOINT_POLICY} nested-segment-decoder"
      ;;
    a3c_warm_nested)
      echo "phase5_timealign_hss_a3c_warm_started_nested_gate TimeAlignOfficialUnified720_A3C_checkpoint_initialized_nested_segment_decoder_multiprefix_${CHECKPOINT_POLICY} checkpoint-initialized-nested-segment-decoder"
      ;;
    a3d_teacher_preserved)
      echo "phase5_timealign_hss_a3d_teacher_preserved_nested_gate TimeAlignOfficialUnified720_A3D_teacher_preserved_nested_w03_${CHECKPOINT_POLICY} checkpoint-initialized-nested-segment-decoder"
      ;;
    a3e_target_conditioned_warm)
      echo "phase5_timealign_hss_a3e_target_conditioned_nested_gate TimeAlignOfficialUnified720_A3E_target_conditioned_nested_warm_${CHECKPOINT_POLICY} target-conditioned-nested-segment-decoder"
      ;;
    a3e_target_conditioned_scratch)
      echo "phase5_timealign_hss_a3e_target_conditioned_nested_gate TimeAlignOfficialUnified720_A3E_target_conditioned_nested_scratch_${CHECKPOINT_POLICY} target-conditioned-nested-segment-decoder"
      ;;
    *)
      echo "Unknown A4S path_id: ${path_id}" >&2
      exit 1
      ;;
  esac
}

checkpoint_path() {
  local phase_root="$1"
  local run_name="$2"
  local dataset="$3"
  echo "${REMOTE_ROOT}/${phase_root}/${CHECKPOINT_POLICY}/${run_name}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}/checkpoint.pt"
}

h1_teacher_checkpoint() {
  local dataset="$1"
  echo "${H1_CHECKPOINT_ROOT}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}/checkpoint.pt"
}

echo "phase5_timealign_hss_a4s_validation_prefix_signal_export_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "remote_root=${REMOTE_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "h1_checkpoint_root=${H1_CHECKPOINT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "datasets=${DATASETS}"
echo "paths=${PATHS}"
echo "max_eval_batches=${MAX_EVAL_BATCHES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_job() {
  local dataset="$1"
  local path_id="$2"
  local gpu="$3"
  local phase_root run_name readout_mode ckpt teacher_ckpt output_dir run_log
  read -r phase_root run_name readout_mode <<< "$(path_config "${path_id}")"
  ckpt="$(checkpoint_path "${phase_root}" "${run_name}" "${dataset}")"
  teacher_ckpt="$(h1_teacher_checkpoint "${dataset}")"
  if [[ ! -s "${ckpt}" ]]; then
    echo "Missing checkpoint for ${path_id}/${dataset}: ${ckpt}" >&2
    exit 1
  fi
  if [[ ! -s "${teacher_ckpt}" ]]; then
    echo "Missing H1 teacher checkpoint for ${dataset}: ${teacher_ckpt}" >&2
    exit 1
  fi
  output_dir="${OUTPUT_ROOT}/${CHECKPOINT_POLICY}/${path_id}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}"
  run_log="${LOG_ROOT}/${path_id}_${dataset}_seed${SEED}.log"
  if [[ -s "${output_dir}/validation_prefix_diagnostics.csv" ]]; then
    echo "skip_existing path_id=${path_id} dataset=${dataset}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "diagnostic_start=$(date -Is) path_id=${path_id} dataset=${dataset} readout_mode=${readout_mode} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/export_timealign_validation_prefix_diagnostics.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode unified \
      --seq-len "${SEQ_LEN}" \
      --pred-len 720 \
      --target-horizons 96,192,336,720 \
      --batch-size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers "${NUM_WORKERS}" \
      --run-name "A4S_${path_id}_${CHECKPOINT_POLICY}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --readout-mode "${readout_mode}" \
      --checkpoint "${ckpt}" \
      --teacher-checkpoint "${teacher_ckpt}" \
      --teacher-readout-mode target-set-decoder 2>&1 | tee "${run_log}"
  echo "diagnostic_done=$(date -Is) path_id=${path_id} dataset=${dataset} gpu=${gpu}"
}

pids=()
next_gpu=0

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

launch_job() {
  local dataset="$1"
  local path_id="$2"
  local gpu="${gpu_ids[${next_gpu}]}"
  next_gpu=$(((next_gpu + 1) % ${#gpu_ids[@]}))
  run_job "${dataset}" "${path_id}" "${gpu}" &
  pids+=("$!")
}

for dataset in "${datasets[@]}"; do
  for path_id in "${paths[@]}"; do
    wait_for_slot
    launch_job "${dataset}" "${path_id}"
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_timealign_hss_a4s_validation_prefix_signal_export_done=$(date -Is)"
find "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" -name validation_prefix_diagnostics.csv -type f | sort
