#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DATASETS_STR="${DATASETS:-Weather ETTm1 ETTh2}"
ARMS_STR="${ARMS:-a6_clean b11_bcf b11_no_basis b11_constant_slot}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
BASIS_FIELD_WINDOW_LEN="${BASIS_FIELD_WINDOW_LEN:-96}"
BASIS_FIELD_STRIDE="${BASIS_FIELD_STRIDE:-48}"
BASIS_FIELD_RANK="${BASIS_FIELD_RANK:-32}"
BASIS_FIELD_TAU="${BASIS_FIELD_TAU:-1.0}"
BASIS_FIELD_GATE_INIT="${BASIS_FIELD_GATE_INIT:--5.0}"

if [[ "${CHECKPOINT_POLICY}" != "official-last" ]]; then
  echo "B11-BCF small gate expects CHECKPOINT_POLICY=official-last." >&2
  exit 2
fi

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a ARMS <<< "${ARMS_STR}"

mkdir -p "${OUTPUT_ROOT}/_logs"

echo "phase5_stage_b_b11_bcf_small_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "datasets=${DATASETS[*]}"
echo "arms=${ARMS[*]}"
echo "basis_field_window_len=${BASIS_FIELD_WINDOW_LEN}"
echo "basis_field_stride=${BASIS_FIELD_STRIDE}"
echo "basis_field_rank=${BASIS_FIELD_RANK}"
echo "basis_field_tau=${BASIS_FIELD_TAU}"
echo "basis_field_gate_init=${BASIS_FIELD_GATE_INIT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

readout_for_arm() {
  local arm="$1"
  case "${arm}" in
    a6_clean)
      echo "learned-basis-forecast-operator"
      ;;
    b11_bcf)
      echo "basis-conditioned-coefficient-field"
      ;;
    b11_no_basis)
      echo "basis-conditioned-coefficient-field-no-basis"
      ;;
    b11_shuffled_basis)
      echo "basis-conditioned-coefficient-field-shuffled-basis"
      ;;
    b11_constant_slot)
      echo "basis-conditioned-coefficient-field-constant-slot"
      ;;
    *)
      echo "Unknown arm: ${arm}" >&2
      return 2
      ;;
  esac
}

run_one() {
  local arm="$1"
  local dataset="$2"
  local gpu="$3"
  local readout_mode
  readout_mode="$(readout_for_arm "${arm}")"
  local run_name="TimeAlignOfficialUnified720_${arm}_${CHECKPOINT_POLICY}"
  local output_dir="${OUTPUT_ROOT}/${CHECKPOINT_POLICY}/${run_name}/${dataset}/mixed_h96_h192_h336_h720/seed${SEED}"
  local run_log="${OUTPUT_ROOT}/_logs/${run_name}_${dataset}_seed${SEED}.log"

  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" && -s "${output_dir}/checkpoint.pt" ]]; then
    echo "skip_existing arm=${arm} dataset=${dataset} output_dir=${output_dir}"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) arm=${arm} dataset=${dataset} gpu=${gpu} output_dir=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode unified \
      --seq-len 720 \
      --pred-len 720 \
      --target-horizons 96,192,336,720 \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --seed "${SEED}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers 0 \
      --run-name "${run_name}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --checkpoint-policy "${CHECKPOINT_POLICY}" \
      --readout-mode "${readout_mode}" \
      --basis-rank 256 \
      --basis-field-window-len "${BASIS_FIELD_WINDOW_LEN}" \
      --basis-field-stride "${BASIS_FIELD_STRIDE}" \
      --basis-field-rank "${BASIS_FIELD_RANK}" \
      --basis-field-tau "${BASIS_FIELD_TAU}" \
      --basis-field-gate-init "${BASIS_FIELD_GATE_INIT}" \
      --pred-loss-mode multi-prefix 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) arm=${arm} dataset=${dataset} gpu=${gpu}"
}

pids=()
gpu_count="${#GPU_IDS[@]}"
idx=0

# Dataset-major order spreads slow datasets across GPUs before moving to shorter datasets.
for dataset in "${DATASETS[@]}"; do
  for arm in "${ARMS[@]}"; do
    gpu="${GPU_IDS[$((idx % gpu_count))]}"
    run_one "${arm}" "${dataset}" "${gpu}" &
    pids+=("$!")
    idx=$((idx + 1))
    if (( ${#pids[@]} >= gpu_count )); then
      wait -n
      remaining=()
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          remaining+=("${pid}")
        fi
      done
      pids=("${remaining[@]}")
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_stage_b_b11_bcf_small_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" -name metrics_by_target_horizon.csv -type f | sort
