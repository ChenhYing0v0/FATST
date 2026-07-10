#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_c0_ettm1_carrier_protocol_gate}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
ARMS_STR="${ARMS:-p1_d256_f256_d09 p1_d384_f96_d09 p5_d52_f256_d09 p5_d52_f2048_d09 p1_d256_f256_d02 p5_d52_f2048_d02}"
SEEDS_STR="${SEEDS:-2021}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
read -r -a ARMS <<< "${ARMS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

if (( ${#GPU_IDS[@]} == 0 )); then
  echo "at least one GPU id is required" >&2
  exit 2
fi
gpu_count="${#GPU_IDS[@]}"
mkdir -p "${OUTPUT_ROOT}/_logs"

echo "phase5_stage_b_c0_ettm1_carrier_protocol_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "arms=${ARMS[*]}"
echo "seeds=${SEEDS[*]}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

arm_config() {
  case "$1" in
    p1_d256_f256_d09) echo "1 256 256 0.9" ;;
    p1_d384_f96_d09) echo "1 384 96 0.9" ;;
    p5_d52_f256_d09) echo "5 52 256 0.9" ;;
    p5_d52_f2048_d09) echo "5 52 2048 0.9" ;;
    p1_d256_f256_d02) echo "1 256 256 0.2" ;;
    p5_d52_f2048_d02) echo "5 52 2048 0.2" ;;
    *) echo "unknown arm: $1" >&2; return 2 ;;
  esac
}

run_one() {
  local arm="$1"
  local seed="$2"
  local gpu="$3"
  local patch_num d_model d_ff dropout
  read -r patch_num d_model d_ff dropout <<< "$(arm_config "${arm}")"

  local run_name="A6_C0_${arm}_dual"
  local output_dir="${OUTPUT_ROOT}/${run_name}/ETTm1/mixed_h96_h192_h336_h720/seed${seed}"
  local run_log="${OUTPUT_ROOT}/_logs/${run_name}_ETTm1_seed${seed}.log"

  if [[ -s "${output_dir}/metrics_last_by_target_horizon.csv" && \
        -s "${output_dir}/metrics_best_val_by_target_horizon.csv" ]]; then
    echo "skip_existing arm=${arm} seed=${seed} output_dir=${output_dir}"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) arm=${arm} seed=${seed} gpu=${gpu} output_dir=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset ETTm1 \
      --mode unified \
      --seq-len 720 \
      --pred-len 720 \
      --target-horizons 96,192,336,720 \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --patience 5 \
      --seed "${seed}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers 0 \
      --run-name "${run_name}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --checkpoint-policy official-last \
      --evaluate-dual-checkpoints \
      --learning-rate 0.0001 \
      --encoder-mode timealign-token-mlp \
      --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" \
      --legacy-dropout "${dropout}" \
      --readout-mode learned-basis-forecast-operator \
      --basis-rank 256 \
      --pred-loss-mode multi-prefix 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) arm=${arm} seed=${seed} output_dir=${output_dir}"
}

pids=()
gpu_cursor=0
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    gpu="${GPU_IDS[$((gpu_cursor % gpu_count))]}"
    run_one "${arm}" "${seed}" "${gpu}" &
    pids+=("$!")
    gpu_cursor=$((gpu_cursor + 1))
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

echo "phase5_stage_b_c0_ettm1_carrier_protocol_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}" -name metrics_last_by_target_horizon.csv -type f | sort
