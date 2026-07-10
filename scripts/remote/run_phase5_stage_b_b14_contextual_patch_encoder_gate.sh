#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_contextual_patch_encoder_gate}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DATASETS_STR="${DATASETS:-Weather ETTm1 ETTh2}"
ARMS_STR="${ARMS:-cpe_p16s8 cpe_p48s24}"
SEEDS_STR="${SEEDS:-2021}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a ARMS <<< "${ARMS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

if (( ${#GPU_IDS[@]} == 0 )); then
  echo "at least one GPU id is required" >&2
  exit 2
fi
gpu_count="${#GPU_IDS[@]}"

mkdir -p "${OUTPUT_ROOT}/_logs"

echo "phase5_stage_b_b14_contextual_patch_encoder_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "datasets=${DATASETS[*]}"
echo "arms=${ARMS[*]}"
echo "seeds=${SEEDS[*]}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

arm_config() {
  case "$1" in
    cpe_p16s8) echo "16 8" ;;
    cpe_p48s24) echo "48 24" ;;
    *) echo "unknown arm: $1" >&2; return 2 ;;
  esac
}

dataset_config() {
  case "$1" in
    ETTh2) echo "16 4 128 0.3" ;;
    ETTm1|Weather) echo "128 16 256 0.2" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

run_one() {
  local dataset="$1"
  local arm="$2"
  local seed="$3"
  local gpu="$4"
  local patch_len patch_stride d_model n_heads d_ff dropout
  read -r patch_len patch_stride <<< "$(arm_config "${arm}")"
  read -r d_model n_heads d_ff dropout <<< "$(dataset_config "${dataset}")"

  local run_name="A6_CPE_LBF_r256_${arm}_best-val"
  local output_dir="${OUTPUT_ROOT}/best-val/${run_name}/${dataset}/mixed_h96_h192_h336_h720/seed${seed}"
  local run_log="${OUTPUT_ROOT}/_logs/${run_name}_${dataset}_seed${seed}.log"

  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing arm=${arm} dataset=${dataset} seed=${seed} output_dir=${output_dir}"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) arm=${arm} dataset=${dataset} seed=${seed} gpu=${gpu} output_dir=${output_dir}"
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
      --patience 5 \
      --seed "${seed}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers 0 \
      --run-name "${run_name}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --checkpoint-policy best-val \
      --learning-rate 0.0001 \
      --encoder-mode contextual-patch-transformer \
      --history-patch-len "${patch_len}" \
      --history-patch-stride "${patch_stride}" \
      --history-d-model "${d_model}" \
      --history-n-heads "${n_heads}" \
      --history-d-ff "${d_ff}" \
      --history-e-layers 3 \
      --history-dropout "${dropout}" \
      --history-attn-dropout 0.0 \
      --history-res-attention \
      --readout-mode learned-basis-forecast-operator \
      --basis-rank 256 \
      --pred-loss-mode multi-prefix 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) arm=${arm} dataset=${dataset} seed=${seed} output_dir=${output_dir}"
}

pids=()
gpu_cursor=0
for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for arm in "${ARMS[@]}"; do
      gpu="${GPU_IDS[$((gpu_cursor % gpu_count))]}"
      run_one "${dataset}" "${arm}" "${seed}" "${gpu}" &
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
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_stage_b_b14_contextual_patch_encoder_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}" -name metrics_by_target_horizon.csv -type f | sort
