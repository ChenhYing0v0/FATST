#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_hierarchical_patch_memory_gate}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_probe_inputs/a6_clean}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DATASETS_STR="${DATASETS:-Weather ETTm1 ETTh2}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "at least one GPU id is required" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "phase5_stage_b_b14_hierarchical_patch_memory_gate_start=$(date -Is)"
echo "git_commit=$(git rev-parse HEAD)"
echo "output_root=${OUTPUT_ROOT}"
echo "reference_root=${REFERENCE_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

pids=()
for idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$idx]}"
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  reference_dir="${REFERENCE_ROOT}/${dataset}/mixed_h96_h192_h336_h720/seed2021"
  output_dir="${OUTPUT_ROOT}/${dataset}/seed2021"
  log="${OUTPUT_ROOT}/_logs/${dataset}.log"
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_phase5_stage_b_b14_hierarchical_patch_memory.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --reference-dir "${reference_dir}" \
      --output-dir "${output_dir}" \
      --batch-size 32 \
      --device cuda > "${log}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "phase5_stage_b_b14_hierarchical_patch_memory_gate_done=$(date -Is)"
find "${OUTPUT_ROOT}" -name equivalence_summary.json -type f -print -exec cat {} \;
