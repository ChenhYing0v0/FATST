#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
D14_ROOT="${D14_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a1_dual_carrier_grouped_mlp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d21_evidence_validity_surface}"
DESIGN="${DESIGN:-configs/stage_c_d21_evidence_validity_surface.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
STATUS_ONLY="${STATUS_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONHASHSEED=2021
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

test -s "${DESIGN}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi

JOBS=()
while IFS= read -r value; do
  JOBS+=("${value}")
done < <(
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
for carrier in d["carriers"]:
    for dataset in d["datasets"]:
        for arm in d["canonical_arms"]:
            for split in ("val", "test"):
                print("\t".join((carrier, dataset, arm, split)))
' "${DESIGN}"
)

output_for() {
  local carrier="$1" dataset="$2" arm="$3" split="$4"
  echo "${OUTPUT_ROOT}/${carrier}/${arm}/${dataset}/seed2021/${split}.npz"
}

source_for() {
  local carrier="$1" dataset="$2" arm="$3"
  echo "${D14_ROOT}/${carrier}/${arm}/${dataset}/h720_full/seed2021"
}

is_complete() {
  local carrier="$1" dataset="$2" arm="$3" split="$4" output
  output="$(output_for "${carrier}" "${dataset}" "${arm}" "${split}")"
  [[ -s "${output}" && -s "${output%.npz}_invariants.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${JOBS[@]}"; do
    IFS=$'\t' read -r carrier dataset arm split <<< "${line}"
    if is_complete "${carrier}" "${dataset}" "${arm}" "${split}"; then
      completed=$((completed + 1))
    fi
  done
  echo "d21_evs_status=$(date -Is) completed=${completed}/${#JOBS[@]}"
  find "${OUTPUT_ROOT}/_logs" -type f -name '*.log' -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_stage_c_d21_evs_step7a.py
  printf '%s\n' "${JOBS[@]}"
  echo "d21_evs_dry_run=pass jobs=${#JOBS[@]} new_training=false"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
{
  echo "stage_c_d21_evs_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "dataset_root=${DATASET_ROOT}"
  echo "d14_root=${D14_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "jobs=${#JOBS[@]}"
  echo "fit_split=validation"
  echo "evaluation_split=test"
  echo "new_forecasting_model_training=false"
  echo "checkpoint_mutation=false"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch.txt"

run_job() {
  local index="$1" line="$2" gpu="$3"
  local carrier dataset arm split source output log descriptor_args=()
  IFS=$'\t' read -r carrier dataset arm split <<< "${line}"
  source="$(source_for "${carrier}" "${dataset}" "${arm}")"
  output="$(output_for "${carrier}" "${dataset}" "${arm}" "${split}")"
  log="${OUTPUT_ROOT}/_logs/${carrier}_${arm}_${dataset}_${split}.log"
  if is_complete "${carrier}" "${dataset}" "${arm}" "${split}"; then
    echo "skip_existing job=$((index + 1))/${#JOBS[@]} ${carrier}/${arm}/${dataset}/${split}"
    return 0
  fi
  test -s "${source}/checkpoint.pt"
  test -s "${source}/effective_config.json"
  mkdir -p "$(dirname "${output}")"
  if [[ "${arm}" == "c_s1" ]]; then
    descriptor_args=(--save-descriptors)
  fi
  echo "d21_evs_job_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} gpu=${gpu} ${carrier}/${arm}/${dataset}/${split}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_d21_evs_checkpoint.py \
      --run-dir "${source}" --design "${DESIGN}" --split "${split}" \
      --output "${output}" --device cuda "${descriptor_args[@]}" \
      >"${log}" 2>&1
  echo "d21_evs_job_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} ${carrier}/${arm}/${dataset}/${split}"
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#JOBS[@]}; index+=${#GPU_IDS[@]})); do
    run_job "${index}" "${JOBS[$index]}" "${gpu}"
  done
}

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
  worker "${worker_index}" "${GPU_IDS[$worker_index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "d21_evs_remote=failed" >&2
  exit "${status}"
fi
echo "d21_evs_remote=complete jobs=${#JOBS[@]}"
