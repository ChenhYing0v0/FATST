#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2}"
D17_ROOT="${D17_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d17_projective_future_context_v1}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/stage_c_siff_equal_attribution_v2.json}"
D17_CONFIG="${D17_CONFIG:-configs/stage_c_d17_projective_future_context_diagnostic.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/yingch/.conda/envs/moe/bin/python}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
PROBE_ROWS="${PROBE_ROWS:-256}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${SOURCE_CONFIG}"
test -s "${D17_CONFIG}"

VALIDATION_ROOT="${D17_ROOT}/validation"
ANALYSIS_ROOT="${D17_ROOT}/analysis"
LOG_ROOT="${D17_ROOT}/logs"
mkdir -p "${VALIDATION_ROOT}" "${ANALYSIS_ROOT}" "${LOG_ROOT}"

mapfile -t JOBS < <(
  "${PYTHON_BIN}" - "${D17_CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
priority = ["Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2"]
for dataset in priority:
    if dataset not in config["datasets"]:
        continue
    for carrier in config["parent_arms"]:
        print(f"{carrier}\t{dataset}")
PY
)

source_dir() {
  local carrier="$1" dataset="$2"
  echo "${SOURCE_ROOT}/${carrier}/${dataset}/h720_full/seed${SEED}"
}

validation_dir() {
  local carrier="$1" dataset="$2"
  echo "${VALIDATION_ROOT}/${carrier}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local carrier="$1" dataset="$2" output_dir
  output_dir="$(validation_dir "${carrier}" "${dataset}")"
  [[ -s "${output_dir}/pcsd_validation_diagnostics.npz" \
    && -s "${output_dir}/trained_invariants.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${JOBS[@]}"; do
    IFS=$'\t' read -r carrier dataset <<< "${line}"
    if is_complete "${carrier}" "${dataset}"; then
      completed=$((completed + 1))
    fi
  done
  echo "d17_status=$(date -Is) validation_complete=${completed}/${#JOBS[@]}"
  find "${LOG_ROOT}" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  if [[ -s "${ANALYSIS_ROOT}/summary.json" ]]; then
    echo "analysis_complete=true"
  else
    echo "analysis_complete=false"
  fi
  exit 0
fi

for line in "${JOBS[@]}"; do
  IFS=$'\t' read -r carrier dataset <<< "${line}"
  test -s "$(source_dir "${carrier}" "${dataset}")/checkpoint.pt"
  test -s "$(source_dir "${carrier}" "${dataset}")/pcsd_test_audit_diagnostics.npz"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    scripts/analyze_stage_c_d17_projective_future_context.py
  echo "d17_dry_run=pass jobs=${#JOBS[@]} probe_rows=${PROBE_ROWS}"
  printf '%s\n' "${JOBS[@]}"
  exit 0
fi

{
  echo "d17_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "source_root=${SOURCE_ROOT}"
  echo "d17_root=${D17_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#JOBS[@]}"
  echo "fit_split=validation"
  echo "evaluation_split=existing_authorized_test_probe"
  echo "checkpoint_mutation=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${D17_ROOT}/launch_record.txt"

run_one() {
  local index="$1" carrier="$2" dataset="$3" gpu="$4"
  local input_dir output_dir log_file checkpoint_before checkpoint_after
  input_dir="$(source_dir "${carrier}" "${dataset}")"
  output_dir="$(validation_dir "${carrier}" "${dataset}")"
  log_file="${LOG_ROOT}/${carrier}_${dataset}.log"
  if is_complete "${carrier}" "${dataset}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#JOBS[@]} carrier=${carrier} dataset=${dataset}"
    return 0
  fi
  mkdir -p "${output_dir}"
  checkpoint_before="$(sha256sum "${input_dir}/checkpoint.pt" | awk '{print $1}')"
  echo "validation_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} carrier=${carrier} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${input_dir}" \
      --artifact-dir "${output_dir}" \
      --design "${SOURCE_CONFIG}" \
      --test-audit-config "${SOURCE_CONFIG}" \
      --evaluation-split val \
      --probe-rows "${PROBE_ROWS}" \
      --device cuda >"${log_file}" 2>&1
  checkpoint_after="$(sha256sum "${input_dir}/checkpoint.pt" | awk '{print $1}')"
  if [[ "${checkpoint_before}" != "${checkpoint_after}" ]]; then
    echo "checkpoint_mutated carrier=${carrier} dataset=${dataset}" >&2
    return 1
  fi
  echo "validation_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} carrier=${carrier} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index carrier dataset
  for ((line_index=worker_index; line_index<${#JOBS[@]}; line_index+=${#GPU_IDS[@]})); do
    IFS=$'\t' read -r carrier dataset <<< "${JOBS[${line_index}]}"
    run_one "${line_index}" "${carrier}" "${dataset}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
  echo "worker_start=$(date -Is) worker=${index} gpu=${GPU_IDS[${index}]} pid=$!"
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" != "0" ]]; then
  echo "d17_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/analyze_stage_c_d17_projective_future_context.py \
  --config "${D17_CONFIG}" \
  --validation-root "${VALIDATION_ROOT}" \
  --test-root "${SOURCE_ROOT}" \
  --output-dir "${ANALYSIS_ROOT}"
echo "d17_done=$(date -Is)"
