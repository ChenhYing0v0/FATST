#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b}"
DESIGN="${DESIGN:-configs/stage_c_pcsd_cf_native_direct.json}"
AUDIT_CONFIG="${AUDIT_CONFIG:-configs/stage_c_pcsd_cf_test_audit.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

ARMS=(
  pcsd_direct pcsd_equal pcsd_static pcsd_random
  pcsd_fixed_1 pcsd_fixed_48 pcsd_fixed_144 pcsd_fixed_360 pcsd_fixed_720
  a6 pcsd_m0 dense_matched
)
DATASETS=(Weather ETTm1 ETTh1 ETTm2 ETTh2)
JOBS=()
for arm in "${ARMS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    JOBS+=("${arm} ${dataset}")
  done
done

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -s "${DESIGN}"
test -s "${AUDIT_CONFIG}"

run_dir() {
  local arm="$1" dataset="$2"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local arm="$1" dataset="$2" directory
  directory="$(run_dir "${arm}" "${dataset}")"
  [[ -s "${directory}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${directory}/test_audit_invariants.json" \
    && -s "${directory}/pcsd_test_audit_diagnostics.npz" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for job in "${JOBS[@]}"; do
    read -r arm dataset <<< "${job}"
    if is_complete "${arm}" "${dataset}"; then
      completed=$((completed + 1))
    fi
  done
  echo "pcsd_test_audit_status=$(date -Is) completed=${completed}/${#JOBS[@]}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 -c '
import json,sys
audit=json.load(open(sys.argv[1]))
assert audit["candidate_version"] == "SC1-PCSD-CF-v1"
assert audit["matrix"]["expected_runs"] == 60
assert audit["authorization"]["user_authorized"] is True
assert audit["authorization"]["checkpoint_retraining_allowed"] is False
' "${AUDIT_CONFIG}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    --synthetic-smoke >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_pcsd_cf_step7b.py \
    --design "${DESIGN}" --evaluation-split test-audit \
    --synthetic-smoke >/dev/null
  echo "pcsd_test_audit_dry_run=pass jobs=${#JOBS[@]} retraining=false"
  exit 0
fi

AUDIT_ROOT="${OUTPUT_ROOT}/_test_audit_seed${SEED}"
LOG_ROOT="${OUTPUT_ROOT}/_test_audit_logs_seed${SEED}"
mkdir -p "${AUDIT_ROOT}" "${LOG_ROOT}"
{
  echo "pcsd_test_audit_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "candidate_version=SC1-PCSD-CF-v1"
  echo "test_role=primary_milestone_effectiveness_gate"
  echo "checkpoint_selection=historical_best_validation_h720_mse"
  echo "checkpoint_retraining=false"
  echo "jobs=${#JOBS[@]}"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${AUDIT_ROOT}/launch_record.txt"

run_one() {
  local index="$1" job="$2" gpu="$3"
  local arm dataset directory log checkpoint_before checkpoint_after
  read -r arm dataset <<< "${job}"
  directory="$(run_dir "${arm}" "${dataset}")"
  log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  test -s "${directory}/checkpoint.pt"
  if is_complete "${arm}" "${dataset}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    return 0
  fi
  checkpoint_before="$(sha256sum "${directory}/checkpoint.pt" | awk '{print $1}')"
  echo "audit_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    --run-dir "${directory}" --design "${DESIGN}" \
    --test-audit-config "${AUDIT_CONFIG}" --evaluation-split test \
    --device cuda >"${log}" 2>&1
  checkpoint_after="$(sha256sum "${directory}/checkpoint.pt" | awk '{print $1}')"
  if [[ "${checkpoint_before}" != "${checkpoint_after}" ]]; then
    echo "checkpoint_mutated arm=${arm} dataset=${dataset}" >&2
    return 1
  fi
  echo "audit_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" job_index
  for ((job_index=worker_index; job_index<${#JOBS[@]}; job_index+=${#GPU_IDS[@]})); do
    run_one "${job_index}" "${JOBS[${job_index}]}" "${gpu}"
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
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "pcsd_test_audit_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_pcsd_cf_step7b.py \
  --raw-root "${OUTPUT_ROOT}" --output-dir "${AUDIT_ROOT}" \
  --design "${DESIGN}" --seed "${SEED}" --evaluation-split test-audit
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_pcsd_cf_step7b_deep_dive.py \
  --raw-root "${OUTPUT_ROOT}" --run-summary "${AUDIT_ROOT}/run_summary.csv" \
  --output-dir "${AUDIT_ROOT}" --seed "${SEED}" \
  --evaluation-split test-audit
echo "pcsd_test_audit_done=$(date -Is) jobs=${#JOBS[@]}"
