#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_bsca_v1_confirmation}"
CONFIG="${CONFIG:-configs/stage_c_iscf_bsca_v1_confirmation.json}"
INNER_RUNNER="scripts/remote/run_stage_c_iscf_bsca_v1.sh"
SEED2022_GPUS="${SEED2022_GPUS:-0 1}"
SEED2023_GPUS="${SEED2023_GPUS:-2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
FORMAL_TEST_ONLY="${FORMAL_TEST_ONLY:-0}"
PROTOCOL_PROFILE="stage_c_iscf_bsca_v1_confirmation"

test -s "${CONFIG}"
test -x "${INNER_RUNNER}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

run_dir() {
  local seed="$1" dataset="$2"
  echo "${OUTPUT_ROOT}/iscf_bsca_v1/${dataset}/h720_full/seed${seed}"
}

training_complete() {
  local directory="$1"
  [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/training_log.csv" \
    && -s "${directory}/metrics_by_target_horizon.csv" \
    && -s "${directory}/effective_config.json" \
    && -s "${directory}/initialization_contract.json" \
    && -s "${directory}/model_diagnostics.json" \
    && -s "${directory}/pcsd_validation_diagnostics.npz" \
    && -s "${directory}/trained_invariants.json" ]]
}

test_complete() {
  local directory="$1"
  [[ -s "${directory}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${directory}/test_audit_invariants.json" \
    && -s "${directory}/pcsd_test_audit_diagnostics.npz" ]] \
    && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' \
      "${directory}/test_audit_invariants.json" 2>/dev/null
}

counts() {
  local trained=0 tested=0 seed dataset directory
  for seed in 2022 2023; do
    for dataset in Weather ETTm1 ETTh1 ETTh2 ETTm2; do
      directory="$(run_dir "${seed}" "${dataset}")"
      if training_complete "${directory}"; then trained=$((trained + 1)); fi
      if test_complete "${directory}"; then tested=$((tested + 1)); fi
    done
  done
  echo "${trained} ${tested}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  read -r trained tested <<< "$(counts)"
  echo "bsca_confirmation_status=$(date -Is) training=${trained}/10 test=${tested}/10"
  find "${OUTPUT_ROOT}" -path '*/_logs_seed*/*.log' -type f -print0 \
    2>/dev/null | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  for seed in 2022 2023; do
    SEED="${seed}" OUTPUT_ROOT="${OUTPUT_ROOT}" CONFIG="${CONFIG}" \
      PROTOCOL_PROFILE="${PROTOCOL_PROFILE}" DRY_RUN=1 \
      bash "${INNER_RUNNER}"
  done
  echo "bsca_confirmation_dry_run=pass jobs=10 effective_runs=30"
  exit 0
fi

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
    echo "RESOURCE_SMOKE and FORMAL_TEST_ONLY are mutually exclusive" >&2
    exit 2
  fi
  SEED=2022 OUTPUT_ROOT="${OUTPUT_ROOT}" CONFIG="${CONFIG}" \
    GPU_IDS="${SEED2022_GPUS%% *}" PROTOCOL_PROFILE="${PROTOCOL_PROFILE}" \
    RESOURCE_SMOKE=1 bash "${INNER_RUNNER}"
  echo "bsca_confirmation_resource_smoke_done=$(date -Is)"
  exit 0
fi

if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
  read -r trained _tested <<< "$(counts)"
  if [[ "${trained}" -ne 10 ]]; then
    echo "confirmation formal test requires complete training: ${trained}/10" >&2
    exit 4
  fi
fi

mkdir -p "${OUTPUT_ROOT}"
mode=training
if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then mode=formal_test; fi
record="${OUTPUT_ROOT}/confirmation_${mode}_launch_record.txt"
{
  echo "bsca_confirmation_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "config_hash=$(sha256_file "${CONFIG}")"
  echo "output_root=${OUTPUT_ROOT}"
  echo "seed2022_gpus=${SEED2022_GPUS}"
  echo "seed2023_gpus=${SEED2023_GPUS}"
  echo "new_runs=10"
  echo "reused_equal_runs=10"
  echo "three_seed_effective_runs=30"
  echo "formal_test_execution_mode=${FORMAL_TEST_ONLY}"
  echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${record}"

run_seed() {
  local seed="$1" gpu_ids="$2"
  SEED="${seed}" OUTPUT_ROOT="${OUTPUT_ROOT}" CONFIG="${CONFIG}" \
    GPU_IDS="${gpu_ids}" PROTOCOL_PROFILE="${PROTOCOL_PROFILE}" \
    FORMAL_TEST_ONLY="${FORMAL_TEST_ONLY}" bash "${INNER_RUNNER}"
}

run_seed 2022 "${SEED2022_GPUS}" &
pid2022=$!
run_seed 2023 "${SEED2023_GPUS}" &
pid2023=$!
status=0
if ! wait "${pid2022}"; then status=1; fi
if ! wait "${pid2023}"; then status=1; fi
if [[ "${status}" != "0" ]]; then
  echo "bsca_confirmation_${mode}_failure=$(date -Is)" >&2
  exit 1
fi
read -r trained tested <<< "$(counts)"
echo "bsca_confirmation_${mode}_done=$(date -Is) training=${trained}/10 test=${tested}/10"
