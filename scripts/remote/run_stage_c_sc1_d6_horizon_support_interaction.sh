#!/usr/bin/env bash
set -euo pipefail

LEGACY_A_ROOT="${LEGACY_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
LEGACY_B_ROOT="${LEGACY_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
LEGACY_C_ROOT="${LEGACY_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
FIVE_A_ROOT="${FIVE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_a}"
FIVE_B_ROOT="${FIVE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_b}"
FIVE_C_ROOT="${FIVE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d6_horizon_support_interaction}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
D6_CONFIG="${D6_CONFIG:-configs/stage_c_sc1_d6_horizon_support_interaction.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
  test -s "${CONTRACT}"
  test -s "${D6_CONFIG}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_sc1_d4_structured_basis.py --synthetic-smoke
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_sc1_d6_horizon_support_interaction.py --synthetic-smoke
  echo "stage_c_sc1_d6_dry_run=pass datasets=5 fits=225"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "stage_c_sc1_d6_start=$(date -Is) commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_dataset() {
  local dataset="$1" gpu="$2"
  if [[ -s "${OUTPUT_ROOT}/${dataset}/d4_probe_metrics.csv" ]]; then
    echo "skip_existing dataset=${dataset}"
    return
  fi
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_sc1_d4_structured_basis.py \
      --phase-a-root "${LEGACY_A_ROOT}" --phase-b-root "${LEGACY_B_ROOT}" \
      --phase-c-root "${LEGACY_C_ROOT}" --five-phase-a-root "${FIVE_A_ROOT}" \
      --five-phase-b-root "${FIVE_B_ROOT}" --five-phase-c-root "${FIVE_C_ROOT}" \
      --contract "${CONTRACT}" --d4-config "${D6_CONFIG}" \
      --output-dir "${OUTPUT_ROOT}" --dataset "${dataset}" --device cuda \
      --val-offset-batches 8 >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1
  echo "worker_done=$(date -Is) dataset=${dataset} gpu=${gpu}"
}

worker() {
  local gpu="$1"
  shift
  local dataset
  for dataset in "$@"; do run_dataset "${dataset}" "${gpu}"; done
}

worker 0 Weather ETTh1 & pid0="$!"
worker 1 ETTm1 & pid1="$!"
worker 2 ETTm2 ETTh2 & pid2="$!"
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc1_d6_horizon_support_interaction.py \
    --input-root "${OUTPUT_ROOT}" --d6-config "${D6_CONFIG}" \
    --output-dir "${OUTPUT_ROOT}"
echo "stage_c_sc1_d6_done=$(date -Is)"
