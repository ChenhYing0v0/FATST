#!/usr/bin/env bash
set -euo pipefail

LEGACY_A_ROOT="${LEGACY_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
LEGACY_B_ROOT="${LEGACY_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
LEGACY_C_ROOT="${LEGACY_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
FIVE_A_ROOT="${FIVE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_a}"
FIVE_B_ROOT="${FIVE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_b}"
FIVE_C_ROOT="${FIVE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d5_conditioning_locality}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
D5_CONFIG="${D5_CONFIG:-configs/stage_c_sc1_d5_conditioning_locality_frontier.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 3 ]]; then
  echo "SC1-D5 requires three GPU ids" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  test -s "${CONTRACT}"
  test -s "${D5_CONFIG}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/run_stage_c_sc1_d4_structured_basis.py --synthetic-smoke
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_sc1_d5_conditioning_locality.py --synthetic-smoke
  echo "stage_c_sc1_d5_dry_run=pass datasets=5 fits=585"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "stage_c_sc1_d5_start=$(date -Is) commit=$(git rev-parse HEAD)"
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
      --contract "${CONTRACT}" --d4-config "${D5_CONFIG}" \
      --output-dir "${OUTPUT_ROOT}" --dataset "${dataset}" --device cuda \
      >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1
  echo "worker_done=$(date -Is) dataset=${dataset} gpu=${gpu}"
}

worker() {
  local gpu="$1"
  shift
  local dataset
  for dataset in "$@"; do run_dataset "${dataset}" "${gpu}"; done
}

worker "${GPU_IDS[0]}" Weather ETTh1 & pid0="$!"
worker "${GPU_IDS[1]}" ETTm1 ETTh2 & pid1="$!"
worker "${GPU_IDS[2]}" ETTm2 & pid2="$!"
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc1_d5_conditioning_locality.py \
    --input-root "${OUTPUT_ROOT}" --d5-config "${D5_CONFIG}" \
    --output-dir "${OUTPUT_ROOT}"
echo "stage_c_sc1_d5_done=$(date -Is)"
