#!/usr/bin/env bash
set -euo pipefail

PHASE_A_ROOT="${PHASE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
PHASE_B_ROOT="${PHASE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
PHASE_C_ROOT="${PHASE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_natural_baseline_test}"
CONTRACT="${CONTRACT:-configs/stage_c_mechanism_control_natural_dataset_profiles.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
DATASETS=(Weather ETTm1 ETTh2)

if [[ "${DRY_RUN}" == "1" ]]; then
  test -s "${CONTRACT}"
  echo "stage_c_natural_baseline_test_dry_run=pass datasets=${#DATASETS[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
echo "natural_baseline_test_start=$(date -Is) commit=$(git rev-parse HEAD)"
pids=()
for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[${index}]}"
  gpu="${GPU_IDS[${index}]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_natural_baseline_test.py \
      --phase-a-root "${PHASE_A_ROOT}" --phase-b-root "${PHASE_B_ROOT}" \
      --phase-c-root "${PHASE_C_ROOT}" --contract "${CONTRACT}" \
      --output-dir "${OUTPUT_ROOT}" --dataset "${dataset}" --device cuda \
      >"${OUTPUT_ROOT}/_logs/${dataset}.log" 2>&1 &
  pids+=("$!")
  echo "worker_start=$(date -Is) dataset=${dataset} gpu=${gpu} pid=$!"
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_natural_baseline_test.py \
    --input-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}"
echo "natural_baseline_test_done=$(date -Is)"
