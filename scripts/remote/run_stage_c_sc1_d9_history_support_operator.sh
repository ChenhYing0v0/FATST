#!/usr/bin/env bash
set -euo pipefail

LEGACY_A_ROOT="${LEGACY_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
LEGACY_B_ROOT="${LEGACY_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
LEGACY_C_ROOT="${LEGACY_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability}"
FIVE_A_ROOT="${FIVE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_a}"
FIVE_B_ROOT="${FIVE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_b}"
FIVE_C_ROOT="${FIVE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d9_history_support_operator}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
DESIGN="${DESIGN:-configs/stage_c_sc1_d9_history_support_operator_audit.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DRY_RUN="${DRY_RUN:-0}"

test -s "${CONTRACT}"
test -s "${DESIGN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_sc1_d9_history_support_operator.py \
      --synthetic-smoke
  echo "stage_c_sc1_d9_dry_run=pass"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
{
  echo "stage_c_sc1_d9_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "role=diagnostic_only"
  echo "reads_data_samples=false"
  echo "uses_test_split=false"
  echo "device=cpu"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc1_d9_history_support_operator.py \
    --legacy-a-root "${LEGACY_A_ROOT}" --legacy-b-root "${LEGACY_B_ROOT}" \
    --legacy-c-root "${LEGACY_C_ROOT}" --five-a-root "${FIVE_A_ROOT}" \
    --five-b-root "${FIVE_B_ROOT}" --five-c-root "${FIVE_C_ROOT}" \
    --contract "${CONTRACT}" --design "${DESIGN}" \
    --output-dir "${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/run.log"

echo "stage_c_sc1_d9_finished=$(date -Is)"
