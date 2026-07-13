#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_BASE="${REMOTE_BASE:-/home/yingch/exp_outputs/r-2026-fatst}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_five_profile_extension_20260713/raw}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_five_dataset_profile_extension.json}"

mkdir -p "${LOCAL_ROOT}"
for phase in a b c; do
  remote_root="${REMOTE_BASE}/stage_c_five_profile_extension_${phase}/"
  local_root="${LOCAL_ROOT}/phase_${phase}/"
  mkdir -p "${local_root}"
  rsync -av --exclude checkpoint.pt --exclude predictions.npz \
    "${REMOTE_HOST}:${remote_root}" "${local_root}"
done

python scripts/analyze_stage_c_five_profile_extension.py \
  --phase a --phase-a-root "${LOCAL_ROOT}/phase_a" \
  --phase-b-root "${LOCAL_ROOT}/phase_b" --phase-c-root "${LOCAL_ROOT}/phase_c" \
  --output-dir "${LOCAL_ROOT}/phase_a/_analysis_local" --config "${CONFIG_PATH}"
python scripts/analyze_stage_c_five_profile_extension.py \
  --phase b --phase-a-root "${LOCAL_ROOT}/phase_a" \
  --phase-b-root "${LOCAL_ROOT}/phase_b" --phase-c-root "${LOCAL_ROOT}/phase_c" \
  --phase-a-summary "${LOCAL_ROOT}/phase_a/_analysis_local/phase_a_summary.json" \
  --output-dir "${LOCAL_ROOT}/phase_b/_analysis_local" --config "${CONFIG_PATH}"
python scripts/analyze_stage_c_five_profile_extension.py \
  --phase c --phase-a-root "${LOCAL_ROOT}/phase_a" \
  --phase-b-root "${LOCAL_ROOT}/phase_b" --phase-c-root "${LOCAL_ROOT}/phase_c" \
  --phase-a-summary "${LOCAL_ROOT}/phase_a/_analysis_local/phase_a_summary.json" \
  --phase-b-summary "${LOCAL_ROOT}/phase_b/_analysis_local/phase_b_summary.json" \
  --output-dir "${LOCAL_ROOT}/phase_c/_analysis_local" --config "${CONFIG_PATH}"
