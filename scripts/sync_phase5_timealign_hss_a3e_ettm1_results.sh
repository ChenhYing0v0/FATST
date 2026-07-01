#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_$(date '+%Y%m%d')}"
SEED="${SEED:-2021}"

sync_tree() {
  local label="$1"
  local remote_path="$2"
  local local_path="${ANALYSIS_ROOT}/raw/${label}"
  mkdir -p "${local_path}"
  rsync -av \
    --include='*/' \
    --include='metrics_by_target_horizon.csv' \
    --include='metrics_by_segment.csv' \
    --include='training_log.csv' \
    --include='effective_config.json' \
    --include='environment.json' \
    --exclude='*' \
    "${REMOTE_HOST}:${remote_path}/" \
    "${local_path}/"
}

echo "sync_phase5_timealign_hss_a3e_ettm1_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "remote_host=${REMOTE_HOST}"
echo "remote_root=${REMOTE_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "analysis_root=${ANALYSIS_ROOT}"

mkdir -p "${ANALYSIS_ROOT}/raw"

sync_tree "official" "${REMOTE_ROOT}/phase5_timealign_official_gate"
sync_tree "h1" "${REMOTE_ROOT}/phase5_timealign_hss_h1_readout_gate"
sync_tree "h1c" "${REMOTE_ROOT}/phase5_timealign_hss_h1c_capacity_preserving_gate"
sync_tree "a2" "${REMOTE_ROOT}/phase5_timealign_hss_a2_interface_gate"
sync_tree "a3c" "${REMOTE_ROOT}/phase5_timealign_hss_a3c_warm_started_nested_gate"
sync_tree "a3d" "${REMOTE_ROOT}/phase5_timealign_hss_a3d_teacher_preserved_nested_gate"
sync_tree "a3e" "${REMOTE_ROOT}/phase5_timealign_hss_a3e_target_conditioned_nested_gate"

python scripts/analyze_phase5_timealign_hss_a3e_ettm1_replacement_gate.py \
  --raw-root "${ANALYSIS_ROOT}/raw" \
  --output-dir "${ANALYSIS_ROOT}" \
  --checkpoint-policy "${CHECKPOINT_POLICY}" \
  --seed "${SEED}"

echo "sync_phase5_timealign_hss_a3e_ettm1_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
