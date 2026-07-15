#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d9_history_support_operator}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='launch_record.txt' \
  --include='run.log' \
  --include='unit_metrics.csv' \
  --include='group_profiles.csv' \
  --include='control_distributions.csv' \
  --include='dataset_metrics.csv' \
  --include='gate.json' \
  --include='environment.json' \
  --include='research_interpretation.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_sc1_d9_sync=pass local_root=${LOCAL_ROOT}"
