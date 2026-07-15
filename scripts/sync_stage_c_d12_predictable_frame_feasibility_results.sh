#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d12_predictable_frame_feasibility_v2}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_d12_predictable_frame_feasibility_20260715/raw_v2}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='*/' \
  --include='launch_record.txt' \
  --include='_logs/*.log' \
  --include='*/fold_metrics.csv' \
  --include='*/subspace_metrics.csv' \
  --include='*/dataset_summary.json' \
  --include='*/metadata.json' \
  --include='*/fold*_training_history.csv' \
  --include='_analysis/*.csv' \
  --include='_analysis/d12_a_gate.json' \
  --include='_analysis/d12_a_result_report.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_d12_sync=pass local_root=${LOCAL_ROOT}"
