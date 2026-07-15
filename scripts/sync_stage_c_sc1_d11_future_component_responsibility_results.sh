#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d11_future_component_responsibility}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_sc1_d11_future_component_responsibility_20260715/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='*/' \
  --include='launch_record.txt' \
  --include='_logs/*.log' \
  --include='*/total_gradient_metrics.csv' \
  --include='*/component_metrics.csv' \
  --include='*/component_group_metrics.csv' \
  --include='*/reachability_metrics.csv' \
  --include='*/metadata.json' \
  --include='_analysis/*.csv' \
  --include='_analysis/gate.json' \
  --include='_analysis/research_interpretation.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_sc1_d11_sync=pass local_root=${LOCAL_ROOT}"
