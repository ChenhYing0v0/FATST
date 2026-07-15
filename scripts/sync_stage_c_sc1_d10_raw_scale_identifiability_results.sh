#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d10_raw_scale_identifiability}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='*/' \
  --include='launch_record.txt' \
  --include='_logs/*.log' \
  --include='*/matrix_cell_metrics.csv' \
  --include='*/binary_cell_metrics.csv' \
  --include='*/metadata.json' \
  --include='_analysis/replicate_metrics.csv' \
  --include='_analysis/dataset_metrics.csv' \
  --include='_analysis/gate.json' \
  --include='_analysis/research_interpretation.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_sc1_d10_sync=pass local_root=${LOCAL_ROOT}"
