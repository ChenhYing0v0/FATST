#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a_output_coupling_granularity}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_d14a_output_coupling_granularity_20260715/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='*/' \
  --include='launch_record.txt' \
  --include='_logs/*.log' \
  --include='*/fold_metrics.csv' \
  --include='*/parameter_budget.csv' \
  --include='*/metadata.json' \
  --include='*/validation_bin_losses_fold*.npz' \
  --include='_analysis/fold_gate_metrics.csv' \
  --include='_analysis/crossing_metrics.csv' \
  --include='_analysis/dataset_metrics.csv' \
  --include='_analysis/gate.json' \
  --include='_analysis/research_interpretation.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_d14a_sync=pass local_root=${LOCAL_ROOT}"
