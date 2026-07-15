#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a1_dual_carrier_grouped_mlp}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av \
  --include='*/' \
  --include='launch_*.txt' \
  --include='jobs_*.tsv' \
  --include='_logs_*/*.log' \
  --include='*/effective_config.json' \
  --include='*/environment.json' \
  --include='*/training_log.csv' \
  --include='*/metrics_by_target_horizon.csv' \
  --include='*/metrics_by_segment.csv' \
  --include='*/model_diagnostics.json' \
  --include='*/trained_invariants.json' \
  --include='*/validation_diagnostics.npz' \
  --include='_analysis_*/dataset_metrics.csv' \
  --include='_analysis_*/gate.json' \
  --include='_analysis_*/research_interpretation.md' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"

echo "stage_c_d14a1_sync=pass local_root=${LOCAL_ROOT}"
