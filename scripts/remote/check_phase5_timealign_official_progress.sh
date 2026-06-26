#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_official_gate}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/${CHECKPOINT_POLICY}}"

echo "phase5_timealign_official_progress=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "log_root=${LOG_ROOT}"

if [[ -d "${LOG_ROOT}" ]]; then
  find "${LOG_ROOT}" -maxdepth 1 -type f -name "*.log" | sort | while read -r log_file; do
    echo "== ${log_file} =="
    tail -n 12 "${log_file}"
  done
else
  echo "missing_log_root=${LOG_ROOT}"
fi

if [[ -d "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" ]]; then
  echo "completed_metrics:"
  find "${OUTPUT_ROOT}/${CHECKPOINT_POLICY}" -path "*/metrics_by_target_horizon.csv" -type f | sort
else
  echo "missing_output_root=${OUTPUT_ROOT}/${CHECKPOINT_POLICY}"
fi
