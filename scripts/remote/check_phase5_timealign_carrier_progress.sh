#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_carrier_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase5_timealign_carrier_gate}"

echo "phase5_timealign_carrier_progress=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "log_root=${LOG_ROOT}"

if [[ -d "${LOG_ROOT}" ]]; then
  find "${LOG_ROOT}" -maxdepth 1 -type f -name "*.log" | sort | while read -r log_file; do
    echo "== ${log_file} =="
    tail -n 10 "${log_file}"
  done
else
  echo "missing_log_root=${LOG_ROOT}"
fi

if [[ -d "${OUTPUT_ROOT}" ]]; then
  echo "completed_metrics:"
  find "${OUTPUT_ROOT}" -path "*/metrics_by_target_horizon.csv" -type f | sort
else
  echo "missing_output_root=${OUTPUT_ROOT}"
fi
