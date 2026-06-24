#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_horizon_decoupled}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_horizon_decoupled_gate}"

echo "phase4_horizon_decoupled_progress=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "log_root=${LOG_ROOT}"

if [[ -d "${LOG_ROOT}" ]]; then
  find "${LOG_ROOT}" -maxdepth 1 -type f -name "*.log" | sort | while read -r log_file; do
    echo "== ${log_file} =="
    tail -n 8 "${log_file}"
  done
else
  echo "missing_log_root=${LOG_ROOT}"
fi

if [[ -d "${OUTPUT_ROOT}" ]]; then
  find "${OUTPUT_ROOT}" -path "*/metrics_by_target_horizon.csv" -type f | sort
else
  echo "missing_output_root=${OUTPUT_ROOT}"
fi
