#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_qdf_upstream_gate}"

echo "phase2_qdf_upstream_progress=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

echo "completed_runs=$(find "${OUTPUT_ROOT}" -type f -name run.done 2>/dev/null | wc -l | tr -d ' ')"
echo "metric_files=$(find "${OUTPUT_ROOT}" -type f -name metrics.npy 2>/dev/null | wc -l | tr -d ' ')"

if [[ -d "${LOG_ROOT}" ]]; then
  echo "recent_logs:"
  find "${LOG_ROOT}" -maxdepth 1 -type f -name '*.log' -print0 \
    | xargs -0 ls -t 2>/dev/null \
    | head -6
  latest_log="$(find "${LOG_ROOT}" -maxdepth 1 -type f -name '*.log' -print0 \
    | xargs -0 ls -t 2>/dev/null \
    | head -1 || true)"
  if [[ -n "${latest_log}" ]]; then
    echo "latest_log=${latest_log}"
    tail -40 "${latest_log}"
  fi
fi
