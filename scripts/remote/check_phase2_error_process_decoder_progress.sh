#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_error_process_decoder}"
RUN_NAME="${RUN_NAME:-PatchEncoderErrorProcessDecoder}"
DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-100}"
PID="${PID:-}"

IFS="," read -r -a horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

log_root="${OUTPUT_ROOT}/_logs/phase2_error_process_decoder_gate"
outer_log="${OUTPUT_ROOT}/_logs/phase2_error_process_decoder_outer.log"
read -r -a datasets <<< "${DATASETS}"

finish_time() {
  local eta_sec="$1"
  local python_bin="${PYTHON_BIN:-}"
  if [[ -z "${python_bin}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
      python_bin="$(command -v python)"
    elif [[ -x "/home/anaconda3/bin/python" ]]; then
      python_bin="/home/anaconda3/bin/python"
    else
      echo "unknown"
      return 0
    fi
  fi
  "${python_bin}" - "$eta_sec" <<'PY'
from __future__ import annotations

from datetime import datetime, timedelta
import sys

eta = float(sys.argv[1])
print((datetime.now().astimezone() + timedelta(seconds=eta)).isoformat(timespec="seconds"))
PY
}

extract_value() {
  local line="$1"
  local key="$2"
  awk -v key="${key}" '{
    for (i = 1; i <= NF; i++) {
      split($i, pair, "=")
      if (pair[1] == key) {
        print pair[2]
        exit
      }
    }
  }' <<< "${line}"
}

echo "check_time=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "run_name=${RUN_NAME}"
echo "datasets=${DATASETS}"
echo "pid=${PID:-unknown}"
if [[ -n "${PID}" ]]; then
  ps -p "${PID}" -o pid,stat,etime,cmd || true
fi

completed=0
running=0
queued=0
total="${#datasets[@]}"

for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  position=$((index + 1))
  run_dir="${OUTPUT_ROOT}/${RUN_NAME}/${dataset}/${horizon_label}/seed${SEED}"
  dataset_log="${log_root}/${RUN_NAME}_${dataset}_mixed_seed${SEED}.log"
  metrics_file="${run_dir}/metrics_by_target_horizon.csv"
  if [[ -s "${metrics_file}" ]]; then
    completed=$((completed + 1))
    echo "dataset_progress position=${position}/${total} dataset=${dataset} status=completed epoch=${EPOCHS}/${EPOCHS}"
    continue
  fi
  if [[ -s "${dataset_log}" ]]; then
    running=$((running + 1))
    last_progress="$(grep 'epoch_progress' "${dataset_log}" | tail -n 1 || true)"
    if [[ -n "${last_progress}" ]]; then
      epoch="$(extract_value "${last_progress}" "epoch")"
      eta_sec="$(extract_value "${last_progress}" "eta_sec")"
      finish_at="$(finish_time "${eta_sec}")"
      echo "dataset_progress position=${position}/${total} dataset=${dataset} status=running epoch=${epoch} eta_sec=${eta_sec} finish_at=${finish_at}"
    else
      echo "dataset_progress position=${position}/${total} dataset=${dataset} status=running epoch=unknown/${EPOCHS} eta_sec=unknown finish_at=unknown"
    fi
    continue
  fi
  queued=$((queued + 1))
  echo "dataset_progress position=${position}/${total} dataset=${dataset} status=queued epoch=0/${EPOCHS}"
done

echo "matrix_progress completed=${completed}/${total} running=${running} queued=${queued}"

if [[ -s "${outer_log}" ]]; then
  echo "outer_log_tail_start"
  tail -n 20 "${outer_log}"
  echo "outer_log_tail_end"
fi
