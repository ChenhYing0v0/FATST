#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions}"
export RUN_NAME="${RUN_NAME:-PatchEncoderPrefixRiskWeighted}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_qdf_alignment_r3_predictions}"
export DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
export TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
export SEED="${SEED:-2021}"
export EPOCHS="${EPOCHS:-100}"

IFS="," read -r -a horizon_array <<< "${TARGET_HORIZONS}"
horizon_label="mixed"
for horizon in "${horizon_array[@]}"; do
  horizon_label="${horizon_label}_h${horizon}"
done

read -r -a datasets <<< "${DATASETS}"

echo "check_time=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "run_name=${RUN_NAME}"
echo "datasets=${DATASETS}"

completed=0
running=0
queued=0
total="${#datasets[@]}"

for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  position=$((index + 1))
  run_dir="${OUTPUT_ROOT}/${RUN_NAME}/${dataset}/${horizon_label}/seed${SEED}"
  dataset_log="${LOG_ROOT}/${RUN_NAME}_${dataset}_mixed_seed${SEED}.log"
  metrics_file="${run_dir}/metrics_by_target_horizon.csv"
  prediction_count="$(find "${run_dir}" -path "*/h*/predictions_test.npz" -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -s "${metrics_file}" && "${prediction_count}" == "${#horizon_array[@]}" ]]; then
    completed=$((completed + 1))
    echo "dataset_progress position=${position}/${total} dataset=${dataset} status=completed epoch=${EPOCHS}/${EPOCHS} predictions=${prediction_count}/${#horizon_array[@]}"
    continue
  fi
  if [[ -s "${dataset_log}" ]]; then
    running=$((running + 1))
    last_progress="$(grep 'epoch_progress' "${dataset_log}" | tail -n 1 || true)"
    if [[ -n "${last_progress}" ]]; then
      echo "dataset_progress position=${position}/${total} dataset=${dataset} status=running ${last_progress#epoch_progress } predictions=${prediction_count}/${#horizon_array[@]}"
    else
      echo "dataset_progress position=${position}/${total} dataset=${dataset} status=running epoch=unknown/${EPOCHS} predictions=${prediction_count}/${#horizon_array[@]}"
    fi
    continue
  fi
  queued=$((queued + 1))
  echo "dataset_progress position=${position}/${total} dataset=${dataset} status=queued epoch=0/${EPOCHS} predictions=${prediction_count}/${#horizon_array[@]}"
done

echo "matrix_progress completed=${completed}/${total} running=${running} queued=${queued}"
