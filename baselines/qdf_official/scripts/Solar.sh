#!/usr/bin/env bash
set -euo pipefail

QDF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${QDF_ROOT}/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QDF_ROOT}/results/solar_fatst}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-dry-run}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2023}"

case "${MODE}" in
  dry-run|resource-smoke|run) ;;
  *) echo "unsupported MODE=${MODE}" >&2; exit 2 ;;
esac

read -r -a GPU_IDS_ARR <<< "${GPU_IDS_STR}"
[[ "${#GPU_IDS_ARR[@]}" -ge 1 ]]
[[ -s "${DATA_ROOT}/Solar/solar_AL.txt" || "${MODE}" == "dry-run" ]]

profile_fields() {
  local horizon="$1"
  case "${horizon}" in
    96)  echo "0.005 0.005 0.1 300 3 2 16" ;;
    192) echo "0.005 0.005 0.1 300 3 1 16" ;;
    336) echo "0.005 0.005 0.02 300 3 2 16" ;;
    720) echo "0.01 0.01 0.05 300 1 1 16" ;;
    *) echo "unsupported horizon=${horizon}" >&2; return 2 ;;
  esac
}

run_one() {
  local horizon="$1" gpu_id="$2"
  local lr inner_lr meta_lr warmup_steps num_tasks meta_inner_steps batch_size
  read -r lr inner_lr meta_lr warmup_steps num_tasks meta_inner_steps batch_size \
    <<< "$(profile_fields "${horizon}")"

  local run_root="${OUTPUT_ROOT}/runs/QDF__Solar__H${horizon}__seed${SEED}"
  local train_epochs=30 patience=5 max_train_batches=0 max_eval_batches=0 final_split=test
  if [[ "${MODE}" == "resource-smoke" ]]; then
    run_root="${OUTPUT_ROOT}/_resource_smoke/QDF__Solar__H${horizon}__seed${SEED}"
    train_epochs=1
    patience=1
    warmup_steps=1
    max_train_batches=2
    max_eval_batches=2
    final_split=none
  fi

  if [[ "${MODE}" == "dry-run" ]]; then
    printf 'QDF__Solar__H%s__seed%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${horizon}" "${SEED}" "${lr}" "${inner_lr}" "${meta_lr}" \
      "${warmup_steps}" "${num_tasks}" "${meta_inner_steps}" "${batch_size}"
    return 0
  fi

  mkdir -p "${run_root}"
  if [[ "${MODE}" == "run" ]] && find "${run_root}/results" -name metrics.npy -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
    echo "skip_complete run_root=${run_root}"
    return 0
  fi

  echo "run_start=$(date --iso-8601=seconds) horizon=${horizon} gpu=${gpu_id} mode=${MODE} run_root=${run_root}"
  (
    cd "${QDF_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u run.py \
      --task_name long_term_forecast_meta_ml3 \
      --is_training 1 \
      --root_path "${DATA_ROOT}/Solar/" \
      --data_path solar_AL.txt \
      --model_id "Solar_96_${horizon}" \
      --model TQNet \
      --data_id Solar \
      --data Solar \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${horizon}" \
      --enc_in 137 \
      --dec_in 137 \
      --c_out 137 \
      --factor 3 \
      --des TQNet \
      --learning_rate "${lr}" \
      --lradj type1 \
      --train_epochs "${train_epochs}" \
      --patience "${patience}" \
      --batch_size "${batch_size}" \
      --test_batch_size 1 \
      --itr 1 \
      --rec_lambda 1.0 \
      --auxi_lambda 0.0 \
      --reg_lambda 0.0 \
      --auxi_batch_size 64 \
      --fix_seed "${SEED}" \
      --checkpoints "${run_root}/checkpoints/" \
      --results "${run_root}/results/" \
      --test_results "${run_root}/test_results/" \
      --log_path "${run_root}/result_long_term_forecast.txt" \
      --rerun 0 \
      --inner_lr "${inner_lr}" \
      --meta_lr "${meta_lr}" \
      --meta_inner_steps "${meta_inner_steps}" \
      --overlap_ratio 0.0 \
      --num_tasks "${num_tasks}" \
      --max_norm 5.0 \
      --auxi_loss MSE \
      --model_type linear \
      --cycle 144 \
      --use_revin 1 \
      --dropout 0.0 \
      --first_order 1 \
      --warmup_steps "${warmup_steps}" \
      --max_train_batches "${max_train_batches}" \
      --max_eval_batches "${max_eval_batches}" \
      --final_evaluation_split "${final_split}" \
      --gpu 0
  ) >"${run_root}/stdout.log" 2>&1
  echo "run_done=$(date --iso-8601=seconds) horizon=${horizon} gpu=${gpu_id} mode=${MODE} run_root=${run_root}"
}

if [[ "${MODE}" == "dry-run" ]]; then
  for horizon in 96 192 336 720; do run_one "${horizon}" 0; done
  exit 0
fi

gpu0="${GPU_IDS_ARR[0]}"
gpu1="${GPU_IDS_ARR[$((1 % ${#GPU_IDS_ARR[@]}))]}"
gpu2="${GPU_IDS_ARR[$((2 % ${#GPU_IDS_ARR[@]}))]}"

(run_one 720 "${gpu0}") & pid0=$!
(run_one 336 "${gpu1}" && run_one 96 "${gpu1}") & pid1=$!
(run_one 192 "${gpu2}") & pid2=$!
wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
echo "qdf_solar_${MODE}_done=$(date --iso-8601=seconds) jobs=4 seed=${SEED}"
