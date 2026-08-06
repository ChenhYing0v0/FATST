#!/usr/bin/env bash
set -euo pipefail

QDF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${QDF_ROOT}/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QDF_ROOT}/results/main_i_l336}"
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

dataset_fields() {
  case "$1" in
    ETTh1) echo "ETTh1 ETT-small ETTh1.csv 7 24 0.5 ETTh1" ;;
    ETTh2) echo "ETTh2 ETT-small ETTh2.csv 7 24 0.5 ETTh2" ;;
    ETTm1) echo "ETTm1 ETT-small ETTm1.csv 7 96 0.5 ETTm1" ;;
    ETTm2) echo "ETTm2 ETT-small ETTm2.csv 7 96 0.5 ETTm2" ;;
    Weather) echo "custom weather weather.csv 21 144 0.5 Weather" ;;
    ECL) echo "custom electricity electricity.csv 321 168 0.0 ECL" ;;
    Solar) echo "Solar Solar solar_AL.txt 137 144 0.0 Solar" ;;
    Exchange) echo "custom exchange_rate exchange_rate.csv 8 1 0.5 Exchange" ;;
    *) echo "unsupported dataset=$1" >&2; return 2 ;;
  esac
}

profile_fields() {
  local dataset="$1" horizon="$2" source_dataset="${dataset}"
  [[ "${dataset}" == "Solar" ]] && source_dataset="ECL"
  [[ "${dataset}" == "Exchange" ]] && source_dataset="ETTh1"
  case "${source_dataset}:${horizon}" in
    ETTh1:96) echo "0.005 0.005 0.2 300 3 3 32" ;;
    ETTh1:192) echo "0.005 0.005 0.02 300 1 3 32" ;;
    ETTh1:336) echo "0.002 0.002 0.01 300 3 4 32" ;;
    ETTh1:720) echo "0.005 0.005 0.01 300 5 2 32" ;;
    ETTh2:96) echo "0.0005 0.0005 0.2 300 1 2 32" ;;
    ETTh2:192) echo "0.0005 0.0005 0.01 300 4 1 32" ;;
    ETTh2:336) echo "0.0005 0.0005 0.2 300 1 2 32" ;;
    ETTh2:720) echo "0.0005 0.0005 0.2 300 2 4 32" ;;
    ETTm1:96) echo "0.001 0.001 0.1 500 2 1 32" ;;
    ETTm1:192) echo "0.001 0.001 0.01 500 2 3 32" ;;
    ETTm1:336) echo "0.001 0.001 0.2 500 4 1 32" ;;
    ETTm1:720) echo "0.001 0.001 0.2 500 2 5 32" ;;
    ETTm2:96) echo "0.0002 0.0002 0.2 500 1 5 32" ;;
    ETTm2:192) echo "0.0002 0.0002 0.2 500 1 2 32" ;;
    ETTm2:336) echo "0.0002 0.0002 0.1 500 3 4 32" ;;
    ETTm2:720) echo "0.0002 0.0002 0.2 500 5 5 32" ;;
    Weather:96) echo "0.002 0.002 0.1 700 5 4 32" ;;
    Weather:192) echo "0.002 0.002 0.02 700 3 1 32" ;;
    Weather:336) echo "0.002 0.002 0.05 700 4 5 32" ;;
    Weather:720) echo "0.002 0.002 0.2 700 3 4 32" ;;
    ECL:96) echo "0.005 0.005 0.1 300 3 2 16" ;;
    ECL:192) echo "0.005 0.005 0.1 300 3 1 16" ;;
    ECL:336) echo "0.005 0.005 0.02 300 3 2 16" ;;
    ECL:720) echo "0.01 0.01 0.05 300 1 1 16" ;;
    *) echo "unsupported profile=${dataset}:${horizon}" >&2; return 2 ;;
  esac
}

run_one() {
  local dataset="$1" horizon="$2" gpu_id="$3"
  local data_name relative_root data_path channels cycle dropout data_id
  local lr inner_lr meta_lr warmup_steps num_tasks meta_inner_steps batch_size
  read -r data_name relative_root data_path channels cycle dropout data_id <<< "$(dataset_fields "${dataset}")"
  read -r lr inner_lr meta_lr warmup_steps num_tasks meta_inner_steps batch_size <<< "$(profile_fields "${dataset}" "${horizon}")"

  local run_root="${OUTPUT_ROOT}/runs/QDF__${dataset}__H${horizon}__seed${SEED}"
  local train_epochs=30 patience=5 max_train_batches=0 max_eval_batches=0 final_split=test is_training=1
  if [[ "${MODE}" == "resource-smoke" ]]; then
    run_root="${OUTPUT_ROOT}/_resource_smoke/QDF__${dataset}__H${horizon}__seed${SEED}"
    train_epochs=1; patience=1; warmup_steps=1
    max_train_batches=2; max_eval_batches=2; final_split=none
  fi

  if [[ "${MODE}" == "dry-run" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "QDF__${dataset}__H${horizon}__seed${SEED}" "${data_name}" "${channels}" "${cycle}" "${dropout}" \
      "${lr}" "${inner_lr}" "${meta_lr}" "${warmup_steps}" "${num_tasks}" "${meta_inner_steps}" "${batch_size}"
    return 0
  fi
  [[ -s "${DATA_ROOT}/${relative_root}/${data_path}" ]]
  mkdir -p "${run_root}"
  if [[ "${MODE}" == "run" ]] && find "${run_root}/results" -name metrics.npy -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
    echo "skip_complete dataset=${dataset} horizon=${horizon}"
    return 0
  fi
  if [[ "${MODE}" == "run" ]] \
    && [[ "$(find "${run_root}/checkpoints" -name checkpoint.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]] \
    && [[ "$(find "${run_root}/checkpoints" -name A.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]]; then
    is_training=0
    if [[ -s "${run_root}/stdout.log" && ! -e "${run_root}/training_stdout_before_test_retry.log" ]]; then
      mv "${run_root}/stdout.log" "${run_root}/training_stdout_before_test_retry.log"
    fi
    echo "evaluation_only_retry dataset=${dataset} horizon=${horizon} checkpoint_retrained=false"
  fi

  echo "run_start=$(date --iso-8601=seconds) dataset=${dataset} horizon=${horizon} gpu=${gpu_id} mode=${MODE}"
  (
    cd "${QDF_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u run.py \
      --task_name long_term_forecast_meta_ml3 --is_training "${is_training}" \
      --root_path "${DATA_ROOT}/${relative_root}/" --data_path "${data_path}" \
      --model_id "${dataset}_336_${horizon}" --model TQNet --data_id "${data_id}" --data "${data_name}" --features M \
      --seq_len 336 --label_len 48 --pred_len "${horizon}" \
      --enc_in "${channels}" --dec_in "${channels}" --c_out "${channels}" \
      --factor 3 --des TQNet --learning_rate "${lr}" --lradj type1 \
      --train_epochs "${train_epochs}" --patience "${patience}" --batch_size "${batch_size}" \
      --test_batch_size 1 --num_workers 0 --itr 1 --rec_lambda 1.0 --auxi_lambda 0.0 \
      --reg_lambda 0.0 --auxi_batch_size 64 --fix_seed "${SEED}" \
      --checkpoints "${run_root}/checkpoints/" --results "${run_root}/results/" \
      --test_results "${run_root}/test_results/" --log_path "${run_root}/result_long_term_forecast.txt" \
      --rerun 0 --inner_lr "${inner_lr}" --meta_lr "${meta_lr}" --meta_inner_steps "${meta_inner_steps}" \
      --overlap_ratio 0.0 --num_tasks "${num_tasks}" --max_norm 5.0 --auxi_loss MSE \
      --model_type linear --cycle "${cycle}" --use_revin 1 --dropout "${dropout}" --first_order 1 \
      --warmup_steps "${warmup_steps}" --max_train_batches "${max_train_batches}" \
      --max_eval_batches "${max_eval_batches}" --final_evaluation_split "${final_split}" --gpu 0
  ) >"${run_root}/stdout.log" 2>&1
  echo "run_done=$(date --iso-8601=seconds) dataset=${dataset} horizon=${horizon} gpu=${gpu_id} mode=${MODE}"
}

if [[ "${MODE}" == "dry-run" ]]; then
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Solar Exchange; do
    for horizon in 96 192 336 720; do run_one "${dataset}" "${horizon}" 0; done
  done
  exit 0
fi

gpu0="${GPU_IDS_ARR[0]}"; gpu1="${GPU_IDS_ARR[$((1 % ${#GPU_IDS_ARR[@]}))]}"; gpu2="${GPU_IDS_ARR[$((2 % ${#GPU_IDS_ARR[@]}))]}"
if [[ "${MODE}" == "resource-smoke" ]]; then
  (run_one ECL 720 "${gpu0}" && run_one ETTh1 720 "${gpu0}" && run_one ETTh2 720 "${gpu0}") & pid0=$!
  (run_one Solar 720 "${gpu1}" && run_one Exchange 720 "${gpu1}") & pid1=$!
  (run_one Weather 720 "${gpu2}" && run_one ETTm1 720 "${gpu2}" && run_one ETTm2 720 "${gpu2}") & pid2=$!
else
  (for job in ECL:720 ECL:336 ECL:192 ECL:96 ETTh1:720 ETTh1:336 ETTh1:192 ETTh1:96; do run_one "${job%:*}" "${job#*:}" "${gpu0}"; done) & pid0=$!
  (for job in Solar:720 Solar:336 Solar:192 Solar:96 Exchange:720 Exchange:336 Exchange:192 Exchange:96 ETTh2:720 ETTh2:336 ETTh2:192 ETTh2:96; do run_one "${job%:*}" "${job#*:}" "${gpu1}"; done) & pid1=$!
  (for job in Weather:720 Weather:336 Weather:192 Weather:96 ETTm1:720 ETTm1:336 ETTm1:192 ETTm1:96 ETTm2:720 ETTm2:336 ETTm2:192 ETTm2:96; do run_one "${job%:*}" "${job#*:}" "${gpu2}"; done) & pid2=$!
fi
wait "${pid0}"; wait "${pid1}"; wait "${pid2}"
echo "qdf_main_i_l336_${MODE}_done=$(date --iso-8601=seconds) seed=${SEED}"
