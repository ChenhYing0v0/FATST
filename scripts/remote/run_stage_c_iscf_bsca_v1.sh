#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_bsca_v1}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_iscf_bsca_v1.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
FORMAL_TEST_ONLY="${FORMAL_TEST_ONLY:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-stage_c_iscf_bsca_v1}"
STANDARD_HORIZONS="96,192,336,720"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
test "${#GPU_IDS[@]}" -ge 1
test -s "${CONFIG}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

CONFIG_HASH="$(sha256_file "${CONFIG}")"
PROFILE_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["path"])' "${CONFIG}")"
PROFILE_HASH="$(sha256_file "${PROFILE_PATH}")"
REMOTE_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"

LINES=()
while IFS= read -r value; do LINES+=("${value}"); done < <(
  python3 -c '
import json,sys
c=json.load(open(sys.argv[1])); p=json.load(open(c["profiles"]["path"]))["dataset_profiles"]
a={x["id"]:x for x in c["arms"]}
for dataset,arm_id in c["launch_order"]:
 arm=a[arm_id]; profile=p[dataset]; rank=c["matched_ranks"][dataset][arm["rank_rule"]]
 print("\t".join(map(str,(dataset,arm_id,arm["readout_mode"],arm["policy_mode"],arm["objective_mode"],arm["partition"],arm["partition_seed"],rank,profile["patch_num"],profile["d_model"],profile["d_ff"]))))
' "${CONFIG}"
)

run_dir_for_line() {
  local dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "$1"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_training_core_complete() {
  local d; d="$(run_dir_for_line "$1")"
  [[ -s "${d}/checkpoint.pt" && -s "${d}/training_log.csv" \
    && -s "${d}/metrics_by_target_horizon.csv" \
    && -s "${d}/effective_config.json" \
    && -s "${d}/initialization_contract.json" \
    && -s "${d}/model_diagnostics.json" ]]
}

is_complete() {
  local d; d="$(run_dir_for_line "$1")"
  is_training_core_complete "$1" \
    && [[ -s "${d}/pcsd_validation_diagnostics.npz" \
    && -s "${d}/trained_invariants.json" ]]
}

is_test_complete() {
  local d; d="$(run_dir_for_line "$1")"
  [[ -s "${d}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${d}/test_audit_invariants.json" \
    && -s "${d}/pcsd_test_audit_diagnostics.npz" ]] \
    && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' \
      "${d}/test_audit_invariants.json" 2>/dev/null
}

status_counts() {
  local trained=0 tested=0 line
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then trained=$((trained + 1)); fi
    if is_test_complete "${line}"; then tested=$((tested + 1)); fi
  done
  echo "${trained} ${tested}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  read -r trained tested <<< "$(status_counts)"
  echo "bsca_status=$(date -Is) training=${trained}/${#LINES[@]} test=${tested}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 2>/dev/null | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "bsca_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

[[ "${REMOTE_AUTHORIZED}" == "true" ]] || { echo "remote training not authorized" >&2; exit 3; }
if [[ "${RESOURCE_SMOKE}" == "1" && "${FORMAL_TEST_ONLY}" == "1" ]]; then
  echo "RESOURCE_SMOKE and FORMAL_TEST_ONLY are mutually exclusive" >&2; exit 2
fi
if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
  [[ "${TEST_AUTHORIZED}" == "true" ]] || { echo "formal test not authorized" >&2; exit 3; }
  read -r trained _tested <<< "$(status_counts)"
  [[ "${trained}" -eq "${#LINES[@]}" ]] || { echo "formal test blocked until 5/5 training: ${trained}/${#LINES[@]}" >&2; exit 4; }
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" log="$4" smoke="$5"
  local dataset arm readout policy objective partition partition_seed rank patch_num d_model d_ff
  local final_args=()
  IFS=$'\t' read -r dataset arm readout policy objective partition partition_seed rank patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    final_args=(--max-train-batches 2 --max-eval-batches 2 --epochs 1 --patience 1 --final-evaluation-split none)
  else
    final_args=(--epochs "${EPOCHS}" --patience "${PATIENCE}" --final-evaluation-split val)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons "${STANDARD_HORIZONS}" --evaluation-horizons "${STANDARD_HORIZONS}" \
      --segment-horizons "${STANDARD_HORIZONS}" --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 --enable-early-stopping \
      --early-stopping-min-delta 0 --seed "${SEED}" --num-workers 0 \
      --run-name "ISCF_BSCA_${arm}" --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" --legacy-dropout 0.1 \
      --legacy-layer-norm 1 --learning-rate 0.0001 --readout-mode "${readout}" --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" --pcsd-policy-history-dim 32 \
      --pcsd-policy-hidden-dim 64 --pcsd-policy-mode "${policy}" --pcsd-fixed-scale 720 \
      --pcsd-partition "${partition}" --pcsd-partition-seed "${partition_seed}" \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --pcc-objective-mode "${objective}" --pred-loss-mode full --no-save-predictions \
      "${final_args[@]}" >"${log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  line="${LINES[0]}"; gpu="${GPU_IDS[0]}"; d="${OUTPUT_ROOT}/_resource_smoke/Weather_seed${SEED}"
  mkdir -p "${d}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  run_training_command "${line}" "${gpu}" "${d}" "${d}/smoke.log" 1
  test -s "${d}/training_log.csv"; test -s "${d}/effective_config.json"
  failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
  if command -v rg >/dev/null 2>&1; then
    ! rg -ni "${failure_pattern}" "${d}/smoke.log"
  else
    ! grep -Ein "${failure_pattern}" "${d}/smoke.log"
  fi
  echo "bsca_resource_smoke_done=$(date -Is) output=${d}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"; mkdir -p "${LOG_ROOT}"
MODE=training; [[ "${FORMAL_TEST_ONLY}" == "1" ]] && MODE=formal_test
{
  echo "bsca_start=$(date -Is)"; echo "commit=$(git rev-parse HEAD)"; echo "output_root=${OUTPUT_ROOT}"
  echo "config_hash=${CONFIG_HASH}"; echo "profile_hash=${PROFILE_HASH}"; echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"; echo "formal_test_execution_mode=${FORMAL_TEST_ONLY}"; echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/${MODE}_jobs_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3" dataset arm rest d log before after
  IFS=$'\t' read -r dataset arm rest <<< "${line}"; d="$(run_dir_for_line "${line}")"
  log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
    if is_test_complete "${line}"; then echo "skip_existing_test ${arm} ${dataset}"; return; fi
    before="$(sha256_file "${d}/checkpoint.pt")"
    echo "test_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py --run-dir "${d}" \
        --design "${CONFIG}" --test-audit-config "${CONFIG}" --evaluation-split test \
        --probe-rows 256 --device cuda >>"${log}" 2>&1
    after="$(sha256_file "${d}/checkpoint.pt")"; [[ "${before}" == "${after}" ]]
    echo "test_done=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} gpu=${gpu}"
  else
    if is_complete "${line}"; then echo "skip_existing ${arm} ${dataset}"; return; fi
    mkdir -p "${d}"
    if is_training_core_complete "${line}"; then
      echo "validation_replay_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} gpu=${gpu}"
    else
      echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} gpu=${gpu}"
      run_training_command "${line}" "${gpu}" "${d}" "${log}" 0
    fi
    before="$(sha256_file "${d}/checkpoint.pt")"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py --run-dir "${d}" \
        --design "${CONFIG}" --test-audit-config "${CONFIG}" --evaluation-split val \
        --probe-rows 256 --device cuda >>"${log}" 2>&1
    after="$(sha256_file "${d}/checkpoint.pt")"; [[ "${before}" == "${after}" ]]
    echo "train_done=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} gpu=${gpu}"
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#LINES[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${LINES[${index}]}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
[[ "${status}" == "0" ]]; echo "bsca_${MODE}_done=$(date -Is)"
