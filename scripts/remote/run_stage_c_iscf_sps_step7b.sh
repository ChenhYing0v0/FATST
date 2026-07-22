#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_sps_v0_step7b}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_iscf_sps_step7b.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-val}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-stage_c_iscf_sps_v0_step7b}"
RUN_LABEL="${RUN_LABEL:-SPS}"
STANDARD_HORIZONS="96,192,336,720"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

test "${#GPU_IDS[@]}" -ge 1
test -s "${CONFIG}"
if [[ "${EVALUATION_SPLIT}" != "val" ]]; then
  echo "SPS Step7B is validation-only; requested split=${EVALUATION_SPLIT}" >&2
  exit 3
fi

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
config=json.load(open(sys.argv[1]))
profiles=json.load(open(config["profiles"]["path"]))["dataset_profiles"]
arms={arm["id"]: arm for arm in config["arms"]}
for dataset,arm_id in config["launch_order"]:
    arm=arms[arm_id]
    profile=profiles[dataset]
    rank=config["matched_ranks"][dataset][arm["rank_rule"]]
    print("\t".join(map(str,(
        dataset,arm_id,arm["readout_mode"],arm["projection_mode"],
        arm["partition"],rank,profile["profile"],profile["patch_num"],
        profile["d_model"],profile["d_ff"],
        arm.get("conditioning_strength", 1.0),
        arm.get("objective_mode", "equal_skill"),
    ))))
' "${CONFIG}"
)

run_dir_for_line() {
  local dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "$1"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local output_dir
  output_dir="$(run_dir_for_line "$1")"
  [[ -s "${output_dir}/checkpoint.pt" \
    && -s "${output_dir}/training_log.csv" \
    && -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/effective_config.json" \
    && -s "${output_dir}/initialization_contract.json" \
    && -s "${output_dir}/model_diagnostics.json" \
    && -s "${output_dir}/pcsd_validation_diagnostics.npz" \
    && -s "${output_dir}/trained_invariants.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  complete=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then complete=$((complete + 1)); fi
  done
  echo "sps_status=$(date -Is) validation=${complete}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 \
    2>/dev/null | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "sps_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED} split=${EVALUATION_SPLIT}"
  exit 0
fi

if [[ "${REMOTE_AUTHORIZED}" != "true" ]]; then
  echo "SPS Step7B remote launch is not authorized by ${CONFIG}" >&2
  exit 3
fi
if [[ "${TEST_AUTHORIZED}" == "true" ]]; then
  echo "SPS Step7B config must keep formal test access disabled" >&2
  exit 3
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4" smoke="$5"
  local dataset arm readout projection partition rank profile patch_num d_model d_ff conditioning_strength objective_mode
  local run_args=()
  IFS=$'\t' read -r dataset arm readout projection partition rank profile \
    patch_num d_model d_ff conditioning_strength objective_mode <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(--max-train-batches 2 --max-eval-batches 2 --epochs 1 \
      --patience 1 --final-evaluation-split none)
  else
    run_args=(--epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --final-evaluation-split val)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons "${STANDARD_HORIZONS}" \
      --evaluation-horizons "${STANDARD_HORIZONS}" \
      --segment-horizons "${STANDARD_HORIZONS}" \
      --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "${RUN_LABEL}_${arm}" \
      --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode "${readout}" --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" \
      --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode direct --pcsd-fixed-scale 720 \
      --pcsd-partition "${partition}" --pcsd-partition-seed 15101 \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --sps-projection-mode "${projection}" \
      --frsc-conditioning-strength "${conditioning_strength}" \
      --pcc-objective-mode "${objective_mode}" --pred-loss-mode full \
      --no-save-predictions "${run_args[@]}" >"${run_log}" 2>&1
}

run_validation_diagnostics() {
  local gpu="$1" output_dir="$2" run_log="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${output_dir}" --design "${CONFIG}" \
      --test-audit-config "${CONFIG}" --evaluation-split val \
      --probe-rows 256 --device cuda >>"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_root="${OUTPUT_ROOT}/_resource_smoke"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  for index in 0 1; do
    line="${LINES[${index}]}"
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    dataset="$(cut -f1 <<< "${line}")"
    arm="$(cut -f2 <<< "${line}")"
    output_dir="${smoke_root}/${arm}_${dataset}_seed${SEED}"
    mkdir -p "${output_dir}"
    run_training_command "${line}" "${gpu}" "${output_dir}" \
      "${output_dir}/smoke.log" 1
    test -s "${output_dir}/training_log.csv"
    failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
    if command -v rg >/dev/null 2>&1; then
      ! rg -ni "${failure_pattern}" "${output_dir}/smoke.log"
    else
      ! grep -Ein "${failure_pattern}" "${output_dir}/smoke.log"
    fi
  done
  echo "sps_resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"
mkdir -p "${LOG_ROOT}"
LAUNCH_RECORD="${OUTPUT_ROOT}/training_launch_record_seed${SEED}.txt"
JOB_RECORD="${OUTPUT_ROOT}/training_jobs_seed${SEED}.tsv"
{
  echo "sps_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "validation_runs=${#LINES[@]}"
  echo "formal_test_access=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${LAUNCH_RECORD}"
printf '%s\n' "${LINES[@]}" >"${JOB_RECORD}"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm rest output_dir run_log
  IFS=$'\t' read -r dataset arm rest <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset}"
    return
  fi
  mkdir -p "${output_dir}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  run_validation_diagnostics "${gpu}" "${output_dir}" "${run_log}"
  echo "train_done=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index
  for ((line_index=WORKER_OFFSET + worker_index; \
    line_index<${#LINES[@]}; line_index+=WORKER_STRIDE)); do
    run_one "${line_index}" "${LINES[${line_index}]}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
[[ "${status}" == "0" ]]
echo "sps_validation_done=$(date -Is)"
