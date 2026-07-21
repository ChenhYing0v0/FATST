#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_v3_tsaf_v1}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_siff_v3_tsaf_step7b.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="stage_c_siff_v3_tsaf_v1"
STANDARD_HORIZONS="96,192,336,720"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
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
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
profiles=json.load(open(config["profiles"]["path"]))["dataset_profiles"]
arms={arm["id"]: arm for arm in config["effective_arms"]}
for dataset,arm_id in config["launch_order"]:
    arm=arms[arm_id]
    if arm["source"] != "new_training":
        raise ValueError(f"launch_order contains reused arm: {arm_id}")
    profile=profiles[dataset]
    rule=arm["rank_rule"]
    rank=256 if rule == "fixed_256" else config["matched_ranks"][dataset][rule]
    print("\t".join(map(str,(
        dataset,arm_id,arm["readout_mode"],arm["policy_mode"],
        arm["objective_mode"],rank,profile["profile"],
        profile["patch_num"],profile["d_model"],profile["d_ff"],
    ))))
' "${CONFIG}"
)

run_dir_for_line() {
  local line="$1" dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local line="$1" output_dir
  output_dir="$(run_dir_for_line "${line}")"
  [[ -s "${output_dir}/checkpoint.pt" \
    && -s "${output_dir}/training_log.csv" \
    && -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/effective_config.json" \
    && -s "${output_dir}/initialization_contract.json" \
    && -s "${output_dir}/model_diagnostics.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "tsaf_status=$(date -Is) completed=${completed}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 \
    2>/dev/null | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "tsaf_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${REMOTE_AUTHORIZED}" != "true" ]]; then
  echo "TSAF remote launch is not authorized by ${CONFIG}" >&2
  exit 3
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4" smoke="$5"
  local dataset arm readout policy objective rank profile patch_num d_model d_ff
  local run_args=()
  IFS=$'\t' read -r dataset arm readout policy objective rank profile \
    patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(
      --max-train-batches 2 --max-eval-batches 2 --epochs 1 --patience 1
      --final-evaluation-split none
    )
  else
    run_args=(
      --epochs "${EPOCHS}" --patience "${PATIENCE}"
      --final-evaluation-split val
    )
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
      --num-workers 0 --run-name "TSAF_${arm}" \
      --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening \
      --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode "${readout}" --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" \
      --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode "${policy}" --pcsd-fixed-scale 720 \
      --pcsd-partition canonical --pcsd-partition-seed 15101 \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --pcc-objective-mode "${objective}" --pred-loss-mode full \
      --no-save-predictions "${run_args[@]}" >"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_lines=()
  for line in "${LINES[@]}"; do
    if [[ "${line}" == $'Weather\ttsaf\t'* \
      || "${line}" == $'ETTm2\tsiff_independent_target_only\t'* ]]; then
      smoke_lines+=("${line}")
    fi
  done
  test "${#smoke_lines[@]}" -eq 2
  smoke_root="${OUTPUT_ROOT}/_resource_smoke"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  smoke_pids=()
  for index in "${!smoke_lines[@]}"; do
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    dataset="$(cut -f1 <<< "${smoke_lines[${index}]}")"
    arm="$(cut -f2 <<< "${smoke_lines[${index}]}")"
    output_dir="${smoke_root}/${arm}_${dataset}_seed${SEED}"
    mkdir -p "${output_dir}"
    run_training_command \
      "${smoke_lines[${index}]}" "${gpu}" "${output_dir}" \
      "${output_dir}/smoke.log" 1 &
    smoke_pids+=("$!")
  done
  for pid in "${smoke_pids[@]}"; do wait "${pid}"; done
  for directory in "${smoke_root}"/*_seed"${SEED}"; do
    test -s "${directory}/training_log.csv"
    test -s "${directory}/effective_config.json"
    failure_pattern="Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)"
    if command -v rg >/dev/null 2>&1; then
      ! rg -ni "${failure_pattern}" "${directory}/smoke.log"
    else
      ! grep -Ein "${failure_pattern}" "${directory}/smoke.log"
    fi
  done
  echo "tsaf_resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"
mkdir -p "${LOG_ROOT}"
{
  echo "tsaf_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "new_training_runs=${#LINES[@]}"
  echo "reused_reference_runs=20"
  echo "effective_runs=45"
  echo "checkpoint_selection=best_val_mean_mse_h96_h192_h336_h720"
  echo "formal_test_executed=false"
  echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm readout policy objective rank profile patch_num d_model d_ff
  local output_dir run_log
  IFS=$'\t' read -r dataset arm readout policy objective rank profile \
    patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu} profile=${profile} rank=${rank}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  echo "run_done=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
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
  echo "worker_start=$(date -Is) worker=${index} gpu=${GPU_IDS[${index}]} pid=$!"
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" != "0" ]]; then
  echo "tsaf_worker_failure=$(date -Is)" >&2
  exit 1
fi
echo "tsaf_training_done=$(date -Is) formal_test_executed=false"
