#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/stage_c_siff_ccsf_temperature_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["remote_root"])' "${CONFIG}")}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"

test -s "${CONFIG}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
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
PILOT_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["validation_temperature_pilot_authorized"]).lower())' "${CONFIG}")"
PILOT_REMOTE_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["pilot_remote_training_authorized"]).lower())' "${CONFIG}")"
FORMAL_TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"
SEED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "${CONFIG}")"

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
profiles=json.load(open(config["profiles"]["path"]))["dataset_profiles"]
for dataset in config["datasets"]:
    profile=profiles[dataset]
    for temperature in config["temperatures"]:
        print("\t".join(map(str,(
            dataset,temperature,profile["profile"],profile["patch_num"],
            profile["d_model"],profile["d_ff"],
        ))))
' "${CONFIG}"
)

temperature_tag() {
  local value="$1"
  value="${value#0.}"
  value="${value//./}"
  echo "${value}"
}

run_dir_for_line() {
  local line="$1" dataset temperature rest tag
  IFS=$'\t' read -r dataset temperature rest <<< "${line}"
  tag="$(temperature_tag "${temperature}")"
  echo "${OUTPUT_ROOT}/tau${tag}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local line="$1" output_dir
  output_dir="$(run_dir_for_line "${line}")"
  [[ -s "${output_dir}/training_log.csv" \
    && -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/checkpoint.pt" \
    && -s "${output_dir}/effective_config.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "ccsf_temperature_pilot_status=$(date -Is) completed=${completed}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "ccsf_temperature_pilot_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} validation_only=true formal_test_authorized=${FORMAL_TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${PILOT_AUTHORIZED}" != "true" || "${PILOT_REMOTE_AUTHORIZED}" != "true" ]]; then
  echo "validation temperature pilot is not authorized by ${CONFIG}" >&2
  exit 3
fi
if [[ "${FORMAL_TEST_AUTHORIZED}" != "false" ]]; then
  echo "temperature pilot requires formal_test_access_authorized=false" >&2
  exit 3
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4" smoke="$5"
  local dataset temperature profile patch_num d_model d_ff
  local run_args=()
  IFS=$'\t' read -r dataset temperature profile patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(
      --max-train-batches 3 --max-eval-batches 1 --epochs 1 --patience 1
    )
  else
    run_args=(--epochs 20 --patience 5)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons 96,192,336,720 \
      --evaluation-horizons 96,192,336,720 \
      --segment-horizons 96,192,336,720 \
      --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size 32 --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "CCSF_TEMP_tau${temperature}" \
      --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening \
      --protocol-profile stage_c_siff_ccsf_temperature_pilot_v1 \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode ccsf-coupling-field --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank 256 \
      --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode direct --pcsd-fixed-scale 720 \
      --pcsd-partition canonical --pcsd-partition-seed 15101 \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --ccsf-correction-hidden-dim 64 \
      --ccsf-calibration-temperature "${temperature}" \
      --ccsf-calibration-weight 0.1 \
      --pcc-objective-mode ccsf_relative_calibration --pred-loss-mode full \
      --final-evaluation-split val --no-official-test-mode \
      --no-save-predictions "${run_args[@]}" >"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_line="${LINES[1]}"
  smoke_root="${OUTPUT_ROOT}/_resource_smoke/weather_tau01_seed${SEED}"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  run_training_command \
    "${smoke_line}" "${GPU_IDS[0]}" "${smoke_root}" "${smoke_root}/smoke.log" 1
  test -s "${smoke_root}/training_log.csv"
  test -s "${smoke_root}/metrics_by_target_horizon.csv"
  echo "ccsf_temperature_resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"
ANALYSIS_ROOT="${OUTPUT_ROOT}/_analysis_seed${SEED}"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"
{
  echo "ccsf_temperature_pilot_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "evaluation_split=validation_only"
  echo "formal_test_access=false"
  echo "selection=shared_macro_mse_five_datasets_four_horizons"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset temperature profile patch_num d_model d_ff output_dir run_log
  IFS=$'\t' read -r dataset temperature profile patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/tau$(temperature_tag "${temperature}")_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} tau=${temperature} gpu=${gpu}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} tau=${temperature} gpu=${gpu} profile=${profile}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  echo "run_done=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} tau=${temperature} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index
  for ((line_index=WORKER_OFFSET + worker_index; line_index<${#LINES[@]}; line_index+=WORKER_STRIDE)); do
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
  echo "ccsf_temperature_pilot_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_siff_ccsf_temperature_pilot.py \
    --config "${CONFIG}" --raw-root "${OUTPUT_ROOT}" \
    --output-dir "${ANALYSIS_ROOT}"
echo "ccsf_temperature_pilot_done=$(date -Is)"
