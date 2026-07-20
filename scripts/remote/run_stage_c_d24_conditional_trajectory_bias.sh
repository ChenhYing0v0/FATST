#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/stage_c_d24_conditional_trajectory_bias.json}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d23_fcmi_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d24_ctb_v1}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
STATUS_ONLY="${STATUS_ONLY:-0}"
export PYTHONHASHSEED=20260720
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -s "${CONFIG}"

read_config() {
  python3 -c "import json; print(json.load(open('${CONFIG}'))${1})"
}

REMOTE_INFERENCE_AUTHORIZED="$(read_config "['authorization']['remote_checkpoint_inference_authorized']")"
REMOTE_TRAINING_AUTHORIZED="$(read_config "['authorization']['remote_training_authorized']")"
TEST_AUTHORIZED="$(read_config "['authorization']['official_test_access_authorized']")"
EVALUATION_SPLIT="$(read_config "['evaluation_split']")"
if [[ "${REMOTE_INFERENCE_AUTHORIZED}" != "True" \
  || "${REMOTE_TRAINING_AUTHORIZED}" != "False" \
  || "${TEST_AUTHORIZED}" != "False" \
  || "${EVALUATION_SPLIT}" != "val" ]]; then
  echo "D24 authorization contract invalid" >&2
  exit 3
fi

mapfile -t JOBS < <(
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
for dataset in config["datasets"]:
    for arm in config["arms"]:
        print(f"{arm}\t{dataset}")
' "${CONFIG}"
)

run_output_dir() {
  local arm="$1" dataset="$2"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}"
}

run_complete() {
  local arm="$1" dataset="$2" directory
  directory="$(run_output_dir "${arm}" "${dataset}")"
  [[ -s "${directory}/metrics.csv" && -s "${directory}/metadata.json" ]]
}

complete_count() {
  local count=0 arm dataset
  for line in "${JOBS[@]}"; do
    IFS=$'\t' read -r arm dataset <<< "${line}"
    if run_complete "${arm}" "${dataset}"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  echo "complete=$(complete_count)/${#JOBS[@]}"
  if [[ -s "${OUTPUT_ROOT}/_analysis/decision.json" ]]; then
    python3 -c "import json; print('decision='+json.load(open('${OUTPUT_ROOT}/_analysis/decision.json'))['decision'])"
  fi
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
{
  echo "launch_time=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
  echo "source_root=${SOURCE_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "evaluation_split=val"
  echo "remote_training_authorized=false"
  echo "official_test_access_authorized=false"
  echo "gpus=${GPU_IDS_STR}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local arm dataset source_dir output_dir log_path
  IFS=$'\t' read -r arm dataset <<< "${line}"
  source_dir="${SOURCE_ROOT}/${arm}/${dataset}/h720_full/seed2021"
  output_dir="$(run_output_dir "${arm}" "${dataset}")"
  log_path="${OUTPUT_ROOT}/_logs/${arm}_${dataset}.log"
  if run_complete "${arm}" "${dataset}"; then
    echo "skip_existing job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset}"
    return
  fi
  test -s "${source_dir}/checkpoint.pt"
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_d24_conditional_trajectory_bias.py \
      --config "${CONFIG}" \
      --run-dir "${source_dir}" \
      --output-dir "${output_dir}" \
      --arm "${arm}" \
      --dataset "${dataset}" \
      --device cuda >"${log_path}" 2>&1
  run_complete "${arm}" "${dataset}"
  echo "run_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" job_index
  for ((job_index=worker_index; job_index<${#JOBS[@]}; job_index+=${#GPU_IDS[@]})); do
    run_one "${job_index}" "${JOBS[${job_index}]}" "${gpu}"
  done
}

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
  worker "${worker_index}" "${GPU_IDS[${worker_index}]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

if [[ "$(complete_count)" != "${#JOBS[@]}" ]]; then
  echo "D24 matrix incomplete after workers" >&2
  exit 4
fi
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_d24_conditional_trajectory_bias.py \
    --config "${CONFIG}" \
    --aggregate-root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/_analysis"
echo "d24_complete=$(date -Is) runs=$(complete_count)/${#JOBS[@]}"
