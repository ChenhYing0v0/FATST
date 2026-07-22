#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/stage_c_iscf_scc_d0.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_d0_validation_replay}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
PYTHON_BIN="${PYTHON_BIN:-/home/yingch/.conda/envs/moe/bin/python}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

test -s "${CONFIG}"
test -x "${PYTHON_BIN}"
test "${#GPU_IDS[@]}" -ge 1

AUTHORIZED="$(${PYTHON_BIN} -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["validation_replay_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$(${PYTHON_BIN} -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"
if [[ "${AUTHORIZED}" != "true" || "${TEST_AUTHORIZED}" != "false" ]]; then
  echo "SCC D0 validation replay authorization boundary failed" >&2
  exit 3
fi

mapfile -t JOBS < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json,sys
config=json.load(open(sys.argv[1]))
for dataset in config["datasets"]:
    for seed in config["seeds"]:
        key="seed2021" if seed == 2021 else "seed2022_2023"
        source=f'{config["source_roots"][key]}/{dataset}/h720_full/seed{seed}'
        print(f'{dataset}\t{seed}\t{source}')
PY
)

is_complete() {
  local dataset="$1" seed="$2" directory
  directory="${OUTPUT_ROOT}/${dataset}/seed${seed}"
  [[ -s "${directory}/pcsd_validation_diagnostics.npz" \
    && -s "${directory}/trained_invariants.json" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  complete=0
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r dataset seed source <<< "${job}"
    if is_complete "${dataset}" "${seed}"; then complete=$((complete + 1)); fi
  done
  echo "iscf_scc_d0_status=$(date -Is) validation_replay=${complete}/${#JOBS[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
{
  echo "iscf_scc_d0_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "jobs=${#JOBS[@]}"
  echo "evaluation_split=val"
  echo "new_training=false"
  echo "formal_test_access=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"

run_one() {
  local index="$1" gpu="$2" dataset seed source artifact_dir log_file
  IFS=$'\t' read -r dataset seed source <<< "${JOBS[${index}]}"
  artifact_dir="${OUTPUT_ROOT}/${dataset}/seed${seed}"
  log_file="${OUTPUT_ROOT}/_logs/${dataset}_seed${seed}.log"
  if is_complete "${dataset}" "${seed}"; then
    echo "skip_existing dataset=${dataset} seed=${seed}"
    return 0
  fi
  test -s "${source}/checkpoint.pt"
  mkdir -p "${artifact_dir}"
  before="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${source}/checkpoint.pt")"
  echo "replay_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} dataset=${dataset} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${source}" --artifact-dir "${artifact_dir}" \
      --design "${CONFIG}" --test-audit-config "${CONFIG}" \
      --evaluation-split val --probe-rows 256 --device cuda \
      >"${log_file}" 2>&1
  after="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${source}/checkpoint.pt")"
  if [[ "${before}" != "${after}" ]]; then
    echo "checkpoint_mutated dataset=${dataset} seed=${seed}" >&2
    return 1
  fi
  echo "replay_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} dataset=${dataset} seed=${seed} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#JOBS[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" != "0" ]]; then exit "${status}"; fi

"${PYTHON_BIN}" scripts/analyze_stage_c_iscf_scc_d0.py \
  --config "${CONFIG}" --validation-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/_analysis"
echo "iscf_scc_d0_done=$(date -Is)"
