#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/timealign_official_ettm2_weather_reproduction.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/ettm2_weather_20260804}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-2}"
MODE="${MODE:-dry-run}"

case "${MODE}" in
  dry-run|resource-smoke|run|status) ;;
  *)
    echo "unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

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
AUTHORIZED="$(
  python3 -c \
    'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' \
    "${CONFIG}"
)"

JOBS_TMP="$(mktemp)"
trap 'rm -f "${JOBS_TMP}"' EXIT
python3 - "${CONFIG}" >"${JOBS_TMP}" <<'PY'
import hashlib
import json
import sys

config = json.load(open(sys.argv[1]))
jobs = {job["run_id"]: job for job in config["jobs"]}
for run_id in config["workload_order"]:
    job = jobs[run_id]
    profile_hash = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fields = [
        job["run_id"], job["dataset"], job["horizon"], job["seed"],
        job["d_model"], job["d_ff"], job["dropout"],
        job["learning_rate"], job["w_align"], job["patch_num"],
        job["layer_norm"], profile_hash,
    ]
    print("\t".join(map(str, fields)))
PY

LINES=()
while IFS= read -r line; do
  LINES+=("${line}")
done <"${JOBS_TMP}"
[[ "${#LINES[@]}" -eq 8 ]]

run_dir_for_line() {
  local run_id rest
  IFS=$'\t' read -r run_id rest <<< "$1"
  echo "${OUTPUT_ROOT}/runs/${run_id}"
}

is_complete() {
  local run_dir
  run_dir="$(run_dir_for_line "$1")"
  [[ -s "${run_dir}/checkpoint.pt" \
    && -s "${run_dir}/effective_config.json" \
    && -s "${run_dir}/environment.json" \
    && -s "${run_dir}/training_log.csv" \
    && -s "${run_dir}/metrics_by_target_horizon.csv" \
    && -s "${run_dir}/metrics_by_segment.csv" \
    && -s "${run_dir}/model_diagnostics.json" \
    && -s "${run_dir}/predictions_test.npz" \
    && -s "${run_dir}/run.log" ]]
}

if [[ "${MODE}" == "dry-run" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "timealign_official_reproduction_dry_run=pass jobs=8 test_jobs=8 seed=2021 config_hash=${CONFIG_HASH} remote_authorized=${AUTHORIZED}"
  exit 0
fi

if [[ "${MODE}" == "status" ]]; then
  complete=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then
      complete=$((complete + 1))
    fi
  done
  echo "timealign_official_reproduction_status=$(date -Is) complete=${complete}/8 test=${complete}/8"
  find "${OUTPUT_ROOT}/runs" -name run.log -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

[[ "${AUTHORIZED}" == "true" ]] || {
  echo "TimeAlign remote reproduction is not authorized" >&2
  exit 3
}

verify_file() {
  local path="$1" expected="$2" label="$3"
  test -s "${path}"
  local actual
  actual="$(sha256_file "${path}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} sha256 mismatch expected=${expected} actual=${actual}" >&2
    exit 4
  }
}

readarray -t PREFLIGHT < <(
  python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
source = config["source_contract"]
for name in (
    "adapter", "ettm2_script", "weather_script", "executed_model",
    "data_loader", "metric",
):
    print(f"source\t{source[name + '_path']}\t{source[name + '_sha256']}\t{name}")
for dataset, item in config["datasets"].items():
    print(f"dataset\t{item['relative_path']}\t{item['remote_sha256']}\t{dataset}")
PY
)
for record in "${PREFLIGHT[@]}"; do
  IFS=$'\t' read -r kind relative_path expected label <<< "${record}"
  if [[ "${kind}" == "source" ]]; then
    verify_file "${relative_path}" "${expected}" "${label}"
  else
    verify_file "${DATASET_ROOT}/${relative_path}" "${expected}" "${label}"
  fi
done

mkdir -p "${OUTPUT_ROOT}"
{
  echo "start=$(date -Is)"
  echo "mode=${MODE}"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "config_hash=${CONFIG_HASH}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=8"
  echo "test_jobs=$([[ "${MODE}" == "run" ]] && echo 8 || echo 0)"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits || true
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/${MODE}_jobs.tsv"

MODE_ROOT="${OUTPUT_ROOT}"
if [[ "${MODE}" == "resource-smoke" ]]; then
  MODE_ROOT="${OUTPUT_ROOT}/_resource_smoke"
fi
QUEUE_STATE="${OUTPUT_ROOT}/.${MODE}_queue_index"
QUEUE_LOCK="${OUTPUT_ROOT}/.${MODE}_queue_lock"
echo 0 >"${QUEUE_STATE}"

next_job() {
  (
    flock -x 9
    local index
    index="$(<"${QUEUE_STATE}")"
    if [[ "${index}" -ge "${#LINES[@]}" ]]; then
      exit 1
    fi
    echo $((index + 1)) >"${QUEUE_STATE}"
    printf '%s' "${index}"
  ) 9>"${QUEUE_LOCK}"
}

run_one() {
  local index="$1" line="$2" gpu="$3"
  local run_id dataset horizon seed d_model d_ff dropout learning_rate
  local w_align patch_num layer_norm profile_hash run_dir log
  local budget_args=() evaluation_args=()
  IFS=$'\t' read -r \
    run_id dataset horizon seed d_model d_ff dropout learning_rate \
    w_align patch_num layer_norm profile_hash <<< "${line}"
  if [[ "${MODE}" == "run" ]] && is_complete "${line}"; then
    echo "skip_existing job=$((index + 1))/8 run_id=${run_id}"
    return
  fi
  if [[ "${MODE}" == "resource-smoke" ]]; then
    run_dir="${MODE_ROOT}/${run_id}"
    budget_args=(--epochs 1 --max-train-batches 2 --max-eval-batches 2)
    evaluation_args=(--final-evaluation-split none --no-save-predictions)
  else
    run_dir="$(run_dir_for_line "${line}")"
    budget_args=(--epochs 10)
    evaluation_args=(--final-evaluation-split test --save-predictions)
  fi
  mkdir -p "${run_dir}"
  log="${run_dir}/run.log"
  echo "run_start=$(date -Is) job=$((index + 1))/8 dataset=${dataset} horizon=${horizon} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode fixed \
      --seq-len 720 \
      --label-len 48 \
      --pred-len "${horizon}" \
      --target-horizons "${horizon}" \
      --validation-horizons "${horizon}" \
      --evaluation-horizons "${horizon}" \
      --segment-horizons "${horizon}" \
      --evaluation-prefix-mode native \
      --e-layers 2 \
      --encoder-mode timealign-token-mlp \
      --readout-mode official \
      --pred-loss-mode full \
      --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" \
      --legacy-dropout "${dropout}" \
      --legacy-layer-norm "${layer_norm}" \
      --learning-rate "${learning_rate}" \
      --weight-decay 0.01 \
      --w-recon 1.0 \
      --w-align "${w_align}" \
      --batch-size 32 \
      --gradient-accumulation-steps 1 \
      --no-enable-early-stopping \
      --seed "${seed}" \
      --num-workers 0 \
      --run-name "${run_id}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --checkpoint-policy official-last \
      --no-evaluate-dual-checkpoints \
      --official-test-mode \
      --protocol-class native_external \
      --protocol-profile TIMEALIGN-OFFICIAL-ETTM2-WEATHER-REPRODUCTION-20260804 \
      --profile-hash "${profile_hash}" \
      --allow-archived-research-modes \
      "${budget_args[@]}" \
      "${evaluation_args[@]}" >"${log}" 2>&1
  failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
  if command -v rg >/dev/null 2>&1; then
    ! rg -ni "${failure_pattern}" "${log}"
  else
    ! grep -Ein "${failure_pattern}" "${log}"
  fi
  test -s "${run_dir}/checkpoint.pt"
  test -s "${run_dir}/training_log.csv"
  test -s "${run_dir}/effective_config.json"
  if [[ "${MODE}" == "run" ]]; then
    is_complete "${line}"
  fi
  echo "run_done=$(date -Is) job=$((index + 1))/8 dataset=${dataset} horizon=${horizon} gpu=${gpu}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do
    run_one "${index}" "${LINES[${index}]}" "${gpu}"
  done
}

pids=()
for gpu in "${GPU_IDS[@]}"; do
  worker "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
[[ "${status}" == "0" ]]
echo "timealign_official_reproduction_${MODE}_done=$(date -Is) jobs=8 test_jobs=$([[ "${MODE}" == "run" ]] && echo 8 || echo 0)"
