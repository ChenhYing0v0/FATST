#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/timealign_official_main_i_reproduction.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/main_i_8dataset_20260806}"
REUSE_ROOT="${REUSE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/ettm2_weather_20260804/runs}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MODE="${MODE:-dry-run}"

case "${MODE}" in
  dry-run|resource-smoke|run|status) ;;
  *) echo "unsupported MODE=${MODE}" >&2; exit 2 ;;
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
AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
JOBS_TMP="$(mktemp)"
trap 'rm -f "${JOBS_TMP}"' EXIT

python3 - "${CONFIG}" >"${JOBS_TMP}" <<'PY'
import hashlib
import json
import sys

config = json.load(open(sys.argv[1]))
horizons = config["matrix"]["horizons"]
reuse = set(config["reuse_contract"]["run_ids"])
profiles = config["profile_contract"]
jobs = {}
for dataset in config["matrix"]["datasets"]:
    for horizon in horizons:
        run_id = f"TimeAlign__{dataset}__H{horizon}__seed2021"
        raw_profile = profiles[dataset]
        profile = raw_profile.get(str(horizon), raw_profile)
        epochs = 1 if dataset == "ETTh1" and horizon == 96 else 10
        batch_size = 16 if dataset == "ECL" else 32
        job = {
            "run_id": run_id,
            "dataset": dataset,
            "horizon": horizon,
            "seed": 2021,
            "epochs": epochs,
            "batch_size": batch_size,
            **profile,
        }
        profile_hash = hashlib.sha256(
            json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        jobs[run_id] = (job, profile_hash)
for run_id in config["workload_order"]:
    job, profile_hash = jobs[run_id]
    fields = [
        run_id, job["dataset"], job["horizon"], job["seed"],
        job["epochs"], job["batch_size"],
        "reuse" if run_id in reuse else "new", profile_hash,
    ]
    print("\t".join(map(str, fields)))
PY

LINES=()
while IFS= read -r line; do LINES+=("${line}"); done <"${JOBS_TMP}"
[[ "${#LINES[@]}" -eq 32 ]]

run_dir_for_line() {
  local run_id dataset horizon seed epochs batch_size artifact_role rest
  IFS=$'\t' read -r run_id dataset horizon seed epochs batch_size artifact_role rest <<< "$1"
  if [[ "${artifact_role}" == "reuse" ]]; then
    echo "${REUSE_ROOT}/${run_id}"
  else
    echo "${OUTPUT_ROOT}/runs/${run_id}"
  fi
}

is_complete() {
  local line="$1" run_dir artifact_role
  run_dir="$(run_dir_for_line "${line}")"
  IFS=$'\t' read -r _ _ _ _ _ _ artifact_role _ <<< "${line}"
  [[ -s "${run_dir}/checkpoint.pt" \
    && -s "${run_dir}/effective_config.json" \
    && -s "${run_dir}/environment.json" \
    && -s "${run_dir}/training_log.csv" \
    && -s "${run_dir}/metrics_by_target_horizon.csv" \
    && -s "${run_dir}/metrics_by_segment.csv" \
    && -s "${run_dir}/model_diagnostics.json" \
    && -s "${run_dir}/run.log" ]] || return 1
  if [[ "${artifact_role}" == "reuse" ]]; then
    [[ -s "${run_dir}/predictions_test.npz" ]]
  fi
}

if [[ "${MODE}" == "dry-run" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "timealign_main_i_dry_run=pass jobs=32 reusable=8 new=24 test_jobs=24 seed=2021 config_hash=${CONFIG_HASH} remote_authorized=${AUTHORIZED}"
  exit 0
fi

if [[ "${MODE}" == "status" ]]; then
  complete=0
  new_complete=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then
      complete=$((complete + 1))
      if [[ "${line}" == *$'\tnew\t'* ]]; then new_complete=$((new_complete + 1)); fi
    fi
  done
  echo "timealign_main_i_status=$(date -Is) complete=${complete}/32 new_complete=${new_complete}/24"
  if [[ -d "${OUTPUT_ROOT}/runs" ]]; then
    while IFS= read -r log; do tail -n 1 "${log}"; done < <(find "${OUTPUT_ROOT}/runs" -name run.log -type f | sort)
  fi
  exit 0
fi

[[ "${AUTHORIZED}" == "true" ]] || { echo "TimeAlign Main I reproduction is not authorized" >&2; exit 3; }

verify_file() {
  local path="$1" expected="$2" label="$3" actual
  test -s "${path}"
  actual="$(sha256_file "${path}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} sha256 mismatch expected=${expected} actual=${actual}" >&2
    exit 4
  }
}

while IFS=$'\t' read -r kind relative_path expected label; do
  if [[ "${kind}" == "source" ]]; then
    verify_file "${relative_path}" "${expected}" "${label}"
  else
    verify_file "${DATASET_ROOT}/${relative_path}" "${expected}" "${label}"
  fi
done < <(python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
source = config["source_contract"]
for name in ("adapter", "executed_model", "data_loader", "metric"):
    print(f"source\t{source[name + '_path']}\t{source[name + '_sha256']}\t{name}")
for dataset, item in source["dataset_scripts"].items():
    print(f"source\t{item['path']}\t{item['sha256']}\t{dataset}_script")
for dataset, item in config["datasets"].items():
    print(f"dataset\t{item['relative_path']}\t{item['remote_sha256']}\t{dataset}")
PY
)

mkdir -p "${OUTPUT_ROOT}"
{
  echo "start=$(date -Is)"
  echo "mode=${MODE}"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "config_hash=${CONFIG_HASH}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "reuse_root=${REUSE_ROOT}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=32"
  echo "reusable_jobs=8"
  echo "new_jobs=24"
  echo "test_jobs=$([[ "${MODE}" == "run" ]] && echo 24 || echo 0)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/${MODE}_jobs.tsv"

MODE_ROOT="${OUTPUT_ROOT}"
if [[ "${MODE}" == "resource-smoke" ]]; then MODE_ROOT="${OUTPUT_ROOT}/_resource_smoke"; fi
QUEUE_STATE="${OUTPUT_ROOT}/.${MODE}_queue_index"
QUEUE_LOCK="${OUTPUT_ROOT}/.${MODE}_queue_lock"
echo 0 >"${QUEUE_STATE}"

next_job() {
  (
    flock -x 9
    local index
    index="$(<"${QUEUE_STATE}")"
    if [[ "${index}" -ge "${#LINES[@]}" ]]; then exit 1; fi
    echo $((index + 1)) >"${QUEUE_STATE}"
    printf '%s' "${index}"
  ) 9>"${QUEUE_LOCK}"
}

run_one() {
  local index="$1" line="$2" gpu="$3"
  local run_id dataset horizon seed epochs batch_size artifact_role profile_hash
  local run_dir log
  local budget_args=() evaluation_args=()
  IFS=$'\t' read -r run_id dataset horizon seed epochs batch_size artifact_role profile_hash <<< "${line}"
  if is_complete "${line}"; then
    echo "skip_existing job=$((index + 1))/32 run_id=${run_id} role=${artifact_role}"
    return
  fi
  if [[ "${artifact_role}" == "reuse" ]]; then
    echo "required reusable artifact is incomplete: ${run_id}" >&2
    return 5
  fi
  if [[ "${MODE}" == "resource-smoke" ]]; then
    run_dir="${MODE_ROOT}/${run_id}"
    budget_args=(--epochs 1 --max-train-batches 2 --max-eval-batches 2)
    evaluation_args=(--final-evaluation-split none --no-save-predictions)
  else
    run_dir="${OUTPUT_ROOT}/runs/${run_id}"
    budget_args=(--epochs "${epochs}")
    evaluation_args=(--final-evaluation-split test --no-save-predictions)
  fi
  mkdir -p "${run_dir}"
  log="${run_dir}/run.log"
  echo "run_start=$(date -Is) job=$((index + 1))/32 dataset=${dataset} horizon=${horizon} gpu=${gpu} epochs=${epochs}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode fixed \
      --seq-len 720 --label-len 48 --pred-len "${horizon}" \
      --target-horizons "${horizon}" --validation-horizons "${horizon}" \
      --evaluation-horizons "${horizon}" --segment-horizons "${horizon}" \
      --evaluation-prefix-mode native --e-layers 2 \
      --encoder-mode timealign-token-mlp --readout-mode official --pred-loss-mode full \
      --grouped-mlp-scale 48 --weight-decay 0.01 --w-recon 1.0 \
      --batch-size "${batch_size}" --gradient-accumulation-steps 1 \
      --no-enable-early-stopping --seed "${seed}" --num-workers 0 \
      --run-name "${run_id}" --output-dir "${run_dir}" --device cuda \
      --checkpoint-policy official-last --no-evaluate-dual-checkpoints \
      --official-test-mode --protocol-class native_external \
      --protocol-profile TIMEALIGN-OFFICIAL-MAIN-I-8DATASET-REPRODUCTION-20260806 \
      --profile-hash "${profile_hash}" --allow-archived-research-modes \
      "${budget_args[@]}" "${evaluation_args[@]}" >"${log}" 2>&1
  failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
  if command -v rg >/dev/null 2>&1; then ! rg -ni "${failure_pattern}" "${log}"; else ! grep -Ein "${failure_pattern}" "${log}"; fi
  test -s "${run_dir}/checkpoint.pt"
  test -s "${run_dir}/training_log.csv"
  test -s "${run_dir}/effective_config.json"
  if [[ "${MODE}" == "run" ]]; then is_complete "${line}"; fi
  echo "run_done=$(date -Is) job=$((index + 1))/32 dataset=${dataset} horizon=${horizon} gpu=${gpu}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do run_one "${index}" "${LINES[${index}]}" "${gpu}"; done
}

pids=()
for gpu in "${GPU_IDS[@]}"; do worker "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
[[ "${status}" == "0" ]]
echo "timealign_main_i_${MODE}_done=$(date -Is) jobs=32 reusable=8 new=24 test_jobs=$([[ "${MODE}" == "run" ]] && echo 24 || echo 0)"
