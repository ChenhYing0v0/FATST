#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo.json}"
PHASE="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["matrix"]["phase"])' \
    "${CONFIG}"
)"
PHASE_LOWER="$(printf '%s' "${PHASE}" | tr '[:upper:]' '[:lower:]')"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/${PHASE_LOWER}}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MODE="${MODE:-dry-run}"
CANARY_ONLY="${CANARY_ONLY:-0}"

case "${MODE}" in
  dry-run|data-audit|resource-smoke|train|status) ;;
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
SEARCH_SPACE_HASH="$(
  python3 - "${CONFIG}" <<'PY'
import hashlib
import json
import sys

config = json.load(open(sys.argv[1]))
payload = {
    "jobs": config["jobs"],
    "base_profiles": config.get("base_profiles", {}),
    "hpo_budget": config["hpo_budget"],
    "selection_contract": config["selection_contract"],
    "architecture_invariants": config["architecture_invariants"],
    "base_config": config.get("base_config"),
    "base_config_sha256": config.get("base_config_sha256"),
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
)"
REMOTE_AUTHORIZED="$(
  python3 - "${CONFIG}" "${PHASE}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
phase = sys.argv[2]
phase_key = f"remote_{phase}_training_authorized"
key = phase_key if phase_key in config["authorization"] else (
    "remote_H2_training_authorized"
    if phase == "H2"
    else "remote_H0_H1_authorized"
)
print(str(config["authorization"][key]).lower())
PY
)"

JOBS_TMP="$(mktemp)"
trap 'rm -f "${JOBS_TMP}"' EXIT
python3 - "${CONFIG}" "${CANARY_ONLY}" >"${JOBS_TMP}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
config = json.load(config_path.open())
canary_only = sys.argv[2] == "1"
base_jobs = {}
base_profiles = config.get("base_profiles", {})
if config.get("base_config"):
    base_path = Path(config["base_config"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_config = json.load(base_path.open())
    base_jobs = {job["trial_id"]: job for job in base_config["jobs"]}

resolved_jobs = {}
for specification in config["jobs"]:
    if specification.get("base_profile_id"):
        job = dict(base_profiles[specification["base_profile_id"]])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_profile_id", "overrides"}
            }
        )
    elif specification.get("base_trial_id"):
        job = dict(base_jobs[specification["base_trial_id"]])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_trial_id", "overrides"}
            }
        )
    else:
        job = dict(specification)
    resolved_jobs[job["trial_id"]] = job

for trial_id in config["provisional_lpt_order"]:
    job = resolved_jobs[trial_id]
    if canary_only and job["dataset"] not in {"ECL", "Solar", "Exchange"}:
        continue
    profile_hash = hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fields = [
        job["trial_id"],
        job["dataset"],
        job["profile_id"],
        job["source_prior"],
        job["seq_len"],
        job["patch_num"],
        job["d_model"],
        job["d_ff"],
        job["dropout"],
        job["learning_rate"],
        job["weight_decay"],
        job["batch_size"],
        job["gradient_accumulation_steps"],
        job["mode_rank"],
        job.get("layer_norm", 1),
        job.get("max_epochs", config["training"]["max_epochs"]),
        job.get(
            "early_stopping_patience",
            config["training"]["early_stopping_patience"],
        ),
        profile_hash,
    ]
    print("\t".join(map(str, fields)))
PY
LINES=()
while IFS= read -r line; do
  LINES+=("${line}")
done <"${JOBS_TMP}"

EXPECTED_JOBS="$(
  python3 - "${CONFIG}" "${CANARY_ONLY}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
canary_only = sys.argv[2] == "1"
print(
    sum(
        1
        for job in config["jobs"]
        if not canary_only
        or job["dataset"] in {"ECL", "Solar", "Exchange"}
    )
)
PY
)"
[[ "${#LINES[@]}" -eq "${EXPECTED_JOBS}" ]]

run_dir_for_line() {
  local trial_id dataset rest
  IFS=$'\t' read -r trial_id dataset rest <<< "$1"
  echo "${OUTPUT_ROOT}/trials/${dataset}/${trial_id}/seed2021"
}

is_complete() {
  local run_dir
  run_dir="$(run_dir_for_line "$1")"
  [[ -s "${run_dir}/checkpoint.pt" \
    && -s "${run_dir}/training_log.csv" \
    && -s "${run_dir}/metrics_by_target_horizon.csv" \
    && -s "${run_dir}/effective_config.json" \
    && -s "${run_dir}/initialization_contract.json" \
    && -s "${run_dir}/model_diagnostics.json" ]]
}

if [[ "${MODE}" == "dry-run" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "iscf_bsca_main_${PHASE_LOWER}_dry_run=pass jobs=${#LINES[@]} test_jobs=0 config_hash=${CONFIG_HASH} search_space_hash=${SEARCH_SPACE_HASH} remote_authorized=${REMOTE_AUTHORIZED} canary_only=${CANARY_ONLY}"
  exit 0
fi

if [[ "${MODE}" == "status" ]]; then
  complete=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then
      complete=$((complete + 1))
    fi
  done
  echo "iscf_bsca_main_${PHASE_LOWER}_status=$(date -Is) complete=${complete}/${#LINES[@]} test=0/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

[[ "${REMOTE_AUTHORIZED}" == "true" ]] || {
  echo "${PHASE} remote execution is not authorized" >&2
  exit 3
}

mkdir -p "${OUTPUT_ROOT}"

if [[ "${MODE}" == "data-audit" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/audit_iscf_bsca_paper_datasets.py \
      --dataset-root "${DATASET_ROOT}" \
      --config "${CONFIG}" \
      --datasets ECL Solar Exchange \
      --construct-loaders \
      --output "${OUTPUT_ROOT}/h0_new_dataset_audit.json"
  exit 0
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" log="$4" smoke="$5"
  local trial_id dataset profile_id source_prior seq_len patch_num d_model d_ff
  local dropout learning_rate weight_decay batch_size accumulation mode_rank
  local layer_norm max_epochs patience profile_hash
  local budget_args=()
  IFS=$'\t' read -r \
    trial_id dataset profile_id source_prior seq_len patch_num d_model d_ff \
    dropout learning_rate weight_decay batch_size accumulation mode_rank \
    layer_norm max_epochs patience profile_hash <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    budget_args=(
      --max-train-batches 2
      --max-eval-batches 2
      --epochs 1
      --patience 1
    )
  else
    budget_args=(--epochs "${max_epochs}" --patience "${patience}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode unified \
      --seq-len "${seq_len}" \
      --pred-len 720 \
      --target-horizons 720 \
      --validation-horizons 96,192,336,720 \
      --evaluation-horizons 96,192,336,720 \
      --segment-horizons 96,192,336,720 \
      --evaluation-prefix-mode full-crop \
      --e-layers 2 \
      --batch-size "${batch_size}" \
      --gradient-accumulation-steps "${accumulation}" \
      --enable-early-stopping \
      --early-stopping-min-delta 0 \
      --seed 2021 \
      --num-workers 0 \
      --run-name "ISCF_BSCA_MAIN_${trial_id}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --checkpoint-policy best-val \
      --no-evaluate-dual-checkpoints \
      --no-official-test-mode \
      --final-evaluation-split val \
      --protocol-class method_screening \
      --protocol-profile "ISCF-BSCA-MAIN-v1-HPO-${PHASE}" \
      --profile-hash "${profile_hash}" \
      --hpo-trial-id "${trial_id}" \
      --hpo-profile-id "${profile_id}" \
      --hpo-profile-hash "${profile_hash}" \
      --hpo-config-hash "${CONFIG_HASH}" \
      --hpo-search-space-hash "${SEARCH_SPACE_HASH}" \
      --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" \
      --legacy-dropout "${dropout}" \
      --legacy-layer-norm "${layer_norm}" \
      --learning-rate "${learning_rate}" \
      --weight-decay "${weight_decay}" \
      --readout-mode siff-independent-scope-control \
      --basis-rank 256 \
      --pcsd-coordinate-dim 4 \
      --pcsd-mode-rank "${mode_rank}" \
      --pcsd-policy-history-dim 32 \
      --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode direct \
      --pcsd-fixed-scale 720 \
      --pcsd-partition canonical \
      --pcsd-partition-seed 15101 \
      --pcsd-group-chunk-size 64 \
      --pcsd-target-chunk-size 128 \
      --pcc-objective-mode equal_uniform_scope_anchor \
      --pred-loss-mode full \
      --no-save-predictions \
      "${budget_args[@]}" >"${log}" 2>&1
}

MODE_ROOT="${OUTPUT_ROOT}"
if [[ "${MODE}" == "resource-smoke" ]]; then
  MODE_ROOT="${OUTPUT_ROOT}/_resource_smoke"
fi
mkdir -p "${MODE_ROOT}" "${OUTPUT_ROOT}/_logs"

{
  echo "start=$(date -Is)"
  echo "mode=${MODE}"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "config_hash=${CONFIG_HASH}"
  echo "search_space_hash=${SEARCH_SPACE_HASH}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "test_jobs=0"
  echo "canary_only=${CANARY_ONLY}"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits || true
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/${MODE}_jobs.tsv"

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
  local trial_id dataset rest run_dir log smoke
  IFS=$'\t' read -r trial_id dataset rest <<< "${line}"
  if [[ "${MODE}" == "resource-smoke" ]]; then
    run_dir="${MODE_ROOT}/${dataset}/${trial_id}/seed2021"
    smoke=1
  else
    run_dir="$(run_dir_for_line "${line}")"
    smoke=0
  fi
  log="${OUTPUT_ROOT}/_logs/${MODE}_${trial_id}.log"
  if [[ "${MODE}" == "train" ]] && is_complete "${line}"; then
    echo "skip_existing job=$((index + 1))/${#LINES[@]} trial=${trial_id}"
    return
  fi
  mkdir -p "${run_dir}"
  echo "run_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} trial=${trial_id} gpu=${gpu}"
  run_training_command "${line}" "${gpu}" "${run_dir}" "${log}" "${smoke}"
  failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
  if command -v rg >/dev/null 2>&1; then
    ! rg -ni "${failure_pattern}" "${log}"
  else
    ! grep -Ein "${failure_pattern}" "${log}"
  fi
  test -s "${run_dir}/training_log.csv"
  test -s "${run_dir}/effective_config.json"
  test -s "${run_dir}/metrics_by_target_horizon.csv"
  echo "run_done=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} trial=${trial_id} gpu=${gpu}"
}

worker() {
  local gpu="$1"
  local index
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
echo "iscf_bsca_main_${PHASE_LOWER}_${MODE}_done=$(date -Is) jobs=${#LINES[@]} test_jobs=0"
