#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_patchtst_test_tuned_formal.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
export PYTHONHASHSEED=2021
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

hash_file() { sha256sum "$1" | awk '{print $1}'; }
json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): value=value[key]
print(value)' "${CONFIG}" "$1"
}

MANIFEST="$(json_value artifact_contract.unique_manifest)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(json_value artifact_contract.remote_output_root)}"
DESIGN_PATH="$(json_value diagnostic_design.path)"
[[ "$(hash_file "${MANIFEST}")" == "$(json_value artifact_contract.unique_manifest_sha256)" ]]
[[ "$(hash_file "${DESIGN_PATH}")" == "$(json_value diagnostic_design.sha256)" ]]
[[ "$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")" == true ]]

ROWS=()
while IFS= read -r line; do ROWS+=("$line"); done < <(
  python3 - "${MANIFEST}" <<'PY'
import csv
import sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if len(rows) != 40 or len({row["checkpoint_sha256"] for row in rows}) != 40:
    raise SystemExit("unique manifest must contain exactly 40 unique hashes")
for row in rows:
    print("\t".join((
        row["dataset"], row["representative_profile_id"], row["checkpoint"],
        row["checkpoint_sha256"], row["existing_formal_artifact_dir"],
    )))
PY
)

formal_dir() {
  local dataset profile checkpoint expected existing
  IFS=$'\t' read -r dataset profile checkpoint expected existing <<< "$1"
  if [[ -n "${existing}" ]]; then
    echo "${existing}"
  else
    echo "${OUTPUT_ROOT}/formal_test/${profile}/${dataset}/seed2021"
  fi
}

test_complete() {
  local out
  out="$(formal_dir "$1")"
  [[ -s "${out}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${out}/pcsd_test_audit_diagnostics.npz" \
    && -s "${out}/test_audit_invariants.json" ]] \
    && python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["pass"] is True and x["evaluation_split"] == "test"' "${out}/test_audit_invariants.json" 2>/dev/null
}

verify_checkpoints() {
  local line dataset profile checkpoint expected existing actual count=0
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r dataset profile checkpoint expected existing <<< "${line}"
    [[ -s "${checkpoint}" ]]
    actual="$(hash_file "${checkpoint}")"
    [[ "${actual}" == "${expected}" ]]
    count=$((count + 1))
  done
  [[ "${count}" == 40 ]]
}

if [[ "${STATUS_ONLY}" == 1 ]]; then
  complete=0
  reused=0
  for line in "${ROWS[@]}"; do
    test_complete "${line}" && complete=$((complete + 1))
    IFS=$'\t' read -r _ _ _ _ existing <<< "${line}"
    [[ -n "${existing}" ]] && reused=$((reused + 1))
  done
  echo "patchtst_test_tuned_formal_status=$(date -Is) unique=${complete}/40 new=$((complete - reused))/35 reused=${reused}/5 cells=$((complete * 4))/160"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  verify_checkpoints
  complete=0
  for line in "${ROWS[@]}"; do test_complete "${line}" && complete=$((complete + 1)); done
  [[ "${complete}" == 5 ]]
  echo "patchtst_test_tuned_formal_dry_run=pass unique=40 reusable=5 new=35 cells=160"
  exit 0
fi

verify_checkpoints
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "${OUTPUT_ROOT}/_formal_logs"
{
  echo "start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "protocol_hash=$(hash_file "${CONFIG}")"
  echo "manifest_hash=$(hash_file "${MANIFEST}")"
  echo "gpus=${GPU_IDS[*]}"
  echo "formal_test=true"
  echo "test_tuned=true"
  echo "unique_jobs=40"
  echo "reused_jobs=5"
  echo "new_jobs=35"
} | tee "${OUTPUT_ROOT}/formal_launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset profile checkpoint expected existing run_dir out log before after
  IFS=$'\t' read -r dataset profile checkpoint expected existing <<< "${line}"
  test_complete "${line}" && return
  run_dir="$(dirname "${checkpoint}")"
  out="$(formal_dir "${line}")"
  log="${OUTPUT_ROOT}/_formal_logs/${dataset}_${profile}.log"
  mkdir -p "${out}"
  before="$(hash_file "${checkpoint}")"
  [[ "${before}" == "${expected}" ]]
  echo "formal_start=$(date -Is) job=$((index + 1))/40 dataset=${dataset} profile=${profile} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    --run-dir "${run_dir}" --artifact-dir "${out}" \
    --design "${DESIGN_PATH}" --test-audit-config "${CONFIG}" \
    --evaluation-split test --probe-rows 64 --device cuda >"${log}" 2>&1
  after="$(hash_file "${checkpoint}")"
  [[ "${before}" == "${after}" ]]
  test_complete "${line}"
  echo "formal_done=$(date -Is) job=$((index + 1))/40 dataset=${dataset} profile=${profile} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#ROWS[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${ROWS[$index]}" "${gpu}"
  done
}

pids=()
for i in "${!GPU_IDS[@]}"; do
  worker "${i}" "${GPU_IDS[$i]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" == 0 ]]
verify_checkpoints
complete=0
for line in "${ROWS[@]}"; do test_complete "${line}" && complete=$((complete + 1)); done
[[ "${complete}" == 40 ]]
echo "patchtst_test_tuned_formal_done=$(date -Is) unique=40/40 new=35/35 cells=160/160"
