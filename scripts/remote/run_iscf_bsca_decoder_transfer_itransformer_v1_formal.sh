#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_itransformer_v1_formal.json}"
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

PROFILE_PATH="$(json_value profiles.path)"
TRAINING_PROTOCOL="$(json_value training_protocol.path)"
DESIGN_PATH="$(json_value diagnostic_design.path)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(json_value artifact_contract.remote_output_root)}"
MANIFEST="${MANIFEST:-$(json_value artifact_contract.remote_training_manifest)}"
EXPECTED_MANIFEST_HASH="$(json_value artifact_contract.training_manifest_sha256)"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"

[[ "$(hash_file "${PROFILE_PATH}")" == "$(json_value profiles.sha256)" ]]
[[ "$(hash_file "${TRAINING_PROTOCOL}")" == "$(json_value training_protocol.sha256)" ]]
[[ "$(hash_file "${DESIGN_PATH}")" == "$(json_value diagnostic_design.sha256)" ]]
[[ "$(hash_file "${MANIFEST}")" == "${EXPECTED_MANIFEST_HASH}" ]]
[[ "${TEST_AUTHORIZED}" == true ]]

ROWS=()
while IFS= read -r line; do ROWS+=("${line}"); done < <(
  python3 - "${MANIFEST}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
if len(rows) != 15:
    raise SystemExit(f"manifest row count mismatch: {len(rows)}/15")
if len({row["checkpoint_sha256"] for row in rows}) != 15:
    raise SystemExit("manifest checkpoint hashes are not unique")
for row in rows:
    print("\t".join((
        row["dataset"],
        row["arm"],
        row["checkpoint"],
        row["checkpoint_sha256"],
    )))
PY
)

formal_dir() {
  local dataset arm _
  IFS=$'\t' read -r dataset arm _ <<< "$1"
  echo "${OUTPUT_ROOT}/formal_test/${arm}/${dataset}/seed2021"
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
  local line dataset arm checkpoint expected actual count=0
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r dataset arm checkpoint expected <<< "${line}"
    [[ -s "${checkpoint}" ]]
    actual="$(hash_file "${checkpoint}")"
    [[ "${actual}" == "${expected}" ]]
    count=$((count + 1))
  done
  [[ "${count}" == 15 ]]
}

if [[ "${STATUS_ONLY}" == 1 ]]; then
  complete=0
  for line in "${ROWS[@]}"; do
    test_complete "${line}" && complete=$((complete + 1))
  done
  echo "itransformer_transfer_formal_status=$(date -Is) formal_test=${complete}/15 cells=$((complete * 4))/60"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  verify_checkpoints
  for line in "${ROWS[@]}"; do
    IFS=$'\t' read -r dataset arm checkpoint expected <<< "${line}"
    echo -e "formal_test\t${dataset}\t${arm}\t${checkpoint}\t${expected}"
  done
  echo "itransformer_transfer_formal_dry_run=pass jobs=15 cells=60"
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
  echo "jobs=15"
  echo "cells=60"
} | tee "${OUTPUT_ROOT}/formal_launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm checkpoint expected run_dir out log before after
  IFS=$'\t' read -r dataset arm checkpoint expected <<< "${line}"
  test_complete "${line}" && return
  run_dir="$(dirname "${checkpoint}")"
  out="$(formal_dir "${line}")"
  log="${OUTPUT_ROOT}/_formal_logs/${arm}_${dataset}.log"
  mkdir -p "${out}"
  before="$(hash_file "${checkpoint}")"
  [[ "${before}" == "${expected}" ]]
  echo "formal_start=$(date -Is) job=$((index + 1))/15 arm=${arm} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    --run-dir "${run_dir}" --artifact-dir "${out}" \
    --design "${DESIGN_PATH}" --test-audit-config "${CONFIG}" \
    --evaluation-split test --probe-rows 64 --device cuda >"${log}" 2>&1
  after="$(hash_file "${checkpoint}")"
  [[ "${before}" == "${after}" ]]
  test_complete "${line}"
  echo "formal_done=$(date -Is) job=$((index + 1))/15 arm=${arm} dataset=${dataset} gpu=${gpu}"
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
for line in "${ROWS[@]}"; do
  test_complete "${line}" && complete=$((complete + 1))
done
[[ "${complete}" == 15 ]]
echo "itransformer_transfer_formal_done=$(date -Is) jobs=15/15 cells=60/60"
