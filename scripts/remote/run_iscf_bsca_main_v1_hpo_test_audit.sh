#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_test_audit.json}"
MANIFEST="${MANIFEST:-analysis/iscf_bsca_main_v1_hpo_20260731/combined_checkpoint_manifest.csv}"
TEST_ROOT="${TEST_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/test_audit}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MODE="${MODE:-dry-run}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

case "${MODE}" in
  dry-run|preflight|test|status) ;;
  *)
    echo "unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
test "${#GPU_IDS[@]}" -ge 1
test -s "${CONFIG}"
test -s "${MANIFEST}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

CONFIG_HASH="$(sha256_file "${CONFIG}")"
MANIFEST_HASH="$(sha256_file "${MANIFEST}")"
EXPECTED_MANIFEST_HASH="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_manifest"]["sha256"])' \
    "${CONFIG}"
)"
[[ "${MANIFEST_HASH}" == "${EXPECTED_MANIFEST_HASH}" ]] || {
  echo "manifest hash mismatch" >&2
  exit 3
}

AUTHORIZED="$(
  python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
authorization = config["authorization"]
authorized = (
    config["status"] == "authorized_prelaunch"
    and config["matrix"]["expected_runs"] > 0
    and authorization["user_authorized"] is True
    and authorization["formal_test_access_count_for_version"] == 1
    and authorization["test_role"]
    == "test-tuned-hyperparameter-selection-and-paper-benchmark"
    and authorization["per_dataset_aggregate_hyperparameter_tuning_allowed"]
    is True
    and authorization["per_horizon_seed_metric_or_cell_tuning_allowed"]
    is False
    and authorization["checkpoint_mutation_during_test_allowed"] is False
)
print(str(authorized).lower())
PY
)"
CANDIDATE_VERSION="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["candidate_version"])' \
    "${CONFIG}"
)"
EXPECTED_RUNS="$(
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["matrix"]["expected_runs"])' \
    "${CONFIG}"
)"
EXPECTED_CELLS="$((EXPECTED_RUNS * 4))"

JOBS_TMP="$(mktemp)"
trap 'rm -f "${JOBS_TMP}"' EXIT
python3 - "${MANIFEST}" "${TEST_ROOT}" "${EXPECTED_RUNS}" >"${JOBS_TMP}" <<'PY'
import csv
from pathlib import Path
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
test_root = Path(sys.argv[2]).resolve()
expected_runs = int(sys.argv[3])
if len(rows) != expected_runs:
    raise SystemExit(f"expected {expected_runs} manifest rows, found {len(rows)}")
for row in rows:
    test_dir = Path(row["test_artifact_dir"]).resolve()
    if test_root not in test_dir.parents:
        raise SystemExit(f"test artifact escapes TEST_ROOT: {test_dir}")
    fields = [
        row["phase"],
        row["dataset"],
        row["trial_id"],
        row["profile_id"],
        row["seed"],
        row["checkpoint_sha256_before_test"],
        row["artifact_dir"],
        row["test_artifact_dir"],
    ]
    if any("\t" in field or "\n" in field for field in fields):
        raise SystemExit("manifest fields must not contain tabs or newlines")
    print("\t".join(fields))
PY

JOB_LINES=()
while IFS= read -r line; do
  JOB_LINES+=("${line}")
done <"${JOBS_TMP}"
[[ "${#JOB_LINES[@]}" -eq "${EXPECTED_RUNS}" ]]

target_artifact_count() {
  local count=0 line phase dataset trial profile seed expected_sha run_dir test_dir
  for line in "${JOB_LINES[@]}"; do
    IFS=$'\t' read -r phase dataset trial profile seed expected_sha run_dir \
      test_dir <<< "${line}"
    if [[ -d "${test_dir}" ]]; then
      count=$((count + $(find "${test_dir}" -type f 2>/dev/null | wc -l)))
    elif [[ -e "${test_dir}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s' "${count}"
}

temporary_artifact_count() {
  if [[ -d "${TEST_ROOT}/_tmp" ]]; then
    find "${TEST_ROOT}/_tmp" -type f 2>/dev/null | wc -l | tr -d ' '
  else
    printf '0'
  fi
}

training_artifact_pass() {
  local line="$1"
  local phase dataset trial profile seed expected_sha run_dir test_dir
  IFS=$'\t' read -r phase dataset trial profile seed expected_sha run_dir \
    test_dir <<< "${line}"
  test -s "${run_dir}/checkpoint.pt"
  test -s "${run_dir}/training_log.csv"
  test -s "${run_dir}/metrics_by_target_horizon.csv"
  test -s "${run_dir}/effective_config.json"
  test -s "${run_dir}/initialization_contract.json"
  test -s "${run_dir}/model_diagnostics.json"
  [[ "$(sha256_file "${run_dir}/checkpoint.pt")" == "${expected_sha}" ]]
  python3 - "${run_dir}" "${trial}" "${dataset}" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
trial = sys.argv[2]
dataset = sys.argv[3]
effective = json.loads((run_dir / "effective_config.json").read_text())
adapter = effective["adapter"]
if adapter["dataset"] != dataset:
    raise SystemExit("dataset provenance mismatch")
if adapter.get("hpo_trial_id") != trial:
    raise SystemExit("trial provenance mismatch")
if adapter["checkpoint_policy"] != "best-val":
    raise SystemExit("checkpoint selector mismatch")
if adapter["final_evaluation_split"] != "val":
    raise SystemExit("training final split mismatch")
if adapter.get("official_test_mode"):
    raise SystemExit("training artifact accessed official test")
rows = list(csv.DictReader((run_dir / "metrics_by_target_horizon.csv").open()))
metrics = {
    int(row["target_horizon"]): (float(row["mse"]), float(row["mae"]))
    for row in rows
}
if set(metrics) != {96, 192, 336, 720}:
    raise SystemExit("validation horizon matrix mismatch")
if not all(math.isfinite(value) for pair in metrics.values() for value in pair):
    raise SystemExit("non-finite validation metric")
PY
}

test_artifact_pass() {
  local line="$1" artifact_override="${2:-}"
  local phase dataset trial profile seed expected_sha run_dir test_dir
  IFS=$'\t' read -r phase dataset trial profile seed expected_sha run_dir \
    test_dir <<< "${line}"
  if [[ -n "${artifact_override}" ]]; then
    test_dir="${artifact_override}"
  fi
  python3 - "${test_dir}" "${expected_sha}" "${dataset}" "${trial}" \
    "${profile}" "${seed}" "${CANDIDATE_VERSION}" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys
import zipfile

artifact_dir = Path(sys.argv[1])
expected_sha = sys.argv[2]
dataset = sys.argv[3]
trial = sys.argv[4]
profile = sys.argv[5]
seed = int(sys.argv[6])
candidate_version = sys.argv[7]
metrics_path = artifact_dir / "test_audit_metrics_by_target_horizon.csv"
invariant_path = artifact_dir / "test_audit_invariants.json"
npz_path = artifact_dir / "pcsd_test_audit_diagnostics.npz"
if not all(path.is_file() and path.stat().st_size for path in (
    metrics_path,
    invariant_path,
    npz_path,
)):
    raise SystemExit(1)
rows = list(csv.DictReader(metrics_path.open()))
if len(rows) != 720 or {int(row["target_horizon"]) for row in rows} != set(
    range(1, 721)
):
    raise SystemExit(1)
if not all(
    math.isfinite(float(row[metric]))
    for row in rows
    for metric in ("mse", "mae")
):
    raise SystemExit(1)
if not all(
    row.get("evaluation_split") == "test"
    and row.get("candidate_version") == candidate_version
    and row.get("hyperparameter_trial_id") == trial
    and row.get("hyperparameter_profile_id") == profile
    and int(row.get("seed", -1)) == seed
    for row in rows
):
    raise SystemExit(1)
invariant = json.loads(invariant_path.read_text())
if not (
    invariant["pass"] is True
    and invariant["dataset"] == dataset
    and invariant["uses_test_split"] is True
    and invariant["test_access_authorized"] is True
    and invariant["checkpoint_sha256"] == expected_sha
    and invariant.get("candidate_version") == candidate_version
    and invariant.get("hyperparameter_trial_id") == trial
    and invariant.get("hyperparameter_profile_id") == profile
    and invariant.get("seed") == seed
):
    raise SystemExit(1)
with zipfile.ZipFile(npz_path) as archive:
    if archive.testzip() is not None:
        raise SystemExit(1)
    names = set(archive.namelist())
if not {"fused_row_bin_mse.npy", "fused_row_bin_mae.npy"} <= names:
    raise SystemExit(1)
PY
}

remote_code_gate() {
  [[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "EXPECTED_COMMIT must be the 40-character pushed commit" >&2
    return 1
  }
  [[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "remote HEAD does not match EXPECTED_COMMIT" >&2
    return 1
  }
  git diff --quiet HEAD -- \
    scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh \
    scripts/analyze_iscf_bsca_main_v1_hpo_test_audit.py \
    scripts/check_iscf_bsca_main_v1_hpo_test_audit.py \
    "${CONFIG}" \
    "${MANIFEST}"
}

gpu_preflight() {
  local gpu output
  for gpu in "${GPU_IDS[@]}"; do
    output="$(nvidia-smi --id="${gpu}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)"
    python3 - "${gpu}" "${output}" <<'PY'
import sys

gpu = sys.argv[1]
used, utilization = (int(value.strip()) for value in sys.argv[2].split(","))
if used > 1024 or utilization > 20:
    raise SystemExit(
        f"GPU {gpu} is not safe: memory.used={used} MiB, utilization={utilization}%"
    )
PY
  done
}

if [[ "${MODE}" == "dry-run" ]]; then
  printf '%s\n' "${JOB_LINES[@]}"
  echo "iscf_bsca_main_test_audit_dry_run=pass jobs=${EXPECTED_RUNS} test_cells=${EXPECTED_CELLS} authorized=${AUTHORIZED} manifest_hash=${MANIFEST_HASH}"
  exit 0
fi

if [[ "${MODE}" == "status" ]]; then
  complete=0
  for line in "${JOB_LINES[@]}"; do
    if test_artifact_pass "${line}"; then
      complete=$((complete + 1))
    fi
  done
  echo "iscf_bsca_main_test_audit_status=$(date -Is) complete=${complete}/${EXPECTED_RUNS} cells=$((complete * 4))/${EXPECTED_CELLS}"
  find "${TEST_ROOT}/_logs" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

[[ "${AUTHORIZED}" == "true" ]] || {
  echo "official-test audit is not authorized" >&2
  exit 4
}

training_complete=0
for line in "${JOB_LINES[@]}"; do
  training_artifact_pass "${line}"
  training_complete=$((training_complete + 1))
done
[[ "${training_complete}" -eq "${EXPECTED_RUNS}" ]]

existing_targets="$(target_artifact_count)"
existing_temporary="$(temporary_artifact_count)"
if [[ "${MODE}" == "preflight" ]]; then
  remote_code_gate
  gpu_preflight
  echo "iscf_bsca_main_test_audit_preflight=pass training=${EXPECTED_RUNS}/${EXPECTED_RUNS} test_target_files=${existing_targets} temporary_files=${existing_temporary} manifest_hash=${MANIFEST_HASH} commit=${EXPECTED_COMMIT}"
  exit 0
fi

remote_code_gate
gpu_preflight
if [[ "${ALLOW_RESUME}" != "1" && ( "${existing_targets}" -ne 0 || "${existing_temporary}" -ne 0 ) ]]; then
  echo "initial test launch requires zero target/temporary files; found target=${existing_targets} temporary=${existing_temporary}" >&2
  exit 5
fi

mkdir -p "${TEST_ROOT}/_logs"
ABORT_SENTINEL="${TEST_ROOT}/ABORT"
if [[ -e "${ABORT_SENTINEL}" ]]; then
  echo "abort sentinel already exists: ${ABORT_SENTINEL}" >&2
  exit 6
fi

{
  echo "start=$(date -Is)"
  echo "mode=${MODE}"
  echo "commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "config_hash=${CONFIG_HASH}"
  echo "manifest=${MANIFEST}"
  echo "manifest_hash=${MANIFEST_HASH}"
  echo "test_root=${TEST_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${EXPECTED_RUNS}"
  echo "test_cells=${EXPECTED_CELLS}"
  echo "allow_resume=${ALLOW_RESUME}"
  echo "expected_commit=${EXPECTED_COMMIT}"
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits || true
} | tee "${TEST_ROOT}/test_launch_record.txt"
printf '%s\n' "${JOB_LINES[@]}" >"${TEST_ROOT}/test_jobs.tsv"

QUEUE_STATE="${TEST_ROOT}/.test_queue_index"
QUEUE_LOCK="${TEST_ROOT}/.test_queue_lock"
echo 0 >"${QUEUE_STATE}"

next_job() {
  (
    flock -x 9
    [[ ! -e "${ABORT_SENTINEL}" ]] || exit 1
    local index
    index="$(<"${QUEUE_STATE}")"
    if [[ "${index}" -ge "${EXPECTED_RUNS}" ]]; then
      exit 1
    fi
    echo $((index + 1)) >"${QUEUE_STATE}"
    printf '%s' "${index}"
  ) 9>"${QUEUE_LOCK}"
}

run_one() {
  local index="$1" line="$2" gpu="$3"
  local phase dataset trial profile seed expected_sha run_dir test_dir log temp_dir
  IFS=$'\t' read -r phase dataset trial profile seed expected_sha run_dir \
    test_dir <<< "${line}"
  log="${TEST_ROOT}/_logs/test_${trial}.log"
  if [[ "${ALLOW_RESUME}" == "1" ]] && test_artifact_pass "${line}"; then
    echo "skip_existing job=$((index + 1))/${EXPECTED_RUNS} trial=${trial}"
    return
  fi
  if [[ -e "${test_dir}" ]]; then
    touch "${ABORT_SENTINEL}"
    echo "refusing incomplete or stale target directory: ${test_dir}" >&2
    return 1
  fi
  temp_dir="${TEST_ROOT}/_tmp/${trial}.worker-${gpu}"
  if [[ -e "${temp_dir}" ]]; then
    touch "${ABORT_SENTINEL}"
    echo "refusing stale temporary directory: ${temp_dir}" >&2
    return 1
  fi
  [[ "$(sha256_file "${run_dir}/checkpoint.pt")" == "${expected_sha}" ]]
  echo "run_start=$(date -Is) job=$((index + 1))/${EXPECTED_RUNS} dataset=${dataset} trial=${trial} gpu=${gpu}"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run \
    --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${run_dir}" \
      --artifact-dir "${temp_dir}" \
      --design configs/iscf_bsca_main_v1_hpo.json \
      --device cuda \
      --evaluation-split test \
      --test-audit-config "${CONFIG}" \
      --probe-rows 256 >"${log}" 2>&1; then
    touch "${ABORT_SENTINEL}"
    echo "test evaluator failed for ${trial}; see ${log}" >&2
    return 1
  fi
  if [[ "$(sha256_file "${run_dir}/checkpoint.pt")" != "${expected_sha}" ]]; then
    touch "${ABORT_SENTINEL}"
    echo "checkpoint mutated during test: ${trial}" >&2
    return 1
  fi
  if ! test_artifact_pass "${line}" "${temp_dir}"; then
    touch "${ABORT_SENTINEL}"
    echo "incomplete or invalid test artifact: ${trial}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${test_dir}")"
  mv "${temp_dir}" "${test_dir}"
  test_artifact_pass "${line}"
  echo "run_done=$(date -Is) job=$((index + 1))/${EXPECTED_RUNS} dataset=${dataset} trial=${trial} gpu=${gpu}" | tee -a "${log}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do
    run_one "${index}" "${JOB_LINES[${index}]}" "${gpu}" || return 1
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

complete=0
for line in "${JOB_LINES[@]}"; do
  test_artifact_pass "${line}"
  complete=$((complete + 1))
done
[[ "${complete}" -eq "${EXPECTED_RUNS}" ]]
echo "iscf_bsca_main_test_audit_done=$(date -Is) jobs=${EXPECTED_RUNS} test_cells=${EXPECTED_CELLS}"
