#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/amd_simpletm_official_main_i_reproduction.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DATA_ROOT="${DATA_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/amd_simpletm_main_i_7dataset_20260806}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MODE="${1:-dry-run}"

case "${MODE}" in
  dry-run|resource-smoke|run|status) ;;
  *) echo "usage: $0 {dry-run|resource-smoke|run|status}" >&2; exit 2 ;;
esac

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
[[ "${#GPU_IDS[@]}" -ge 1 ]]

JOBS=()
while IFS= read -r job; do
  JOBS+=("${job}")
done < <(python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
for item in config["remote_execution"]["workload_order"]:
    baseline, dataset = item.split(":", 1)
    print(f"{baseline}\t{dataset}")
PY
)
[[ "${#JOBS[@]}" -eq 14 ]]

if [[ "${MODE}" == "dry-run" ]]; then
  printf '%s\n' "${JOBS[@]}"
  python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
assert config["authorization"]["remote_training_authorized"] is True
assert config["authorization"]["formal_test_authorized"] is True
assert config["evaluation_contract"]["expected_table_cells"] == 56
assert config["evaluation_contract"]["expected_training_repetitions"] == 110
assert config["baselines"]["AMD"]["expected_metric_rows"] == 28
assert config["baselines"]["SimpleTM"]["expected_metric_rows"] == 82
print("amd_simpletm_dry_run=pass units=14 cells=56 repetitions=110")
PY
  exit 0
fi

unit_dir() {
  local root="$1" baseline="$2" dataset="$3"
  printf '%s/%s__%s' "${root}" "${baseline}" "${dataset}"
}

is_complete() {
  local root="$1" baseline="$2" dataset="$3" directory
  directory="$(unit_dir "${root}" "${baseline}" "${dataset}")"
  [[ -s "${directory}/complete.json" && -s "${directory}/run.log" ]] || return 1
  if [[ "${root}" == "${OUTPUT_ROOT}/runs" ]]; then
    [[ -s "${directory}/metrics.csv" ]] || return 1
  fi
}

if [[ "${MODE}" == "status" ]]; then
  for scope in _resource_smoke runs; do
    complete=0
    root="${OUTPUT_ROOT}/${scope}"
    for job in "${JOBS[@]}"; do
      IFS=$'\t' read -r baseline dataset <<< "${job}"
      if is_complete "${root}" "${baseline}" "${dataset}"; then
        complete=$((complete + 1))
        printf '%s\t%s\t%s\tcomplete\n' "${scope}" "${baseline}" "${dataset}"
      else
        printf '%s\t%s\t%s\tincomplete\n' "${scope}" "${baseline}" "${dataset}"
      fi
    done
    echo "${scope}_complete=${complete}/14"
  done
  exit 0
fi

AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
[[ "${AUTHORIZED}" == "true" ]] || { echo "remote training not authorized" >&2; exit 3; }

if [[ "${MODE}" == "run" ]]; then
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r baseline dataset <<< "${job}"
    is_complete "${OUTPUT_ROOT}/_resource_smoke" "${baseline}" "${dataset}" || {
      echo "resource smoke gate incomplete for ${baseline}:${dataset}" >&2
      exit 4
    }
  done
fi

mkdir -p "${OUTPUT_ROOT}/_upstream"

ensure_checkout() {
  local baseline="$1" url commit destination actual
  read -r url commit < <(python3 - "${CONFIG}" "${baseline}" <<'PY'
import json
import sys

spec = json.load(open(sys.argv[1]))["baselines"][sys.argv[2]]
print(spec["source_repository"], spec["source_commit"])
PY
)
  destination="${OUTPUT_ROOT}/_upstream/${baseline}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${destination}" fetch --quiet origin "${commit}"
  fi
  git -C "${destination}" checkout --quiet --detach "${commit}"
  actual="$(git -C "${destination}" rev-parse HEAD)"
  [[ "${actual}" == "${commit}" ]]
  [[ -z "$(git -C "${destination}" status --short)" ]]
}

ensure_checkout AMD
ensure_checkout SimpleTM

mkdir -p "${OUTPUT_ROOT}"
{
  echo "launch_time=$(date -Is)"
  echo "mode=${MODE}"
  echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
  echo "adapter_sha256=$(sha256sum "${REPO_ROOT}/scripts/run_amd_simpletm_official_dataset.py" | awk '{print $1}')"
  echo "output_root=${OUTPUT_ROOT}"
  echo "data_root=${DATA_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "units=14"
  echo "formal_cells=$([[ "${MODE}" == "run" ]] && echo 56 || echo 0)"
  echo "formal_training_repetitions=$([[ "${MODE}" == "run" ]] && echo 110 || echo 0)"
  quota -s 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c 'import platform,torch,numpy,pandas,pywt; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(numpy.__version__); print(pandas.__version__); print(pywt.__version__)'
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record.txt"

MODE_ROOT="${OUTPUT_ROOT}/runs"
if [[ "${MODE}" == "resource-smoke" ]]; then
  MODE_ROOT="${OUTPUT_ROOT}/_resource_smoke"
fi
mkdir -p "${MODE_ROOT}"

QUEUE_STATE="${OUTPUT_ROOT}/.${MODE}_queue_index"
QUEUE_LOCK="${OUTPUT_ROOT}/.${MODE}_queue_lock"
echo 0 > "${QUEUE_STATE}"

next_job() {
  (
    flock -x 9
    local index
    index="$(<"${QUEUE_STATE}")"
    if [[ "${index}" -ge "${#JOBS[@]}" ]]; then exit 1; fi
    echo $((index + 1)) > "${QUEUE_STATE}"
    printf '%s' "${index}"
  ) 9>"${QUEUE_LOCK}"
}

run_one() {
  local index="$1" line="$2" gpu="$3"
  local baseline dataset directory
  IFS=$'\t' read -r baseline dataset <<< "${line}"
  directory="$(unit_dir "${MODE_ROOT}" "${baseline}" "${dataset}")"
  if is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"; then
    echo "skip_existing job=$((index + 1))/14 baseline=${baseline} dataset=${dataset}"
    return
  fi
  if [[ -e "${directory}" ]]; then
    echo "incomplete unit exists; refusing overwrite: ${directory}" >&2
    return 5
  fi
  echo "run_start=$(date -Is) job=$((index + 1))/14 baseline=${baseline} dataset=${dataset} gpu=${gpu} mode=${MODE}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/run_amd_simpletm_official_dataset.py" \
      --config "${CONFIG}" \
      --baseline "${baseline}" \
      --dataset "${dataset}" \
      --mode "${MODE}" \
      --source-root "${OUTPUT_ROOT}/_upstream/${baseline}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${directory}" \
      --gpu "${gpu}"
  is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"
  echo "run_done=$(date -Is) job=$((index + 1))/14 baseline=${baseline} dataset=${dataset} gpu=${gpu} mode=${MODE}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do
    run_one "${index}" "${JOBS[${index}]}" "${gpu}"
  done
}

pids=()
for gpu in "${GPU_IDS[@]}"; do
  worker "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" == "0" ]]

complete=0
for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r baseline dataset <<< "${job}"
  if is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"; then
    complete=$((complete + 1))
  fi
done
[[ "${complete}" -eq 14 ]]
echo "amd_simpletm_${MODE}_done=$(date -Is) units=14 complete=14"
