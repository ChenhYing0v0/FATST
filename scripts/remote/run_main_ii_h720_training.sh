#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/iscf_bsca_main_ii_h720_execution.json}"
PROTOCOL="${PROTOCOL:-${REPO_ROOT}/configs/iscf_bsca_main_ii_h720_prefix_protocol.json}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${REPO_ROOT}/analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/checkpoint_and_source_manifest.csv}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DATA_ROOT="${DATA_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${OUTPUT_ROOT}/_workspaces_v3}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MODE="${1:-dry-run}"

case "${MODE}" in
  dry-run|resource-smoke|formal-training|formal-test-new|status) ;;
  *) echo "usage: $0 {dry-run|resource-smoke|formal-training|formal-test-new|status}" >&2; exit 2 ;;
esac

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
[[ "${#GPU_IDS[@]}" -ge 1 ]]

JOBS=()
while IFS= read -r job; do JOBS+=("${job}"); done < <(
  python3 - "${CONFIG}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1]))["workload_order"]:
    baseline, dataset = item.split(":", 1)
    print(f"{baseline}\t{dataset}")
PY
)
[[ "${#JOBS[@]}" -eq 21 ]]

unit_dir() {
  local root="$1" baseline="$2" dataset="$3"
  printf '%s/%s__%s' "${root}" "${baseline}" "${dataset}"
}

is_complete() {
  local root="$1" baseline="$2" dataset="$3" directory
  directory="$(unit_dir "${root}" "${baseline}" "${dataset}")"
  [[ -s "${directory}/DONE" && -s "${directory}/artifact_manifest.json" && -s "${directory}/run.log" ]] || return 1
  if [[ "${root}" == "${OUTPUT_ROOT}/formal_test_new" ]]; then
    [[ -s "${directory}/prefix/prefix_metrics.csv" && -s "${directory}/prefix/prefix_metrics.json" ]]
  fi
}

if [[ "${MODE}" == "status" ]]; then
  for scope in _resource_smoke training formal_test_new; do
    complete=0
    for job in "${JOBS[@]}"; do
      IFS=$'\t' read -r baseline dataset <<< "${job}"
      if is_complete "${OUTPUT_ROOT}/${scope}" "${baseline}" "${dataset}"; then
        complete=$((complete + 1))
      fi
    done
    echo "${scope}_complete=${complete}/21"
  done
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_upstream" "${WORKSPACE_ROOT}"

ensure_checkout() {
  local baseline="$1" destination repository commit actual
  read -r repository commit < <(
    python3 - "${CONFIG}" "${baseline}" <<'PY'
import json
import sys
spec = json.load(open(sys.argv[1]))["training_baselines"][sys.argv[2]]
print(spec["repository"], spec["commit"])
PY
  )
  destination="${OUTPUT_ROOT}/_upstream/${baseline}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${repository}" "${destination}"
  fi
  if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${destination}" fetch --quiet origin "${commit}"
  fi
  git -C "${destination}" checkout --quiet --detach "${commit}"
  actual="$(git -C "${destination}" rev-parse HEAD)"
  [[ "${actual}" == "${commit}" ]]
  [[ -z "$(git -C "${destination}" status --short)" ]]
}

ensure_checkout iTransformer
ensure_checkout PatchTST
ensure_checkout DLinear

ensure_workspace() {
  local baseline="$1"
  local workspace="${WORKSPACE_ROOT}/${baseline}"
  if [[ -s "${workspace}/fatst_runtime_patch_manifest.json" ]]; then return; fi
  [[ ! -e "${workspace}" ]] || { echo "incomplete workspace exists: ${workspace}" >&2; exit 3; }
  args=(
    --config "${CONFIG}" --action prepare --baseline "${baseline}"
    --source-root "${OUTPUT_ROOT}/_upstream/${baseline}" --workspace "${workspace}"
  )
  if [[ "${baseline}" != "iTransformer" ]]; then
    args+=(--itransformer-source-root "${OUTPUT_ROOT}/_upstream/iTransformer")
  fi
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/run_main_ii_h720_training_job.py" "${args[@]}"
}

ensure_workspace iTransformer
ensure_workspace PatchTST
ensure_workspace DLinear

if [[ "${MODE}" == "dry-run" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/check_main_ii_h720_prelaunch.py" \
      --config "${CONFIG}" --protocol "${PROTOCOL}" \
      --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
      --itransformer-workspace "${WORKSPACE_ROOT}/iTransformer" \
      --patchtst-workspace "${WORKSPACE_ROOT}/PatchTST" \
      --dlinear-workspace "${WORKSPACE_ROOT}/DLinear" \
      --data-root "${DATA_ROOT}" \
      --output "${OUTPUT_ROOT}/remote_prelaunch_gate.json"
  exit 0
fi

{
  echo "launch_time=$(date -Is)"
  echo "mode=${MODE}"
  echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
  echo "adapter_sha256=$(sha256sum "${REPO_ROOT}/scripts/run_main_ii_h720_training_job.py" | awk '{print $1}')"
  echo "evaluator_sha256=$(sha256sum "${REPO_ROOT}/scripts/evaluate_main_ii_h720_prefix_arrays.py" | awk '{print $1}')"
  echo "output_root=${OUTPUT_ROOT}"
  echo "data_root=${DATA_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  quota -s 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c 'import platform,torch,numpy,pandas; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(numpy.__version__); print(pandas.__version__)'
} | tee "${OUTPUT_ROOT}/${MODE}_launch_record.txt"

if [[ "${MODE}" == "formal-training" ]]; then
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r baseline dataset <<< "${job}"
    is_complete "${OUTPUT_ROOT}/_resource_smoke" "${baseline}" "${dataset}" || {
      echo "resource smoke incomplete: ${baseline}:${dataset}" >&2; exit 4;
    }
  done
fi

if [[ "${MODE}" == "formal-test-new" ]]; then
  authorized="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_prefix_test"]).lower())' "${PROTOCOL}")"
  [[ "${authorized}" == "true" ]]
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r baseline dataset <<< "${job}"
    is_complete "${OUTPUT_ROOT}/training" "${baseline}" "${dataset}" || {
      echo "formal checkpoint incomplete: ${baseline}:${dataset}" >&2; exit 5;
    }
  done
  mkdir -p "${OUTPUT_ROOT}/formal_test_new"
  gpu="${GPU_IDS[0]}"
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r baseline dataset <<< "${job}"
    output="$(unit_dir "${OUTPUT_ROOT}/formal_test_new" "${baseline}" "${dataset}")"
    training="$(unit_dir "${OUTPUT_ROOT}/training" "${baseline}" "${dataset}")"
    if is_complete "${OUTPUT_ROOT}/formal_test_new" "${baseline}" "${dataset}"; then continue; fi
    [[ ! -e "${output}" ]] || { echo "incomplete test unit exists: ${output}" >&2; exit 6; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/run_main_ii_h720_training_job.py" \
        --config "${CONFIG}" --action run --baseline "${baseline}" \
        --workspace "${WORKSPACE_ROOT}/${baseline}" \
        --dataset "${dataset}" --data-root "${DATA_ROOT}" \
        --output-dir "${output}" --checkpoint-dir "${training}" --mode formal-test
    checkpoint_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "${output}/artifact_manifest.json")"
    prediction_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prediction_path"])' "${output}/artifact_manifest.json")"
    target_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_path"])' "${output}/artifact_manifest.json")"
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/evaluate_main_ii_h720_prefix_arrays.py" \
        --prediction "${prediction_path}" --target "${target_path}" --layout NTC \
        --system "${baseline}" --dataset "${dataset}" --repeat 0 \
        --checkpoint-sha256 "${checkpoint_sha}" --output-dir "${output}/prefix" \
        --remove-input-arrays-after-success
  done
  echo "main_ii_formal_test_new_done=$(date -Is) units=21 raw_rows=84"
  exit 0
fi

MODE_ROOT="${OUTPUT_ROOT}/_resource_smoke"
RUN_MODE="resource-smoke"
if [[ "${MODE}" == "formal-training" ]]; then
  MODE_ROOT="${OUTPUT_ROOT}/training"
  RUN_MODE="formal-training"
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
  local index="$1" line="$2" gpu="$3" baseline dataset output
  IFS=$'\t' read -r baseline dataset <<< "${line}"
  output="$(unit_dir "${MODE_ROOT}" "${baseline}" "${dataset}")"
  if is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"; then return; fi
  [[ ! -e "${output}" ]] || { echo "incomplete unit exists: ${output}" >&2; return 7; }
  echo "run_start=$(date -Is) job=$((index + 1))/21 baseline=${baseline} dataset=${dataset} gpu=${gpu} mode=${RUN_MODE}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/run_main_ii_h720_training_job.py" \
      --config "${CONFIG}" --action run --baseline "${baseline}" \
      --workspace "${WORKSPACE_ROOT}/${baseline}" \
      --dataset "${dataset}" --data-root "${DATA_ROOT}" \
      --output-dir "${output}" --mode "${RUN_MODE}"
  is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"
  echo "run_done=$(date -Is) job=$((index + 1))/21 baseline=${baseline} dataset=${dataset} gpu=${gpu} mode=${RUN_MODE}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do run_one "${index}" "${JOBS[${index}]}" "${gpu}"; done
}

pids=()
for gpu in "${GPU_IDS[@]}"; do worker "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
[[ "${status}" == "0" ]]

complete=0
for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r baseline dataset <<< "${job}"
  if is_complete "${MODE_ROOT}" "${baseline}" "${dataset}"; then complete=$((complete + 1)); fi
done
[[ "${complete}" -eq 21 ]]

size_kib="$(du -sk "${OUTPUT_ROOT}" | awk '{print $1}')"
budget_kib=$((40 * 1024 * 1024))
[[ "${size_kib}" -le "${budget_kib}" ]]
echo "main_ii_${MODE}_done=$(date -Is) units=21 complete=21 size_kib=${size_kib}"
