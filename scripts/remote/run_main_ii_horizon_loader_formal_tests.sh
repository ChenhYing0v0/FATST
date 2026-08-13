#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
OLD_ROOT="${OLD_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/main_ii_horizon_loader_20260813}"
TIMEALIGN_MAIN_ROOT="${TIMEALIGN_MAIN_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/main_i_8dataset_20260806/runs}"
TIMEALIGN_REUSE_ROOT="${TIMEALIGN_REUSE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/ettm2_weather_20260804/runs}"
QDF_ROOT="${QDF_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/qdf_main_i_seq336_20260806}"
AMD_SIMPLETM_ROOT="${AMD_SIMPLETM_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/amd_simpletm_main_i_7dataset_20260807_recovery}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
MODE="${1:-dry-run}"

case "${MODE}" in
  dry-run|run|status) ;;
  *) echo "usage: $0 {dry-run|run|status}" >&2; exit 2 ;;
esac
[[ -n "${EXPECTED_COMMIT}" ]] || { echo "EXPECTED_COMMIT is required" >&2; exit 3; }
actual_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "commit mismatch expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2
  exit 4
}

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
EXECUTION_CONFIG="${REPO_ROOT}/configs/iscf_bsca_main_ii_h720_execution.json"
PROTOCOL="${REPO_ROOT}/configs/iscf_bsca_main_ii_horizon_loader_protocol.json"
CHECKPOINT_MANIFEST="${REPO_ROOT}/analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/checkpoint_and_source_manifest.csv"
REUSED_MANIFEST="${OUTPUT_ROOT}/reused_h720_checkpoint_manifest.json"
JOB_MANIFEST="${OUTPUT_ROOT}/formal_job_manifest.json"
WORKSPACE_ROOT="${OLD_ROOT}/_workspaces_v3"
TRAINING_ROOT="${OLD_ROOT}/training"

mkdir -p "${OUTPUT_ROOT}"

build_manifest() {
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/check_main_ii_reused_artifacts.py" \
      --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
      --timealign-main-root "${TIMEALIGN_MAIN_ROOT}" \
      --timealign-reuse-root "${TIMEALIGN_REUSE_ROOT}" \
      --qdf-root "${QDF_ROOT}" \
      --amd-simpletm-root "${AMD_SIMPLETM_ROOT}" \
      --iscf-reuse-json "${OLD_ROOT}/formal_test_reused/ISCF-BSCA-MAIN-v1/prefix_metrics.json" \
      --output "${REUSED_MANIFEST}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/build_main_ii_horizon_loader_job_manifest.py" \
      --reused-manifest "${REUSED_MANIFEST}" \
      --training-root "${TRAINING_ROOT}" \
      --workspace-root "${WORKSPACE_ROOT}" \
      --execution-config "${EXECUTION_CONFIG}" \
      --output "${JOB_MANIFEST}"
}

is_complete() {
  local directory="$1"
  [[ -s "${directory}/prefix_metrics.csv" && -s "${directory}/prefix_metrics.json" ]] || return 1
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
import json, sys
p=json.load(open(sys.argv[1]))
assert p["gate"] == "pass"
assert p["input_only_inference"] is True
assert p["future_label_used_as_model_input"] is False
' "${directory}/prefix_metrics.json" >/dev/null
}

if [[ "${MODE}" == "status" ]]; then
  complete="$(find "${OUTPUT_ROOT}/formal" -mindepth 2 -maxdepth 2 -name prefix_metrics.json -type f 2>/dev/null | wc -l | tr -d ' ')"
  echo "formal_evaluations_complete=${complete}/252"
  exit 0
fi

build_manifest
if [[ "${MODE}" == "dry-run" ]]; then
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
import json, sys
p=json.load(open(sys.argv[1]))
assert p["gate"] == "pass" and p["formal_evaluations"] == 252
assert p["checkpoint_objects"] == p["unique_checkpoint_hashes"] == 63
print("main_ii_horizon_loader_dry_run=pass evaluations=252 checkpoints=63")
' "${JOB_MANIFEST}"
  exit 0
fi

authorized="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test"]).lower())' "${PROTOCOL}")"
[[ "${authorized}" == "true" ]] || { echo "formal test not authorized" >&2; exit 5; }

mkdir -p "${OUTPUT_ROOT}/formal"
{
  echo "launch_time=$(date -Is)"
  echo "repo_commit=${actual_commit}"
  echo "protocol_sha256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
  echo "job_manifest_sha256=$(sha256sum "${JOB_MANIFEST}" | awk '{print $1}')"
  echo "gpu_ids=${GPU_IDS[*]}"
  quota -s 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
} >"${OUTPUT_ROOT}/formal_launch_record.txt"

mapfile -t JOBS < <("${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
import json, sys
for i,j in enumerate(json.load(open(sys.argv[1]))["jobs"]):
 print("\t".join(map(str,(i,j["system"],j["dataset"],j["repeat"],j["loader_horizon"],j["evaluator_family"]))))
' "${JOB_MANIFEST}")
[[ "${#JOBS[@]}" -eq 252 ]]

QUEUE_STATE="${OUTPUT_ROOT}/.queue_index"
QUEUE_LOCK="${OUTPUT_ROOT}/.queue_lock"
echo 0 >"${QUEUE_STATE}"

next_job() {
  (
    flock -x 9
    local index
    index="$(<"${QUEUE_STATE}")"
    if [[ "${index}" -ge "${#JOBS[@]}" ]]; then exit 1; fi
    echo $((index + 1)) >"${QUEUE_STATE}"
    printf '%s' "${index}"
  ) 9>"${QUEUE_LOCK}"
}

job_json() {
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
import json,sys
print(json.dumps(json.load(open(sys.argv[1]))["jobs"][int(sys.argv[2])]))
' "${JOB_MANIFEST}" "$1"
}

run_one() {
  local queue_index="$1" gpu="$2" payload system dataset repeat horizon family output
  payload="$(job_json "${queue_index}")"
  read -r system dataset repeat horizon family < <(python3 -c '
import json,sys
j=json.loads(sys.argv[1]); print(j["system"],j["dataset"],j["repeat"],j["loader_horizon"],j["evaluator_family"])
' "${payload}")
  output="${OUTPUT_ROOT}/formal/${system}__${dataset}__r${repeat}__H${horizon}"
  if is_complete "${output}"; then return; fi
  [[ ! -e "${output}" ]] || { echo "incomplete output exists: ${output}" >&2; return 6; }
  echo "test_start=$(date -Is) job=$((queue_index + 1))/252 system=${system} dataset=${dataset} H=${horizon} repeat=${repeat} gpu=${gpu}"
  if [[ "${family}" == "reused_native" ]]; then
    if [[ "${system}" == "TimeAlign" ]]; then
      read -r effective checkpoint anchor < <(python3 -c 'import json,sys;j=json.loads(sys.argv[1]);print(j["effective_config"],j["checkpoint"],j["anchor_metrics"])' "${payload}")
      CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python "${REPO_ROOT}/scripts/evaluate_main_ii_timealign_checkpoint.py" \
          --effective-config "${effective}" --checkpoint "${checkpoint}" \
          --anchor-metrics "${anchor}" --dataset "${dataset}" --repeat "${repeat}" \
          --loader-horizon "${horizon}" --output-dir "${output}"
    elif [[ "${system}" == "QDF" ]]; then
      read -r config checkpoint anchor expected < <(python3 -c 'import json,sys;j=json.loads(sys.argv[1]);print(j["config_yaml"],j["checkpoint"],j["anchor_metrics"],j["checkpoint_sha256"])' "${payload}")
      CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python "${REPO_ROOT}/scripts/evaluate_main_ii_qdf_checkpoint.py" \
          --config-yaml "${config}" --checkpoint "${checkpoint}" \
          --anchor-metrics "${anchor}" --expected-checkpoint-sha256 "${expected}" \
          --dataset "${dataset}" --loader-horizon "${horizon}" --output-dir "${output}"
    else
      run_dir="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["run_dir"])' "${payload}")"
      CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python "${REPO_ROOT}/scripts/evaluate_main_ii_amd_simpletm_checkpoint.py" \
          --baseline "${system}" --run-dir "${run_dir}" --dataset "${dataset}" \
          --repeat "${repeat}" --loader-horizon "${horizon}" --output-dir "${output}"
    fi
  else
    read -r workspace training < <(python3 -c 'import json,sys;j=json.loads(sys.argv[1]);print(j["workspace"],j["training_dir"])' "${payload}")
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/evaluate_main_ii_horizon_loader_upstream_checkpoint.py" \
        --baseline "${system}" --config "${EXECUTION_CONFIG}" --workspace "${workspace}" \
        --training-dir "${training}" --dataset "${dataset}" \
        --loader-horizon "${horizon}" --output-dir "${output}"
  fi
  is_complete "${output}"
  observed_drop="$(python3 -c 'import json,sys;p=json.load(open(sys.argv[1]));print(str(p.get("loader_drop_last",p.get("row",{}).get("loader_drop_last"))).lower())' "${output}/prefix_metrics.json")"
  expected_drop="$(python3 -c 'import json,sys;print(str(json.loads(sys.argv[1])["expected_drop_last"]).lower())' "${payload}")"
  [[ "${observed_drop}" == "${expected_drop}" ]]
  echo "test_done=$(date -Is) job=$((queue_index + 1))/252 system=${system} dataset=${dataset} H=${horizon} repeat=${repeat} gpu=${gpu}"
}

worker() {
  local gpu="$1" index
  while index="$(next_job)"; do run_one "${index}" "${gpu}"; done
}

pids=()
for gpu in "${GPU_IDS[@]}"; do worker "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]]
complete="$(find "${OUTPUT_ROOT}/formal" -mindepth 2 -maxdepth 2 -name prefix_metrics.json -type f | wc -l | tr -d ' ')"
[[ "${complete}" == "252" ]]
echo "main_ii_horizon_loader_formal_done=$(date -Is) evaluations=252"
