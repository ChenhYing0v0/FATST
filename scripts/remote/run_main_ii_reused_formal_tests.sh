#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/main_ii_h720_prefix_20260808}"
TIMEALIGN_MAIN_ROOT="${TIMEALIGN_MAIN_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/main_i_8dataset_20260806/runs}"
TIMEALIGN_REUSE_ROOT="${TIMEALIGN_REUSE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/ettm2_weather_20260804/runs}"
QDF_ROOT="${QDF_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/qdf_main_i_seq336_20260806}"
AMD_SIMPLETM_ROOT="${AMD_SIMPLETM_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/amd_simpletm_main_i_7dataset_20260807_recovery}"
MODE="${1:-dry-run}"

case "${MODE}" in
  dry-run|run|status) ;;
  *) echo "usage: $0 {dry-run|run|status}" >&2; exit 2 ;;
esac

PROTOCOL="${REPO_ROOT}/configs/iscf_bsca_main_ii_h720_prefix_protocol.json"
CHECKPOINT_MANIFEST="${REPO_ROOT}/analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/checkpoint_and_source_manifest.csv"
ISCF_JSON="${OUTPUT_ROOT}/formal_test_reused/ISCF-BSCA-MAIN-v1/prefix_metrics.json"
EVALUATION_MANIFEST="${OUTPUT_ROOT}/reused_evaluation_manifest.json"

run_preflight() {
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/check_main_ii_reused_artifacts.py" \
      --checkpoint-manifest "${CHECKPOINT_MANIFEST}" \
      --timealign-main-root "${TIMEALIGN_MAIN_ROOT}" \
      --timealign-reuse-root "${TIMEALIGN_REUSE_ROOT}" \
      --qdf-root "${QDF_ROOT}" \
      --amd-simpletm-root "${AMD_SIMPLETM_ROOT}" \
      --iscf-reuse-json "${ISCF_JSON}" \
      --output "${EVALUATION_MANIFEST}"
}

is_complete() {
  local directory="$1"
  [[ -s "${directory}/prefix_metrics.csv" && -s "${directory}/prefix_metrics.json" ]] || return 1
  python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["gate"] == "pass"' \
    "${directory}/prefix_metrics.json"
}

if [[ "${MODE}" == "status" ]]; then
  complete=0
  for path in "${OUTPUT_ROOT}"/formal_test_reused/*/prefix_metrics.json; do
    [[ -e "${path}" ]] || continue
    if is_complete "$(dirname "${path}")"; then complete=$((complete + 1)); fi
  done
  echo "reused_formal_test_complete=${complete}/43_directories"
  echo "checkpoint_evaluations_complete=$((complete == 0 ? 0 : complete + 6))/49_including_ISCF7"
  exit 0
fi

run_preflight
if [[ "${MODE}" == "dry-run" ]]; then
  python3 - "${EVALUATION_MANIFEST}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert payload["new_test_access_planned"] == 42
assert payload["total_reused_checkpoint_evaluations"] == 49
print("main_ii_reused_formal_test_dry_run=pass new_tests=42 completed_ISCF=7 total=49")
PY
  exit 0
fi

authorized="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_prefix_test"]).lower())' "${PROTOCOL}")"
[[ "${authorized}" == "true" ]] || { echo "formal prefix test not authorized" >&2; exit 3; }
training_complete="$(find "${OUTPUT_ROOT}/training" -mindepth 2 -maxdepth 2 -name DONE -type f 2>/dev/null | wc -l | tr -d ' ')"
[[ "${training_complete}" == "21" ]] || {
  echo "new H720 formal training incomplete: ${training_complete}/21" >&2
  exit 4
}

mkdir -p "${OUTPUT_ROOT}/formal_test_reused"
{
  echo "launch_time=$(date -Is)"
  echo "mode=run"
  echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "protocol_sha256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
  echo "preflight_sha256=$(sha256sum "${EVALUATION_MANIFEST}" | awk '{print $1}')"
  echo "gpu=${GPU}"
  quota -s 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
} >"${OUTPUT_ROOT}/reused_formal_test_launch_record.txt"

while IFS=$'\t' read -r system dataset repeat checkpoint expected path_a path_b; do
  output="${OUTPUT_ROOT}/formal_test_reused/${system}__${dataset}__r${repeat}"
  if is_complete "${output}"; then
    echo "skip_existing system=${system} dataset=${dataset} repeat=${repeat}"
    continue
  fi
  [[ ! -e "${output}" ]] || {
    echo "incomplete formal-test output exists: ${output}" >&2
    exit 5
  }
  echo "test_start=$(date -Is) system=${system} dataset=${dataset} repeat=${repeat} gpu=${GPU}"
  if [[ "${system}" == "TimeAlign" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/evaluate_main_ii_timealign_checkpoint.py" \
        --effective-config "${path_a}" --checkpoint "${checkpoint}" \
        --anchor-metrics "${path_b}" --dataset "${dataset}" \
        --repeat "${repeat}" --output-dir "${output}"
  elif [[ "${system}" == "QDF" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/evaluate_main_ii_qdf_checkpoint.py" \
        --config-yaml "${path_a}" --checkpoint "${checkpoint}" \
        --anchor-metrics "${path_b}" --expected-checkpoint-sha256 "${expected}" \
        --dataset "${dataset}" --output-dir "${output}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python "${REPO_ROOT}/scripts/evaluate_main_ii_amd_simpletm_checkpoint.py" \
        --baseline "${system}" --run-dir "${path_a}" --dataset "${dataset}" \
        --repeat "${repeat}" --output-dir "${output}"
  fi
  is_complete "${output}"
  echo "test_done=$(date -Is) system=${system} dataset=${dataset} repeat=${repeat} gpu=${GPU}"
done < <(python3 - "${EVALUATION_MANIFEST}" <<'PY'
import json
import sys

for job in json.load(open(sys.argv[1]))["jobs"]:
    system = job["system"]
    if system == "TimeAlign":
        path_a, path_b = job["effective_config"], job["anchor_metrics"]
    elif system == "QDF":
        path_a, path_b = job["config_yaml"], job["anchor_metrics"]
    else:
        path_a, path_b = job["run_dir"], "unused"
    print("\t".join(map(str, (
        system, job["dataset"], job["repeat"], job["checkpoint"],
        job["checkpoint_sha256"], path_a, path_b,
    ))))
PY
)

complete="$(find "${OUTPUT_ROOT}/formal_test_reused" -mindepth 2 -maxdepth 2 -name prefix_metrics.json -type f | wc -l | tr -d ' ')"
[[ "${complete}" == "43" ]]
echo "main_ii_reused_formal_test_done=$(date -Is) directories=43 checkpoint_evaluations=49 raw_rows=196"
