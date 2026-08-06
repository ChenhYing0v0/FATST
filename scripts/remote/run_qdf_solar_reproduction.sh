#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DATA_ROOT="${DATA_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/qdf_solar_reproduction_20260806}"
GPU_IDS="${GPU_IDS:-0 1 2}"
MODE="${1:-dry-run}"
CONFIG="${REPO_ROOT}/configs/qdf_solar_reproduction.json"
SOURCE_ROOT="${REPO_ROOT}/baselines/qdf_official"

case "${MODE}" in
  dry-run|resource-smoke|run|status) ;;
  *) echo "usage: $0 {dry-run|resource-smoke|run|status}" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${CONDA_ENV}"
python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["authorization"]["remote_training_authorized"]; assert c["authorization"]["formal_test_authorized"]' "${CONFIG}"
expected_data_sha="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_contract"]["data_sha256"])' "${CONFIG}")"
actual_data_sha="$(sha256sum "${DATA_ROOT}/Solar/solar_AL.txt" | awk '{print $1}')"
[[ "${actual_data_sha}" == "${expected_data_sha}" ]]

status_one() {
  local mode_root="$1" horizon="$2"
  local run_root="${mode_root}/QDF__Solar__H${horizon}__seed2023"
  local checkpoint_count=0 a_count=0 metrics_count=0 config_count=0
  [[ -d "${run_root}" ]] || { printf 'H%s\tmissing_run_root\n' "${horizon}"; return; }
  checkpoint_count="$(find "${run_root}/checkpoints" -name checkpoint.pth -type f -size +0c 2>/dev/null | wc -l)"
  a_count="$(find "${run_root}/checkpoints" -name A.pth -type f -size +0c 2>/dev/null | wc -l)"
  metrics_count="$(find "${run_root}/results" -name metrics.npy -type f -size +0c 2>/dev/null | wc -l)"
  config_count="$(find "${run_root}/results" -name config.yaml -type f -size +0c 2>/dev/null | wc -l)"
  printf 'H%s\tcheckpoint=%s\tA=%s\tmetrics=%s\tconfig=%s\n' \
    "${horizon}" "${checkpoint_count}" "${a_count}" "${metrics_count}" "${config_count}"
}

if [[ "${MODE}" == "status" ]]; then
  echo "formal:"
  for horizon in 96 192 336 720; do status_one "${OUTPUT_ROOT}/runs" "${horizon}"; done
  echo "resource_smoke:"
  for horizon in 96 192 336 720; do status_one "${OUTPUT_ROOT}/_resource_smoke" "${horizon}"; done
  exit 0
fi

if [[ "${MODE}" == "run" ]]; then
  smoke_ok=0
  for horizon in 96 192 336 720; do
    run_root="${OUTPUT_ROOT}/_resource_smoke/QDF__Solar__H${horizon}__seed2023"
    [[ "$(find "${run_root}/checkpoints" -name checkpoint.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]] || smoke_ok=1
    [[ "$(find "${run_root}/checkpoints" -name A.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]] || smoke_ok=1
    [[ "$(find "${run_root}/results" -name metrics.npy -type f -size +0c 2>/dev/null | wc -l)" == 0 ]] || smoke_ok=1
  done
  [[ "${smoke_ok}" == 0 ]] || { echo "resource smoke gate is incomplete or contaminated by test" >&2; exit 3; }
fi

mkdir -p "${OUTPUT_ROOT}"
{
  date --iso-8601=seconds
  git rev-parse HEAD
  python -c 'import platform,torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'
  sha256sum "${DATA_ROOT}/Solar/solar_AL.txt"
} >"${OUTPUT_ROOT}/environment_${MODE}.txt"

export DATA_ROOT OUTPUT_ROOT GPU_IDS
export PYTHON_BIN=python
export MODE
bash "${SOURCE_ROOT}/scripts/Solar.sh" | tee "${OUTPUT_ROOT}/queue_${MODE}.log"
