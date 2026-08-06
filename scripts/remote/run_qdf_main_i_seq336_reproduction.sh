#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yingch/projects/FATST}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
DATA_ROOT="${DATA_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/qdf_main_i_seq336_20260806}"
GPU_IDS="${GPU_IDS:-0 1 2}"
MODE="${1:-dry-run}"
CONFIG="${REPO_ROOT}/configs/qdf_main_i_seq336_reproduction.json"
SOURCE_ROOT="${REPO_ROOT}/baselines/qdf_official"

case "${MODE}" in
  dry-run|resource-smoke|run|status) ;;
  *) echo "usage: $0 {dry-run|resource-smoke|run|status}" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${CONDA_ENV}"
python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["authorization"]["remote_training_authorized"]; assert c["authorization"]["formal_test_authorized"]' "${CONFIG}"

verify_data() {
  python - "${CONFIG}" "${DATA_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

config = json.load(open(sys.argv[1]))
root = pathlib.Path(sys.argv[2])
for dataset, spec in config["dataset_contracts"].items():
    path = root / spec["relative_path"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        raise SystemExit(f"{dataset} data hash mismatch: {actual}")
    print(f"{dataset}\t{actual}\t{path}")
PY
}

status_one() {
  local mode_root="$1" dataset="$2" horizon="$3"
  local run_root="${mode_root}/QDF__${dataset}__H${horizon}__seed2023"
  local checkpoint_count=0 a_count=0 metrics_count=0 config_count=0
  [[ -d "${run_root}" ]] || { printf '%s\tH%s\tmissing_run_root\n' "${dataset}" "${horizon}"; return; }
  checkpoint_count="$(find "${run_root}/checkpoints" -name checkpoint.pth -type f -size +0c 2>/dev/null | wc -l)"
  a_count="$(find "${run_root}/checkpoints" -name A.pth -type f -size +0c 2>/dev/null | wc -l)"
  metrics_count="$(find "${run_root}/results" -name metrics.npy -type f -size +0c 2>/dev/null | wc -l)"
  config_count="$(find "${run_root}/results" -name config.yaml -type f -size +0c 2>/dev/null | wc -l)"
  printf '%s\tH%s\tcheckpoint=%s\tA=%s\tmetrics=%s\tconfig=%s\n' \
    "${dataset}" "${horizon}" "${checkpoint_count}" "${a_count}" "${metrics_count}" "${config_count}"
}

if [[ "${MODE}" == "status" ]]; then
  echo "formal:"
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Solar Exchange; do
    for horizon in 96 192 336 720; do status_one "${OUTPUT_ROOT}/runs" "${dataset}" "${horizon}"; done
  done
  echo "resource_smoke:"
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Solar Exchange; do status_one "${OUTPUT_ROOT}/_resource_smoke" "${dataset}" 720; done
  exit 0
fi

data_manifest="$(verify_data)"
if [[ "${MODE}" == "run" ]]; then
  smoke_failure=0
  for dataset in ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Solar Exchange; do
    run_root="${OUTPUT_ROOT}/_resource_smoke/QDF__${dataset}__H720__seed2023"
    [[ "$(find "${run_root}/checkpoints" -name checkpoint.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]] || smoke_failure=1
    [[ "$(find "${run_root}/checkpoints" -name A.pth -type f -size +0c 2>/dev/null | wc -l)" == 1 ]] || smoke_failure=1
    [[ "$(find "${run_root}/results" -name metrics.npy -type f -size +0c 2>/dev/null | wc -l)" == 0 ]] || smoke_failure=1
  done
  [[ "${smoke_failure}" == 0 ]] || { echo "resource smoke gate incomplete or contaminated by test" >&2; exit 3; }
fi

mkdir -p "${OUTPUT_ROOT}"
{
  date --iso-8601=seconds
  git rev-parse HEAD
  python -c 'import platform,torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'
  printf '%s\n' "${data_manifest}"
} >"${OUTPUT_ROOT}/environment_${MODE}.txt"

export DATA_ROOT OUTPUT_ROOT GPU_IDS
export PYTHON_BIN=python MODE
bash "${SOURCE_ROOT}/scripts/MainI_L336.sh" | tee "${OUTPUT_ROOT}/queue_${MODE}.log"
