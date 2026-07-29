#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1}"
CONFIG="${CONFIG:-configs/intro_evidence_visualization_pilot_v1.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DATASET="${DATASET:-Weather}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 3 ]]; then
  echo "visualization pilot requires three GPU ids" >&2
  exit 2
fi
test -s "${CONFIG}"

python3 - "${CONFIG}" "${DATASET}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
dataset = sys.argv[2]
allowed = [config["scope_reduction"]["initial_dataset"], *config["scope_reduction"]["fallback_order"]]
assert dataset in allowed, (dataset, allowed)
assert config["authorization"]["remote_training_initial_9_runs"] is True
assert config["authorization"]["formal_test"] is False
assert config["prefix_disagreement"]["test_accessed"] is False
assert config["sharing_demand"]["test_accessed"] is False
PY

neutral_dir() {
  local scale="$1"
  echo "${OUTPUT_ROOT}/neutral/NeutralSharingExtent/${DATASET}/s${scale}/seed${SEED}"
}

dlinear_dir() {
  local horizon="$1"
  echo "${OUTPUT_ROOT}/dlinear/IntroDLinearPrefixViz/${DATASET}/h${horizon}/seed${SEED}"
}

count_complete() {
  local neutral=0 dlinear=0 scale horizon directory
  for scale in 1 8 32 128 720; do
    directory="$(neutral_dir "${scale}")"
    if [[ -s "${directory}/checkpoint.pt" \
      && -s "${directory}/metrics_val.json" \
      && -s "${directory}/predictions_val.npz" ]]; then
      neutral=$((neutral + 1))
    fi
  done
  for horizon in 96 192 336 720; do
    directory="$(dlinear_dir "${horizon}")"
    if [[ -s "${directory}/checkpoint.pt" \
      && -s "${directory}/metrics_val.json" \
      && -s "${directory}/predictions_val.npz" ]]; then
      dlinear=$((dlinear + 1))
    fi
  done
  echo "${neutral} ${dlinear}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  read -r neutral dlinear <<< "$(count_complete)"
  echo "intro_viz_status=$(date -Is) dataset=${DATASET} neutral=${neutral}/5 dlinear=${dlinear}/4"
  find "${OUTPUT_ROOT}/_logs" -type f -name '*.log' -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  for scale in 1 8 32 128 720; do
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python baselines/intro_evidence_neutral/train.py \
      --sharing-extent "${scale}" --synthetic-smoke --device cpu
  done
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/dlinear/train.py --help >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_prefix_disagreement.py --help >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_sharing_demand.py --help >/dev/null
  echo "intro_evidence_visualization_dry_run=pass jobs=9 dataset=${DATASET}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis/${DATASET}"

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/intro_evidence_neutral/train.py \
    --sharing-extent 720 --synthetic-smoke --device cuda
  echo "intro_evidence_visualization_resource_smoke=pass gpu=${GPU_IDS[0]}"
  exit 0
fi

{
  echo "intro_evidence_visualization_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "dataset=${DATASET}"
  echo "seed=${SEED}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config=${CONFIG}"
  echo "role=exploratory_visualization_only"
  echo "validation_role=checkpoint_selection_and_explanatory_visualization"
  echo "test_accessed=false"
  echo "new_runs=9"
  echo "fallback_authorized=false"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_${DATASET}.txt"

run_neutral() {
  local scale="$1" gpu="$2" directory log
  directory="$(neutral_dir "${scale}")"
  log="${OUTPUT_ROOT}/_logs/neutral_${DATASET}_s${scale}_seed${SEED}.log"
  if [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/metrics_val.json" \
    && -s "${directory}/predictions_val.npz" ]]; then
    echo "skip_existing neutral dataset=${DATASET} scale=${scale}"
    return
  fi
  echo "run_start=$(date -Is) family=neutral dataset=${DATASET} scale=${scale} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/intro_evidence_neutral/train.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${DATASET}" \
    --seq-len 96 --pred-len 720 --sharing-extent "${scale}" \
    --history-dim 64 --step-dim 32 --hidden-dim 128 --state-dim 64 \
    --batch-size 16 --epochs 20 --patience 4 \
    --learning-rate 0.001 --weight-decay 0.0001 --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}/neutral" --device cuda \
    >"${log}" 2>&1
  echo "run_done=$(date -Is) family=neutral dataset=${DATASET} scale=${scale} gpu=${gpu}"
}

run_dlinear() {
  local horizon="$1" gpu="$2" directory log
  directory="$(dlinear_dir "${horizon}")"
  log="${OUTPUT_ROOT}/_logs/dlinear_${DATASET}_h${horizon}_seed${SEED}.log"
  if [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/metrics_val.json" \
    && -s "${directory}/predictions_val.npz" ]]; then
    echo "skip_existing dlinear dataset=${DATASET} horizon=${horizon}"
    return
  fi
  echo "run_start=$(date -Is) family=dlinear dataset=${DATASET} horizon=${horizon} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/dlinear/train.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${DATASET}" \
    --seq-len 96 --pred-len "${horizon}" --batch-size 128 \
    --epochs 50 --patience 8 --learning-rate 0.0001 --seed "${SEED}" \
    --init-mode pytorch_default --run-name IntroDLinearPrefixViz \
    --output-root "${OUTPUT_ROOT}/dlinear" --device cuda --skip-test \
    >"${log}" 2>&1
  echo "run_done=$(date -Is) family=dlinear dataset=${DATASET} horizon=${horizon} gpu=${gpu}"
}

worker0() {
  run_neutral 1 "${GPU_IDS[0]}"
  run_neutral 128 "${GPU_IDS[0]}"
  run_dlinear 96 "${GPU_IDS[0]}"
}

worker1() {
  run_neutral 8 "${GPU_IDS[1]}"
  run_neutral 720 "${GPU_IDS[1]}"
  run_dlinear 192 "${GPU_IDS[1]}"
}

worker2() {
  run_neutral 32 "${GPU_IDS[2]}"
  run_dlinear 336 "${GPU_IDS[2]}"
  run_dlinear 720 "${GPU_IDS[2]}"
}

worker0 & pid0="$!"
worker1 & pid1="$!"
worker2 & pid2="$!"
status=0
if ! wait "${pid0}"; then status=1; fi
if ! wait "${pid1}"; then status=1; fi
if ! wait "${pid2}"; then status=1; fi
if [[ "${status}" != "0" ]]; then
  echo "intro_evidence_visualization_worker_failure=$(date -Is)" >&2
  exit 1
fi

read -r neutral dlinear <<< "$(count_complete)"
if [[ "${neutral}" -ne 5 || "${dlinear}" -ne 4 ]]; then
  echo "incomplete visualization matrix neutral=${neutral}/5 dlinear=${dlinear}/4" >&2
  exit 4
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_intro_prefix_disagreement.py \
  --input-root "${OUTPUT_ROOT}/dlinear" \
  --output-dir "${OUTPUT_ROOT}/_analysis/${DATASET}/prefix" \
  --dataset "${DATASET}" --seed "${SEED}" --sample-quantile 0.85

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_intro_sharing_demand.py \
  --input-root "${OUTPUT_ROOT}/neutral" \
  --output-dir "${OUTPUT_ROOT}/_analysis/${DATASET}/sharing" \
  --dataset "${DATASET}" --seed "${SEED}"

echo "intro_evidence_visualization_done=$(date -Is) dataset=${DATASET} neutral=5/5 dlinear=4/4"
