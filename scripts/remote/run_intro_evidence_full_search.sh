#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1}"
CONFIG="${CONFIG:-configs/intro_evidence_visualization_pilot_v1.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather)
SCALES=(1 8 32 128 720)
HORIZONS=(96 192 336 720)

if [[ "${#GPU_IDS[@]}" -ne 3 ]]; then
  echo "full search requires exactly three GPU ids" >&2
  exit 2
fi
test -s "${CONFIG}"

python3 - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
full = config["full_search"]
assert full["protocol"] == "SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1"
assert full["datasets"] == ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
assert full["new_runs"] == 31
assert full["scheduler"] == "global_dynamic_three_gpu"
assert config["authorization"]["remaining_dataset_full_search"] is True
assert config["authorization"]["formal_test"] is False
assert full["test_accessed"] is False
PY

neutral_dir() {
  local dataset="$1" scale="$2"
  echo "${OUTPUT_ROOT}/neutral/NeutralSharingExtent/${dataset}/s${scale}/seed${SEED}"
}

dlinear_dir() {
  local dataset="$1" horizon="$2"
  echo "${OUTPUT_ROOT}/dlinear/IntroDLinearPrefixViz/${dataset}/h${horizon}/seed${SEED}"
}

neutral_complete() {
  local directory
  directory="$(neutral_dir "$1" "$2")"
  [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/metrics_val.json" \
    && -s "${directory}/predictions_val.npz" \
    && -s "${directory}/effective_config.json" ]]
}

dlinear_complete() {
  local directory
  directory="$(dlinear_dir "$1" "$2")"
  [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/metrics_val.json" \
    && -s "${directory}/predictions_val.npz" ]]
}

count_complete() {
  local neutral=0 dlinear=0 dataset scale horizon
  for dataset in "${DATASETS[@]}"; do
    for scale in "${SCALES[@]}"; do
      if neutral_complete "${dataset}" "${scale}"; then
        neutral=$((neutral + 1))
      fi
    done
    for horizon in "${HORIZONS[@]}"; do
      if dlinear_complete "${dataset}" "${horizon}"; then
        dlinear=$((dlinear + 1))
      fi
    done
  done
  echo "${neutral} ${dlinear}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  read -r neutral dlinear <<< "$(count_complete)"
  echo "intro_full_search_status=$(date -Is) neutral=${neutral}/25 dlinear=${dlinear}/20"
  find "${OUTPUT_ROOT}/_logs" -type f -name '*.log' -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_intro_evidence_visualization_pilot.py >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_prefix_disagreement.py --help >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_sharing_demand.py --help >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_sharing_sample_candidates.py --help >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/rank_intro_visualization_candidates.py --help >/dev/null
  echo "intro_evidence_full_search_dry_run=pass"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis_full"
{
  echo "intro_evidence_full_search_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "protocol=SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1"
  echo "datasets=${DATASETS[*]}"
  echo "seed=${SEED}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "new_runs_expected=31"
  echo "scheduler=global_dynamic_three_gpu"
  echo "selection=maximum_disclosed_validation_candidate"
  echo "test_accessed=false"
  echo "gpu_ids=${GPU_IDS[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_full_search.txt"

run_neutral() {
  local dataset="$1" scale="$2" gpu="$3" directory log
  directory="$(neutral_dir "${dataset}" "${scale}")"
  log="${OUTPUT_ROOT}/_logs/neutral_${dataset}_s${scale}_seed${SEED}.log"
  if neutral_complete "${dataset}" "${scale}"; then
    echo "skip_existing family=neutral dataset=${dataset} scale=${scale}"
    return
  fi
  echo "run_start=$(date -Is) family=neutral dataset=${dataset} scale=${scale} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/intro_evidence_neutral/train.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" \
    --seq-len 96 --pred-len 720 --sharing-extent "${scale}" \
    --history-dim 64 --step-dim 32 --hidden-dim 128 --state-dim 64 \
    --batch-size 16 --epochs 20 --patience 4 \
    --learning-rate 0.001 --weight-decay 0.0001 --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}/neutral" --device cuda \
    >"${log}" 2>&1
  echo "run_done=$(date -Is) family=neutral dataset=${dataset} scale=${scale} gpu=${gpu}"
}

run_dlinear() {
  local dataset="$1" horizon="$2" gpu="$3" directory log
  directory="$(dlinear_dir "${dataset}" "${horizon}")"
  log="${OUTPUT_ROOT}/_logs/dlinear_${dataset}_h${horizon}_seed${SEED}.log"
  if dlinear_complete "${dataset}" "${horizon}"; then
    echo "skip_existing family=dlinear dataset=${dataset} horizon=${horizon}"
    return
  fi
  echo "run_start=$(date -Is) family=dlinear dataset=${dataset} horizon=${horizon} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
    -n "${CONDA_ENV}" python baselines/dlinear/train.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" \
    --seq-len 96 --pred-len "${horizon}" --batch-size 128 \
    --epochs 50 --patience 8 --learning-rate 0.0001 --seed "${SEED}" \
    --init-mode pytorch_default --run-name IntroDLinearPrefixViz \
    --output-root "${OUTPUT_ROOT}/dlinear" --device cuda --skip-test \
    >"${log}" 2>&1
  echo "run_done=$(date -Is) family=dlinear dataset=${dataset} horizon=${horizon} gpu=${gpu}"
}

run_job() {
  local descriptor="$1" gpu="$2" family dataset value
  IFS='|' read -r family dataset value <<< "${descriptor}"
  if [[ "${family}" == "neutral" ]]; then
    run_neutral "${dataset}" "${value}" "${gpu}"
  else
    run_dlinear "${dataset}" "${value}" "${gpu}"
  fi
}

# Longest expected jobs are queued first. A global queue immediately gives the
# next job to whichever GPU finishes, avoiding a fixed GPU-0 critical path.
JOBS=(
  "neutral|ETTm2|1" "neutral|ETTh1|1" "neutral|ETTh2|1"
  "dlinear|ETTm2|720" "dlinear|ETTm1|720" "dlinear|ETTh1|720"
  "dlinear|ETTh2|720" "neutral|ETTm2|8" "neutral|ETTh1|8"
  "neutral|ETTh2|8" "neutral|ETTm2|32" "neutral|ETTh1|32"
  "neutral|ETTh2|32" "neutral|ETTm2|128" "neutral|ETTh1|128"
  "neutral|ETTh2|128" "neutral|ETTm2|720" "neutral|ETTh1|720"
  "neutral|ETTh2|720" "dlinear|ETTm2|336" "dlinear|ETTm1|336"
  "dlinear|ETTh1|336" "dlinear|ETTh2|336" "dlinear|ETTm2|192"
  "dlinear|ETTm1|192" "dlinear|ETTh1|192" "dlinear|ETTh2|192"
  "dlinear|ETTm2|96" "dlinear|ETTm1|96" "dlinear|ETTh1|96"
  "dlinear|ETTh2|96"
)

slot_pids=("" "" "")
slot_jobs=("" "" "")
next_job=0
active_jobs=0
failed=0
while [[ "${next_job}" -lt "${#JOBS[@]}" || "${active_jobs}" -gt 0 ]]; do
  for slot in 0 1 2; do
    if [[ -z "${slot_pids[slot]}" && "${next_job}" -lt "${#JOBS[@]}" ]]; then
      descriptor="${JOBS[next_job]}"
      run_job "${descriptor}" "${GPU_IDS[slot]}" &
      slot_pids[slot]="$!"
      slot_jobs[slot]="${descriptor}"
      next_job=$((next_job + 1))
      active_jobs=$((active_jobs + 1))
      echo "queue_launch=$(date -Is) slot=${slot} gpu=${GPU_IDS[slot]} job=${descriptor}"
    fi
  done
  if [[ "${active_jobs}" -gt 0 ]]; then
    sleep 5
  fi
  for slot in 0 1 2; do
    pid="${slot_pids[slot]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        echo "queue_complete=$(date -Is) slot=${slot} job=${slot_jobs[slot]}"
      else
        echo "queue_failed=$(date -Is) slot=${slot} job=${slot_jobs[slot]}" >&2
        failed=1
      fi
      slot_pids[slot]=""
      slot_jobs[slot]=""
      active_jobs=$((active_jobs - 1))
    fi
  done
done
if [[ "${failed}" -ne 0 ]]; then
  exit 3
fi

read -r neutral dlinear <<< "$(count_complete)"
if [[ "${neutral}" -ne 25 || "${dlinear}" -ne 20 ]]; then
  echo "incomplete matrix neutral=${neutral}/25 dlinear=${dlinear}/20" >&2
  exit 4
fi

for dataset in "${DATASETS[@]}"; do
  analysis_dir="${OUTPUT_ROOT}/_analysis_full/${dataset}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_prefix_disagreement.py \
    --input-root "${OUTPUT_ROOT}/dlinear" \
    --output-dir "${analysis_dir}/prefix" \
    --dataset "${dataset}" --seed "${SEED}" --selection-mode maximum
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_sharing_demand.py \
    --input-root "${OUTPUT_ROOT}/neutral" \
    --output-dir "${analysis_dir}/sharing_aggregate" \
    --dataset "${dataset}" --seed "${SEED}"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_intro_sharing_sample_candidates.py \
    --input-root "${OUTPUT_ROOT}/neutral" \
    --output-dir "${analysis_dir}/sharing_sample" \
    --dataset "${dataset}" --seed "${SEED}"
done

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/rank_intro_visualization_candidates.py \
  --analysis-root "${OUTPUT_ROOT}/_analysis_full" \
  --output-dir "${OUTPUT_ROOT}/_analysis_full/ranking"

echo "intro_evidence_full_search_done=$(date -Is) neutral=${neutral}/25 dlinear=${dlinear}/20"
