#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a1_dual_carrier_grouped_mlp}"
DESIGN="${DESIGN:-configs/stage_c_d14a1_dual_carrier_grouped_mlp.json}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
CARRIER="${CARRIER:-neutral_raw}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
PROTOCOL_PROFILE="stage_c_d14a1_dual_carrier_grouped_mlp_v1"
EVALUATION_HORIZONS="48,96,144,192,288,336,512,720"
SEGMENT_HORIZONS="144,360,720"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

if [[ "${CARRIER}" != "neutral_raw" && "${CARRIER}" != "a6_natural" ]]; then
  echo "CARRIER must be neutral_raw or a6_natural" >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -s "${DESIGN}"
test -s "${CONTRACT}"

profile_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
DESIGN_HASH="$(profile_hash "${DESIGN}")"
CONTRACT_HASH="$(profile_hash "${CONTRACT}")"

if [[ "${CARRIER}" == "a6_natural" ]]; then
  NEUTRAL_GATE="${OUTPUT_ROOT}/_analysis_neutral_raw_seed${SEED}/gate.json"
  python3 -c '
import json,sys
gate=json.load(open(sys.argv[1]))
expected="neutral_problem_pass_authorize_a6_sensitivity"
if gate.get("decision") != expected:
    raise SystemExit(f"A6 sensitivity held: neutral decision={gate.get('decision')!r}")
' "${NEUTRAL_GATE}"
fi

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
contract=json.load(open(sys.argv[1]))
carrier=sys.argv[2]
arms=[
    ("c_s1", 1, "canonical"),
    ("c_s48", 48, "canonical"),
    ("c_s144", 144, "canonical"),
    ("c_s360", 360, "canonical"),
    ("c_s720", 720, "canonical"),
    ("r_s48", 48, "random"),
    ("r_s144", 144, "random"),
    ("r_s360", 360, "random"),
]
if carrier == "a6_natural":
    arms.append(("a6_lbf", 144, "canonical"))
for dataset in ("Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2"):
    profile=contract["dataset_profiles"][dataset]
    for arm,scale,partition in arms:
        print("\t".join(map(str,(
            dataset,arm,scale,partition,profile["profile"],
            profile["patch_num"],profile["d_model"],profile["d_ff"],
        ))))
' "${CONTRACT}" "${CARRIER}"
)

run_dir_for_line() {
  local line="$1" dataset arm scale partition profile patch_num d_model d_ff
  IFS=$'\t' read -r dataset arm scale partition profile patch_num d_model d_ff <<< "${line}"
  echo "${OUTPUT_ROOT}/${CARRIER}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local line="$1" output_dir arm
  output_dir="$(run_dir_for_line "${line}")"
  IFS=$'\t' read -r _dataset arm _rest <<< "${line}"
  if [[ "${arm}" == "a6_lbf" ]]; then
    [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]]
  else
    [[ -s "${output_dir}/metrics_by_target_horizon.csv" \
      && -s "${output_dir}/validation_diagnostics.npz" \
      && -s "${output_dir}/trained_invariants.json" ]]
  fi
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "d14a1_status=$(date -Is) carrier=${CARRIER} seed=${SEED} completed=${completed}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_${CARRIER}_seed${SEED}" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_stage_c_d14a1_step7a.py >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_d14a1.py --synthetic-smoke >/dev/null
  printf '%s\n' "${LINES[@]}"
  echo "stage_c_d14a1_dry_run=pass carrier=${CARRIER} jobs=${#LINES[@]} test=false"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_${CARRIER}_seed${SEED}"
ANALYSIS_ROOT="${OUTPUT_ROOT}/_analysis_${CARRIER}_seed${SEED}"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"
{
  echo "stage_c_d14a1_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "carrier=${CARRIER}"
  echo "seed=${SEED}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "design_hash=${DESIGN_HASH}"
  echo "natural_profile_hash=${CONTRACT_HASH}"
  echo "protocol_profile=${PROTOCOL_PROFILE}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "worker_offset=${WORKER_OFFSET}"
  echo "worker_stride=${WORKER_STRIDE}"
  echo "jobs=${#LINES[@]}"
  echo "initialization=from_scratch_end_to_end"
  echo "objective=full_h720_pointwise_l1"
  echo "checkpoint=best_validation_h720_mse"
  echo "evaluation=validation_full_crop"
  echo "test_used=false"
  echo "a6_failure_can_reject_scale_hypothesis=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_${CARRIER}_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs_${CARRIER}_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm scale partition profile patch_num d_model d_ff output_dir run_log
  local encoder readout
  IFS=$'\t' read -r dataset arm scale partition profile patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset}"
    return 0
  fi
  mkdir -p "${output_dir}"
  if [[ "${CARRIER}" == "neutral_raw" ]]; then
    encoder="raw-history-identity"
  else
    encoder="timealign-token-mlp"
  fi
  if [[ "${arm}" == "a6_lbf" ]]; then
    readout="learned-basis-forecast-operator"
  else
    readout="grouped-mlp"
  fi
  echo "run_start=$(date -Is) job=$((index + 1))/${#LINES[@]} carrier=${CARRIER} arm=${arm} dataset=${dataset} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons 720 --evaluation-horizons "${EVALUATION_HORIZONS}" \
      --segment-horizons "${SEGMENT_HORIZONS}" --evaluation-prefix-mode full-crop \
      --encoder-mode "${encoder}" --readout-mode "${readout}" --basis-rank 256 \
      --grouped-mlp-scale "${scale:-144}" --grouped-mlp-point-hidden-width 4 \
      --grouped-mlp-partition "${partition:-canonical}" --grouped-mlp-partition-seed 14101 \
      --legacy-patch-num "${patch_num}" --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" --legacy-dropout 0.1 --legacy-layer-norm 1 \
      --e-layers 2 --learning-rate 0.0001 --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps 1 --epochs "${EPOCHS}" --patience "${PATIENCE}" \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "D14A1_${CARRIER}_${arm}" --output-dir "${output_dir}" \
      --device cuda --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --final-evaluation-split val --protocol-class method_screening \
      --protocol-profile "${PROTOCOL_PROFILE}" --profile-hash "${DESIGN_HASH}" \
      --pred-loss-mode full --no-save-predictions >"${run_log}" 2>&1
  if [[ "${arm}" != "a6_lbf" ]]; then
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/evaluate_stage_c_d14a1_checkpoint.py \
        --run-dir "${output_dir}" --design "${DESIGN}" --device cuda \
        >>"${run_log}" 2>&1
  fi
  echo "run_done=$(date -Is) job=$((index + 1))/${#LINES[@]} carrier=${CARRIER} arm=${arm} dataset=${dataset}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index
  for ((line_index=WORKER_OFFSET + worker_index; line_index<${#LINES[@]}; line_index+=WORKER_STRIDE)); do
    run_one "${line_index}" "${LINES[${line_index}]}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
  echo "worker_start=$(date -Is) worker=${index} gpu=${GPU_IDS[${index}]} pid=$!"
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" != "0" ]]; then
  echo "stage_c_d14a1_worker_failure=$(date -Is) carrier=${CARRIER}" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_d14a1.py \
    --raw-root "${OUTPUT_ROOT}" --design "${DESIGN}" \
    --output-dir "${ANALYSIS_ROOT}" --carrier "${CARRIER}" --seed "${SEED}"
echo "stage_c_d14a1_done=$(date -Is) carrier=${CARRIER} seed=${SEED}"
