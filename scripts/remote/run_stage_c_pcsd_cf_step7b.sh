#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
DESIGN="${DESIGN:-configs/stage_c_pcsd_cf_native_direct.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="stage_c_pcsd_cf_step7b_validation_screen_v1"
DENSE_HORIZONS="$(seq -s, 1 720)"
SEGMENT_HORIZONS="48,96,144,192,288,336,512,720"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -s "${CONTRACT}"
test -s "${DESIGN}"

profile_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
PROFILE_HASH="$(profile_hash "${CONTRACT}")"
DESIGN_HASH="$(profile_hash "${DESIGN}")"

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
contract=json.load(open(sys.argv[1]))
arms=(
    ("a6", "learned-basis-forecast-operator", "control", 720, "control"),
    ("pcsd_m0", "pcsd-coupling-field-m0", "control", 720, "control"),
    ("pcsd_fixed_1", "pcsd-coupling-field", "fixed", 1, "canonical"),
    ("pcsd_fixed_48", "pcsd-coupling-field", "fixed", 48, "canonical"),
    ("pcsd_fixed_144", "pcsd-coupling-field", "fixed", 144, "canonical"),
    ("pcsd_fixed_360", "pcsd-coupling-field", "fixed", 360, "canonical"),
    ("pcsd_fixed_720", "pcsd-coupling-field", "fixed", 720, "canonical"),
    ("pcsd_equal", "pcsd-coupling-field", "equal", 720, "canonical"),
    ("pcsd_static", "pcsd-coupling-field", "static-target", 720, "canonical"),
    ("pcsd_direct", "pcsd-coupling-field", "direct", 720, "canonical"),
    ("pcsd_random", "pcsd-coupling-field", "direct", 720, "random"),
    ("dense_matched", "pcsd-dense-nonlinear-matched", "control", 720, "control"),
)
datasets=("Weather", "ETTm1", "ETTh1", "ETTm2", "ETTh2")
for arm,readout,policy,fixed_scale,partition in arms:
    for dataset in datasets:
        profile=contract["dataset_profiles"][dataset]
        print("\t".join(map(str,(
            dataset,arm,readout,policy,fixed_scale,partition,profile["profile"],
            profile["patch_num"],profile["d_model"],profile["d_ff"],
        ))))
' "${CONTRACT}"
)

run_dir_for_line() {
  local line="$1" dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local line="$1" output_dir
  output_dir="$(run_dir_for_line "${line}")"
  [[ -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/trained_invariants.json" \
    && -s "${output_dir}/pcsd_validation_diagnostics.npz" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "pcsd_step7b_status=$(date -Is) completed=${completed}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 -c '
import json
gate=json.load(open("analysis/stage_c_pcsd_cf_step7a_local_20260716/step7a_local_gate.json"))
if gate.get("overall_pass") is not True:
    raise SystemExit("historical Step7A gate is not passed")
'
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_stage_c_pcsd_cf_step7b.py >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py --synthetic-smoke >/dev/null
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_pcsd_cf_step7b.py --synthetic-smoke >/dev/null
  printf '%s\n' "${LINES[@]}"
  echo "pcsd_step7b_dry_run=pass jobs=${#LINES[@]} profile_hash=${PROFILE_HASH} design_hash=${DESIGN_HASH} test=false"
  exit 0
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4"
  local smoke="$5"
  local dataset arm readout policy fixed_scale partition profile patch_num d_model d_ff
  local smoke_args=()
  IFS=$'\t' read -r dataset arm readout policy fixed_scale partition profile patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    smoke_args=(
      --max-train-batches 1 --max-eval-batches 1 --epochs 1 --patience 1
      --final-evaluation-split none
    )
  else
    smoke_args=(
      --epochs "${EPOCHS}" --patience "${PATIENCE}"
      --final-evaluation-split val
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons 720 --evaluation-horizons "${DENSE_HORIZONS}" \
      --segment-horizons "${SEGMENT_HORIZONS}" --evaluation-prefix-mode full-crop \
      --e-layers 2 --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "PCSD_STEP7B_${arm}" --output-dir "${output_dir}" \
      --device cuda --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode "${readout}" --basis-rank 256 --pcsd-coordinate-dim 4 \
      --pcsd-mode-rank 256 --pcsd-policy-history-dim 32 \
      --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode "$([[ "${policy}" == "control" ]] && echo direct || echo "${policy}")" \
      --pcsd-fixed-scale "${fixed_scale}" \
      --pcsd-partition "$([[ "${partition}" == "control" ]] && echo canonical || echo "${partition}")" \
      --pcsd-partition-seed 15101 --pcsd-group-chunk-size 64 \
      --pcsd-target-chunk-size 128 --pred-loss-mode full --no-save-predictions \
      "${smoke_args[@]}" >"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_line=""
  for line in "${LINES[@]}"; do
    if [[ "${line}" == $'Weather\tpcsd_direct\t'* ]]; then
      smoke_line="${line}"
      break
    fi
  done
  test -n "${smoke_line}"
  smoke_root="${OUTPUT_ROOT}/_resource_smoke/pcsd_direct_weather_seed${SEED}"
  mkdir -p "${smoke_root}"
  echo "resource_smoke_start=$(date -Is) gpu=${GPU_IDS[0]} batch_size=${BATCH_SIZE}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  run_training_command "${smoke_line}" "${GPU_IDS[0]}" "${smoke_root}" "${smoke_root}/smoke.log" 1
  echo "resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"
ANALYSIS_ROOT="${OUTPUT_ROOT}/_analysis_seed${SEED}"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"
{
  echo "pcsd_step7b_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "design_hash=${DESIGN_HASH}"
  echo "protocol_profile=${PROTOCOL_PROFILE}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "initialization=from_scratch_paired_by_seed"
  echo "training_objective=full_h720_pointwise_l1"
  echo "checkpoint_selection=best_val_h720_mse"
  echo "primary_metric=validation_dense_h1_h720_mse_auc"
  echo "evaluation=validation_dense_h1_h720_full_crop"
  echo "test_used=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm readout policy fixed_scale partition profile patch_num d_model d_ff
  local output_dir run_log
  IFS=$'\t' read -r dataset arm readout policy fixed_scale partition profile patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu} profile=${profile}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${output_dir}" --design "${DESIGN}" --device cuda \
      >>"${run_log}" 2>&1
  echo "run_done=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
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
  echo "pcsd_step7b_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_pcsd_cf_step7b.py \
    --raw-root "${OUTPUT_ROOT}" --output-dir "${ANALYSIS_ROOT}" \
    --design "${DESIGN}" --seed "${SEED}"
echo "pcsd_step7b_done=$(date -Is)"
