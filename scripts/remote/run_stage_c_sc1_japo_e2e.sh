#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_japo_e2e}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONTRACT="${CONTRACT:-configs/stage_c_five_dataset_natural_profiles.json}"
DESIGN_CONTRACT="${DESIGN_CONTRACT:-configs/stage_c_sc1_japo_step6_design.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="stage_c_sc1_japo_e2e_v1"
SEGMENT_HORIZONS="48,96,192,336,720"
DENSE_HORIZONS="$(seq -s, 1 720)"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

profile_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
PROFILE_HASH="$(profile_hash "${CONTRACT}")"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
contract=json.load(open(sys.argv[1]))
arms=(
    ("a6", "learned-basis-forecast-operator"),
    ("joint_geo", "japo-joint-geo"),
    ("uniform", "japo-uniform"),
    ("history", "japo-history"),
    ("atom", "japo-atom"),
    ("joint_perm", "japo-joint-perm"),
    ("joint_random", "japo-joint-random"),
)
for arm,readout in arms:
    for dataset in ("Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2"):
        profile=contract["dataset_profiles"][dataset]
        print("\t".join(map(str,(
            dataset,arm,readout,profile["profile"],profile["patch_num"],
            profile["d_model"],profile["d_ff"],
        ))))
' "${CONTRACT}"
)

run_dir_for_line() {
  local line="$1" dataset arm readout profile patch_num d_model d_ff
  IFS=$'\t' read -r dataset arm readout profile patch_num d_model d_ff <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  invariant_complete=0
  for line in "${LINES[@]}"; do
    output_dir="$(run_dir_for_line "${line}")"
    [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]] \
      && completed=$((completed + 1))
    [[ -s "${output_dir}/trained_invariants.json" ]] \
      && invariant_complete=$((invariant_complete + 1))
  done
  echo "sc1_japo_status=$(date -Is) completed=${completed}/${#LINES[@]} invariants=${invariant_complete}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  test -s "${CONTRACT}"
  test -s "${DESIGN_CONTRACT}"
  for readout in \
    learned-basis-forecast-operator \
    japo-joint-geo \
    japo-uniform \
    japo-history \
    japo-atom \
    japo-joint-perm \
    japo-joint-random; do
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/check_stage_c_sc1_japo_checkpoint_invariants.py \
      --synthetic-readout "${readout}" --seed "${SEED}" >/dev/null
  done
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_sc1_japo_e2e.py \
    --synthetic-smoke >/dev/null
  printf '%s\n' "${LINES[@]}"
  echo "stage_c_sc1_japo_dry_run=pass jobs=${#LINES[@]} profile_hash=${PROFILE_HASH} final_split=val"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
{
  echo "stage_c_sc1_japo_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "cwd=$(pwd)"
  echo "dataset_root=${DATASET_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "contract=${CONTRACT}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "protocol_profile=${PROTOCOL_PROFILE}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "initialization=from_scratch_paired_by_seed"
  echo "training_objective=full_h720_pointwise_l1"
  echo "checkpoint_selection=best_val_h720_mse"
  echo "evaluation=validation_dense_h1_h720_full_crop_mse_mae"
  echo "test_used=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm readout profile patch_num d_model d_ff output_dir run_log
  IFS=$'\t' read -r dataset arm readout profile patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${OUTPUT_ROOT}/_logs/${arm}_${dataset}_seed${SEED}.log"
  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/trained_invariants.json" ]]; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu} profile=${profile}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons 720 --evaluation-horizons "${DENSE_HORIZONS}" \
      --segment-horizons "${SEGMENT_HORIZONS}" --evaluation-prefix-mode full-crop \
      --e-layers 2 --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --epochs "${EPOCHS}" --patience "${PATIENCE}" --enable-early-stopping \
      --early-stopping-min-delta 0 --seed "${SEED}" --num-workers 0 \
      --run-name "SC1_JAPO_${arm}" --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --final-evaluation-split val --protocol-class method_screening \
      --protocol-profile "${PROTOCOL_PROFILE}" --profile-hash "${PROFILE_HASH}" \
      --legacy-patch-num "${patch_num}" --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" --legacy-dropout 0.1 --legacy-layer-norm 1 \
      --learning-rate 0.0001 --readout-mode "${readout}" --basis-rank 256 \
      --plgo-global-rank 16 --plgo-latent-width 256 \
      --plgo-permutation-seed 7101 --plgo-random-descriptor-seed 7102 \
      --japo-expert-count 2 --japo-expert-rank 256 --japo-router-width 32 \
      --japo-router-output-init-std 0.01 --patch-diagnostic-batches 8 \
      --pred-loss-mode full --no-save-predictions >"${run_log}" 2>&1
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_stage_c_sc1_japo_checkpoint_invariants.py \
      --run-dir "${output_dir}" --seed "${SEED}" >>"${run_log}" 2>&1
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
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "stage_c_sc1_japo_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc1_japo_e2e.py \
    --raw-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/_analysis" \
    --seed "${SEED}"
echo "stage_c_sc1_japo_done=$(date -Is)"
