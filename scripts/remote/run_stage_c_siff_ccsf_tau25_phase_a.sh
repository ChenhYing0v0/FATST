#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_ccsf_v1_tau25_phase_a}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json}"
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
PROBE_ROWS="${PROBE_ROWS:-256}"
STANDARD_HORIZONS="96,192,336,720"
PROTOCOL_PROFILE="stage_c_siff_ccsf_v1_tau25_phase_a"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
WORKER_STRIDE="${WORKER_STRIDE:-${#GPU_IDS[@]}}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -s "${CONFIG}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

CONFIG_HASH="$(sha256_file "${CONFIG}")"
PROFILE_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["path"])' "${CONFIG}")"
PROFILE_HASH="$(sha256_file "${PROFILE_PATH}")"
REMOTE_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"
PHASE_A_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_phase_a_authorized"]).lower())' "${CONFIG}")"
TEMPERATURE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_objective"]["calibration_temperature"])' "${CONFIG}")"
CALIBRATION_WEIGHT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_objective"]["calibration_weight"])' "${CONFIG}")"

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
profiles=json.load(open(config["profiles"]["path"]))["dataset_profiles"]
for dataset in config["datasets"]:
    profile=profiles[dataset]
    for arm in config["arms"]:
        rank=256 if arm["rank_rule"] == "fixed_256" else config["matched_ranks"][dataset]
        print("\t".join(map(str,(
            dataset,arm["id"],arm["readout_mode"],arm["objective_mode"],rank,
            profile["profile"],profile["patch_num"],profile["d_model"],profile["d_ff"],
        ))))
' "${CONFIG}"
)

run_dir_for_line() {
  local line="$1" dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

is_complete() {
  local line="$1" output_dir
  output_dir="$(run_dir_for_line "${line}")"
  [[ -s "${output_dir}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${output_dir}/test_audit_invariants.json" \
    && -s "${output_dir}/pcsd_test_audit_diagnostics.npz" \
    && -s "${output_dir}/checkpoint.pt" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "ccsf_tau25_phase_a_status=$(date -Is) completed=${completed}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs_seed${SEED}" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "ccsf_tau25_phase_a_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} tau=${TEMPERATURE} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  if [[ "${REMOTE_AUTHORIZED}" != "true" ]]; then
    echo "remote resource smoke is not authorized by ${CONFIG}" >&2
    exit 3
  fi
elif [[ "${REMOTE_AUTHORIZED}" != "true" || "${TEST_AUTHORIZED}" != "true" || "${PHASE_A_AUTHORIZED}" != "true" ]]; then
  echo "formal Phase-A remote/test launch is not authorized by ${CONFIG}" >&2
  exit 3
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4" smoke="$5"
  local dataset arm readout objective rank profile patch_num d_model d_ff
  local run_args=()
  IFS=$'\t' read -r dataset arm readout objective rank profile patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(
      --max-train-batches 3 --max-eval-batches 1 --epochs 1 --patience 1
      --final-evaluation-split none
    )
  else
    run_args=(
      --epochs "${EPOCHS}" --patience "${PATIENCE}"
      --final-evaluation-split val
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons "${STANDARD_HORIZONS}" \
      --evaluation-horizons "${STANDARD_HORIZONS}" \
      --segment-horizons "${STANDARD_HORIZONS}" \
      --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "CCSF_TAU25_${arm}" \
      --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode "${readout}" --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" \
      --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode direct --pcsd-fixed-scale 720 \
      --pcsd-partition canonical --pcsd-partition-seed 15101 \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --ccsf-correction-hidden-dim 64 \
      --ccsf-calibration-temperature "${TEMPERATURE}" \
      --ccsf-calibration-weight "${CALIBRATION_WEIGHT}" \
      --pcc-objective-mode "${objective}" --pred-loss-mode full \
      --no-save-predictions "${run_args[@]}" >"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_line=""
  for line in "${LINES[@]}"; do
    if [[ "${line}" == $'Weather\tccsf_relcal\t'* ]]; then
      smoke_line="${line}"
      break
    fi
  done
  test -n "${smoke_line}"
  smoke_root="${OUTPUT_ROOT}/_resource_smoke/ccsf_relcal_weather_seed${SEED}"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  run_training_command \
    "${smoke_line}" "${GPU_IDS[0]}" "${smoke_root}" "${smoke_root}/smoke.log" 1
  test -s "${smoke_root}/training_log.csv"
  test -s "${smoke_root}/effective_config.json"
  python3 -c '
import csv,json,math,sys
root=sys.argv[1]
effective=json.load(open(root+"/effective_config.json"))["adapter"]
rows=list(csv.DictReader(open(root+"/training_log.csv")))
assert effective["max_train_batches"] == 3
assert rows
assert all(math.isfinite(float(value)) for row in rows for key,value in row.items() if key.endswith("loss") and value not in ("",None))
' "${smoke_root}"
  echo "resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs_seed${SEED}"
ANALYSIS_ROOT="${OUTPUT_ROOT}/_analysis_seed${SEED}"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"
{
  echo "ccsf_tau25_phase_a_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "temperature=${TEMPERATURE}"
  echo "initialization=from_scratch_paired_by_seed"
  echo "checkpoint_selection=best_val_mean_mse_h96_h192_h336_h720"
  echo "formal_evaluation=test_h96_h192_h336_h720"
  echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record_seed${SEED}.txt"
printf '%s\n' "${LINES[@]}" >"${OUTPUT_ROOT}/jobs_seed${SEED}.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm readout objective rank profile patch_num d_model d_ff
  local output_dir run_log checkpoint_before checkpoint_after
  IFS=$'\t' read -r dataset arm readout objective rank profile patch_num d_model d_ff <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} gpu=${gpu} profile=${profile} rank=${rank}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  checkpoint_before="$(sha256_file "${output_dir}/checkpoint.pt")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${output_dir}" --design "${CONFIG}" \
      --test-audit-config "${CONFIG}" --evaluation-split test \
      --probe-rows "${PROBE_ROWS}" --device cuda >>"${run_log}" 2>&1
  checkpoint_after="$(sha256_file "${output_dir}/checkpoint.pt")"
  if [[ "${checkpoint_before}" != "${checkpoint_after}" ]]; then
    echo "checkpoint_mutated arm=${arm} dataset=${dataset}" >&2
    return 1
  fi
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
  echo "ccsf_tau25_phase_a_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_siff_ccsf_tau25_phase_a.py \
    --raw-root "${OUTPUT_ROOT}" --output-dir "${ANALYSIS_ROOT}" \
    --config "${CONFIG}" --seed "${SEED}"
echo "ccsf_tau25_phase_a_done=$(date -Is)"
