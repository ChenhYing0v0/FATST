#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d18_soft_projectivity_cost_v1}"
CONTROL_ROOT="${CONTROL_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_d18_soft_projectivity_cost.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/yingch/.conda/envs/moe/bin/python}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="stage_c_d18_soft_projectivity_cost_v1"
PROBE_ROWS="256"
export PYTHONHASHSEED="${SEED}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi
test -x "${PYTHON_BIN}"
test -s "${CONFIG}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

CONFIG_HASH="$(sha256_file "${CONFIG}")"
PROFILE_PATH="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["path"])' "${CONFIG}")"
PROFILE_HASH="$(sha256_file "${PROFILE_PATH}")"
REMOTE_AUTHORIZED="$("${PYTHON_BIN}" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$("${PYTHON_BIN}" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"

JOBS=()
while IFS= read -r value; do
  JOBS+=("${value}")
done < <(
  "${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
profiles = json.load(open(config["profiles"]["path"]))["dataset_profiles"]
priority = ["Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2"]
for dataset in priority:
    profile = profiles[dataset]
    for arm in config["arms"]:
        print("\t".join(map(str, (
            dataset,
            arm["id"],
            "train" if arm["training_new"] else "reuse",
            arm.get("own_horizon", 720),
            ",".join(map(str, arm["target_horizons"])),
            ",".join(map(str, arm["validation_horizons"])),
            arm["pred_loss_mode"],
            arm["pcc_objective_mode"],
            profile["profile"],
            profile["patch_num"],
            profile["d_model"],
            profile["d_ff"],
        ))))
PY
)

artifact_dir() {
  local line="$1" dataset arm rest
  IFS=$'\t' read -r dataset arm rest <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
}

source_dir() {
  local line="$1" dataset arm job_type rest
  IFS=$'\t' read -r dataset arm job_type rest <<< "${line}"
  if [[ "${job_type}" == "reuse" ]]; then
    echo "${CONTROL_ROOT}/${arm}/${dataset}/h720_full/seed${SEED}"
  else
    artifact_dir "${line}"
  fi
}

is_complete() {
  local line="$1" directory
  directory="$(artifact_dir "${line}")"
  [[ -s "${directory}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${directory}/test_audit_invariants.json" \
    && -s "${directory}/pcsd_test_audit_diagnostics.npz" ]]
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  completed=0
  for line in "${JOBS[@]}"; do
    if is_complete "${line}"; then completed=$((completed + 1)); fi
  done
  echo "d18_status=$(date -Is) complete=${completed}/${#JOBS[@]}"
  find "${OUTPUT_ROOT}/logs" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  [[ -s "${OUTPUT_ROOT}/analysis/summary.json" ]] \
    && echo "analysis_complete=true" \
    || echo "analysis_complete=false"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    baselines/timealign_official/train_repo.py \
    scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
    scripts/analyze_stage_c_d18_soft_projectivity_cost.py
  "${PYTHON_BIN}" scripts/analyze_stage_c_d18_soft_projectivity_cost.py \
    --config "${CONFIG}" --synthetic-smoke >/dev/null
  printf '%s\n' "${JOBS[@]}"
  echo "d18_dry_run=pass jobs=${#JOBS[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${REMOTE_AUTHORIZED}" != "true" || "${TEST_AUTHORIZED}" != "true" ]]; then
  echo "D18 remote/test access is not authorized by ${CONFIG}" >&2
  exit 3
fi

run_training() {
  local line="$1" gpu="$2" directory="$3" log_file="$4" smoke="$5"
  local dataset arm job_type own_horizon target_horizons validation_horizons
  local pred_loss objective profile patch_num d_model d_ff
  local run_args=()
  IFS=$'\t' read -r dataset arm job_type own_horizon target_horizons \
    validation_horizons pred_loss objective profile patch_num d_model d_ff \
    <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(
      --max-train-batches 1 --max-eval-batches 1
      --epochs 1 --patience 1 --final-evaluation-split none
    )
  else
    run_args=(
      --epochs "${EPOCHS}" --patience "${PATIENCE}"
      --final-evaluation-split val
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons "${target_horizons}" \
      --validation-horizons "${validation_horizons}" \
      --evaluation-horizons "96,192,336,720" \
      --segment-horizons "96,192,336,720" \
      --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${SEED}" \
      --num-workers 0 --run-name "D18_${arm}" \
      --output-dir "${directory}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening \
      --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode learned-basis-forecast-operator --basis-rank 256 \
      --pcc-objective-mode "${objective}" --pred-loss-mode "${pred_loss}" \
      --no-save-predictions "${run_args[@]}" >"${log_file}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_line=""
  for line in "${JOBS[@]}"; do
    if [[ "${line}" == $'Weather\ta6_spec96\ttrain\t'* ]]; then
      smoke_line="${line}"
      break
    fi
  done
  test -n "${smoke_line}"
  smoke_root="${OUTPUT_ROOT}/resource_smoke/a6_spec96_weather_seed${SEED}"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  run_training \
    "${smoke_line}" "${GPU_IDS[0]}" "${smoke_root}" \
    "${smoke_root}/smoke.log" 1
  test -s "${smoke_root}/training_log.csv"
  test -s "${smoke_root}/effective_config.json"
  echo "d18_resource_smoke=pass output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/logs"
ANALYSIS_ROOT="${OUTPUT_ROOT}/analysis"
mkdir -p "${LOG_ROOT}" "${ANALYSIS_ROOT}"
{
  echo "d18_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "control_root=${CONTROL_ROOT}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "jobs=${#JOBS[@]}"
  echo "new_training_runs=15"
  echo "reused_control_runs=10"
  echo "checkpoint_selection=arm_specific_validation_mse"
  echo "test_role=primary_problem_existence_diagnostic"
  echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_record.txt"
printf '%s\n' "${JOBS[@]}" >"${OUTPUT_ROOT}/jobs.tsv"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset arm job_type own_horizon rest directory source log_file
  local checkpoint_before checkpoint_after
  IFS=$'\t' read -r dataset arm job_type own_horizon rest <<< "${line}"
  directory="$(artifact_dir "${line}")"
  source="$(source_dir "${line}")"
  log_file="${LOG_ROOT}/${arm}_${dataset}_seed${SEED}.log"
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset}"
    return 0
  fi
  mkdir -p "${directory}"
  if [[ "${job_type}" == "train" ]]; then
    echo "train_start=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
    run_training "${line}" "${gpu}" "${directory}" "${log_file}" 0
  else
    test -s "${source}/checkpoint.pt"
    "${PYTHON_BIN}" - "${source}" "${directory}/source_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[2]).write_text(
    json.dumps({"source_run_dir": sys.argv[1]}, indent=2) + "\n"
)
PY
  fi
  checkpoint_before="$(sha256_file "${source}/checkpoint.pt")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${source}" --artifact-dir "${directory}" \
      --design "${CONFIG}" --test-audit-config "${CONFIG}" \
      --evaluation-split test --probe-rows "${PROBE_ROWS}" \
      --device cuda >>"${log_file}" 2>&1
  checkpoint_after="$(sha256_file "${source}/checkpoint.pt")"
  if [[ "${checkpoint_before}" != "${checkpoint_after}" ]]; then
    echo "checkpoint_mutated arm=${arm} dataset=${dataset}" >&2
    return 1
  fi
  echo "run_done=$(date -Is) job=$((index + 1))/${#JOBS[@]} arm=${arm} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index
  for ((line_index=worker_index; line_index<${#JOBS[@]}; line_index+=${#GPU_IDS[@]})); do
    run_one "${line_index}" "${JOBS[${line_index}]}" "${gpu}"
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
  echo "d18_worker_failure=$(date -Is)" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/analyze_stage_c_d18_soft_projectivity_cost.py \
  --raw-root "${OUTPUT_ROOT}" --output-dir "${ANALYSIS_ROOT}" \
  --config "${CONFIG}" --seed "${SEED}"
echo "d18_done=$(date -Is)"
