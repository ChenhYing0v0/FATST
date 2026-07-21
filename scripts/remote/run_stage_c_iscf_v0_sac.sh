#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_v0_sac_v1}"
SEED2021_ROOT="${SEED2021_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2}"
FCC_ROOT="${FCC_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_v2_fcc_v1}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/stage_c_iscf_v0_scope_attribution_confirmation.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
FORMAL_TEST_ONLY="${FORMAL_TEST_ONLY:-0}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL_PROFILE="stage_c_iscf_v0_sac_v1"
STANDARD_HORIZONS="96,192,336,720"
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

LINES=()
while IFS= read -r value; do
  LINES+=("${value}")
done < <(
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
profiles=json.load(open(config["profiles"]["path"]))["dataset_profiles"]
arms={arm["id"]: arm for arm in config["arms"]}
for seed,dataset,arm_id in config["launch_order"]:
    arm=arms[arm_id]
    profile=profiles[dataset]
    rank=config["matched_ranks"][dataset][arm["rank_rule"]]
    print("\t".join(map(str,(
        seed,dataset,arm_id,arm["readout_mode"],arm["policy_mode"],
        arm["objective_mode"],arm["partition"],arm["partition_seed"],
        rank,profile["profile"],profile["patch_num"],profile["d_model"],
        profile["d_ff"],
    ))))
' "${CONFIG}"
)

run_dir_for_line() {
  local line="$1" seed dataset arm rest
  IFS=$'\t' read -r seed dataset arm rest <<< "${line}"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/h720_full/seed${seed}"
}

is_complete() {
  local output_dir
  output_dir="$(run_dir_for_line "$1")"
  [[ -s "${output_dir}/checkpoint.pt" \
    && -s "${output_dir}/training_log.csv" \
    && -s "${output_dir}/metrics_by_target_horizon.csv" \
    && -s "${output_dir}/effective_config.json" \
    && -s "${output_dir}/initialization_contract.json" \
    && -s "${output_dir}/model_diagnostics.json" ]]
}

is_test_complete() {
  local output_dir
  output_dir="$(run_dir_for_line "$1")"
  [[ -s "${output_dir}/test_audit_metrics_by_target_horizon.csv" \
    && -s "${output_dir}/test_audit_invariants.json" \
    && -s "${output_dir}/pcsd_test_audit_diagnostics.npz" ]]
}

status_counts() {
  local trained=0 tested=0 line
  for line in "${LINES[@]}"; do
    if is_complete "${line}"; then trained=$((trained + 1)); fi
    if is_test_complete "${line}"; then tested=$((tested + 1)); fi
  done
  echo "${trained} ${tested}"
}

if [[ "${STATUS_ONLY}" == "1" ]]; then
  read -r trained tested <<< "$(status_counts)"
  echo "iscf_sac_status=$(date -Is) training=${trained}/${#LINES[@]} test=${tested}/${#LINES[@]}"
  find "${OUTPUT_ROOT}/_logs" -name '*.log' -type f -print0 2>/dev/null \
    | xargs -0 -r tail -n 1
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/analyze_stage_c_iscf_v0_sac.py \
    --config "${CONFIG}" --synthetic-smoke >/dev/null
  printf '%s\n' "${LINES[@]}"
  echo "iscf_sac_dry_run=pass jobs=${#LINES[@]} config_hash=${CONFIG_HASH} profile_hash=${PROFILE_HASH} remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${REMOTE_AUTHORIZED}" != "true" ]]; then
  echo "SAC remote launch is not authorized by ${CONFIG}" >&2
  exit 3
fi
if [[ "${FORMAL_TEST_ONLY}" == "1" && "${TEST_AUTHORIZED}" != "true" ]]; then
  echo "SAC formal test is not authorized by ${CONFIG}" >&2
  exit 3
fi
if [[ "${RESOURCE_SMOKE}" == "1" && "${FORMAL_TEST_ONLY}" == "1" ]]; then
  echo "RESOURCE_SMOKE and FORMAL_TEST_ONLY are mutually exclusive" >&2
  exit 2
fi
if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
  read -r trained _tested <<< "$(status_counts)"
  if [[ "${trained}" -ne "${#LINES[@]}" ]]; then
    echo "SAC formal test requires complete training: ${trained}/${#LINES[@]}" >&2
    exit 4
  fi
  "${PYTHON_BIN}" -c '
import json, sys
design = json.load(open(sys.argv[1]))
bins = design.get("diagnostic_protocol", {}).get("future_bins", [])
actual = [(item["start"], item["end"]) for item in bins]
expected = [(0, 48), (48, 96), (96, 144), (144, 192),
            (192, 288), (288, 336), (336, 512), (512, 720)]
assert actual == expected, (actual, expected)
' "${CONFIG}"
fi

run_training_command() {
  local line="$1" gpu="$2" output_dir="$3" run_log="$4" smoke="$5"
  local seed dataset arm readout policy objective partition partition_seed
  local rank profile patch_num d_model d_ff
  local run_args=()
  IFS=$'\t' read -r seed dataset arm readout policy objective partition \
    partition_seed rank profile patch_num d_model d_ff <<< "${line}"
  if [[ "${smoke}" == "1" ]]; then
    run_args=(
      --max-train-batches 2 --max-eval-batches 2 --epochs 1 --patience 1
      --final-evaluation-split none
    )
  else
    run_args=(
      --epochs "${EPOCHS}" --patience "${PATIENCE}"
      --final-evaluation-split val
    )
  fi
  PYTHONHASHSEED="${seed}" CUDA_VISIBLE_DEVICES="${gpu}" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 \
      --validation-horizons "${STANDARD_HORIZONS}" \
      --evaluation-horizons "${STANDARD_HORIZONS}" \
      --segment-horizons "${STANDARD_HORIZONS}" \
      --evaluation-prefix-mode full-crop --e-layers 2 \
      --batch-size "${BATCH_SIZE}" --gradient-accumulation-steps 1 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${seed}" \
      --num-workers 0 --run-name "ISCF_SAC_${arm}_seed${seed}" \
      --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints \
      --protocol-class method_screening \
      --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" \
      --legacy-dropout 0.1 --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode "${readout}" --basis-rank 256 \
      --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" \
      --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
      --pcsd-policy-mode "${policy}" --pcsd-fixed-scale 720 \
      --pcsd-partition "${partition}" \
      --pcsd-partition-seed "${partition_seed}" \
      --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 \
      --pcc-objective-mode "${objective}" --pred-loss-mode full \
      --no-save-predictions "${run_args[@]}" >"${run_log}" 2>&1
}

if [[ "${RESOURCE_SMOKE}" == "1" ]]; then
  smoke_lines=()
  for line in "${LINES[@]}"; do
    if [[ "${line}" == $'2021\tWeather\tiscf_random_partition\t'* \
      || "${line}" == $'2023\tETTm2\tiscf_q1_wide\t'* ]]; then
      smoke_lines+=("${line}")
    fi
  done
  test "${#smoke_lines[@]}" -eq 2
  smoke_root="${OUTPUT_ROOT}/_resource_smoke"
  mkdir -p "${smoke_root}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
  smoke_pids=()
  for index in "${!smoke_lines[@]}"; do
    line="${smoke_lines[${index}]}"
    gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
    IFS=$'\t' read -r seed dataset arm _rest <<< "${line}"
    output_dir="${smoke_root}/${arm}_${dataset}_seed${seed}"
    mkdir -p "${output_dir}"
    run_training_command \
      "${line}" "${gpu}" "${output_dir}" "${output_dir}/smoke.log" 1 &
    smoke_pids+=("$!")
  done
  for pid in "${smoke_pids[@]}"; do wait "${pid}"; done
  for directory in "${smoke_root}"/*_seed*; do
    test -s "${directory}/training_log.csv"
    test -s "${directory}/effective_config.json"
    failure_pattern='Traceback|CUDA out of memory|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'
    if command -v rg >/dev/null 2>&1; then
      ! rg -ni "${failure_pattern}" "${directory}/smoke.log"
    else
      ! grep -Ein "${failure_pattern}" "${directory}/smoke.log"
    fi
  done
  echo "iscf_sac_resource_smoke_done=$(date -Is) output=${smoke_root}"
  exit 0
fi

LOG_ROOT="${OUTPUT_ROOT}/_logs"
mkdir -p "${LOG_ROOT}"
MODE_LABEL="training"
if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then MODE_LABEL="formal_test"; fi
LAUNCH_RECORD="${OUTPUT_ROOT}/${MODE_LABEL}_launch_record.txt"
JOB_RECORD="${OUTPUT_ROOT}/${MODE_LABEL}_jobs.tsv"
{
  echo "iscf_sac_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "seed2021_root=${SEED2021_ROOT}"
  echo "fcc_root=${FCC_ROOT}"
  echo "config_hash=${CONFIG_HASH}"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "new_runs=${#LINES[@]}"
  echo "historical_runs=35"
  echo "effective_runs=60"
  echo "checkpoint_selection=best_val_mean_mse_h96_h192_h336_h720"
  echo "formal_test_execution_mode=${FORMAL_TEST_ONLY}"
  echo "test_informed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${LAUNCH_RECORD}"
printf '%s\n' "${LINES[@]}" >"${JOB_RECORD}"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local seed dataset arm rest output_dir run_log before after
  IFS=$'\t' read -r seed dataset arm rest <<< "${line}"
  output_dir="$(run_dir_for_line "${line}")"
  run_log="${LOG_ROOT}/${arm}_${dataset}_seed${seed}.log"
  if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
    if is_test_complete "${line}"; then
      echo "skip_existing_test=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed}"
      return
    fi
    before="$(sha256_file "${output_dir}/checkpoint.pt")"
    echo "test_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output \
      -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
        --run-dir "${output_dir}" --design "${CONFIG}" \
        --test-audit-config "${CONFIG}" --evaluation-split test \
        --device cuda >>"${run_log}" 2>&1
    after="$(sha256_file "${output_dir}/checkpoint.pt")"
    [[ "${before}" == "${after}" ]]
    echo "test_done=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed} gpu=${gpu}"
    return
  fi
  if is_complete "${line}"; then
    echo "skip_existing=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed}"
    return
  fi
  mkdir -p "${output_dir}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed} gpu=${gpu}"
  run_training_command "${line}" "${gpu}" "${output_dir}" "${run_log}" 0
  echo "train_done=$(date -Is) job=$((index + 1))/${#LINES[@]} arm=${arm} dataset=${dataset} seed=${seed} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" line_index
  for ((line_index=WORKER_OFFSET + worker_index; \
    line_index<${#LINES[@]}; line_index+=WORKER_STRIDE)); do
    run_one "${line_index}" "${LINES[${line_index}]}" "${gpu}"
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[${index}]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then status=1; fi; done
[[ "${status}" == "0" ]]

if [[ "${FORMAL_TEST_ONLY}" == "1" ]]; then
  analysis_root="${OUTPUT_ROOT}/_analysis_three_seed"
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_iscf_v0_sac.py \
      --config "${CONFIG}" --new-root "${OUTPUT_ROOT}" \
      --seed2021-root "${SEED2021_ROOT}" --fcc-root "${FCC_ROOT}" \
      --output-dir "${analysis_root}"
  echo "iscf_sac_formal_test_done=$(date -Is) analysis=${analysis_root}"
else
  echo "iscf_sac_training_done=$(date -Is) formal_test_executed=false"
fi
