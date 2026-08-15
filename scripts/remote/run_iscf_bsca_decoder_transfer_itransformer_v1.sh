#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_itransformer_v1_20260815}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_itransformer_v1.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
export PYTHONHASHSEED=2021
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

hash_file() { sha256sum "$1" | awk '{print $1}'; }
PROFILE_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["path"])' "${CONFIG}")"
PROFILE_HASH="$(hash_file "${PROFILE_PATH}")"
EXPECTED_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["sha256"])' "${CONFIG}")"
[[ "${PROFILE_HASH}" == "${EXPECTED_HASH}" ]]

LINES=()
while IFS= read -r line; do LINES+=("${line}"); done < <(python3 - "${CONFIG}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
p = json.load(open(c["profiles"]["path"]))
arms = {row["id"]: row for row in c["arms"]}
common = p["common"]
for dataset, arm_id in c["launch_order"]:
    arm = arms[arm_id]
    dataset_profile = p["dataset_profiles"][dataset]
    mode_rank = (
        dataset_profile["mode_rank"]
        if arm["readout_mode"] == "siff-independent-scope-control"
        else 256
    )
    values = [
        dataset,
        arm_id,
        arm["readout_mode"],
        arm["policy_mode"],
        arm["objective_mode"],
        mode_rank,
        dataset_profile["d_model"],
        common["n_heads"],
        dataset_profile["d_ff"],
        dataset_profile["e_layers"],
        common["dropout"],
        common["learning_rate"],
        common["batch_size"],
        common["max_epochs"],
        common["patience"],
        common["seq_len"],
    ]
    print("\t".join(map(str, values)))
PY
)

run_dir() {
  IFS=$'\t' read -r dataset arm _ <<< "$1"
  echo "${OUTPUT_ROOT}/${arm}/${dataset}/seed2021"
}

training_complete() {
  local directory dataset arm
  IFS=$'\t' read -r dataset arm _ <<< "$1"
  directory="$(run_dir "$1")"
  [[ -s "${directory}/checkpoint.pt" \
    && -s "${directory}/effective_config.json" \
    && -s "${directory}/initialization_contract.json" \
    && -s "${directory}/metrics_by_target_horizon.csv" \
    && -s "${directory}/training_log.csv" ]] || return 1
  if [[ "${arm}" != "itransformer_original" ]]; then
    [[ -s "${directory}/trained_invariants.json" ]] \
      && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' "${directory}/trained_invariants.json" 2>/dev/null
  fi
}

status_count() {
  local trained=0 line
  for line in "${LINES[@]}"; do
    training_complete "${line}" && trained=$((trained + 1))
  done
  echo "${trained}"
}

if [[ "${STATUS_ONLY}" == 1 ]]; then
  echo "itransformer_transfer_status=$(date -Is) training=$(status_count)/15 test=0/15"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "itransformer_transfer_dry_run=pass jobs=${#LINES[@]} profile_hash=${PROFILE_HASH}"
  exit 0
fi

run_training() {
  local line="$1" gpu="$2" output="$3" log="$4" smoke="$5"
  local dataset arm readout policy objective rank dmodel heads dff layers dropout lr batch epochs patience seq_len
  IFS=$'\t' read -r dataset arm readout policy objective rank dmodel heads dff layers dropout lr batch epochs patience seq_len <<< "${line}"
  local budget=(--epochs "${epochs}" --patience "${patience}" --final-evaluation-split val)
  if [[ "${smoke}" == 1 ]]; then
    budget=(--epochs 1 --patience 1 --max-train-batches 2 --max-eval-batches 2 --final-evaluation-split none)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
    --seq-len "${seq_len}" --pred-len 720 --target-horizons 720 \
    --validation-horizons 96,192,336,720 --evaluation-horizons 96,192,336,720 \
    --segment-horizons 96,192,336,720 --evaluation-prefix-mode full-crop \
    --encoder-mode itransformer-variate-attention --readout-mode "${readout}" \
    --history-d-model "${dmodel}" --history-n-heads "${heads}" \
    --history-d-ff "${dff}" --history-e-layers "${layers}" \
    --history-dropout "${dropout}" --batch-size "${batch}" \
    --learning-rate "${lr}" --weight-decay 0 --gradient-accumulation-steps 1 \
    --enable-early-stopping --checkpoint-policy best-val \
    --protocol-class method_screening \
    --protocol-profile iscf_bsca_decoder_transfer_itransformer_v1_20260815 \
    --profile-hash "${PROFILE_HASH}" --seed 2021 --num-workers 0 \
    --run-name "ITRANSFORMER_TRANSFER_${arm}" --output-dir "${output}" \
    --device cuda --no-save-predictions --pcsd-coordinate-dim 4 \
    --pcsd-mode-rank "${rank}" --pcsd-policy-history-dim 32 \
    --pcsd-policy-hidden-dim 64 --pcsd-policy-mode "${policy}" \
    --pcsd-fixed-scale 720 --pcsd-partition canonical \
    --pcsd-partition-seed 15101 --pcsd-group-chunk-size 64 \
    --pcsd-target-chunk-size 128 --pcc-objective-mode "${objective}" \
    --pred-loss-mode full "${budget[@]}" >"${log}" 2>&1
}

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

if [[ "${RESOURCE_SMOKE}" == 1 ]]; then
  for i in 0 1 2; do
    line="${LINES[$i]}"
    IFS=$'\t' read -r dataset arm _ <<< "${line}"
    output="${OUTPUT_ROOT}/_resource_smoke/${arm}_${dataset}"
    mkdir -p "${output}"
    run_training "${line}" "${GPU_IDS[$i]}" "${output}" "${output}/smoke.log" 1
  done
  echo "itransformer_transfer_resource_smoke=pass jobs=3"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
{
  echo "start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "profile_hash=${PROFILE_HASH}"
  echo "gpus=${GPU_IDS[*]}"
  echo "formal_test=false"
} | tee "${OUTPUT_ROOT}/launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3" output log
  IFS=$'\t' read -r dataset arm _ <<< "${line}"
  output="$(run_dir "${line}")"
  log="${OUTPUT_ROOT}/_logs/${arm}_${dataset}.log"
  mkdir -p "${output}"
  training_complete "${line}" && return
  echo "train_start=$(date -Is) job=$((index + 1))/15 arm=${arm} dataset=${dataset} gpu=${gpu}"
  run_training "${line}" "${gpu}" "${output}" "${log}" 0
  echo "job_done=$(date -Is) job=$((index + 1))/15 arm=${arm} dataset=${dataset} gpu=${gpu}"
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#LINES[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${LINES[$index]}" "${gpu}"
  done
}

pids=()
for i in "${!GPU_IDS[@]}"; do
  worker "${i}" "${GPU_IDS[$i]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == 0 ]]
echo "itransformer_transfer_training_done=$(date -Is) training=$(status_count)/15 test=0/15"
