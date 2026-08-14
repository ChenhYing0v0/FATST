#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_patchtst_hpo_v2.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
export PYTHONHASHSEED=2021
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

hash_file() { sha256sum "$1" | awk '{print $1}'; }
PROFILE_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["backbone"]["profile_path"])' "${CONFIG}")"
PROFILE_HASH="$(hash_file "${PROFILE_PATH}")"
EXPECTED_PROFILE_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["backbone"]["profile_sha256"])' "${CONFIG}")"
[[ "${PROFILE_HASH}" == "${EXPECTED_PROFILE_HASH}" ]]
SEARCH_HASH="$(hash_file "${CONFIG}")"

LINES=()
while IFS= read -r line; do LINES+=("${line}"); done < <(python3 - "${CONFIG}" "${PROFILE_PATH}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
source = json.load(open(sys.argv[2]))
reference_ranks = config["reference_profile"]["mode_rank_by_dataset"]
dataset_order = config["datasets"]
profiles = config["search_profiles"]
source_family = source["backbones"]["patchtst_style"]
for dataset in dataset_order:
    dataset_profile = source_family["dataset_profiles"][dataset]
    for profile in profiles:
        rank = max(1, round(reference_ranks[dataset] * profile["rank_scale"]))
        values = [
            profile["id"], dataset, rank,
            profile["readout_learning_rate_multiplier"],
            profile["readout_weight_decay"],
            dataset_profile["learning_rate"], dataset_profile["batch_size"],
            dataset_profile["patience"], dataset_profile["d_model"],
            dataset_profile["n_heads"], dataset_profile["d_ff"],
            dataset_profile["dropout"], source_family["history_patch_len"],
            source_family["history_patch_stride"], source_family["history_e_layers"],
        ]
        print("\t".join(map(str, values)))
PY
)

run_dir() {
  IFS=$'\t' read -r profile dataset _ <<< "$1"
  echo "${OUTPUT_ROOT}/${dataset}/${profile}/seed2021"
}

training_complete() {
  local out
  out="$(run_dir "$1")"
  [[ -s "${out}/checkpoint.pt" && -s "${out}/training_log.csv" && -s "${out}/metrics_by_target_horizon.csv" && -s "${out}/effective_config.json" && -s "${out}/trained_invariants.json" ]] && \
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' "${out}/trained_invariants.json" 2>/dev/null
}

if [[ "${STATUS_ONLY}" == 1 ]]; then
  complete=0
  for line in "${LINES[@]}"; do training_complete "${line}" && complete=$((complete + 1)); done
  echo "patchtst_hpo_v2_status=$(date -Is) training=${complete}/${#LINES[@]} test=0"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  printf '%s\n' "${LINES[@]}"
  echo "patchtst_hpo_v2_dry_run=pass jobs=${#LINES[@]} search_hash=${SEARCH_HASH}"
  exit 0
fi

run_training() {
  local line="$1" gpu="$2" out="$3" log="$4" smoke="$5"
  local profile dataset rank readout_lr_scale readout_wd base_lr batch patience dmodel heads dff dropout patch stride layers
  IFS=$'\t' read -r profile dataset rank readout_lr_scale readout_wd base_lr batch patience dmodel heads dff dropout patch stride layers <<< "${line}"
  local budget=(--epochs 100 --patience "${patience}" --final-evaluation-split val)
  [[ "${smoke}" == 1 ]] && budget=(--epochs 1 --patience 1 --max-train-batches 2 --max-eval-batches 2 --final-evaluation-split none)
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified --seq-len 336 --pred-len 720 \
    --target-horizons 720 --validation-horizons 96,192,336,720 --evaluation-horizons 96,192,336,720 \
    --segment-horizons 96,192,336,720 --evaluation-prefix-mode full-crop \
    --encoder-mode contextual-patch-transformer --readout-mode siff-independent-scope-control \
    --batch-size "${batch}" --learning-rate "${base_lr}" --weight-decay 0 \
    --readout-learning-rate-multiplier "${readout_lr_scale}" --readout-weight-decay "${readout_wd}" \
    --gradient-accumulation-steps 1 --enable-early-stopping --checkpoint-policy best-val \
    --allow-archived-research-modes \
    --protocol-class method_screening --protocol-profile iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815 \
    --profile-hash "${PROFILE_HASH}" --hpo-trial-id "${profile}__${dataset}" --hpo-profile-id "${profile}" \
    --hpo-search-space-hash "${SEARCH_HASH}" --seed 2021 --num-workers 0 \
    --run-name "PATCHTST_HPO_V2_${profile}" --output-dir "${out}" --device cuda --no-save-predictions --no-official-test-mode \
    --history-patch-len "${patch}" --history-patch-stride "${stride}" --history-d-model "${dmodel}" \
    --history-n-heads "${heads}" --history-d-ff "${dff}" --history-e-layers "${layers}" \
    --history-dropout "${dropout}" --history-attn-dropout 0 --history-res-attention \
    --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" --pcsd-policy-history-dim 32 \
    --pcsd-policy-hidden-dim 64 --pcsd-policy-mode direct --pcsd-fixed-scale 720 \
    --pcsd-partition canonical --pcsd-partition-seed 15101 --pcsd-group-chunk-size 64 \
    --pcsd-target-chunk-size 128 --pcc-objective-mode equal_uniform_scope_anchor --pred-loss-mode full \
    "${budget[@]}" >"${log}" 2>&1
}

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

if [[ "${RESOURCE_SMOKE}" == 1 ]]; then
  smoke_indices=(0 8 49)
  for i in "${!smoke_indices[@]}"; do
    line="${LINES[${smoke_indices[$i]}]}"
    IFS=$'\t' read -r profile dataset _ <<< "${line}"
    out="${OUTPUT_ROOT}/_resource_smoke/${dataset}_${profile}"
    mkdir -p "${out}"
    run_training "${line}" "${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}" "${out}" "${out}/smoke.log" 1
  done
  echo "patchtst_hpo_v2_resource_smoke=pass jobs=${#smoke_indices[@]}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
{
  echo "start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "source_profile_hash=${PROFILE_HASH}"
  echo "search_hash=${SEARCH_HASH}"
  echo "gpus=${GPU_IDS[*]}"
  echo "jobs=${#LINES[@]}"
  echo "formal_test=0"
} | tee "${OUTPUT_ROOT}/launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3" out log
  IFS=$'\t' read -r profile dataset _ <<< "${line}"
  out="$(run_dir "${line}")"
  log="${OUTPUT_ROOT}/_logs/${dataset}_${profile}.log"
  training_complete "${line}" && return
  mkdir -p "${out}"
  echo "train_start=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} profile=${profile} gpu=${gpu}"
  run_training "${line}" "${gpu}" "${out}" "${log}" 0
  echo "job_done=$(date -Is) job=$((index + 1))/${#LINES[@]} dataset=${dataset} profile=${profile} gpu=${gpu}"
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
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" == 0 ]]
echo "patchtst_hpo_v2_training_done=$(date -Is) jobs=${#LINES[@]} formal_test=0"
