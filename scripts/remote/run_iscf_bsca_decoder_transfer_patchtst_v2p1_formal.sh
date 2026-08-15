#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_patchtst_v2p1_formal.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
RESOURCE_SMOKE="${RESOURCE_SMOKE:-0}"
FORMAL_TEST_ONLY="${FORMAL_TEST_ONLY:-0}"
export PYTHONHASHSEED=2021
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

hash_file() { sha256sum "$1" | awk '{print $1}'; }
json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): value=value[key]
print(value)' "${CONFIG}" "$1"
}

PROFILE_PATH="$(json_value profiles.path)"
PROFILE_HASH="$(hash_file "${PROFILE_PATH}")"
EXPECTED_PROFILE_HASH="$(json_value profiles.sha256)"
DESIGN_PATH="$(json_value diagnostic_design.path)"
DESIGN_HASH="$(hash_file "${DESIGN_PATH}")"
EXPECTED_DESIGN_HASH="$(json_value diagnostic_design.sha256)"
SELECTION_PATH="$(json_value selection_artifact.path)"
SELECTION_HASH="$(hash_file "${SELECTION_PATH}")"
EXPECTED_SELECTION_HASH="$(json_value selection_artifact.sha256)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(json_value artifact_contract.remote_output_root)}"
HPO_ROOT="${HPO_ROOT:-$(json_value artifact_contract.hpo_output_root)}"
MANIFEST="${MANIFEST:-$(json_value artifact_contract.training_manifest)}"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"
[[ "${PROFILE_HASH}" == "${EXPECTED_PROFILE_HASH}" ]]
[[ "${SELECTION_HASH}" == "${EXPECTED_SELECTION_HASH}" ]]
PROTOCOL_HASH="$(hash_file "${CONFIG}")"
TRAINING_SEARCH_HASH="$(json_value matched_training_execution.search_space_hash)"
[[ "${DESIGN_HASH}" == "${EXPECTED_DESIGN_HASH}" ]]

SELECTED=()
while IFS= read -r line; do SELECTED+=("${line}"); done < <(python3 - "${CONFIG}" "${PROFILE_PATH}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
profiles = json.load(open(sys.argv[2]))
family = profiles["backbones"]["patchtst_style"]
for selected in config["selected_profiles"]:
    dataset = selected["dataset"]
    dataset_profile = family["dataset_profiles"][dataset]
    values = [
        dataset,
        selected["profile_id"],
        selected["mode_rank"],
        selected["readout_learning_rate_multiplier"],
        selected["readout_weight_decay"],
        selected["bsca_checkpoint_sha256"],
        dataset_profile["learning_rate"],
        dataset_profile["batch_size"],
        dataset_profile["patience"],
        dataset_profile["d_model"],
        dataset_profile["n_heads"],
        dataset_profile["d_ff"],
        dataset_profile["dropout"],
        family["history_patch_len"],
        family["history_patch_stride"],
        family["history_e_layers"],
    ]
    print("\t".join(map(str, values)))
PY
)

matched_dir() {
  local dataset
  IFS=$'\t' read -r dataset _ <<< "$1"
  echo "${OUTPUT_ROOT}/training/patchtst_iscf/${dataset}/seed2021"
}

bsca_dir() {
  local dataset profile
  IFS=$'\t' read -r dataset profile _ <<< "$1"
  echo "${HPO_ROOT}/${dataset}/${profile}/seed2021"
}

formal_dir() {
  local line="$1" arm="$2" dataset
  IFS=$'\t' read -r dataset _ <<< "${line}"
  echo "${OUTPUT_ROOT}/formal_test/${arm}/${dataset}/seed2021"
}

training_complete() {
  local out name
  out="$(matched_dir "$1")"
  for name in checkpoint.pt training_log.csv metrics_by_target_horizon.csv effective_config.json initialization_contract.json model_diagnostics.json environment.json; do
    [[ -s "${out}/${name}" ]] || return 1
  done
}

test_complete() {
  local out
  out="$(formal_dir "$1" "$2")"
  [[ -s "${out}/test_audit_invariants.json" ]] \
    && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' "${out}/test_audit_invariants.json" 2>/dev/null
}

if [[ "${STATUS_ONLY}" == 1 ]]; then
  trained=0
  tested=0
  for line in "${SELECTED[@]}"; do
    training_complete "${line}" && trained=$((trained + 1))
    for arm in patchtst_iscf patchtst_iscf_bsca; do
      test_complete "${line}" "${arm}" && tested=$((tested + 1))
    done
  done
  echo "patchtst_v2p1_status=$(date -Is) matched_training=${trained}/5 formal_test=${tested}/10"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  for line in "${SELECTED[@]}"; do
    IFS=$'\t' read -r dataset profile rank lr_scale readout_wd _ <<< "${line}"
    echo -e "train\t${dataset}\tpatchtst_iscf\t${profile}\trank=${rank}\tlr_scale=${lr_scale}\twd=${readout_wd}"
    echo -e "formal_test\t${dataset}\tpatchtst_iscf\t${profile}"
    echo -e "formal_test\t${dataset}\tpatchtst_iscf_bsca\t${profile}"
  done
  echo "patchtst_v2p1_dry_run=pass matched_train=5 formal_test=10 protocol_hash=${PROTOCOL_HASH}"
  exit 0
fi

run_training() {
  local line="$1" gpu="$2" out="$3" log="$4" smoke="$5"
  local dataset profile rank lr_scale readout_wd expected_hash base_lr batch patience dmodel heads dff dropout patch stride layers
  IFS=$'\t' read -r dataset profile rank lr_scale readout_wd expected_hash base_lr batch patience dmodel heads dff dropout patch stride layers <<< "${line}"
  local budget=(--epochs 100 --patience "${patience}" --final-evaluation-split val)
  [[ "${smoke}" == 1 ]] && budget=(--epochs 1 --patience 1 --max-train-batches 2 --max-eval-batches 2 --final-evaluation-split none)
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified --seq-len 336 --pred-len 720 \
    --target-horizons 720 --validation-horizons 96,192,336,720 --evaluation-horizons 96,192,336,720 \
    --segment-horizons 96,192,336,720 --evaluation-prefix-mode full-crop \
    --encoder-mode contextual-patch-transformer --readout-mode siff-independent-scope-control \
    --batch-size "${batch}" --learning-rate "${base_lr}" --weight-decay 0 \
    --readout-learning-rate-multiplier "${lr_scale}" --readout-weight-decay "${readout_wd}" \
    --gradient-accumulation-steps 1 --enable-early-stopping --checkpoint-policy best-val \
    --allow-archived-research-modes \
    --protocol-class method_screening --protocol-profile iscf_bsca_decoder_transfer_patchtst_v2p1_20260815 \
    --profile-hash "${PROFILE_HASH}" --hpo-trial-id "matched_iscf__${profile}__${dataset}" \
    --hpo-profile-id "${profile}" --hpo-search-space-hash "${TRAINING_SEARCH_HASH}" \
    --seed 2021 --num-workers 0 --run-name "PATCHTST_V2P1_MATCHED_ISCF_${profile}" \
    --output-dir "${out}" --device cuda --no-save-predictions --no-official-test-mode \
    --history-patch-len "${patch}" --history-patch-stride "${stride}" --history-d-model "${dmodel}" \
    --history-n-heads "${heads}" --history-d-ff "${dff}" --history-e-layers "${layers}" \
    --history-dropout "${dropout}" --history-attn-dropout 0 --history-res-attention \
    --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" --pcsd-policy-history-dim 32 \
    --pcsd-policy-hidden-dim 64 --pcsd-policy-mode direct --pcsd-fixed-scale 720 \
    --pcsd-partition canonical --pcsd-partition-seed 15101 --pcsd-group-chunk-size 64 \
    --pcsd-target-chunk-size 128 --pcc-objective-mode measure_only --pred-loss-mode full \
    "${budget[@]}" >"${log}" 2>&1
}

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

if [[ "${RESOURCE_SMOKE}" == 1 ]]; then
  smoke_indices=(0 2 4)
  for i in "${!smoke_indices[@]}"; do
    line="${SELECTED[${smoke_indices[$i]}]}"
    IFS=$'\t' read -r dataset profile _ <<< "${line}"
    out="${OUTPUT_ROOT}/_resource_smoke/${dataset}_${profile}"
    mkdir -p "${out}"
    run_training "${line}" "${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}" "${out}" "${out}/smoke.log" 1
  done
  echo "patchtst_v2p1_resource_smoke=pass jobs=${#smoke_indices[@]}"
  exit 0
fi

if [[ "${FORMAL_TEST_ONLY}" == 1 ]]; then
  [[ "${TEST_AUTHORIZED}" == true ]]
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/check_iscf_bsca_decoder_transfer_patchtst_v2p1_artifacts.py \
    --config "${CONFIG}" --manifest "${MANIFEST}" --verify-manifest
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
{
  echo "start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "profile_hash=${PROFILE_HASH}"
  echo "diagnostic_design_hash=${DESIGN_HASH}"
  echo "selection_hash=${SELECTION_HASH}"
  echo "protocol_hash=${PROTOCOL_HASH}"
  echo "training_search_hash=${TRAINING_SEARCH_HASH}"
  echo "gpus=${GPU_IDS[*]}"
  echo "formal_test=${FORMAL_TEST_ONLY}"
} | tee "${OUTPUT_ROOT}/launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3"
  local dataset profile out log before after source_dir artifact_dir arm
  IFS=$'\t' read -r dataset profile _ <<< "${line}"
  if [[ "${FORMAL_TEST_ONLY}" == 1 ]]; then
    for arm in patchtst_iscf patchtst_iscf_bsca; do
      test_complete "${line}" "${arm}" && continue
      if [[ "${arm}" == patchtst_iscf ]]; then
        source_dir="$(matched_dir "${line}")"
      else
        source_dir="$(bsca_dir "${line}")"
      fi
      artifact_dir="$(formal_dir "${line}" "${arm}")"
      log="${OUTPUT_ROOT}/_logs/formal_${arm}_${dataset}.log"
      mkdir -p "${artifact_dir}"
      before="$(hash_file "${source_dir}/checkpoint.pt")"
      echo "formal_start=$(date -Is) job=$((index + 1))/5 arm=${arm} dataset=${dataset} gpu=${gpu}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
        --run-dir "${source_dir}" --artifact-dir "${artifact_dir}" \
        --design "${DESIGN_PATH}" --test-audit-config "${CONFIG}" \
        --evaluation-split test --probe-rows 64 --device cuda >>"${log}" 2>&1
      after="$(hash_file "${source_dir}/checkpoint.pt")"
      [[ "${before}" == "${after}" ]]
      echo "formal_done=$(date -Is) job=$((index + 1))/5 arm=${arm} dataset=${dataset} gpu=${gpu}"
    done
  else
    training_complete "${line}" && return
    out="$(matched_dir "${line}")"
    log="${OUTPUT_ROOT}/_logs/train_patchtst_iscf_${dataset}.log"
    mkdir -p "${out}"
    echo "train_start=$(date -Is) job=$((index + 1))/5 dataset=${dataset} profile=${profile} gpu=${gpu}"
    run_training "${line}" "${gpu}" "${out}" "${log}" 0
    before="$(hash_file "${out}/checkpoint.pt")"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py \
      --run-dir "${out}" --design "${DESIGN_PATH}" --test-audit-config "${CONFIG}" \
      --evaluation-split val --probe-rows 64 --device cuda >>"${log}" 2>&1
    after="$(hash_file "${out}/checkpoint.pt")"
    [[ "${before}" == "${after}" ]]
    echo "train_done=$(date -Is) job=$((index + 1))/5 dataset=${dataset} profile=${profile} gpu=${gpu}"
  fi
}

worker() {
  local worker_index="$1" gpu="$2" index
  for ((index=worker_index; index<${#SELECTED[@]}; index+=${#GPU_IDS[@]})); do
    run_one "${index}" "${SELECTED[$index]}" "${gpu}"
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
echo "patchtst_v2p1_done=$(date -Is) formal_test=${FORMAL_TEST_ONLY}"
