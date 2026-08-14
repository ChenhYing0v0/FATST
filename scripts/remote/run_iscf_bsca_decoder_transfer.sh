#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_20260814}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG="${CONFIG:-configs/iscf_bsca_decoder_transfer_protocol.json}"
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
PROFILE_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["path"])' "${CONFIG}")"
PROFILE_HASH="$(hash_file "${PROFILE_PATH}")"
EXPECTED_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profiles"]["sha256"])' "${CONFIG}")"
[[ "${PROFILE_HASH}" == "${EXPECTED_HASH}" ]]
MANIFEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_contract"]["training_manifest"])' "${CONFIG}")"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"

LINES=()
while IFS= read -r line; do LINES+=("${line}"); done < <(python3 - "${CONFIG}" <<'PY'
import json, sys
c=json.load(open(sys.argv[1])); p=json.load(open(c["profiles"]["path"])); arms={x["id"]:x for x in c["arms"]}
for backbone,dataset,arm_id in c["launch_order"]:
    a=arms[arm_id]; b=p["backbones"][backbone]; d=b["dataset_profiles"][dataset]
    rank=c["matched_ranks"][dataset] if a["readout_mode"] == "siff-independent-scope-control" else 256
    vals=[backbone,dataset,arm_id,b["encoder_mode"],a["readout_mode"],a["policy_mode"],a["objective_mode"],rank,d["learning_rate"],d["batch_size"],d["patience"],d.get("d_model",0),d.get("n_heads",0),d.get("d_ff",0),d.get("dropout",0.0),b.get("moving_average",25),b.get("history_patch_len",16),b.get("history_patch_stride",8),b.get("history_e_layers",3)]
    print("\t".join(map(str,vals)))
PY
)

run_dir() { IFS=$'\t' read -r b d a _ <<< "$1"; echo "${OUTPUT_ROOT}/${b}/${a}/${d}/seed2021"; }
training_complete() { local d; d="$(run_dir "$1")"; [[ -s "${d}/checkpoint.pt" && -s "${d}/trained_invariants.json" ]] && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' "${d}/trained_invariants.json" 2>/dev/null; }
test_complete() { local d; d="$(run_dir "$1")"; [[ -s "${d}/test_audit_invariants.json" ]] && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["pass"] is True' "${d}/test_audit_invariants.json" 2>/dev/null; }

status_counts() {
  local trained=0 tested=0 line
  for line in "${LINES[@]}"; do training_complete "${line}" && trained=$((trained+1)); test_complete "${line}" && tested=$((tested+1)); done
  echo "${trained} ${tested}"
}
if [[ "${STATUS_ONLY}" == 1 ]]; then read -r a b <<< "$(status_counts)"; echo "decoder_transfer_status=$(date -Is) training=${a}/30 test=${b}/30"; exit 0; fi
if [[ "${DRY_RUN}" == 1 ]]; then printf '%s\n' "${LINES[@]}"; echo "decoder_transfer_dry_run=pass jobs=${#LINES[@]} profile_hash=${PROFILE_HASH}"; exit 0; fi

if [[ "${FORMAL_TEST_ONLY}" == 1 ]]; then
  [[ "${TEST_AUTHORIZED}" == true ]]
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python scripts/check_iscf_bsca_decoder_transfer_training_artifacts.py --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" --manifest "${MANIFEST}" --verify-manifest
  read -r trained _ <<< "$(status_counts)"; [[ "${trained}" -eq 30 ]]
fi

run_training() {
  local line="$1" gpu="$2" out="$3" log="$4" smoke="$5"
  local backbone dataset arm encoder readout policy objective rank lr batch patience dmodel heads dff dropout moving patch stride layers
  IFS=$'\t' read -r backbone dataset arm encoder readout policy objective rank lr batch patience dmodel heads dff dropout moving patch stride layers <<< "${line}"
  local extra=(--dlinear-moving-avg "${moving}")
  if [[ "${backbone}" == patchtst_style ]]; then
    extra=(--history-patch-len "${patch}" --history-patch-stride "${stride}" --history-d-model "${dmodel}" --history-n-heads "${heads}" --history-d-ff "${dff}" --history-e-layers "${layers}" --history-dropout "${dropout}" --history-attn-dropout 0 --history-res-attention)
  fi
  local budget=(--epochs 100 --patience "${patience}" --final-evaluation-split val)
  [[ "${smoke}" == 1 ]] && budget=(--epochs 1 --patience 1 --max-train-batches 2 --max-eval-batches 2 --final-evaluation-split none)
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python baselines/timealign_official/train_repo.py \
    --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified --seq-len 336 --pred-len 720 \
    --target-horizons 720 --validation-horizons 96,192,336,720 --evaluation-horizons 96,192,336,720 --segment-horizons 96,192,336,720 \
    --evaluation-prefix-mode full-crop --encoder-mode "${encoder}" --readout-mode "${readout}" --batch-size "${batch}" \
    --learning-rate "${lr}" --weight-decay 0 --gradient-accumulation-steps 1 --enable-early-stopping --checkpoint-policy best-val \
    --protocol-class method_screening --protocol-profile iscf_bsca_decoder_transfer_20260814 --profile-hash "${PROFILE_HASH}" \
    --seed 2021 --num-workers 0 --run-name "DECODER_TRANSFER_${arm}" --output-dir "${out}" --device cuda --no-save-predictions \
    --pcsd-coordinate-dim 4 --pcsd-mode-rank "${rank}" --pcsd-policy-history-dim 32 --pcsd-policy-hidden-dim 64 \
    --pcsd-policy-mode "${policy}" --pcsd-fixed-scale 720 --pcsd-partition canonical --pcsd-partition-seed 15101 \
    --pcsd-group-chunk-size 64 --pcsd-target-chunk-size 128 --pcc-objective-mode "${objective}" --pred-loss-mode full \
    "${extra[@]}" "${budget[@]}" >"${log}" 2>&1
}

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
if [[ "${RESOURCE_SMOKE}" == 1 ]]; then
  smoke=(0 2 9 15 19 29)
  for i in "${!smoke[@]}"; do line="${LINES[${smoke[$i]}]}"; IFS=$'\t' read -r b d a _ <<< "${line}"; out="${OUTPUT_ROOT}/_resource_smoke/${a}_${d}"; mkdir -p "${out}"; run_training "${line}" "${GPU_IDS[$((i%${#GPU_IDS[@]}))]}" "${out}" "${out}/smoke.log" 1; done
  echo "decoder_transfer_resource_smoke=pass jobs=${#smoke[@]}"; exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"
{
  echo "start=$(date -Is)"; echo "commit=$(git rev-parse HEAD)"; echo "profile_hash=${PROFILE_HASH}"; echo "gpus=${GPU_IDS[*]}"; echo "formal_test=${FORMAL_TEST_ONLY}"
} | tee "${OUTPUT_ROOT}/launch_record_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local index="$1" line="$2" gpu="$3" out log before after
  IFS=$'\t' read -r b d a _ <<< "${line}"; out="$(run_dir "${line}")"; log="${OUTPUT_ROOT}/_logs/${a}_${d}.log"; mkdir -p "${out}"
  if [[ "${FORMAL_TEST_ONLY}" == 1 ]]; then
    test_complete "${line}" && return
    before="$(hash_file "${out}/checkpoint.pt")"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py --run-dir "${out}" --design "${CONFIG}" --test-audit-config "${CONFIG}" --evaluation-split test --probe-rows 64 --device cuda >>"${log}" 2>&1
    after="$(hash_file "${out}/checkpoint.pt")"; [[ "${before}" == "${after}" ]]
  else
    training_complete "${line}" && return
    echo "train_start=$(date -Is) job=$((index+1))/30 backbone=${b} arm=${a} dataset=${d} gpu=${gpu}"
    run_training "${line}" "${gpu}" "${out}" "${log}" 0
    before="$(hash_file "${out}/checkpoint.pt")"
    CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python scripts/evaluate_stage_c_pcsd_cf_checkpoint.py --run-dir "${out}" --design "${CONFIG}" --test-audit-config "${CONFIG}" --evaluation-split val --probe-rows 64 --device cuda >>"${log}" 2>&1
    after="$(hash_file "${out}/checkpoint.pt")"; [[ "${before}" == "${after}" ]]
  fi
  echo "job_done=$(date -Is) job=$((index+1))/30 backbone=${b} arm=${a} dataset=${d} gpu=${gpu}"
}

worker() { local wi="$1" gpu="$2" i; for ((i=wi;i<${#LINES[@]};i+=${#GPU_IDS[@]})); do run_one "${i}" "${LINES[$i]}" "${gpu}"; done; }
pids=(); for i in "${!GPU_IDS[@]}"; do worker "${i}" "${GPU_IDS[$i]}" & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" == 0 ]]
echo "decoder_transfer_done=$(date -Is) formal_test=${FORMAL_TEST_ONLY}"
