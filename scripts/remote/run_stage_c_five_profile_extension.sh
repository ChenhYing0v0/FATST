#!/usr/bin/env bash
set -euo pipefail

PHASE_A_ROOT="${PHASE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_a}"
PHASE_B_ROOT="${PHASE_B_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_b}"
PHASE_C_ROOT="${PHASE_C_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_c}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_five_dataset_profile_extension.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "GPU_IDS cannot be empty" >&2
  exit 2
fi

PROFILE_HASH="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${CONFIG_PATH}")"
read -r PROTOCOL_PROFILE < <(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_profile"])' "${CONFIG_PATH}")

analyze() {
  local phase="$1" output_dir="$2"
  local extra=()
  if [[ "${phase}" == "b" || "${phase}" == "c" ]]; then
    extra+=(--phase-a-summary "${PHASE_A_ROOT}/_analysis/phase_a_summary.json")
  fi
  if [[ "${phase}" == "c" ]]; then
    extra+=(--phase-b-summary "${PHASE_B_ROOT}/_analysis/phase_b_summary.json")
  fi
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_five_profile_extension.py \
      --phase "${phase}" --phase-a-root "${PHASE_A_ROOT}" \
      --phase-b-root "${PHASE_B_ROOT}" --phase-c-root "${PHASE_C_ROOT}" \
      --output-dir "${output_dir}" --config "${CONFIG_PATH}" "${extra[@]}"
}

run_one() {
  local root="$1" run_prefix="$2" dataset="$3" profile="$4"
  local patch_num="$5" d_model="$6" d_ff="$7" seed="$8" gpu="$9"
  local run_name="${run_prefix}_${profile}"
  local output_dir="${root}/${run_name}/${dataset}/h720_full/seed${seed}"
  local run_log="${root}/_logs/${run_name}_${dataset}_seed${seed}.log"
  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing dataset=${dataset} profile=${profile} seed=${seed}"
    return
  fi
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) dataset=${dataset} profile=${profile} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 --validation-horizons 720 \
      --evaluation-horizons 48,96,144,192,288,336,512,720 --e-layers 2 \
      --batch-size 32 --gradient-accumulation-steps 1 --epochs 20 --patience 5 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed "${seed}" --num-workers 0 \
      --run-name "${run_name}" --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints --final-evaluation-split val \
      --protocol-class mechanism_control --protocol-profile "${PROTOCOL_PROFILE}" \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" --legacy-dropout 0.1 \
      --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode learned-basis-forecast-operator --basis-rank 256 --pred-loss-mode full \
      >"${run_log}" 2>&1
  echo "run_done=$(date -Is) dataset=${dataset} profile=${profile} seed=${seed} gpu=${gpu}"
}

run_matrix() {
  local root="$1" run_prefix="$2"
  shift 2
  local lines=("$@") pids=() worker_index gpu
  mkdir -p "${root}/_logs" "${root}/_analysis"
  matrix_worker() {
    local start="$1" assigned_gpu="$2" line_index line
    local dataset profile patch_num d_model d_ff seed
    for ((line_index=start; line_index<${#lines[@]}; line_index+=${#GPU_IDS[@]})); do
      line="${lines[${line_index}]}"
      read -r dataset profile patch_num d_model d_ff seed <<< "${line}"
      run_one "${root}" "${run_prefix}" "${dataset}" "${profile}" \
        "${patch_num}" "${d_model}" "${d_ff}" "${seed}" "${assigned_gpu}"
    done
  }
  for worker_index in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[${worker_index}]}"
    matrix_worker "${worker_index}" "${gpu}" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_five_profile_extension.py \
      --phase a --phase-a-root /tmp/a --phase-b-root /tmp/b --phase-c-root /tmp/c \
      --output-dir /tmp/out --config "${CONFIG_PATH}" --synthetic-smoke
  echo "stage_c_five_profile_extension_dry_run=pass runs=14 hash=${PROFILE_HASH}"
  exit 0
fi

echo "stage_c_five_profile_extension_start=$(date -Is) commit=$(git rev-parse HEAD) hash=${PROFILE_HASH}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

PHASE_A_LINES=()
while IFS= read -r line; do PHASE_A_LINES+=("${line}"); done < <(python3 -c '
import json,sys
c=json.load(open(sys.argv[1]))
for dataset in c["datasets"]:
 for name,x in c["phase_a_patch_screen"]["profiles"].items():
  print(dataset,name,x["patch_num"],x["d_model"],x["d_ff"],c["common"]["screen_seed"])
' "${CONFIG_PATH}")
run_matrix "${PHASE_A_ROOT}" SC0FIVE_R2A "${PHASE_A_LINES[@]}"
analyze a "${PHASE_A_ROOT}/_analysis"

PHASE_B_LINES=()
while IFS= read -r line; do PHASE_B_LINES+=("${line}"); done < <(python3 -c '
import json,sys
c=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); a=c["phase_a_patch_screen"]["profiles"]
for dataset in c["datasets"]:
 p=a[s["selected_profiles"][dataset]]["patch_num"]
 for width in ("narrow","wide"):
  x=c["phase_b_width_screen"]["widths"][width]
  name="r2b_p{}_d{}_ff{}_{}".format(p,x["d_model"],x["d_ff"],width)
  print(dataset,name,p,x["d_model"],x["d_ff"],c["common"]["screen_seed"])
' "${CONFIG_PATH}" "${PHASE_A_ROOT}/_analysis/phase_a_summary.json")
run_matrix "${PHASE_B_ROOT}" SC0FIVE_R2B "${PHASE_B_LINES[@]}"
analyze b "${PHASE_B_ROOT}/_analysis"

PHASE_C_LINES=()
while IFS= read -r line; do PHASE_C_LINES+=("${line}"); done < <(python3 -c '
import json,re,sys
c=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); pattern=re.compile(r"r2b_p(\d+)_d(\d+)_ff(\d+)_\w+")
for dataset in c["datasets"]:
 profile=s["selected_profiles"][dataset]; p,d,ff=pattern.fullmatch(profile).groups()
 for seed in c["common"]["confirmation_seeds"]: print(dataset,profile,p,d,ff,seed)
' "${CONFIG_PATH}" "${PHASE_B_ROOT}/_analysis/phase_b_summary.json")
run_matrix "${PHASE_C_ROOT}" SC0FIVE_R2C "${PHASE_C_LINES[@]}"
analyze c "${PHASE_C_ROOT}/_analysis"
echo "stage_c_five_profile_extension_done=$(date -Is)"
