#!/usr/bin/env bash
set -euo pipefail

PHASE_A_ROOT="${PHASE_A_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2b_width_screen}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_dataset_profile_calibration_r2.json}"
R2A_SUMMARY="${R2A_SUMMARY:-${PHASE_A_ROOT}/_analysis/r2a_summary.json}"
CONDA_ENV="${CONDA_ENV:-moe}"; CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"; DRY_RUN="${DRY_RUN:-0}"
PROFILE_HASH="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${CONFIG_PATH}")"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

LINES=()
while IFS= read -r value; do LINES+=("${value}"); done < <(
python3 -c '
import json,sys
c=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); a=c["phase_a_patch_screen"]["profiles"]
for dataset in c["datasets"]:
 p=a[s["selected_profiles"][dataset]]["patch_num"]
 for name in ("narrow","wide"):
  w=c["phase_b_width_screen"]["widths"][name]
  print("\t".join(map(str,[dataset,p,name,w["d_model"],w["d_ff"]])))
' "${CONFIG_PATH}" "${R2A_SUMMARY}")

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "stage_c_dap_r2b_dry_run=pass profile_hash=${PROFILE_HASH} new_runs=${#LINES[@]}"; exit 0
fi
mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
echo "stage_c_dap_r2b_start=$(date -Is) commit=$(git rev-parse HEAD) profile_hash=${PROFILE_HASH}"

run_one() {
  local line="$1" gpu="$2" dataset patch_num width_name d_model d_ff profile run_name output_dir
  IFS=$'\t' read -r dataset patch_num width_name d_model d_ff <<< "${line}"
  profile="r2b_p${patch_num}_d${d_model}_ff${d_ff}_${width_name}"
  run_name="SC0DAP_R2B_${profile}"
  output_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/h720_full/seed2021"
  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]]; then echo "skip ${dataset} ${profile}"; return; fi
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) dataset=${dataset} profile=${profile} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len 720 --pred-len 720 --target-horizons 720 --validation-horizons 720 \
      --evaluation-horizons 48,96,144,192,288,336,512,720 --e-layers 2 \
      --batch-size 32 --gradient-accumulation-steps 1 --epochs 20 --patience 5 \
      --enable-early-stopping --early-stopping-min-delta 0 --seed 2021 --num-workers 0 \
      --run-name "${run_name}" --output-dir "${output_dir}" --device cuda \
      --checkpoint-policy best-val --no-evaluate-dual-checkpoints --final-evaluation-split val \
      --protocol-class mechanism_control --protocol-profile stage_c_dataset_profile_calibration_r2 \
      --profile-hash "${PROFILE_HASH}" --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" --legacy-d-ff "${d_ff}" --legacy-dropout 0.1 \
      --legacy-layer-norm 1 --learning-rate 0.0001 \
      --readout-mode learned-basis-forecast-operator --basis-rank 256 --pred-loss-mode full \
      >"${OUTPUT_ROOT}/_logs/${run_name}_${dataset}.log" 2>&1
  echo "run_done=$(date -Is) dataset=${dataset} profile=${profile} gpu=${gpu}"
}

pids=(); index=0
for line in "${LINES[@]}"; do
  gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"; run_one "${line}" "${gpu}" & pids+=("$!"); index=$((index+1))
  if (( ${#pids[@]} >= ${#GPU_IDS[@]} )); then
    wait -n; remaining=(); for pid in "${pids[@]}"; do if kill -0 "${pid}" 2>/dev/null; then remaining+=("${pid}"); fi; done; pids=("${remaining[@]}")
  fi
done
for pid in "${pids[@]}"; do wait "${pid}"; done
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_dap_r2b_width_screen.py \
    --phase-a-root "${PHASE_A_ROOT}" --phase-b-root "${OUTPUT_ROOT}" \
    --phase-a-summary "${R2A_SUMMARY}" --output-dir "${OUTPUT_ROOT}/_analysis" --config "${CONFIG_PATH}"
echo "stage_c_dap_r2b_done=$(date -Is)"
