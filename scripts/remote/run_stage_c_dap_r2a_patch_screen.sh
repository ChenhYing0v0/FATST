#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_dataset_profile_calibration_r2.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRY_RUN="${DRY_RUN:-0}"

PROFILE_HASH="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${CONFIG_PATH}")"
read -r PROFILE SEQ_LEN PRED_LEN TARGETS VAL_HORIZONS EVAL_HORIZONS BASIS_RANK E_LAYERS DROPOUT LAYER_NORM LR BATCH ACCUM EPOCHS PATIENCE MIN_DELTA LOSS_MODE CHECKPOINT FINAL_SPLIT SEED < <(
  "${PYTHON_BIN}" -c '
import json,sys
c=json.load(open(sys.argv[1])); x=c["common"]
v=[c["protocol_profile"],x["seq_len"],x["pred_len"],
   ",".join(map(str,x["target_horizons"])),
   ",".join(map(str,x["validation_horizons"])),
   ",".join(map(str,x["evaluation_horizons"])),
   x["basis_rank"],x["e_layers"],x["dropout"],x["layer_norm"],x["learning_rate"],
   x["batch_size"],x["gradient_accumulation_steps"],x["max_epochs"],
   x["early_stopping_patience"],x["early_stopping_min_delta"],x["pred_loss_mode"],
   x["checkpoint_policy"],x["final_evaluation_split"],x["screen_seed"]]
print(" ".join(map(str,v)))
' "${CONFIG_PATH}"
)
DATASETS=()
while IFS= read -r value; do DATASETS+=("${value}"); done < <(
  "${PYTHON_BIN}" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["datasets"]))' "${CONFIG_PATH}"
)
PROFILE_LINES=()
while IFS= read -r value; do PROFILE_LINES+=("${value}"); done < <(
  "${PYTHON_BIN}" -c '
import json,sys
c=json.load(open(sys.argv[1]))
for name,x in c["phase_a_patch_screen"]["profiles"].items():
    print("\t".join(map(str,[name,x["patch_num"],x["d_model"],x["d_ff"]])))
' "${CONFIG_PATH}"
)
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${FINAL_SPLIT}" != "val" ]]; then echo "R2A requires validation-only" >&2; exit 2; fi
if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then echo "GPU_IDS cannot be empty" >&2; exit 2; fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "stage_c_dap_r2a_dry_run=pass profile_hash=${PROFILE_HASH}"
  echo "datasets=${DATASETS[*]} profiles=${#PROFILE_LINES[@]} seed=${SEED}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
cp "${CONFIG_PATH}" "${OUTPUT_ROOT}/r2_config_${PROFILE_HASH}.json"
echo "stage_c_dap_r2a_start=$(date -Is) commit=$(git rev-parse HEAD) profile_hash=${PROFILE_HASH}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

run_one() {
  local dataset="$1" profile_line="$2" gpu="$3"
  local profile patch_num d_model d_ff run_name output_dir run_log
  IFS=$'\t' read -r profile patch_num d_model d_ff <<< "${profile_line}"
  run_name="SC0DAP_R2A_${profile}"
  output_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/h720_full/seed${SEED}"
  run_log="${OUTPUT_ROOT}/_logs/${run_name}_${dataset}_seed${SEED}.log"
  if [[ -s "${output_dir}/metrics_by_target_horizon.csv" ]]; then
    echo "skip_existing dataset=${dataset} profile=${profile}"; return 0
  fi
  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) dataset=${dataset} profile=${profile} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" --dataset "${dataset}" --mode unified \
      --seq-len "${SEQ_LEN}" --pred-len "${PRED_LEN}" --target-horizons "${TARGETS}" \
      --validation-horizons "${VAL_HORIZONS}" --evaluation-horizons "${EVAL_HORIZONS}" \
      --e-layers "${E_LAYERS}" --batch-size "${BATCH}" \
      --gradient-accumulation-steps "${ACCUM}" --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" --enable-early-stopping \
      --early-stopping-min-delta "${MIN_DELTA}" --seed "${SEED}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers "${NUM_WORKERS}" --run-name "${run_name}" --output-dir "${output_dir}" \
      --device cuda --checkpoint-policy "${CHECKPOINT}" --no-evaluate-dual-checkpoints \
      --final-evaluation-split "${FINAL_SPLIT}" --protocol-class mechanism_control \
      --protocol-profile "${PROFILE}" --profile-hash "${PROFILE_HASH}" \
      --legacy-patch-num "${patch_num}" --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" --legacy-dropout "${DROPOUT}" \
      --legacy-layer-norm "${LAYER_NORM}" --learning-rate "${LR}" \
      --readout-mode learned-basis-forecast-operator --basis-rank "${BASIS_RANK}" \
      --pred-loss-mode "${LOSS_MODE}" 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) dataset=${dataset} profile=${profile} gpu=${gpu}"
}

pids=(); job_index=0; gpu_count="${#GPU_IDS[@]}"
for dataset in "${DATASETS[@]}"; do
  for profile_line in "${PROFILE_LINES[@]}"; do
    gpu="${GPU_IDS[$((job_index % gpu_count))]}"
    run_one "${dataset}" "${profile_line}" "${gpu}" & pids+=("$!")
    job_index=$((job_index+1))
    if (( ${#pids[@]} >= gpu_count )); then
      wait -n
      remaining=(); for pid in "${pids[@]}"; do if kill -0 "${pid}" 2>/dev/null; then remaining+=("${pid}"); fi; done
      pids=("${remaining[@]}")
    fi
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_dap_r2a_patch_screen.py \
    --raw-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/_analysis" --config "${CONFIG_PATH}"
echo "stage_c_dap_r2a_done=$(date -Is)"
