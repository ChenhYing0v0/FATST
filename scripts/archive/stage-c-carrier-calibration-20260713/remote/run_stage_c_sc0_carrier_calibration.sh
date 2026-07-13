#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration}"
DATASET_ROOT="${DATASET_ROOT:-/home/yingch/dataset}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage_c_mechanism_control.json}"
CONDA_ENV="${CONDA_ENV:-moe}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_STR="${GPU_IDS:-0 1 2}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "missing config: ${CONFIG_PATH}" >&2
  exit 2
fi

PROFILE_HASH="$(${PYTHON_BIN} -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${CONFIG_PATH}")"
read -r PROFILE SEQ_LEN PRED_LEN TARGET_HORIZONS VALIDATION_HORIZONS EVALUATION_HORIZONS BASIS_RANK E_LAYERS DROPOUT LAYER_NORM LEARNING_RATE BATCH_SIZE ACCUM_STEPS EPOCHS PRED_LOSS_MODE CHECKPOINT_POLICY FINAL_SPLIT SEED < <(
  "${PYTHON_BIN}" -c '
import json, sys
c = json.load(open(sys.argv[1]))
x = c["common"]
values = [
    c["protocol_profile"], x["seq_len"], x["pred_len"],
    ",".join(map(str, x["target_horizons"])),
    ",".join(map(str, x["validation_horizons"])),
    ",".join(map(str, x["evaluation_horizons"])),
    x["basis_rank"], x["e_layers"], x["dropout"], x["layer_norm"],
    x["learning_rate"], x["batch_size"], x["gradient_accumulation_steps"],
    x["epochs"], x["pred_loss_mode"], x["checkpoint_policy"],
    x["final_evaluation_split"], x["initial_seed"],
]
print(" ".join(map(str, values)))
' "${CONFIG_PATH}"
)
DATASETS=()
while IFS= read -r dataset; do
  DATASETS+=("${dataset}")
done < <(
  "${PYTHON_BIN}" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["datasets"]))' "${CONFIG_PATH}"
)
ARM_LINES=()
while IFS= read -r arm_line; do
  ARM_LINES+=("${arm_line}")
done < <(
  "${PYTHON_BIN}" -c '
import json, sys
c = json.load(open(sys.argv[1]))
for name, arm in c["arms"].items():
    print("\t".join(map(str, [name, arm["patch_num"], arm["d_model"], arm["d_ff"]])))
' "${CONFIG_PATH}"
)
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

if [[ "${FINAL_SPLIT}" != "val" ]]; then
  echo "SC0 calibration requires final_evaluation_split=val" >&2
  exit 2
fi
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "stage_c_sc0_dry_run=pass"
  echo "protocol_profile=${PROFILE} profile_hash=${PROFILE_HASH}"
  echo "datasets=${DATASETS[*]} arms=${ARM_LINES[*]}"
  echo "target_horizons=${TARGET_HORIZONS} validation_horizons=${VALIDATION_HORIZONS} evaluation_horizons=${EVALUATION_HORIZONS}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs" "${OUTPUT_ROOT}/_analysis"
cp "${CONFIG_PATH}" "${OUTPUT_ROOT}/stage_c_mechanism_control_${PROFILE_HASH}.json"

echo "stage_c_sc0_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "git_status=$(git status --short | tr '\n' ';')"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "config_path=${CONFIG_PATH}"
echo "protocol_profile=${PROFILE}"
echo "profile_hash=${PROFILE_HASH}"
echo "conda_env=${CONDA_ENV}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "datasets=${DATASETS[*]}"
echo "seed=${SEED} epochs=${EPOCHS} effective_batch=$((BATCH_SIZE * ACCUM_STEPS))"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

run_one() {
  local dataset="$1"
  local arm_line="$2"
  local gpu="$3"
  local arm patch_num d_model d_ff run_name output_dir run_log
  IFS=$'\t' read -r arm patch_num d_model d_ff <<< "${arm_line}"
  run_name="SC0_${arm}_validation_only"
  output_dir="${OUTPUT_ROOT}/${run_name}/${dataset}/h720_full/seed${SEED}"
  run_log="${OUTPUT_ROOT}/_logs/${run_name}_${dataset}_seed${SEED}.log"

  if [[ -s "${output_dir}/metrics_best_val_by_target_horizon.csv" ]]; then
    echo "skip_existing dataset=${dataset} arm=${arm} output_dir=${output_dir}"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "run_start=$(date -Is) dataset=${dataset} arm=${arm} gpu=${gpu} output_dir=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python baselines/timealign_official/train_repo.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset "${dataset}" \
      --mode unified \
      --seq-len "${SEQ_LEN}" \
      --pred-len "${PRED_LEN}" \
      --target-horizons "${TARGET_HORIZONS}" \
      --validation-horizons "${VALIDATION_HORIZONS}" \
      --evaluation-horizons "${EVALUATION_HORIZONS}" \
      --e-layers "${E_LAYERS}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${ACCUM_STEPS}" \
      --epochs "${EPOCHS}" \
      --patience "${EPOCHS}" \
      --seed "${SEED}" \
      --max-train-batches "${MAX_TRAIN_BATCHES}" \
      --max-eval-batches "${MAX_EVAL_BATCHES}" \
      --num-workers "${NUM_WORKERS}" \
      --run-name "${run_name}" \
      --output-dir "${output_dir}" \
      --device cuda \
      --checkpoint-policy "${CHECKPOINT_POLICY}" \
      --evaluate-dual-checkpoints \
      --final-evaluation-split "${FINAL_SPLIT}" \
      --protocol-class mechanism_control \
      --protocol-profile "${PROFILE}" \
      --profile-hash "${PROFILE_HASH}" \
      --legacy-patch-num "${patch_num}" \
      --legacy-d-model "${d_model}" \
      --legacy-d-ff "${d_ff}" \
      --legacy-dropout "${DROPOUT}" \
      --legacy-layer-norm "${LAYER_NORM}" \
      --learning-rate "${LEARNING_RATE}" \
      --readout-mode learned-basis-forecast-operator \
      --basis-rank "${BASIS_RANK}" \
      --pred-loss-mode "${PRED_LOSS_MODE}" 2>&1 | tee "${run_log}"
  echo "run_done=$(date -Is) dataset=${dataset} arm=${arm} gpu=${gpu} output_dir=${output_dir}"
}

pids=()
gpu_count="${#GPU_IDS[@]}"
job_index=0
for dataset in "${DATASETS[@]}"; do
  for arm_line in "${ARM_LINES[@]}"; do
    gpu="${GPU_IDS[$((job_index % gpu_count))]}"
    run_one "${dataset}" "${arm_line}" "${gpu}" &
    pids+=("$!")
    job_index=$((job_index + 1))
    if (( ${#pids[@]} >= gpu_count )); then
      wait -n
      remaining=()
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          remaining+=("${pid}")
        fi
      done
      pids=("${remaining[@]}")
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

analysis_dir="${OUTPUT_ROOT}/_analysis/seed${SEED}"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_sc0_carrier_calibration.py \
    --raw-root "${OUTPUT_ROOT}" \
    --output-dir "${analysis_dir}" \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}"

echo "stage_c_sc0_done=$(date -Is) analysis_dir=${analysis_dir}"
find "${OUTPUT_ROOT}" -name metrics_best_val_by_target_horizon.csv -type f | sort
