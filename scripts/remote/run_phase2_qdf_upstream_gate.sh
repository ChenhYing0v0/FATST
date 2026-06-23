#!/usr/bin/env bash
set -euo pipefail

QDF_REPO_URL="${QDF_REPO_URL:-https://github.com/Master-PLC/QDF.git}"
QDF_REPO_DIR="${QDF_REPO_DIR:-/home/yingch/projects/QDF}"
CLONE_IF_MISSING="${CLONE_IF_MISSING:-1}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_qdf_upstream_gate}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-moe}}"
CONDA_BIN="${CONDA_BIN:-}"
GPU_IDS="${GPU_IDS:-1}"
SEED="${SEED:-2023}"
DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
HORIZONS="${HORIZONS:-96 192 336 720}"
META_TYPES="${META_TYPES:-all}"
RERUN="${RERUN:-0}"
KEEP_HEAVY_ARTIFACTS="${KEEP_HEAVY_ARTIFACTS:-0}"

mkdir -p "${LOG_ROOT}"

if [[ -f "/home/anaconda3/etc/profile.d/conda.sh" ]]; then
  # Non-interactive SSH shells on 529_Lab-3090 do not load the zsh conda hook.
  # shellcheck source=/dev/null
  . "/home/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "/home/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/home/anaconda3/bin/conda"
  elif [[ -x "/data/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/data/anaconda3/bin/conda"
  else
    echo "Unable to locate conda. Set CONDA_BIN=/path/to/conda." >&2
    exit 1
  fi
fi

if [[ ! -d "${QDF_REPO_DIR}/.git" ]]; then
  if [[ "${CLONE_IF_MISSING}" != "1" ]]; then
    echo "QDF repo missing at ${QDF_REPO_DIR}; set CLONE_IF_MISSING=1 or clone it manually." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${QDF_REPO_DIR}")"
  git clone "${QDF_REPO_URL}" "${QDF_REPO_DIR}"
fi

cd "${QDF_REPO_DIR}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
from pathlib import Path

path = Path("exp/exp_long_term_forecasting_meta_ml3.py")
lines = path.read_text().splitlines()
updated_lines = []
for line in lines:
    if "self.A = torch.load(best_A_path)" in line and "weights_only" not in line:
        indent = line[: len(line) - len(line.lstrip())]
        line = f"{indent}self.A = torch.load(best_A_path, weights_only=False)"
    elif "self.A = torch.load(os.path.join(ckpt_dir," in line and "A.pth" in line and "weights_only" not in line:
        indent = line[: len(line) - len(line.lstrip())]
        line = f"{indent}self.A = torch.load(os.path.join(ckpt_dir, \"A.pth\"), weights_only=False)"
    updated_lines.append(line)
updated = "\n".join(updated_lines) + "\n"
text = "\n".join(lines) + "\n"
if updated != text:
    path.write_text(updated)
    print("patched_qdf_torch_load_weights_only=1")
else:
    print("patched_qdf_torch_load_weights_only=already")
'

echo "phase2_qdf_upstream_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "qdf_repo_url=${QDF_REPO_URL}"
echo "qdf_git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conda_env=${CONDA_ENV}"
echo "conda_bin=${CONDA_BIN}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "datasets=${DATASETS}"
echo "horizons=${HORIZONS}"
echo "meta_types=${META_TYPES}"
echo "rerun=${RERUN}"
echo "keep_heavy_artifacts=${KEEP_HEAVY_ARTIFACTS}"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

if ! "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import cupy" >/dev/null 2>&1; then
  SHIM_ROOT="${OUTPUT_ROOT}/_shims"
  mkdir -p "${SHIM_ROOT}"
  cat > "${SHIM_ROOT}/cupy.py" <<'PY'
class _Random:
    def seed(self, seed=None):
        return None


random = _Random()
PY
  export PYTHONPATH="${SHIM_ROOT}:${PYTHONPATH:-}"
  echo "cupy_shim=${SHIM_ROOT}/cupy.py"
fi

read -r -a gpu_ids <<< "${GPU_IDS}"
read -r -a datasets <<< "${DATASETS}"
read -r -a horizons <<< "${HORIZONS}"
read -r -a meta_types <<< "${META_TYPES}"

dataset_args() {
  local dataset="$1"
  case "${dataset}" in
    ETTh2)
      echo "--root_path ${DATASET_ROOT}/ETT-small/ --data_path ETTh2.csv --data_id ETTh2 --data ETTh2 --enc_in 7 --dec_in 7 --c_out 7 --cycle 24"
      ;;
    ETTm1)
      echo "--root_path ${DATASET_ROOT}/ETT-small/ --data_path ETTm1.csv --data_id ETTm1 --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --cycle 96"
      ;;
    Weather)
      echo "--root_path ${DATASET_ROOT}/weather/ --data_path weather.csv --data_id Weather --data custom --enc_in 21 --dec_in 21 --c_out 21 --cycle 144"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

qdf_hparams() {
  local dataset="$1"
  local horizon="$2"
  case "${dataset}_${horizon}" in
    ETTh2_96) echo "0.0005 0.0005 0.2 300 1 2 32" ;;
    ETTh2_192) echo "0.0005 0.0005 0.01 300 4 1 32" ;;
    ETTh2_336) echo "0.0005 0.0005 0.2 300 1 2 32" ;;
    ETTh2_720) echo "0.0005 0.0005 0.2 300 2 4 32" ;;
    ETTm1_96) echo "0.001 0.001 0.1 500 2 1 32" ;;
    ETTm1_192) echo "0.001 0.001 0.01 500 2 3 32" ;;
    ETTm1_336) echo "0.001 0.001 0.2 500 4 1 32" ;;
    ETTm1_720) echo "0.001 0.001 0.2 500 2 5 32" ;;
    Weather_96) echo "0.002 0.002 0.1 700 5 4 32" ;;
    Weather_192) echo "0.002 0.002 0.02 700 3 1 32" ;;
    Weather_336) echo "0.002 0.002 0.05 700 4 5 32" ;;
    Weather_720) echo "0.002 0.002 0.2 700 3 4 32" ;;
    *)
      echo "Unsupported dataset/horizon: ${dataset}/${horizon}" >&2
      return 1
      ;;
  esac
}

run_one() {
  local meta_type="$1"
  local dataset="$2"
  local horizon="$3"
  local gpu_id="$4"
  local model_name="TQNet"
  local des="TQNet"
  local train_epochs=30
  local patience=5
  local test_batch_size=1
  local use_revin=1
  local model_type="linear"
  local dropout=0.5
  local first_order=1
  local auxi_loss="MSE"
  local overlap_ratio=0.0
  local reg_lambda=0.0
  local rec_lambda=1.0
  local auxi_lambda=0.0
  local auxi_batch_size=64
  local lradj="type1"
  local max_norm=5.0
  local lr lr_inner lr_meta meta_steps num_tasks meta_inner_steps batch_size
  read -r lr lr_inner lr_meta meta_steps num_tasks meta_inner_steps batch_size < <(qdf_hparams "${dataset}" "${horizon}")
  local dataset_extra
  dataset_extra="$(dataset_args "${dataset}")"
  local run_dir="${OUTPUT_ROOT}/${meta_type}/${dataset}/h${horizon}/seed${SEED}"
  local log_path="${LOG_ROOT}/QDF_${meta_type}_${dataset}_h${horizon}_seed${SEED}.log"
  local checkpoints="${run_dir}/checkpoints"
  local results="${run_dir}/results"
  local test_results="${run_dir}/test_results"
  local marker="${run_dir}/run.done"

  if [[ "${RERUN}" != "1" && -f "${marker}" ]]; then
    echo "skip_existing meta_type=${meta_type} dataset=${dataset} horizon=${horizon}"
    return 0
  fi
  if [[ "${RERUN}" == "1" ]]; then
    rm -rf "${run_dir}"
  fi
  mkdir -p "${run_dir}" "${checkpoints}" "${results}" "${test_results}"

  echo "run_start=$(date -Is) meta_type=${meta_type} dataset=${dataset} horizon=${horizon} gpu=${gpu_id}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -u run.py \
      --task_name long_term_forecast_meta_ml3 \
      --is_training 1 \
      ${dataset_extra} \
      --model_id "${dataset}_96_${horizon}" \
      --model "${model_name}" \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${horizon}" \
      --factor 3 \
      --des "${des}" \
      --learning_rate "${lr}" \
      --lradj "${lradj}" \
      --train_epochs "${train_epochs}" \
      --patience "${patience}" \
      --batch_size "${batch_size}" \
      --test_batch_size "${test_batch_size}" \
      --itr 1 \
      --rec_lambda "${rec_lambda}" \
      --auxi_lambda "${auxi_lambda}" \
      --reg_lambda "${reg_lambda}" \
      --auxi_batch_size "${auxi_batch_size}" \
      --fix_seed "${SEED}" \
      --checkpoints "${checkpoints}" \
      --results "${results}" \
      --test_results "${test_results}" \
      --log_path "${run_dir}/result_long_term_forecast.txt" \
      --rerun "${RERUN}" \
      --inner_lr "${lr_inner}" \
      --meta_lr "${lr_meta}" \
      --meta_inner_steps "${meta_inner_steps}" \
      --overlap_ratio "${overlap_ratio}" \
      --num_tasks "${num_tasks}" \
      --max_norm "${max_norm}" \
      --auxi_loss "${auxi_loss}" \
      --model_type "${model_type}" \
      --cycle "$(awk '{for(i=1;i<=NF;i++) if($i=="--cycle") print $(i+1)}' <<< "${dataset_extra}")" \
      --use_revin "${use_revin}" \
      --dropout "${dropout}" \
      --first_order "${first_order}" \
      --warmup_steps "${meta_steps}" \
      --meta_type "${meta_type}" 2>&1 | tee "${log_path}"

  if [[ "${KEEP_HEAVY_ARTIFACTS}" != "1" ]]; then
    find "${run_dir}" -type f -name "checkpoint.pth" -delete
  fi
  date -Is > "${marker}"
  echo "run_done=$(date -Is) meta_type=${meta_type} dataset=${dataset} horizon=${horizon} gpu=${gpu_id}"
}

job_index=0
active_jobs=0
max_jobs="${#gpu_ids[@]}"

for meta_type in "${meta_types[@]}"; do
  for dataset in "${datasets[@]}"; do
    for horizon in "${horizons[@]}"; do
      gpu_id="${gpu_ids[$((job_index % max_jobs))]}"
      run_one "${meta_type}" "${dataset}" "${horizon}" "${gpu_id}" &
      job_index=$((job_index + 1))
      active_jobs=$((active_jobs + 1))
      if (( active_jobs >= max_jobs )); then
        wait -n
        active_jobs=$((active_jobs - 1))
      fi
    done
  done
done

wait
echo "phase2_qdf_upstream_gate_done=$(date -Is)"
