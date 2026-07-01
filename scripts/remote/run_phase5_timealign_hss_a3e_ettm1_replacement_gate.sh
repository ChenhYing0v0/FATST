#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT:-/home/yingch/dataset}}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-official-last}"
GPU_IDS="${GPU_IDS:-0 1 2}"
SEED="${SEED:-2021}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
SEQ_LEN="${SEQ_LEN:-720}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LAUNCH_ROOT="${LAUNCH_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3e_ettm1_replacement_gate/_launcher}"

mkdir -p "${LAUNCH_ROOT}"

run_step() {
  local label="$1"
  shift
  echo "step_start=$(date -Is) label=${label}"
  "$@"
  echo "step_done=$(date -Is) label=${label}"
}

common_env=(
  DATASET_ROOT="${DATASET_ROOT}"
  CHECKPOINT_POLICY="${CHECKPOINT_POLICY}"
  GPU_IDS="${GPU_IDS}"
  SEED="${SEED}"
  EPOCHS="${EPOCHS}"
  PATIENCE="${PATIENCE}"
  SEQ_LEN="${SEQ_LEN}"
  BATCH_SIZE="${BATCH_SIZE}"
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES}"
  MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES}"
  NUM_WORKERS="${NUM_WORKERS}"
)

echo "phase5_timealign_hss_a3e_ettm1_replacement_gate_start=$(date -Is)"
echo "cwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_root=${DATASET_ROOT}"
echo "checkpoint_policy=${CHECKPOINT_POLICY}"
echo "gpu_ids=${GPU_IDS}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "patience=${PATIENCE}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size=${BATCH_SIZE}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_eval_batches=${MAX_EVAL_BATCHES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

run_step "ettm1_official_fixed_reference" \
  env "${common_env[@]}" DATASETS="ETTm1" HORIZONS="96 192 336 720" \
  bash scripts/remote/run_phase5_timealign_official_gate.sh

run_step "ettm1_h1_target_set_checkpoint" \
  env "${common_env[@]}" DATASETS="ETTm1" ARMS="target_set_decoder_multiprefix" \
  bash scripts/remote/run_phase5_timealign_hss_h1_readout_gate.sh

run_step "ettm1_h1c_row_gated_reference" \
  env "${common_env[@]}" DATASETS="ETTm1" ARMS="row_gated_dense_head_multiprefix" \
  bash scripts/remote/run_phase5_timealign_hss_h1c_capacity_preserving_gate.sh

run_step "ettm1_a2_nested_reference" \
  env "${common_env[@]}" DATASETS="ETTm1" ARMS="nested_segment_decoder_multiprefix" \
  bash scripts/remote/run_phase5_timealign_hss_a2_interface_gate.sh

run_step "ettm1_a3c_warm_reference" \
  env "${common_env[@]}" DATASETS="ETTm1" \
  bash scripts/remote/run_phase5_timealign_hss_a3c_warm_started_nested_gate.sh

run_step "ettm1_a3d_teacher_w03_reference" \
  env "${common_env[@]}" DATASETS="ETTm1" ARMS="teacher_preserved_nested_w03" \
  bash scripts/remote/run_phase5_timealign_hss_a3d_teacher_preserved_nested_gate.sh

run_step "a3e_target_conditioned_nested_gate" \
  env "${common_env[@]}" DATASETS="Weather ETTm1 ETTh2" \
  bash scripts/remote/run_phase5_timealign_hss_a3e_target_conditioned_nested_gate.sh

echo "phase5_timealign_hss_a3e_ettm1_replacement_gate_done=$(date -Is)"
