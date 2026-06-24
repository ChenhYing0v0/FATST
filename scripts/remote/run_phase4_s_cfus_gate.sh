#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_s_cfus_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_s_cfus_gate}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-conditioned_future_unit_scheduling full_time_mse r3_prefix_risk}"
DATASETS="${DATASETS:-ETTh2 Weather}"
GPU_IDS="${GPU_IDS:-1 2}"
EPOCHS="${EPOCHS:-100}"
SUPERVISION_CONDITION="${SUPERVISION_CONDITION:-label_novelty}"
SUPERVISION_CONDITION_TOP_RATIO="${SUPERVISION_CONDITION_TOP_RATIO:-0.25}"
SUPERVISION_AUX_WEIGHT="${SUPERVISION_AUX_WEIGHT:-0.1}"

export OUTPUT_ROOT
export LOG_ROOT
export SUPERVISION_STRATEGIES
export DATASETS
export GPU_IDS
export EPOCHS

echo "phase4_s_cfus_gate_start=$(date -Is)"
echo "supervision_condition=${SUPERVISION_CONDITION}"
echo "supervision_condition_top_ratio=${SUPERVISION_CONDITION_TOP_RATIO}"
echo "supervision_aux_weight=${SUPERVISION_AUX_WEIGHT}"

EXTRA_ARGS=(
  --supervision-condition "${SUPERVISION_CONDITION}"
  --supervision-condition-top-ratio "${SUPERVISION_CONDITION_TOP_RATIO}"
  --supervision-aux-weight "${SUPERVISION_AUX_WEIGHT}"
)

PHASE4_EXTRA_ARGS="${EXTRA_ARGS[*]}" bash scripts/remote/run_phase4_horizon_decoupled_gate.sh

echo "phase4_s_cfus_gate_done=$(date -Is)"
