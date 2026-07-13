#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_s_predictability_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_s_predictability_gate}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-predictability_downweight full_time_mse r3_prefix_risk}"
DATASETS="${DATASETS:-ETTh2 Weather}"
GPU_IDS="${GPU_IDS:-1 2}"
EPOCHS="${EPOCHS:-100}"
SUPERVISION_CONDITION_TOP_RATIO="${SUPERVISION_CONDITION_TOP_RATIO:-0.25}"
SUPERVISION_AUX_WEIGHT="${SUPERVISION_AUX_WEIGHT:-0.1}"
SUPERVISION_PREDICTABILITY_FLOOR_WEIGHT="${SUPERVISION_PREDICTABILITY_FLOOR_WEIGHT:-0.5}"

export OUTPUT_ROOT
export LOG_ROOT
export SUPERVISION_STRATEGIES
export DATASETS
export GPU_IDS
export EPOCHS

echo "phase4_s_predictability_gate_start=$(date -Is)"
echo "supervision_condition_top_ratio=${SUPERVISION_CONDITION_TOP_RATIO}"
echo "supervision_aux_weight=${SUPERVISION_AUX_WEIGHT}"
echo "supervision_predictability_floor_weight=${SUPERVISION_PREDICTABILITY_FLOOR_WEIGHT}"

EXTRA_ARGS=(
  --supervision-condition-top-ratio "${SUPERVISION_CONDITION_TOP_RATIO}"
  --supervision-aux-weight "${SUPERVISION_AUX_WEIGHT}"
  --supervision-predictability-floor-weight "${SUPERVISION_PREDICTABILITY_FLOOR_WEIGHT}"
)

PHASE4_EXTRA_ARGS="${EXTRA_ARGS[*]}" bash scripts/remote/run_phase4_horizon_decoupled_gate.sh

echo "phase4_s_predictability_gate_done=$(date -Is)"
