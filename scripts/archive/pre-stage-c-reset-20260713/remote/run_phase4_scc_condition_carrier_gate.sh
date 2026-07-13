#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_scc_condition_carrier_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_scc_condition_carrier_gate}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-single_720_prefix_risk r3_prefix_risk dynamic_residual_stability_routing scc_condition_delta_detached scc_condition_delta_state_open}"
DATASETS="${DATASETS:-ETTh2 Weather}"
GPU_IDS="${GPU_IDS:-1 2}"
EPOCHS="${EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
SUPERVISION_CONDITION_TOP_RATIO="${SUPERVISION_CONDITION_TOP_RATIO:-0.25}"
SUPERVISION_AUX_WEIGHT="${SUPERVISION_AUX_WEIGHT:-0.1}"
SUPERVISION_RESIDUAL_PERIODS="${SUPERVISION_RESIDUAL_PERIODS:-24,48,96,168}"

export OUTPUT_ROOT
export LOG_ROOT
export SUPERVISION_STRATEGIES
export DATASETS
export GPU_IDS
export EPOCHS
export LEARNING_RATE

echo "phase4_scc_condition_carrier_gate_start=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "learning_rate=${LEARNING_RATE}"
echo "supervision_strategies=${SUPERVISION_STRATEGIES}"
echo "datasets=${DATASETS}"
echo "gpu_ids=${GPU_IDS}"
echo "supervision_condition_top_ratio=${SUPERVISION_CONDITION_TOP_RATIO}"
echo "supervision_aux_weight=${SUPERVISION_AUX_WEIGHT}"
echo "supervision_residual_periods=${SUPERVISION_RESIDUAL_PERIODS}"

EXTRA_ARGS=(
  --learning-rate "${LEARNING_RATE}"
  --supervision-condition-top-ratio "${SUPERVISION_CONDITION_TOP_RATIO}"
  --supervision-aux-weight "${SUPERVISION_AUX_WEIGHT}"
  --supervision-residual-periods "${SUPERVISION_RESIDUAL_PERIODS}"
)

PHASE4_EXTRA_ARGS="${EXTRA_ARGS[*]}" bash scripts/remote/run_phase4_horizon_decoupled_gate.sh

echo "phase4_scc_condition_carrier_gate_done=$(date -Is)"
