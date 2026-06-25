#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_late_conflict_adapter_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_late_conflict_adapter_gate}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-late_conflict_adapter_routing full_time_mse r3_prefix_risk}"
DATASETS="${DATASETS:-ETTh2 Weather}"
GPU_IDS="${GPU_IDS:-1 2}"
EPOCHS="${EPOCHS:-100}"
SUPERVISION_AUX_WEIGHT="${SUPERVISION_AUX_WEIGHT:-0.1}"
SUPERVISION_ADAPTER_START_STEP="${SUPERVISION_ADAPTER_START_STEP:-337}"

export OUTPUT_ROOT
export LOG_ROOT
export SUPERVISION_STRATEGIES
export DATASETS
export GPU_IDS
export EPOCHS

echo "phase4_late_conflict_adapter_gate_start=$(date -Is)"
echo "supervision_aux_weight=${SUPERVISION_AUX_WEIGHT}"
echo "supervision_adapter_start_step=${SUPERVISION_ADAPTER_START_STEP}"

EXTRA_ARGS=(
  --supervision-aux-weight "${SUPERVISION_AUX_WEIGHT}"
  --supervision-adapter-start-step "${SUPERVISION_ADAPTER_START_STEP}"
)

PHASE4_EXTRA_ARGS="${EXTRA_ARGS[*]}" bash scripts/remote/run_phase4_horizon_decoupled_gate.sh

echo "phase4_late_conflict_adapter_gate_done=$(date -Is)"
