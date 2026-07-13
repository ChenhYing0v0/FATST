#!/usr/bin/env bash
set -euo pipefail

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_protocol_calibration_gate}"
LEARNING_RATES="${LEARNING_RATES:-0.0001 0.00005 0.00003}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-single_720_prefix_risk r3_prefix_risk hssg_region_routed_readout}"

echo "phase4_protocol_calibration_gate_start=$(date -Is)"
echo "base_output_root=${BASE_OUTPUT_ROOT}"
echo "learning_rates=${LEARNING_RATES}"
echo "supervision_strategies=${SUPERVISION_STRATEGIES}"

for learning_rate in ${LEARNING_RATES}; do
  lr_label="${learning_rate//./p}"
  lr_label="${lr_label//-/m}"
  lr_output_root="${BASE_OUTPUT_ROOT}/lr_${lr_label}"
  lr_log_root="${lr_output_root}/_logs/phase4_protocol_calibration_gate"
  echo "phase4_protocol_calibration_lr_start=$(date -Is) learning_rate=${learning_rate} output_root=${lr_output_root}"
  OUTPUT_ROOT="${lr_output_root}" \
  LOG_ROOT="${lr_log_root}" \
  LEARNING_RATE="${learning_rate}" \
  SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES}" \
  bash scripts/remote/run_phase4_hssg_gradient_routing_gate.sh
  echo "phase4_protocol_calibration_lr_done=$(date -Is) learning_rate=${learning_rate} output_root=${lr_output_root}"
done

echo "phase4_protocol_calibration_gate_done=$(date -Is)"
