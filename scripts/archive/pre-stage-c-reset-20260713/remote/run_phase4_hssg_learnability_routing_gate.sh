#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_hssg_learnability_routing_gate}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_hssg_learnability_routing_gate}"
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES:-full_time_mse single_720_prefix_risk r3_prefix_risk hssg_region_routed_readout hssg_learnability_region_routing}"

echo "phase4_hssg_learnability_routing_gate_start=$(date -Is)"
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
SUPERVISION_STRATEGIES="${SUPERVISION_STRATEGIES}" \
bash scripts/remote/run_phase4_hssg_gradient_routing_gate.sh
echo "phase4_hssg_learnability_routing_gate_done=$(date -Is)"
