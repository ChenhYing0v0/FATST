#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_fsa_f2_anchor_pressure_gate}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_fsa_f2_anchor_pressure_gate}"

bash scripts/remote/check_phase4_future_state_anchor_progress.sh
