#!/usr/bin/env bash
set -euo pipefail

export REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective}"
export RUN_NAME="${RUN_NAME:-PatchEncoderOffdiagBlockQuadratic}"
export ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis/phase2_offdiag_block_quadratic_gate_20260623}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-phase2_offdiag_block_quadratic}"

exec bash scripts/sync_phase2_region_balanced_results.sh
