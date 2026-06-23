#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_offdiag_block_quadratic_objective}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_offdiag_block_quadratic_gate}"
export RUN_NAME="${RUN_NAME:-PatchEncoderOffdiagBlockQuadratic}"
export STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-offdiag_block_quadratic}"
export STEP_LOSS_ALPHA="${STEP_LOSS_ALPHA:-0.5}"
export OFFDIAG_BLOCK_SIZE="${OFFDIAG_BLOCK_SIZE:-48}"
export OFFDIAG_QUADRATIC_WEIGHT="${OFFDIAG_QUADRATIC_WEIGHT:-0.05}"
export OFFDIAG_RIDGE_EPS="${OFFDIAG_RIDGE_EPS:-1e-3}"

EXTRA_ARGS=(
  --offdiag-block-size "${OFFDIAG_BLOCK_SIZE}"
  --offdiag-quadratic-weight "${OFFDIAG_QUADRATIC_WEIGHT}"
  --offdiag-ridge-eps "${OFFDIAG_RIDGE_EPS}"
)

export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-${EXTRA_ARGS[*]}}"

exec bash scripts/remote/run_phase2_region_balanced_gate.sh
