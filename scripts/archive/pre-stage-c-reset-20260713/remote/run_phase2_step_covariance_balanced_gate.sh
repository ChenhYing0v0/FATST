#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_step_covariance_balanced_objective}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_step_covariance_balanced_gate}"
export RUN_NAME="${RUN_NAME:-PatchEncoderStepCovarianceBalanced}"
export STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-step_covariance_balanced}"
export STEP_COVARIANCE_BETA="${STEP_COVARIANCE_BETA:-0.5}"
export STEP_COVARIANCE_ETA="${STEP_COVARIANCE_ETA:-0.5}"
export STEP_COVARIANCE_EPS="${STEP_COVARIANCE_EPS:-1e-6}"

exec bash scripts/remote/run_phase2_region_balanced_gate.sh
