#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase3_regime_segment_operator}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase3_regime_segment_operator_gate}"
export DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
export TARGET_HORIZONS="${TARGET_HORIZONS:-96,720}"
export RUN_NAME="${RUN_NAME:-PatchEncoderRegimeSegmentTargetOperator}"
export MODEL_VARIANT="${MODEL_VARIANT:-regime_segment_operator}"
export USE_WINDOW_POSITION="${USE_WINDOW_POSITION:-1}"
export STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-prefix_risk}"
export STEP_LOSS_ALPHA="${STEP_LOSS_ALPHA:-0.5}"
export REGIME_HIDDEN_DIM="${REGIME_HIDDEN_DIM:-64}"
export REGIME_DROPOUT="${REGIME_DROPOUT:-0.0}"

exec bash scripts/remote/run_phase1_target_set_decoder_gate.sh
