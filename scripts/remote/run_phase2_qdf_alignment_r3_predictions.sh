#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_alignment_r3_predictions}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_qdf_alignment_r3_predictions}"
export RUN_NAME="${RUN_NAME:-PatchEncoderPrefixRiskWeighted}"
export STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-prefix_risk}"
export STEP_LOSS_ALPHA="${STEP_LOSS_ALPHA:-0.5}"
export TARGET_HORIZONS="${TARGET_HORIZONS:-96,192,336,720}"
export EPOCHS="${EPOCHS:-100}"
export SEED="${SEED:-2021}"
export GPU_IDS="${GPU_IDS:-1 2}"
export KEEP_HEAVY_ARTIFACTS="${KEEP_HEAVY_ARTIFACTS:-1}"
export SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

exec bash scripts/remote/run_phase1_target_set_decoder_gate.sh
