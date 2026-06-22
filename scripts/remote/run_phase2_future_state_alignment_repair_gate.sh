#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase2_future_state_alignment_repair}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase2_future_state_alignment_repair_gate}"
export RUN_NAME="${RUN_NAME:-PatchEncoderFutureStateAlignmentConfWeighted}"
export DATASETS="${DATASETS:-ETTm1 Weather ETTh2}"
export STEP_LOSS_WEIGHTING="${STEP_LOSS_WEIGHTING:-prefix_risk}"
export STEP_LOSS_ALPHA="${STEP_LOSS_ALPHA:-0.5}"
export FUTURE_TEACHER_LAYERS="${FUTURE_TEACHER_LAYERS:-1}"
export FUTURE_TEACHER_HEADS="${FUTURE_TEACHER_HEADS:-8}"
export FUTURE_TEACHER_D_FF="${FUTURE_TEACHER_D_FF:-256}"
export FUTURE_ALIGN_WEIGHT="${FUTURE_ALIGN_WEIGHT:-0.02}"
export FUTURE_RELATION_WEIGHT="${FUTURE_RELATION_WEIGHT:-0.01}"
export FUTURE_RECON_WEIGHT="${FUTURE_RECON_WEIGHT:-0.001}"
export FUTURE_RECON_NORMALIZATION="${FUTURE_RECON_NORMALIZATION:-target_energy}"
export FUTURE_ALIGN_WEIGHTING="${FUTURE_ALIGN_WEIGHTING:-reconstruction_confidence}"
export FUTURE_CONFIDENCE_TEMPERATURE="${FUTURE_CONFIDENCE_TEMPERATURE:-1.0}"
export FUTURE_CONFIDENCE_FLOOR="${FUTURE_CONFIDENCE_FLOOR:-0.05}"
export FUTURE_RECON_EPS="${FUTURE_RECON_EPS:-1e-6}"

exec bash scripts/remote/run_phase2_future_state_alignment_gate.sh
