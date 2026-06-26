#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/phase4_fsa_f2_anchor_pressure_gate}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs/phase4_fsa_f2_anchor_pressure_gate}"
export ARMS="${ARMS:-F2-A0 F2-A1}"
export DATASETS="${DATASETS:-Weather ETTh2}"
export GPU_IDS="${GPU_IDS:-1 2}"
export JOB_ORDER="${JOB_ORDER:-dataset_major}"
export FUTURE_CONFIDENCE_FLOOR="${FUTURE_CONFIDENCE_FLOOR:-0.0}"
export FUTURE_ALIGN_WEIGHT="${FUTURE_ALIGN_WEIGHT:-0.01}"
export FUTURE_RELATION_WEIGHT="${FUTURE_RELATION_WEIGHT:-0.0}"
export FUTURE_RECON_WEIGHT="${FUTURE_RECON_WEIGHT:-0.001}"

echo "phase4_fsa_f2_anchor_pressure_gate_start=$(date -Is)"
echo "output_root=${OUTPUT_ROOT}"
echo "arms=${ARMS}"
echo "datasets=${DATASETS}"
echo "gpu_ids=${GPU_IDS}"
echo "job_order=${JOB_ORDER}"
echo "future_confidence_floor=${FUTURE_CONFIDENCE_FLOOR}"
echo "future_align_weight=${FUTURE_ALIGN_WEIGHT}"
echo "future_relation_weight=${FUTURE_RELATION_WEIGHT}"
echo "future_recon_weight=${FUTURE_RECON_WEIGHT}"

bash scripts/remote/run_phase4_future_state_anchor_gate.sh

echo "phase4_fsa_f2_anchor_pressure_gate_done=$(date -Is)"
