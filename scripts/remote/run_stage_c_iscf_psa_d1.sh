#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_psa_d1}"
export CONFIG="${CONFIG:-configs/stage_c_iscf_psa_d1.json}"
export PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-stage_c_iscf_psa_d1_control_v0}"
export RUN_LABEL="${RUN_LABEL:-PSA_D1}"
export RESOURCE_SMOKE_JOBS="${RESOURCE_SMOKE_JOBS:-1}"

exec bash scripts/remote/run_stage_c_iscf_sps_step7b.sh
