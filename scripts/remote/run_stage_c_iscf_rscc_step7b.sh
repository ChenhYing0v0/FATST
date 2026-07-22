#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_rscc_v1_step7b}"
export CONFIG="${CONFIG:-configs/stage_c_iscf_rscc_step7b.json}"
export PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-stage_c_iscf_rscc_v1_step7b}"
export RUN_LABEL="${RUN_LABEL:-RSCC}"

exec bash scripts/remote/run_stage_c_iscf_sps_step7b.sh
