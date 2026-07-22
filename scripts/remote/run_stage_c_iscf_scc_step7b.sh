#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_v0_step7b}"
export CONFIG="${CONFIG:-configs/stage_c_iscf_scc_step7b.json}"
export PROTOCOL_PROFILE="${PROTOCOL_PROFILE:-stage_c_iscf_scc_v0_step7b}"
export RUN_LABEL="${RUN_LABEL:-SCC}"

exec bash scripts/remote/run_stage_c_iscf_sps_step7b.sh
