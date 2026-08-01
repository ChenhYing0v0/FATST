#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a.json}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/ecl_solar_h3a}" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo.sh
