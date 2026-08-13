#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_etth1_h5d.json}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5d}" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo.sh
