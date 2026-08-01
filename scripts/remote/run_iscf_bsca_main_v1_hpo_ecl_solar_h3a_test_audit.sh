#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a_test_audit.json}" \
MANIFEST="${MANIFEST:-analysis/iscf_bsca_main_v1_hpo_20260731/h3a_checkpoint_manifest.csv}" \
TEST_ROOT="${TEST_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/ecl_solar_h3a/test_audit}" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh
