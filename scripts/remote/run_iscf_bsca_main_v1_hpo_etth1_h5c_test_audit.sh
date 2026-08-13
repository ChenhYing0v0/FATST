#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_etth1_h5c_test_audit.json}" \
MANIFEST="${MANIFEST:-analysis/iscf_bsca_main_v1_hpo_20260731/h5c_checkpoint_manifest.csv}" \
TEST_ROOT="${TEST_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5c/test_audit}" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh
