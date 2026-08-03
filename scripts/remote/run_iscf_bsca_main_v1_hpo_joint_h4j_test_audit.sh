#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/iscf_bsca_main_v1_hpo_joint_h4j_test_audit.json}" \
MANIFEST="${MANIFEST:-analysis/iscf_bsca_main_v1_hpo_20260731/h4j_checkpoint_manifest.csv}" \
TEST_ROOT="${TEST_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4j_r1/test_audit_r2}" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh
