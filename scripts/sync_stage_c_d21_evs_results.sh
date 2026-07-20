#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-529_Lab-3090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d21_evidence_validity_surface}"
LOCAL_ROOT="${LOCAL_ROOT:-analysis/stage_c_d21_evidence_validity_surface_20260720/raw}"

mkdir -p "${LOCAL_ROOT}"
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='*.npz' \
  --include='*.json' \
  --include='*.log' \
  --include='*.txt' \
  --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"
echo "d21_evs_sync=pass local_root=${LOCAL_ROOT}"
