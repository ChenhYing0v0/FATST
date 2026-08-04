#!/usr/bin/env bash
set -euo pipefail

HPO_ROOT="/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo"
MODE="${MODE:-audit}"
CONFIRM_CLEANUP="${CONFIRM_CLEANUP:-}"
EXPECTED_CONFIRMATION="delete_nonselected_hpo_smokes_and_diagnostics_20260804"

RESOURCE_SMOKE_DIRS=(
  "${HPO_ROOT}/h1/_resource_smoke"
  "${HPO_ROOT}/h2/_resource_smoke"
  "${HPO_ROOT}/ecl_solar_h3a/_resource_smoke"
  "${HPO_ROOT}/solar_h3b/_resource_smoke"
  "${HPO_ROOT}/h4j/_resource_smoke"
  "${HPO_ROOT}/h4k/_resource_smoke"
  "${HPO_ROOT}/h4l/_resource_smoke"
)

SELECTED_TRIALS=(
  "ECL__h2_intermediate_capacity"
  "ETTh1__h4j_lr3e4"
  "ETTh2__h2_lr5e4"
  "ETTm1__h2_table5_capacity"
  "ETTm2__h4l_wd1e3"
  "Solar__h4j_patch4_lr3e4"
  "Weather__h4k_current_lr5e5_patch24_dropout0"
  "Exchange__h2_lookback336"
)

[[ "${MODE}" == "audit" || "${MODE}" == "apply" ]] || {
  echo "MODE must be audit or apply" >&2
  exit 2
}
[[ -d "${HPO_ROOT}" ]] || {
  echo "missing HPO root: ${HPO_ROOT}" >&2
  exit 3
}
[[ "$(realpath "${HPO_ROOT}")" == "${HPO_ROOT}" ]] || {
  echo "refusing noncanonical HPO root: ${HPO_ROOT}" >&2
  exit 4
}

is_selected_diagnostic() {
  local path="$1" trial
  for trial in "${SELECTED_TRIALS[@]}"; do
    if [[ "${path}" == *"/${trial}/seed2021/pcsd_test_audit_diagnostics.npz" ]]; then
      return 0
    fi
  done
  return 1
}

resource_count=0
resource_bytes=0
for directory in "${RESOURCE_SMOKE_DIRS[@]}"; do
  if [[ -d "${directory}" ]]; then
    bytes="$(du -sb "${directory}" | cut -f1)"
    resource_count=$((resource_count + 1))
    resource_bytes=$((resource_bytes + bytes))
    printf 'resource_smoke\t%s\t%s\n' "${bytes}" "${directory}"
  fi
done

diagnostic_count=0
diagnostic_bytes=0
selected_count=0
selected_bytes=0
delete_count=0
delete_bytes=0
declare -a DELETE_DIAGNOSTICS=()
while IFS= read -r -d '' path; do
  bytes="$(stat -c %s "${path}")"
  diagnostic_count=$((diagnostic_count + 1))
  diagnostic_bytes=$((diagnostic_bytes + bytes))
  if is_selected_diagnostic "${path}"; then
    selected_count=$((selected_count + 1))
    selected_bytes=$((selected_bytes + bytes))
    printf 'selected_diagnostic_keep\t%s\t%s\n' "${bytes}" "${path}"
  else
    delete_count=$((delete_count + 1))
    delete_bytes=$((delete_bytes + bytes))
    DELETE_DIAGNOSTICS+=("${path}")
  fi
done < <(
  find "${HPO_ROOT}" -xdev -type f \
    -name 'pcsd_test_audit_diagnostics.npz' -print0
)

metrics_count="$(find "${HPO_ROOT}" -xdev -type f -name 'test_audit_metrics_by_target_horizon.csv' | wc -l | tr -d ' ')"
invariants_count="$(find "${HPO_ROOT}" -xdev -type f -name 'test_audit_invariants.json' | wc -l | tr -d ' ')"

echo "cleanup_audit mode=${MODE} resource_dirs=${resource_count} resource_bytes=${resource_bytes} diagnostics=${diagnostic_count} diagnostic_bytes=${diagnostic_bytes} selected_keep=${selected_count} selected_bytes=${selected_bytes} delete_diagnostics=${delete_count} delete_bytes=${delete_bytes} metrics=${metrics_count} invariants=${invariants_count}"

if [[ "${MODE}" == "audit" ]]; then
  exit 0
fi

[[ "${CONFIRM_CLEANUP}" == "${EXPECTED_CONFIRMATION}" ]] || {
  echo "apply mode requires CONFIRM_CLEANUP=${EXPECTED_CONFIRMATION}" >&2
  exit 5
}
[[ "${diagnostic_count}" -eq 165 ]] || {
  echo "expected 165 diagnostic files before first apply; found ${diagnostic_count}" >&2
  exit 6
}
[[ "${selected_count}" -eq 8 ]] || {
  echo "expected 8 selected diagnostics; found ${selected_count}" >&2
  exit 7
}
[[ "${metrics_count}" -eq 165 && "${invariants_count}" -eq 165 ]] || {
  echo "expected 165 retained metrics and invariants before cleanup" >&2
  exit 8
}

for directory in "${RESOURCE_SMOKE_DIRS[@]}"; do
  if [[ -d "${directory}" ]]; then
    rm -rf -- "${directory}"
  fi
done
for path in "${DELETE_DIAGNOSTICS[@]}"; do
  rm -f -- "${path}"
done

post_diagnostics="$(find "${HPO_ROOT}" -xdev -type f -name 'pcsd_test_audit_diagnostics.npz' | wc -l | tr -d ' ')"
post_metrics="$(find "${HPO_ROOT}" -xdev -type f -name 'test_audit_metrics_by_target_horizon.csv' | wc -l | tr -d ' ')"
post_invariants="$(find "${HPO_ROOT}" -xdev -type f -name 'test_audit_invariants.json' | wc -l | tr -d ' ')"
post_resource_dirs=0
for directory in "${RESOURCE_SMOKE_DIRS[@]}"; do
  [[ ! -e "${directory}" ]] || post_resource_dirs=$((post_resource_dirs + 1))
done

[[ "${post_diagnostics}" -eq 8 ]] || {
  echo "expected 8 selected diagnostics after cleanup; found ${post_diagnostics}" >&2
  exit 9
}
[[ "${post_metrics}" -eq 165 && "${post_invariants}" -eq 165 ]] || {
  echo "retained metric/invariant count changed unexpectedly" >&2
  exit 10
}
[[ "${post_resource_dirs}" -eq 0 ]] || {
  echo "resource-smoke directories remain after cleanup" >&2
  exit 11
}

echo "cleanup_apply=pass removed_resource_dirs=${resource_count} removed_diagnostics=${delete_count} retained_selected_diagnostics=${post_diagnostics} retained_metrics=${post_metrics} retained_invariants=${post_invariants} estimated_bytes_removed=$((resource_bytes + delete_bytes))"
