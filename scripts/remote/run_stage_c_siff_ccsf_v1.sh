#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/stage_c_siff_ccsf_step7a.json}"
DRY_RUN="${DRY_RUN:-0}"

test -s "${CONFIG}"

REMOTE_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["remote_training_authorized"]).lower())' "${CONFIG}")"
TEST_AUTHORIZED="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["authorization"]["formal_test_access_authorized"]).lower())' "${CONFIG}")"

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 -c '
import json,sys
config=json.load(open(sys.argv[1]))
for dataset in config["datasets"]:
    for arm in config["arms"]:
        print("\t".join((dataset,arm["id"],arm["readout_mode"],arm["objective_mode"])))
' "${CONFIG}"
  echo "ccsf_step7a_dry_run=pass remote_authorized=${REMOTE_AUTHORIZED} test_authorized=${TEST_AUTHORIZED}"
  exit 0
fi

if [[ "${REMOTE_AUTHORIZED}" != "true" || "${TEST_AUTHORIZED}" != "true" ]]; then
  echo "remote/test launch is not authorized by ${CONFIG}; complete Step7B prelaunch first" >&2
  exit 3
fi

echo "Step7B production runner is not frozen; refusing launch" >&2
exit 4
