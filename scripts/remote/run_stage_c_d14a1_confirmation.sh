#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yingch/exp_outputs/r-2026-fatst/stage_c_d14a1_dual_carrier_grouped_mlp}"
DESIGN="${DESIGN:-configs/stage_c_d14a1_dual_carrier_grouped_mlp.json}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-moe}"
GPU_IDS="${GPU_IDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
STATUS_ONLY="${STATUS_ONLY:-0}"
SEEDS=(2022 2023)

if [[ "${STATUS_ONLY}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    for carrier in neutral_raw a6_natural; do
      CARRIER="${carrier}" SEED="${seed}" STATUS_ONLY=1 GPU_IDS="${GPU_IDS}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" DESIGN="${DESIGN}" \
        bash scripts/remote/run_stage_c_d14a1_dual_carrier_grouped_mlp.sh
    done
  done
  test -s "${OUTPUT_ROOT}/_analysis_multiseed/gate.json" \
    && cat "${OUTPUT_ROOT}/_analysis_multiseed/gate.json"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_d14a1_multiseed.py --synthetic-smoke
  echo "stage_c_d14a1_confirmation_dry_run=pass seeds=${SEEDS[*]} test=false"
  exit 0
fi

python3 -c '
import json,sys
root=sys.argv[1]
expected={
    "_analysis_neutral_raw_seed2021/gate.json": "neutral_problem_pass_authorize_a6_sensitivity",
    "_analysis_a6_natural_seed2021/gate.json": "a6_sensitivity_confirming",
}
for relative,decision in expected.items():
    gate=json.load(open(f"{root}/{relative}"))
    if gate.get("decision") != decision:
        raise SystemExit(f"confirmation held: {relative} decision={gate.get('decision')!r}")
' "${OUTPUT_ROOT}"

{
  echo "stage_c_d14a1_confirmation_start=$(date -Is)"
  echo "commit=$(git rev-parse HEAD)"
  echo "gpu_ids=${GPU_IDS}"
  echo "seeds=${SEEDS[*]}"
  echo "execution=neutral_then_a6_per_seed"
  echo "a6_lbf_performance_is_problem_gate=false"
  echo "d14b_automatic=false"
  echo "test_used=false"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "${OUTPUT_ROOT}/launch_confirmation.txt"

for carrier in neutral_raw a6_natural; do
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/analyze_stage_c_d14a1.py \
      --raw-root "${OUTPUT_ROOT}" --design "${DESIGN}" \
      --output-dir "${OUTPUT_ROOT}/_analysis_${carrier}_seed2021" \
      --carrier "${carrier}" --seed 2021
done

for seed in "${SEEDS[@]}"; do
  CARRIER=neutral_raw SEED="${seed}" GPU_IDS="${GPU_IDS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" DESIGN="${DESIGN}" \
    bash scripts/remote/run_stage_c_d14a1_dual_carrier_grouped_mlp.sh
  CARRIER=a6_natural SEED="${seed}" GPU_IDS="${GPU_IDS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" DESIGN="${DESIGN}" \
    bash scripts/remote/run_stage_c_d14a1_dual_carrier_grouped_mlp.sh
done

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python scripts/analyze_stage_c_d14a1_multiseed.py \
    --raw-root "${OUTPUT_ROOT}" --design "${DESIGN}" \
    --output-dir "${OUTPUT_ROOT}/_analysis_multiseed"
echo "stage_c_d14a1_confirmation_done=$(date -Is)"
