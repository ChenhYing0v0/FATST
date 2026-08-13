#!/usr/bin/env python3
"""Check a frozen H5B/H5C formal-test contract before test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "iscf_bsca_main_v1_hpo_etth1_h5b_test_audit.json"
)
DEFAULT_MANIFEST = (
    ROOT
    / "analysis"
    / "iscf_bsca_main_v1_hpo_20260731"
    / "h5b_checkpoint_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--runner",
        default="scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5b_test_audit.sh",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_runs = config["matrix"]["expected_runs"]
    expected_cells = config["matrix"]["expected_standard_horizon_cells"]
    phase = config["protocol_id"].split("-ETTH1-")[1].split("-")[0]

    assert config["status"] == "authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert expected_cells == expected_runs * 4
    assert sha256(args.manifest) == config["checkpoint_manifest"]["sha256"]
    assert len(rows) == expected_runs
    assert len({row["trial_id"] for row in rows}) == expected_runs
    assert len({row["checkpoint_sha256_before_test"] for row in rows}) == expected_runs
    assert {row["dataset"] for row in rows} == {"ETTh1"}
    assert all(row["phase"] == phase and row["seed"] == "2021" for row in rows)

    authorization = config["authorization"]
    assert authorization["user_authorized"] is True
    assert authorization["formal_test_access_count_for_version"] == 1
    assert authorization["checkpoint_retraining_allowed"] is False
    assert authorization["checkpoint_mutation_during_test_allowed"] is False
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    extension_key = (
        "automatic_H5D_extension_authorized"
        if phase == "H5C"
        else "automatic_H5C_extension_authorized"
    )
    assert authorization[extension_key] is False
    assert authorization["additional_seed_authorized"] is False
    assert authorization["automatic_Main_I_or_Main_II_table_mutation_authorized"] is False

    selection = config["hyperparameter_selection"]
    guard_suffix = "h5b" if phase == "H5C" else "h5a"
    mse_guard = f"eligibility_guard_mean_mse_relative_to_{guard_suffix}_max"
    mae_guard = f"eligibility_guard_mean_mae_relative_to_{guard_suffix}_max"
    assert selection[mse_guard] > 1.0
    assert selection[mae_guard] > 1.0
    assert (
        selection["minimum_Main_II_best_cells"]
        > selection["current_Main_II_best_cells"]
    )
    assert selection["per_horizon_profile_selection"] is False
    assert selection["metric_specific_selection"] is False
    assert selection["cell_specific_selection"] is False
    assert selection["all_trials_retained"] is True

    surfaces = config["frozen_comparison_surfaces"]
    for key in ("main_i", "main_ii"):
        path = ROOT / surfaces[f"{key}_table_data"]
        assert sha256(path) == surfaces[f"{key}_table_data_sha256"]

    result = subprocess.run(
        ["bash", args.runner],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_test_audit_dry_run=pass" in result
    assert f"jobs={expected_runs}" in result and f"test_cells={expected_cells}" in result
    assert "authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "training_checkpoints": expected_runs,
                "standard_horizon_test_cells": expected_cells,
                "profiles_per_dataset": {"ETTh1": expected_runs},
                "manifest_sha256": sha256(args.manifest),
                "formal_test_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
