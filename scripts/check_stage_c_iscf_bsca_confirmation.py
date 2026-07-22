#!/usr/bin/env python3
"""Check the frozen ISCF-BSCA-v1 confirmation contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_stage_c_pcsd_cf_checkpoint import test_audit_authorized  # noqa: E402


def main() -> None:
    config_path = ROOT / "configs/stage_c_iscf_bsca_v1_confirmation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["seeds"] == [2022, 2023]
    assert config["all_effective_seeds"] == [2021, 2022, 2023]
    assert config["matrix"]["new_training_runs"] == 10
    assert config["matrix"]["expected_runs"] == 20
    assert config["matrix"]["three_seed_test_cells"] == 60
    assert config["training"]["post_seed2021_tuning"] is False
    assert config["authorization"]["confirmation_seeds_authorized"] is True
    assert config["authorization"]["formal_test_access_count_for_version"] == 1
    assert test_audit_authorized(config)

    local_reference = (
        ROOT
        / "analysis/stage_c_post_d21_unconstrained_reset_20260720"
        / "siff_v2_fcc_v1/raw_lite/siff_independent_equal"
    )
    reference_count = 0
    for seed in config["seeds"]:
        for dataset in config["datasets"]:
            directory = local_reference / dataset / "h720_full" / f"seed{seed}"
            for name in (
                "metrics_by_target_horizon.csv",
                "initialization_contract.json",
                "test_audit_metrics_by_target_horizon.csv",
                "test_audit_invariants.json",
            ):
                assert (directory / name).is_file(), directory / name
            invariant = json.loads(
                (directory / "test_audit_invariants.json").read_text(
                    encoding="utf-8"
                )
            )
            assert invariant["pass"] is True
            reference_count += 1
    assert reference_count == 10

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": str(config_path),
            "OUTPUT_ROOT": "/tmp/fatst_bsca_confirmation_contract",
            "DRY_RUN": "1",
        }
    )
    dry = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_iscf_bsca_v1_confirmation.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "bsca_confirmation_dry_run=pass jobs=10" in dry.stdout

    guard_env = env.copy()
    guard_env["DRY_RUN"] = "0"
    guard_env["FORMAL_TEST_ONLY"] = "1"
    guard = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_iscf_bsca_v1_confirmation.sh"],
        cwd=ROOT,
        env=guard_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert guard.returncode == 4, guard.stderr
    assert "complete training: 0/10" in guard.stderr

    print(
        json.dumps(
            {
                "candidate": config["candidate_version"],
                "new_training_runs": 10,
                "reused_equal_references": reference_count,
                "three_seed_test_cells": 60,
                "test_authorization_contract": "pass",
                "dry_run": "pass",
                "ten_of_ten_formal_guard": "pass",
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
