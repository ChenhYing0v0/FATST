#!/usr/bin/env python3
"""Check the authorized frozen Weather H4N formal-test contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_weather_h4n_test_audit.json"
MANIFEST = (
    ROOT
    / "analysis"
    / "iscf_bsca_main_v1_hpo_20260731"
    / "h4n_checkpoint_manifest.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = config["matrix"]["profiles_per_dataset"]

    assert config["status"] == "authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["matrix"]["expected_runs"] == 40
    assert config["matrix"]["expected_standard_horizon_cells"] == 160
    assert sha256(MANIFEST) == config["checkpoint_manifest"]["sha256"]
    assert len(rows) == 40
    assert len({row["trial_id"] for row in rows}) == 40
    assert len({row["checkpoint_sha256_before_test"] for row in rows}) == 40
    assert Counter(row["dataset"] for row in rows) == Counter(expected)
    assert all(row["phase"] == "H4N" for row in rows)
    assert all(row["seed"] == "2021" for row in rows)

    authorization = config["authorization"]
    assert authorization["user_authorized"] is True
    assert authorization["authorization_date"] == "2026-08-06"
    assert authorization["formal_test_access_count_for_version"] == 1
    assert authorization["checkpoint_retraining_allowed"] is False
    assert authorization["checkpoint_mutation_during_test_allowed"] is False
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    assert authorization["automatic_H4O_extension_authorized"] is False
    assert authorization["additional_seed_authorized"] is False

    selection = config["hyperparameter_selection"]
    assert selection["all_229_trials_in_final_selector"] is True
    assert selection["all_96_weather_trials_in_final_selector"] is True
    assert selection["near_tie_band_pct"] == 0.1
    assert selection["per_horizon_profile_selection"] is False
    assert selection["metric_specific_selection"] is False
    assert selection["weather_combined_lead_gate"] == "at_least_6_of_8"
    assert selection["global_combined_lead_gate"] == "at_least_40_of_56"

    result = subprocess.run(
        [
            "bash",
            "scripts/remote/run_iscf_bsca_main_v1_hpo_weather_h4n_test_audit.sh",
        ],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_test_audit_dry_run=pass" in result
    assert "jobs=40" in result and "test_cells=160" in result
    assert "authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "training_checkpoints": 40,
                "standard_horizon_test_cells": 160,
                "profiles_per_dataset": expected,
                "manifest_sha256": sha256(MANIFEST),
                "formal_test_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
