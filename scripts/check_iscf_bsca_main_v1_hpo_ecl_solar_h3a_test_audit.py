#!/usr/bin/env python3
"""Check the frozen nine-checkpoint ECL/Solar H3A test audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "iscf_bsca_main_v1_hpo_ecl_solar_h3a_test_audit.json"
)
MANIFEST = (
    ROOT
    / "analysis"
    / "iscf_bsca_main_v1_hpo_20260731"
    / "h3a_checkpoint_manifest.csv"
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
    assert config["status"] == "authorized_prelaunch"
    assert config["test_tuned"] is True
    assert config["test_informed"] is True
    assert config["matrix"]["expected_runs"] == 9
    assert config["matrix"]["explicit_manifest_rows"] == 9
    assert config["matrix"]["profiles_per_dataset"] == {
        "ECL": 1,
        "Solar": 8,
    }
    assert sha256(MANIFEST) == config["checkpoint_manifest"]["sha256"]
    assert len(rows) == 9
    assert len({row["trial_id"] for row in rows}) == 9
    assert Counter(row["dataset"] for row in rows) == Counter(
        {"ECL": 1, "Solar": 8}
    )
    assert all(row["phase"] == "H3A" for row in rows)
    assert all(row["seed"] == "2021" for row in rows)
    assert all(len(row["checkpoint_sha256_before_test"]) == 64 for row in rows)
    authorization = config["authorization"]
    assert authorization["user_authorized"] is True
    assert authorization["checkpoint_mutation_during_test_allowed"] is False
    assert authorization[
        "per_dataset_aggregate_hyperparameter_tuning_allowed"
    ] is True
    assert authorization[
        "per_horizon_seed_metric_or_cell_tuning_allowed"
    ] is False
    result = subprocess.run(
        [
            "bash",
            "scripts/remote/"
            "run_iscf_bsca_main_v1_hpo_ecl_solar_h3a_test_audit.sh",
        ],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_test_audit_dry_run=pass" in result
    assert "jobs=9" in result
    assert "test_cells=36" in result
    assert "authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "training_checkpoints": 9,
                "standard_horizon_test_cells": 36,
                "profiles_per_dataset": config["matrix"][
                    "profiles_per_dataset"
                ],
                "manifest_sha256": sha256(MANIFEST),
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
