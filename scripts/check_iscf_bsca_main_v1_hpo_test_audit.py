#!/usr/bin/env python3
"""Check the frozen 40-checkpoint ISCF-BSCA HPO test-audit contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_test_audit.json"
MANIFEST = (
    ROOT
    / "analysis"
    / "iscf_bsca_main_v1_hpo_20260731"
    / "combined_checkpoint_manifest.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail_closed_check() -> None:
    project_python = Path("/opt/anaconda3/envs/r2026-fsa/bin/python")
    python_executable = os.environ.get(
        "R2026_FSA_PYTHON",
        str(project_python if project_python.is_file() else Path(sys.executable)),
    )
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        invalid_config = temp_dir / "invalid_test_audit.json"
        artifact_dir = temp_dir / "must_not_be_created"
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["authorization"]["user_authorized"] = False
        invalid_config.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                python_executable,
                "scripts/evaluate_stage_c_pcsd_cf_checkpoint.py",
                "--run-dir",
                str(temp_dir / "nonexistent_run"),
                "--artifact-dir",
                str(artifact_dir),
                "--design",
                str(temp_dir / "nonexistent_design.json"),
                "--device",
                "cpu",
                "--evaluation-split",
                "test",
                "--test-audit-config",
                str(invalid_config),
                "--probe-rows",
                "1",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "authorization failed before test loader access" in output
        assert not artifact_dir.exists()
        assert "nonexistent_design" not in output


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert config["status"] == "authorized_prelaunch"
    assert config["test_tuned"] is True
    assert config["matrix"]["expected_runs"] == 40
    assert config["matrix"]["partial_test_execution_allowed"] is False
    assert config["matrix"]["partial_profile_selection_allowed"] is False
    assert sha256(MANIFEST) == config["checkpoint_manifest"]["sha256"]
    assert len(rows) == 40
    assert len({row["trial_id"] for row in rows}) == 40
    assert Counter(row["phase"] for row in rows) == Counter(
        {"H1": 16, "H2": 24}
    )
    assert Counter(row["dataset"] for row in rows) == Counter(
        {dataset: 5 for dataset in config["datasets"]}
    )
    for row in rows:
        assert row["seed"] == "2021"
        assert len(row["checkpoint_sha256_before_test"]) == 64
        assert row["artifact_dir"] != row["test_artifact_dir"]
        assert "/test_audit/" in row["test_artifact_dir"]
    authorization = config["authorization"]
    assert authorization["user_authorized"] is True
    assert authorization["checkpoint_mutation_during_test_allowed"] is False
    assert authorization[
        "per_dataset_aggregate_hyperparameter_tuning_allowed"
    ] is True
    assert authorization[
        "per_horizon_seed_metric_or_cell_tuning_allowed"
    ] is False
    assert authorization["selected_profile_confirmation_authorized"] is False
    assert authorization["final_paper_reporting_audit_authorized"] is False
    dry_run = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_test_audit_dry_run=pass" in dry_run
    assert "jobs=40" in dry_run
    assert "test_cells=160" in dry_run
    assert "authorized=true" in dry_run
    fail_closed_check()
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "manifest_sha256": sha256(MANIFEST),
                "training_checkpoints": 40,
                "standard_horizon_test_cells": 160,
                "dataset_level_shared_profile_selection": "pass",
                "checkpoint_immutability_contract": "pass",
                "unauthorized_test_fail_closed": "pass",
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
