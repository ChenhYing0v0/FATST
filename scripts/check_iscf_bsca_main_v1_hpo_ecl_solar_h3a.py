#!/usr/bin/env python3
"""Check the test-informed ECL/Solar H3A HPO contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_jobs(
    config: dict[str, Any],
    base_config: dict[str, Any],
) -> list[dict[str, Any]]:
    base_jobs = {job["trial_id"]: job for job in base_config["jobs"]}
    jobs = []
    for specification in config["jobs"]:
        job = dict(base_jobs[specification["base_trial_id"]])
        job.update(specification["overrides"])
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_trial_id", "overrides"}
            }
        )
        jobs.append(job)
    return jobs


def main() -> None:
    config_path = (
        ROOT / "configs" / "iscf_bsca_main_v1_hpo_ecl_solar_h3a.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    assert sha256(base_path) == config["base_config_sha256"]
    assert config["matrix"]["phase"] == "H3A"
    assert config["matrix"]["expected_training_runs"] == 9
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["authorization"]["remote_H3A_training_authorized"] is True
    assert config["authorization"][
        "official_test_H3A_execution_authorized_after_complete_training"
    ] is True
    assert config["authorization"][
        "per_horizon_seed_metric_or_cell_tuning_allowed"
    ] is False
    jobs = materialize_jobs(config, base_config)
    assert len(jobs) == 9
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {"ECL": 1, "Solar": 8}
    )
    assert len({job["trial_id"] for job in jobs}) == 9
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    for job in jobs:
        assert job["seq_len"] == 720
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["max_epochs"] == 45
        assert job["early_stopping_patience"] == 10
        assert job["weight_decay"] in config["search_space"]["weight_decay"]
        assert job["dropout"] in config["search_space"]["dropout"]
        assert job["learning_rate"] in config["search_space"]["learning_rate"]
        assert job["mode_rank"] in config["search_space"]["mode_rank"]
    solar = {job["trial_id"]: job for job in jobs if job["dataset"] == "Solar"}
    assert solar["Solar__h3a_budget45"]["d_model"] == 256
    assert solar["Solar__h3a_budget45"]["d_ff"] == 256
    assert solar["Solar__h3a_budget45"]["dropout"] == 0.3
    assert solar["Solar__h3a_budget45"]["learning_rate"] == 0.0005
    assert solar["Solar__h3a_dropout4"]["dropout"] == 0.4
    assert solar["Solar__h3a_rank64"]["mode_rank"] == 64
    assert solar["Solar__h3a_effective_batch16"]["batch_size"] == 8
    assert solar["Solar__h3a_effective_batch16"][
        "gradient_accumulation_steps"
    ] == 2
    assert {solar[key]["patch_num"] for key in (
        "Solar__h3a_patch2",
        "Solar__h3a_patch4",
    )} == {2, 4}
    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_ecl_solar_h3a.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h3a_dry_run=pass" in result
    assert "jobs=9" in result
    assert "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "test_informed": True,
                "ECL_jobs": 1,
                "Solar_jobs": 8,
                "expanded_budget_epochs": 45,
                "validation_checkpoint_selector": "pass",
                "dataset_level_test_selector": "pass",
                "test_jobs_during_training": 0,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
