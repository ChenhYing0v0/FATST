#!/usr/bin/env python3
"""Check the terminal test-informed Solar H3B contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config_path = ROOT / "configs" / "iscf_bsca_main_v1_hpo_solar_h3b.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_jobs = {job["trial_id"]: job for job in base["jobs"]}
    assert sha256(base_path) == config["base_config_sha256"]
    assert config["matrix"]["phase"] == "H3B"
    assert config["matrix"]["expected_training_runs"] == 4
    assert config["test_informed"] is True
    assert config["hpo_budget"]["terminal_bounded_batch"] is True
    assert config["authorization"]["remote_H3B_training_authorized"] is True
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
    assert len(jobs) == 4
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    assert all(job["dataset"] == "Solar" for job in jobs)
    assert all(job["seq_len"] == 720 and job["patch_num"] == 1 for job in jobs)
    assert all(job["max_epochs"] == 45 for job in jobs)
    assert all(job["early_stopping_patience"] == 10 for job in jobs)
    by_id = {job["trial_id"]: job for job in jobs}
    assert by_id["Solar__h3b_lr2e4"]["learning_rate"] == 0.0002
    assert by_id["Solar__h3b_lr4e4"]["learning_rate"] == 0.0004
    assert by_id["Solar__h3b_lr3e4_dropout4"]["dropout"] == 0.4
    assert by_id["Solar__h3b_lr3e4_rank64"]["mode_rank"] == 64
    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_solar_h3b.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h3b_dry_run=pass" in result
    assert "jobs=4" in result
    assert "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "Solar_jobs": 4,
                "terminal_bounded_batch": True,
                "validation_checkpoint_selector": "pass",
                "dataset_level_test_selector": "pass",
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
