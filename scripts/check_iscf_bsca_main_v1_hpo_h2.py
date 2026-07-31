#!/usr/bin/env python3
"""Check the frozen H2 contract for ISCF-BSCA-MAIN-v1 HPO."""

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


def dry_run(canary_only: bool) -> str:
    environment = {
        "PATH": os.environ["PATH"],
        "MODE": "dry-run",
    }
    if canary_only:
        environment["CANARY_ONLY"] = "1"
    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_h2.sh"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    config_path = ROOT / "configs" / "iscf_bsca_main_v1_hpo_h2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    assert sha256(base_path) == config["base_config_sha256"]
    assert config["matrix"]["phase"] == "H2"
    assert config["matrix"]["expected_training_runs"] == 24
    assert config["matrix"]["expected_total_trials_including_H1"] == 40
    assert config["matrix"]["expected_test_runs_before_H2_completion"] == 0
    assert config["hpo_budget"]["H2_search_space_frozen"] is True
    assert config["authorization"]["remote_H2_training_authorized"] is True
    assert (
        config["authorization"][
            "official_test_HPO_execution_authorized_after_complete_H2"
        ]
        is True
    )
    jobs = materialize_jobs(config, base_config)
    assert len(jobs) == 24
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {dataset: 3 for dataset in config["datasets"]}
    )
    assert len({job["trial_id"] for job in jobs}) == len(jobs)
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    assert not {job["trial_id"] for job in jobs} & {
        job["trial_id"] for job in base_config["jobs"]
    }
    for job in jobs:
        assert job["seq_len"] in config["search_space"]["seq_len"]
        assert job["patch_num"] in config["search_space"]["patch_num"]
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] in config["search_space"]["d_model"]
        assert job["d_ff"] in config["search_space"]["d_ff"]
        assert job["dropout"] in config["search_space"]["dropout"]
        assert job["learning_rate"] in config["search_space"][
            "learning_rate"
        ]
        assert job["weight_decay"] == 0.01
        assert job["batch_size"] > 0
        assert job["gradient_accumulation_steps"] > 0
        assert job["mode_rank"] > 0
    selection = config["selection_contract"]
    assert selection["trial_checkpoint_split"] == "validation"
    assert selection["hyperparameter_selection_split"] == "official_test"
    assert selection["per_horizon_profile_selection"] is False
    assert selection["best_seed_selection"] is False
    full = dry_run(canary_only=False)
    assert "iscf_bsca_main_h2_dry_run=pass" in full
    assert "jobs=24" in full
    assert "test_jobs=0" in full
    canary = dry_run(canary_only=True)
    assert "jobs=9" in canary
    assert "test_jobs=0" in canary
    print(
        json.dumps(
            {
                "candidate": config["candidate_id"],
                "datasets": len(config["datasets"]),
                "h2_additional_jobs": len(jobs),
                "total_trials_including_h1": 40,
                "new_dataset_canary_jobs": 9,
                "validation_checkpoint_selector": "pass",
                "test_tuned_profile_selector": "pass",
                "test_jobs_before_h2_completion": 0,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
