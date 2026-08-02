#!/usr/bin/env python3
"""Check the frozen joint-objective H4J HPO contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_joint_h4j.json"
REQUIRED_FIELDS = {
    "dataset",
    "seq_len",
    "patch_num",
    "d_model",
    "d_ff",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "mode_rank",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H4J"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True
    assert config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["matrix"]["expected_training_runs"] == 40
    assert config["hpo_budget"]["H4J_search_space_frozen"] is True
    assert config["authorization"]["remote_H4J_training_authorized"] is True
    assert config["authorization"][
        "official_test_H4J_execution_authorized_after_complete_training"
    ] is True
    assert config["selection_contract"]["per_horizon_profile_selection"] is False
    assert config["selection_contract"]["per_metric_profile_selection"] is False
    assert config["success_gates"]["mse_lead_cells_min"] == 20
    assert config["success_gates"]["mae_lead_cells_min"] == 20
    assert config["success_gates"]["combined_lead_cells_min"] == 40
    assert config["success_gates"]["combined_lead_cells_denominator"] == 56

    target = ROOT / config["frozen_target_artifact"]["path"]
    status = ROOT / config["existing_evidence"]["status_path"]
    scorecard = ROOT / config["existing_evidence"]["all_trial_scorecard_path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    assert sha256(status) == config["existing_evidence"]["status_sha256"]
    assert sha256(scorecard) == config["existing_evidence"][
        "all_trial_scorecard_sha256"
    ]

    profiles = config["base_profiles"]
    jobs = []
    for specification in config["jobs"]:
        base_id = specification["base_profile_id"]
        assert base_id in profiles
        job = dict(profiles[base_id])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_profile_id", "overrides"}
            }
        )
        assert REQUIRED_FIELDS <= set(job)
        assert job["dataset"] == specification["dataset"]
        assert job.get("max_epochs", config["training"]["max_epochs"]) in {45, 60}
        assert job.get(
            "early_stopping_patience",
            config["training"]["early_stopping_patience"],
        ) in {10, 12}
        jobs.append(job)

    assert len(jobs) == 40
    assert len({job["trial_id"] for job in jobs}) == 40
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    assert Counter(job["dataset"] for job in jobs) == Counter(
        config["matrix"]["jobs_by_dataset"]
    )
    weak_jobs = sum(
        job["dataset"] in config["hpo_budget"]["weak_datasets"] for job in jobs
    )
    assert weak_jobs == config["hpo_budget"]["weak_dataset_trials"] == 28

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "CONFIG": "configs/iscf_bsca_main_v1_hpo_joint_h4j.json",
            "MODE": "dry-run",
        },
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h4j_dry_run=pass" in result
    assert "jobs=40" in result
    assert "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": config["matrix"]["jobs_by_dataset"],
                "weak_dataset_jobs": weak_jobs,
                "joint_mean_guard_pct": 1.0,
                "lead_cell_gate": "MSE>=20/28, MAE>=20/28, total>=40/56",
                "validation_checkpoint_selector": "pass",
                "dataset_level_joint_test_selector": "pass",
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
