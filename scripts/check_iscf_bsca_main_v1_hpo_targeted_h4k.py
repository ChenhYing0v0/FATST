#!/usr/bin/env python3
"""Check the frozen targeted H4K HPO contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_targeted_h4k.json"
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


def resolve_jobs(config_path: Path) -> dict[str, dict[str, object]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_jobs: dict[str, dict[str, object]] = {}
    if config.get("base_config"):
        base_path = ROOT / config["base_config"]
        base_jobs = resolve_jobs(base_path)
    base_profiles = config.get("base_profiles", {})
    resolved = {}
    for specification in config["jobs"]:
        if specification.get("base_profile_id"):
            job = dict(base_profiles[specification["base_profile_id"]])
        elif specification.get("base_trial_id"):
            job = dict(base_jobs[specification["base_trial_id"]])
        else:
            job = {}
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key
                not in {"base_profile_id", "base_trial_id", "overrides"}
            }
        )
        resolved[str(job["trial_id"])] = job
    return resolved


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H4K"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["matrix"]["expected_training_runs"] == 24
    assert config["hpo_budget"]["H4K_search_space_frozen"] is True
    authorization = config["authorization"]
    assert authorization["remote_H4K_resource_smoke_authorized"] is True
    assert authorization["remote_H4K_training_authorized"] is True
    assert (
        authorization[
            "official_test_H4K_execution_authorized_after_complete_training"
        ]
        is False
    )
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    assert config["selection_contract"]["per_horizon_profile_selection"] is False
    assert config["selection_contract"]["per_metric_profile_selection"] is False
    assert config["success_gates"]["mse_lead_cells_min"] == 20
    assert config["success_gates"]["mae_lead_cells_min"] == 20
    assert config["success_gates"]["combined_lead_cells_min"] == 40
    assert config["success_gates"]["ettm2_combined_lead_cells_min"] == 4
    assert config["success_gates"]["weather_combined_lead_cells_min"] == 6
    assert config["success_gates"]["h720_combined_lead_cells_min"] == 6

    target = ROOT / config["frozen_target_artifact"]["path"]
    status = ROOT / config["existing_evidence"]["status_path"]
    scorecard = ROOT / config["existing_evidence"]["all_trial_scorecard_path"]
    selected = ROOT / config["existing_evidence"]["selected_profiles_path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    assert sha256(status) == config["existing_evidence"]["status_sha256"]
    assert sha256(scorecard) == config["existing_evidence"][
        "all_trial_scorecard_sha256"
    ]
    assert sha256(selected) == config["existing_evidence"][
        "selected_profiles_sha256"
    ]
    with scorecard.open(encoding="utf-8", newline="") as handle:
        existing_trial_ids = {row["trial_id"] for row in csv.DictReader(handle)}
    assert len(existing_trial_ids) == 93

    profiles = config["base_profiles"]
    anchor_sources = {
        "ETTm2_rank64": (
            "configs/iscf_bsca_main_v1_hpo_joint_h4j.json",
            "ETTm2__h4j_rank64",
        ),
        "Weather_current": (
            "configs/iscf_bsca_main_v1_hpo.json",
            "Weather__h1_current",
        ),
        "Weather_timealign_dropout3": (
            "configs/iscf_bsca_main_v1_hpo_joint_h4j.json",
            "Weather__h4j_timealign_dropout3",
        ),
        "ETTh1_lr3e4": (
            "configs/iscf_bsca_main_v1_hpo_joint_h4j.json",
            "ETTh1__h4j_lr3e4",
        ),
        "ETTh2_lr5e4": (
            "configs/iscf_bsca_main_v1_hpo_h2.json",
            "ETTh2__h2_lr5e4",
        ),
        "ETTm1_table5_capacity": (
            "configs/iscf_bsca_main_v1_hpo_h2.json",
            "ETTm1__h2_table5_capacity",
        ),
        "ECL_exact": (
            "configs/iscf_bsca_main_v1_hpo.json",
            "ECL__h1_timealign",
        ),
        "Solar_lr3e4": (
            "configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a.json",
            "Solar__h3a_lr3e4",
        ),
    }
    for profile_id, (source_path, trial_id) in anchor_sources.items():
        source = resolve_jobs(ROOT / source_path)[trial_id]
        profile = profiles[profile_id]
        for field in REQUIRED_FIELDS | {"layer_norm"}:
            assert profile.get(field, 1) == source.get(field, 1), (
                profile_id,
                field,
                profile.get(field, 1),
                source.get(field, 1),
            )

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
        assert job["seq_len"] % job["patch_num"] == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0
        assert job["mode_rank"] > 0
        assert job.get("max_epochs", config["training"]["max_epochs"]) in {45, 60}
        assert job.get(
            "early_stopping_patience",
            config["training"]["early_stopping_patience"],
        ) in {10, 12}
        jobs.append(job)

    assert len(jobs) == 24
    trial_ids = {job["trial_id"] for job in jobs}
    assert len(trial_ids) == 24
    assert not trial_ids & existing_trial_ids
    assert trial_ids == set(config["provisional_lpt_order"])
    assert Counter(job["dataset"] for job in jobs) == Counter(
        config["matrix"]["jobs_by_dataset"]
    )
    primary_jobs = sum(
        job["dataset"] in config["hpo_budget"]["primary_gap_datasets"]
        for job in jobs
    )
    supplementary_jobs = sum(
        job["dataset"] in config["hpo_budget"]["h720_supplementary_datasets"]
        for job in jobs
    )
    assert primary_jobs == config["hpo_budget"]["primary_gap_trials"] == 14
    assert supplementary_jobs == config["hpo_budget"][
        "h720_supplementary_trials"
    ] == 10

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "CONFIG": "configs/iscf_bsca_main_v1_hpo_targeted_h4k.json",
            "MODE": "dry-run",
        },
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h4k_dry_run=pass" in result
    assert "jobs=24" in result
    assert "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": config["matrix"]["jobs_by_dataset"],
                "primary_gap_jobs": primary_jobs,
                "h720_supplementary_jobs": supplementary_jobs,
                "lead_cell_gate": "MSE>=20/28, MAE>=20/28, total>=40/56",
                "formal_test_authorized": False,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
