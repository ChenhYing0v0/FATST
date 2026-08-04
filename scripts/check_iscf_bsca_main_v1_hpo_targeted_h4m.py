#!/usr/bin/env python3
"""Check the frozen high-impact H4M HPO contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_targeted_h4m.json"
LEDGER_PHASES = ("h1", "h2", "h3a", "h3b", "h4j", "h4k", "h4l")
PROFILE_FIELDS = (
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
    "layer_norm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = config["base_profiles"]
    jobs = []
    for specification in config["jobs"]:
        job = dict(profiles[specification["base_profile_id"]])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_profile_id", "overrides"}
            }
        )
        jobs.append(job)
    return jobs


def fingerprint(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        job.get(field, 1 if field == "layer_norm" else None)
        for field in PROFILE_FIELDS
    )


def prior_trials() -> tuple[set[str], set[tuple[Any, ...]]]:
    trial_ids: set[str] = set()
    fingerprints: set[tuple[Any, ...]] = set()
    for phase in LEDGER_PHASES:
        path = (
            ROOT
            / "analysis"
            / "iscf_bsca_main_v1_hpo_20260731"
            / f"{phase}_artifact_audit"
            / "trial_ledger.jsonl"
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            trial_ids.add(row["trial_id"])
            fingerprints.add(fingerprint(row["hyperparameters"]))
    return trial_ids, fingerprints


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H4M"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["matrix"]["expected_training_runs"] == 24
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H4M_search_space_frozen"] is True
    assert config["hpo_budget"]["automatic_H4N_extension_authorized"] is False

    authorization = config["authorization"]
    assert authorization["remote_H4M_resource_smoke_authorized"] is True
    assert authorization["remote_H4M_training_authorized"] is True
    assert authorization["official_test_H4M_execution_authorized_after_complete_training"] is True
    assert authorization["formal_test_requires_complete_checkpoint_manifest"] is True
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    assert config["selection_contract"]["per_horizon_profile_selection"] is False
    assert config["selection_contract"]["per_metric_profile_selection"] is False
    assert config["training"]["official_test_during_training"] is False

    target = ROOT / config["frozen_target_artifact"]["path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    for key in ("status", "all_trial_scorecard", "selected_profiles", "h4l_test_ledger"):
        path = ROOT / config["existing_evidence"][f"{key}_path"]
        assert sha256(path) == config["existing_evidence"][f"{key}_sha256"]
    with (ROOT / config["existing_evidence"]["all_trial_scorecard_path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 165

    impact = config["impact_evidence"]
    assert impact["ETTm2"]["patch_num_joint_span_pct"] > 3.0
    assert impact["ETTm2"]["learning_rate_joint_span_pct"] > 3.0
    assert impact["Weather"]["seq_len_x_patch_num_joint_span_pct"] > 4.0
    assert impact["Weather"]["learning_rate_joint_span_pct"] > 1.0
    assert "weight_decay" in impact["frozen_low_impact_parameters"]

    jobs = resolve_jobs(config)
    assert len(jobs) == 24
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {"ETTm2": 12, "Weather": 12}
    )
    trial_ids = {job["trial_id"] for job in jobs}
    assert len(trial_ids) == 24
    assert trial_ids == set(config["provisional_lpt_order"])

    prior_ids, prior_fingerprints = prior_trials()
    assert len(prior_ids) == 165
    assert not trial_ids & prior_ids
    new_fingerprints = [fingerprint(job) for job in jobs]
    assert len(set(new_fingerprints)) == 24
    assert not set(new_fingerprints) & prior_fingerprints

    for job in jobs:
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] % 2 == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0.0
        assert job["weight_decay"] >= 0.0
        assert job["batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["mode_rank"] > 0
        if job["dataset"] == "Weather":
            assert job["max_epochs"] == 90
            assert job["early_stopping_patience"] == 18
        else:
            assert job.get("max_epochs", config["training"]["max_epochs"]) == 60

    ettm2_grid = {
        (job["patch_num"], job["learning_rate"])
        for job in jobs
        if job["dataset"] == "ETTm2" and job["mode_rank"] == 64
        and job["seq_len"] == 720
    }
    assert {
        (patch, learning_rate)
        for patch in (2, 4, 6)
        for learning_rate in (0.00001, 0.00002, 0.00005)
    } <= ettm2_grid

    weather_pairs = {
        (job["seq_len"], job["patch_num"])
        for job in jobs
        if job["dataset"] == "Weather"
        and job["learning_rate"] == 0.00005
        and job["mode_rank"] == 116
        and job["layer_norm"] == 1
    }
    assert {(384, 12), (640, 20), (768, 24), (960, 30), (1152, 36)} <= weather_pairs

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_targeted_h4m.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h4m_dry_run=pass" in result
    assert "jobs=24" in result and "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": config["matrix"]["jobs_by_dataset"],
                "prior_trials_audited": len(prior_ids),
                "prior_profile_duplicates": 0,
                "ETTm2_budget": "60 epochs / patience 12",
                "Weather_budget": "90 epochs / patience 18",
                "training_test_jobs": 0,
                "formal_test_after_manifest_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
