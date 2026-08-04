#!/usr/bin/env python3
"""Check the frozen wide H4L HPO contract."""

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
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_wide_h4l.json"
LEDGER_PHASES = ("h1", "h2", "h3a", "h3b", "h4j", "h4k")
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
    return tuple(job.get(field, 1 if field == "layer_norm" else None) for field in PROFILE_FIELDS)


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


def values(jobs: list[dict[str, Any]], dataset: str, field: str) -> set[Any]:
    return {job.get(field, 1 if field == "layer_norm" else None) for job in jobs if job["dataset"] == dataset}


def assert_source_tuple(job: dict[str, Any], expected: dict[str, Any]) -> None:
    for field, value in expected.items():
        assert job.get(field, 1 if field == "layer_norm" else None) == value, (field, job, expected)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H4L"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["matrix"]["expected_training_runs"] == 48
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H4L_search_space_frozen"] is True
    assert config["hpo_budget"]["automatic_H4M_extension_authorized"] is False

    authorization = config["authorization"]
    assert authorization["remote_H4L_resource_smoke_authorized"] is True
    assert authorization["remote_H4L_training_authorized"] is True
    assert authorization["official_test_H4L_execution_authorized_after_complete_training"] is False
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    assert config["selection_contract"]["per_horizon_profile_selection"] is False
    assert config["selection_contract"]["per_metric_profile_selection"] is False
    assert config["training"]["official_test_during_training"] is False
    assert config["training"]["max_epochs"] == 60
    assert config["training"]["early_stopping_patience"] == 12

    assert config["success_gates"]["mse_lead_cells_min"] == 20
    assert config["success_gates"]["mae_lead_cells_min"] == 20
    assert config["success_gates"]["combined_lead_cells_min"] == 40
    assert config["success_gates"]["ettm2_combined_lead_cells_min"] == 4
    assert config["success_gates"]["weather_combined_lead_cells_min"] == 6

    target = ROOT / config["frozen_target_artifact"]["path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    for key in ("status", "all_trial_scorecard", "selected_profiles", "h4k_test_ledger"):
        path = ROOT / config["existing_evidence"][f"{key}_path"]
        assert sha256(path) == config["existing_evidence"][f"{key}_sha256"]
    with (ROOT / config["existing_evidence"]["all_trial_scorecard_path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 117

    source = config["source_inspiration"]
    for key in ("ettm2_script", "weather_script", "local_official_preset"):
        assert sha256(ROOT / source[f"{key}_path"]) == source[f"{key}_sha256"]

    jobs = resolve_jobs(config)
    assert len(jobs) == 48
    assert Counter(job["dataset"] for job in jobs) == Counter({"ETTm2": 24, "Weather": 24})
    trial_ids = {job["trial_id"] for job in jobs}
    assert len(trial_ids) == 48
    assert trial_ids == set(config["provisional_lpt_order"])
    assert len(config["provisional_lpt_order"]) == 48

    prior_ids, prior_fingerprints = prior_trials()
    assert len(prior_ids) == 117
    assert not trial_ids & prior_ids
    new_fingerprints = [fingerprint(job) for job in jobs]
    assert len(set(new_fingerprints)) == 48
    duplicates = set(new_fingerprints) & prior_fingerprints
    assert not duplicates, duplicates

    for job in jobs:
        assert all(job.get(field, 1 if field == "layer_norm" else None) is not None for field in PROFILE_FIELDS)
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] % 2 == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0.0
        assert job["weight_decay"] >= 0.0
        assert job["batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["mode_rank"] > 0

    expected_coverage = {
        "ETTm2": {
            "seq_len": {96, 192, 720},
            "patch_num": {1, 4, 8, 12, 24, 48, 72, 120},
            "d_model": {32, 64, 128, 256},
            "d_ff": {64, 128, 256, 512},
            "dropout": {0.0, 0.2, 0.3, 0.9},
            "learning_rate": {0.00001, 0.0001, 0.0003, 0.0005},
            "weight_decay": {0.0, 0.001, 0.01, 0.05},
            "mode_rank": {8, 16, 32, 64, 128, 256},
            "layer_norm": {0, 1},
        },
        "Weather": {
            "seq_len": {192, 512, 720},
            "patch_num": {1, 4, 8, 12, 16, 24, 48, 72, 120},
            "d_model": {32, 64, 128, 256},
            "d_ff": {64, 128, 256, 512},
            "dropout": {0.0, 0.1, 0.2, 0.5},
            "learning_rate": {0.00001, 0.00005, 0.0003, 0.0005},
            "weight_decay": {0.0, 0.001, 0.01, 0.05},
            "mode_rank": {16, 32, 64, 116, 128, 192, 256},
            "layer_norm": {0, 1},
        },
    }
    for dataset, coverage in expected_coverage.items():
        for field, expected in coverage.items():
            assert expected <= values(jobs, dataset, field), (dataset, field, values(jobs, dataset, field))

    by_id = {job["trial_id"]: job for job in jobs}
    assert_source_tuple(by_id["ETTm2__h4l_timealign_short_rank128_wd0"], source["ettm2_short_encoder_tuple"])
    assert_source_tuple(by_id["ETTm2__h4l_timealign_long_wd1e3"], source["ettm2_long_encoder_tuple"])
    assert_source_tuple(by_id["Weather__h4l_timealign_short_r64_wd1e3"], source["weather_short_encoder_tuple"])
    assert_source_tuple(by_id["Weather__h4l_timealign_long_r192_lr5e5"], source["weather_long_encoder_tuple"])

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo.sh"],
        cwd=ROOT,
        env={
            **os.environ,
            "CONFIG": "configs/iscf_bsca_main_v1_hpo_wide_h4l.json",
            "MODE": "dry-run",
        },
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h4l_dry_run=pass" in result
    assert "jobs=48" in result
    assert "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": config["matrix"]["jobs_by_dataset"],
                "prior_trials_audited": len(prior_ids),
                "prior_profile_duplicates": 0,
                "timealign_encoder_inspired_profiles": 4,
                "training_budget": "60 epochs / patience 12",
                "formal_test_authorized": False,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
