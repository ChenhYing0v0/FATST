#!/usr/bin/env python3
"""Check the frozen Weather-only H4N HPO contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_weather_h4n.json"
LEDGER_PHASES = ("h1", "h2", "h3a", "h3b", "h4j", "h4k", "h4l", "h4m")
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
HORIZONS = (96, 192, 336, 720)


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


def prior_trials() -> tuple[set[str], set[tuple[Any, ...]], int]:
    trial_ids: set[str] = set()
    fingerprints: set[tuple[Any, ...]] = set()
    weather_trials = 0
    for phase in LEDGER_PHASES:
        path = (
            ROOT
            / "analysis"
            / "iscf_bsca_main_v1_hpo_20260731"
            / f"{phase}_artifact_audit"
            / "trial_ledger.jsonl"
        )
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            trial_ids.add(row["trial_id"])
            fingerprints.add(fingerprint(row["hyperparameters"]))
            weather_trials += int(row["dataset"] == "Weather")
    return trial_ids, fingerprints, weather_trials


def target_map(path: Path) -> dict[str, dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["dataset"] == "Weather"
            and row["horizon"] != "Avg."
            and row["model"] != "ISCF-BSCA"
        ]
    for horizon in HORIZONS:
        selected = [row for row in rows if int(row["horizon"]) == horizon]
        targets[str(horizon)] = {
            "mse": min(float(row["mse"]) for row in selected),
            "mae": min(float(row["mae"]) for row in selected),
        }
    return targets


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H4N"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["matrix"]["expected_training_runs"] == 40
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H4N_search_space_frozen"] is True
    assert config["hpo_budget"]["automatic_H4O_extension_authorized"] is False

    authorization = config["authorization"]
    assert authorization["remote_H4N_resource_smoke_authorized"] is True
    assert authorization["remote_H4N_training_authorized"] is True
    assert authorization[
        "official_test_H4N_execution_authorized_after_complete_training"
    ] is True
    assert authorization["formal_test_requires_complete_checkpoint_manifest"] is True
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False

    selection = config["selection_contract"]
    assert selection["per_horizon_profile_selection"] is False
    assert selection["per_metric_profile_selection"] is False
    assert selection["best_seed_selection"] is False
    assert selection["near_tie_band"].endswith("0.1_percent")
    assert config["training"]["official_test_during_training"] is False

    target = ROOT / config["frozen_target_artifact"]["path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    observed_targets = target_map(target)
    for horizon, metrics in config["frozen_target_artifact"][
        "weather_targets"
    ].items():
        for metric, value in metrics.items():
            assert math.isclose(
                observed_targets[horizon][metric], value, rel_tol=0.0, abs_tol=1e-15
            )

    evidence = config["existing_evidence"]
    for key in ("status", "all_trial_scorecard", "selected_profiles", "h4m_test_ledger"):
        path = ROOT / evidence[f"{key}_path"]
        assert sha256(path) == evidence[f"{key}_sha256"]
    with (ROOT / evidence["all_trial_scorecard_path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 189
    assert sum(row["dataset"] == "Weather" for row in rows) == 56

    jobs = resolve_jobs(config)
    assert len(jobs) == 40
    assert Counter(job["dataset"] for job in jobs) == Counter({"Weather": 40})
    source_counts = Counter(job["source_prior"] for job in jobs)
    assert source_counts == Counter(
        {
            "context_lr_interpolation": 16,
            "low_lr_wide_boundary": 8,
            "patch_geometry_at_frontiers": 8,
            "rank_at_mae_frontier": 5,
            "capacity_at_mae_frontier": 3,
        }
    )
    trial_ids = {job["trial_id"] for job in jobs}
    assert len(trial_ids) == 40
    assert trial_ids == set(config["provisional_lpt_order"])

    prior_ids, prior_fingerprints, weather_trials = prior_trials()
    assert len(prior_ids) == 189
    assert weather_trials == 56
    assert not trial_ids & prior_ids
    new_fingerprints = [fingerprint(job) for job in jobs]
    assert len(set(new_fingerprints)) == 40
    assert not set(new_fingerprints) & prior_fingerprints

    for job in jobs:
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] % 2 == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0.0
        assert job["weight_decay"] >= 0.0
        assert job["batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["mode_rank"] > 0
        assert job["max_epochs"] == 120
        assert job["early_stopping_patience"] == 24

    context_jobs = [job for job in jobs if job["source_prior"] == "context_lr_interpolation"]
    assert {job["seq_len"] for job in context_jobs} == {
        448,
        480,
        544,
        576,
        608,
        640,
        672,
        704,
    }
    assert all(job["seq_len"] // job["patch_num"] == 32 for job in context_jobs)
    assert {job["learning_rate"] for job in jobs} >= {0.000005, 0.0002}
    assert {job["patch_num"] for job in jobs} >= {4, 64}
    assert {job["mode_rank"] for job in jobs} >= {80, 96, 128, 160, 192}
    assert {job["d_model"] for job in jobs} >= {32, 64, 96, 128}

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_weather_h4n.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h4n_dry_run=pass" in result
    assert "jobs=40" in result and "test_jobs=0" in result
    assert "remote_authorized=true" in result
    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": config["matrix"]["jobs_by_dataset"],
                "prior_trials_audited": len(prior_ids),
                "prior_weather_trials_audited": weather_trials,
                "prior_profile_duplicates": 0,
                "search_blocks": dict(source_counts),
                "Weather_budget": "120 epochs / patience 24",
                "training_test_jobs": 0,
                "formal_test_after_manifest_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
