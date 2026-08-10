#!/usr/bin/env python3
"""Check the frozen ETTh1/ECL/Solar Main-II H5A HPO contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_main_ii_h5a.json"
DATASETS = ("ETTh1", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for specification in config["jobs"]:
        job = dict(config["base_profiles"][specification["base_profile_id"]])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_profile_id", "overrides"}
            }
        )
        job.setdefault("max_epochs", config["training"]["max_epochs"])
        job.setdefault(
            "early_stopping_patience",
            config["training"]["early_stopping_patience"],
        )
        jobs.append(job)
    return jobs


def fingerprint(job: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(job.get(field, 1 if field == "layer_norm" else None) for field in PROFILE_FIELDS)


def rounded(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def current_best_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for dataset in DATASETS:
        for horizon in HORIZONS:
            cell = [
                row
                for row in rows
                if row["dataset"] == dataset and int(row["horizon"]) == horizon
            ]
            candidate = next(
                row for row in cell if row["system"] == "ISCF-BSCA-MAIN-v1"
            )
            external = [
                row for row in cell if row["system"] != "ISCF-BSCA-MAIN-v1"
            ]
            for metric in ("mse", "mae"):
                target = min(rounded(float(row[metric])) for row in external)
                counts[dataset] += rounded(float(candidate[metric])) <= target
    return dict(counts)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H5A"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["test_tuned"] is True and config["test_informed"] is True
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["matrix"]["expected_training_runs"] == 48
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H5A_search_space_frozen"] is True
    assert config["hpo_budget"]["automatic_H5B_extension_authorized"] is False
    assert config["training"]["official_test_during_training"] is False

    authorization = config["authorization"]
    assert authorization["remote_H5A_resource_smoke_authorized"] is True
    assert authorization["remote_H5A_training_authorized"] is True
    assert authorization[
        "official_test_H5A_execution_authorized_after_complete_training_manifest"
    ] is True
    assert authorization["per_horizon_seed_metric_or_cell_tuning_allowed"] is False
    assert authorization["automatic_Main_I_or_Main_II_table_mutation_authorized"] is False

    selection = config["selection_contract"]
    assert selection["profile_granularity"].startswith("one_profile_per_dataset")
    assert selection["per_horizon_profile_selection"] is False
    assert selection["per_metric_profile_selection"] is False
    assert selection["cell_specific_profile_selection"] is False
    assert selection["best_seed_selection"] is False
    assert selection["all_trials_retained"] is True
    guard = selection["eligibility_guard"]
    assert guard["mean_mse_relative_to_current_profile_max"] == 1.005
    assert guard["mean_mae_relative_to_current_profile_max"] == 1.005

    target = ROOT / config["frozen_target_artifact"]["path"]
    assert sha256(target) == config["frozen_target_artifact"]["sha256"]
    target_rows = [row for row in read_csv(target) if row["horizon"] != "Avg."]
    assert current_best_counts(target_rows) == config["success_gates"][
        "current_best_cells_by_dataset"
    ]

    evidence = config["existing_evidence"]
    scorecard = ROOT / evidence["historical_scorecard_path"]
    selected = ROOT / evidence["selected_profile_manifest_path"]
    assert sha256(scorecard) == evidence["historical_scorecard_sha256"]
    assert sha256(selected) == evidence["selected_profile_manifest_sha256"]
    historical_rows = read_csv(scorecard)
    assert len(historical_rows) == evidence["historical_trial_count"] == 189
    historical_ids = {row["trial_id"] for row in historical_rows}

    jobs = resolve_jobs(config)
    assert len(jobs) == 48
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {"ETTh1": 16, "ECL": 16, "Solar": 16}
    )
    trial_ids = {job["trial_id"] for job in jobs}
    assert len(trial_ids) == 48
    assert trial_ids == set(config["provisional_lpt_order"])
    assert not trial_ids & historical_ids
    assert len({fingerprint(job) + (job["max_epochs"], job["early_stopping_patience"]) for job in jobs}) == 48

    for job in jobs:
        assert all(field in job for field in PROFILE_FIELDS)
        assert job["seq_len"] in config["search_space"]["seq_len"]
        assert job["patch_num"] in config["search_space"]["patch_num"]
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] % 2 == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0.0
        assert job["weight_decay"] >= 0.0
        assert job["batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["max_epochs"] == 90
        assert job["early_stopping_patience"] == 18

    assert {job["patch_num"] for job in jobs if job["dataset"] == "ECL"} >= {1, 2, 4, 8, 12}
    assert {job["learning_rate"] for job in jobs if job["dataset"] == "Solar"} >= {
        0.00015,
        0.00025,
        0.0003,
        0.00035,
    }
    assert {job["seq_len"] for job in jobs if job["dataset"] == "ETTh1"} == {
        336,
        512,
        720,
    }

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_main_ii_h5a.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h5a_dry_run=pass" in result
    assert "jobs=48" in result and "test_jobs=0" in result
    assert "remote_authorized=true" in result

    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "jobs_by_dataset": dict(Counter(job["dataset"] for job in jobs)),
                "current_best_cells": config["success_gates"][
                    "current_best_cells_by_dataset"
                ],
                "minimum_target_best_cells": config["success_gates"][
                    "minimum_best_cells_by_dataset"
                ],
                "historical_trials_audited": len(historical_rows),
                "training_test_jobs": 0,
                "formal_test_after_manifest_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
