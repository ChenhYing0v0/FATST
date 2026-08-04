#!/usr/bin/env python3
"""Check the frozen TimeAlign ETTm2/Weather reproduction contract."""

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
CONFIG = ROOT / "configs" / "timealign_official_ettm2_weather_reproduction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_means(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, list[dict[str, str]]] = {"ETTm2": [], "Weather": []}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in rows:
                rows[row["dataset"]].append(row)
    return {
        dataset: {
            metric: sum(float(row[metric]) for row in dataset_rows)
            / len(dataset_rows)
            for metric in ("mse", "mae")
        }
        for dataset, dataset_rows in rows.items()
    }


def expected_profile(dataset: str, horizon: int) -> dict[str, Any]:
    if dataset == "ETTm2":
        return {
            "d_model": 128,
            "d_ff": 128,
            "dropout": 0.3 if horizon in {96, 192} else 0.9,
            "learning_rate": 0.0001,
            "w_align": 1.0,
            "patch_num": 12,
            "layer_norm": 1,
        }
    return {
        "d_model": 128,
        "d_ff": 128 if horizon == 720 else 256,
        "dropout": 0.5 if horizon == 720 else 0.1,
        "learning_rate": 0.0001,
        "w_align": 0.1,
        "patch_num": 48,
        "layer_norm": 0,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["role"] == "native_external_fixed_h_published_baseline_reproduction"
    assert config["source_contract"]["raw_official_run_py_used"] is False
    assert config["source_contract"]["license_status"] == "license_unresolved"
    assert config["matrix"]["expected_runs"] == 8
    assert config["matrix"]["expected_test_rows"] == 8
    assert config["matrix"]["fixed_h_independent_models"] is True
    assert config["matrix"]["full_matrix_required"] is True
    assert config["matrix"]["partial_reporting_allowed"] is False

    training = config["training_contract"]
    assert training["seq_len"] == 720 and training["label_len"] == 48
    assert training["epochs"] == 10 and training["early_stopping"] is False
    assert training["checkpoint_policy"] == "official-last"
    assert training["test_access"] == "once_after_training"
    assert training["test_during_epoch"] is False
    assert training["official_test_mode"] is True

    authorization = config["authorization"]
    assert authorization["remote_resource_smoke_authorized"] is True
    assert authorization["remote_training_authorized"] is True
    assert authorization["official_test_execution_authorized"] is True
    assert authorization["additional_seeds_authorized"] is False
    assert authorization["additional_datasets_authorized"] is False

    source = config["source_contract"]
    for name in (
        "adapter",
        "ettm2_script",
        "weather_script",
        "executed_model",
        "data_loader",
        "metric",
    ):
        path = ROOT / source[f"{name}_path"]
        assert sha256(path) == source[f"{name}_sha256"], name

    jobs = config["jobs"]
    assert len(jobs) == 8
    assert len({job["run_id"] for job in jobs}) == 8
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {"ETTm2": 4, "Weather": 4}
    )
    assert {
        (job["dataset"], job["horizon"], job["seed"])
        for job in jobs
    } == {
        (dataset, horizon, 2021)
        for dataset in ("ETTm2", "Weather")
        for horizon in (96, 192, 336, 720)
    }
    assert {job["run_id"] for job in jobs} == set(config["workload_order"])
    for job in jobs:
        for field, value in expected_profile(
            job["dataset"], job["horizon"]
        ).items():
            assert job[field] == value, (job["run_id"], field, job[field], value)

    historical = config["historical_reference"]
    historical_path = ROOT / historical["path"]
    assert sha256(historical_path) == historical["sha256"]
    means = historical_means(historical_path)
    for dataset in ("ETTm2", "Weather"):
        for metric in ("mse", "mae"):
            expected = historical[f"{dataset}_four_h_mean_local"][metric]
            assert abs(means[dataset][metric] - expected) < 1e-12

    result = subprocess.run(
        ["bash", "scripts/remote/run_timealign_official_ettm2_weather_reproduction.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "timealign_official_reproduction_dry_run=pass" in result
    assert "jobs=8" in result and "test_jobs=8" in result
    assert "seed=2021" in result and "remote_authorized=true" in result

    runner = (
        ROOT
        / "scripts"
        / "remote"
        / "run_timealign_official_ettm2_weather_reproduction.sh"
    ).read_text(encoding="utf-8")
    assert "--legacy-" not in runner
    assert "--learning-rate" not in runner and "--w-align" not in runner
    assert "--readout-mode official" in runner
    assert "--checkpoint-policy official-last" in runner

    print(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "jobs": len(jobs),
                "test_rows": config["matrix"]["expected_test_rows"],
                "source_label": source["execution_label"],
                "license_status": source["license_status"],
                "historical_reference_artifact_complete": False,
                "remote_dataset_hashes_frozen": 2,
                "official_presets_loaded_without_legacy_overrides": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
