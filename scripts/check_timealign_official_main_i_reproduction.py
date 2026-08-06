#!/usr/bin/env python3
"""Check the frozen eight-dataset TimeAlign Main I reproduction contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "timealign_official_main_i_reproduction.json"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar", "Exchange")
HORIZONS = (96, 192, 336, 720)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_authorized_prelaunch"
    assert tuple(config["matrix"]["datasets"]) == DATASETS
    assert tuple(config["matrix"]["horizons"]) == HORIZONS
    assert config["matrix"]["seeds"] == [2021]
    assert config["matrix"]["expected_runs"] == 32
    assert config["matrix"]["reusable_runs"] == 8
    assert config["matrix"]["new_runs"] == 24
    assert config["matrix"]["full_matrix_required"] is True
    assert config["matrix"]["partial_reporting_allowed"] is False

    training = config["training_contract"]
    assert training["seq_len"] == 720 and training["label_len"] == 48
    assert training["default_epochs"] == 10
    assert training["etth1_h96_epochs"] == 1
    assert training["early_stopping"] is False
    assert training["checkpoint_policy"] == "official-last"
    assert training["test_access"] == "once_after_training"
    assert training["test_during_epoch"] is False
    assert training["new_run_predictions_retained"] is False

    source = config["source_contract"]
    for name in ("adapter", "executed_model", "data_loader", "metric"):
        assert sha256(ROOT / source[f"{name}_path"]) == source[f"{name}_sha256"]
    assert set(source["dataset_scripts"]) == set(DATASETS)
    for dataset, item in source["dataset_scripts"].items():
        assert sha256(ROOT / item["path"]) == item["sha256"], dataset
    assert source["dataset_scripts"]["Exchange"]["preset_role"] == (
        "source_informed_etth1_bootstrap_not_official"
    )
    assert all(
        source["dataset_scripts"][dataset]["preset_role"] == "official"
        for dataset in DATASETS
        if dataset != "Exchange"
    )

    reuse = config["reuse_contract"]
    assert len(set(reuse["run_ids"])) == 8
    assert sha256(ROOT / reuse["prior_config_path"]) == reuse["prior_config_sha256"]
    assert sha256(ROOT / reuse["prior_artifact_manifest_path"]) == reuse["prior_artifact_manifest_sha256"]
    assert len(config["workload_order"]) == 32
    assert len(set(config["workload_order"])) == 32
    assert set(config["workload_order"]) == {
        f"TimeAlign__{dataset}__H{horizon}__seed2021"
        for dataset in DATASETS
        for horizon in HORIZONS
    }

    authorization = config["authorization"]
    assert authorization["remote_resource_smoke_authorized"] is True
    assert authorization["remote_training_authorized"] is True
    assert authorization["official_test_execution_authorized"] is True
    assert authorization["additional_seeds_authorized"] is False
    assert authorization["other_baseline_training_authorized"] is False

    result = subprocess.run(
        ["bash", "scripts/remote/run_timealign_official_main_i_reproduction.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "timealign_main_i_dry_run=pass" in result
    assert "jobs=32" in result and "reusable=8" in result and "new=24" in result
    assert "test_jobs=24" in result and "remote_authorized=true" in result

    runner = (ROOT / "scripts" / "remote" / "run_timealign_official_main_i_reproduction.sh").read_text(encoding="utf-8")
    assert "--legacy-" not in runner
    assert "--learning-rate" not in runner and "--w-align" not in runner
    assert "--readout-mode official" in runner
    assert "--checkpoint-policy official-last" in runner
    assert "--no-save-predictions" in runner

    print(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "datasets": len(DATASETS),
                "runs": 32,
                "reusable_runs": 8,
                "new_runs": 24,
                "source_hashes_frozen": 12,
                "dataset_hashes_frozen": 8,
                "exchange_preset_role": "source_informed_not_official",
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
