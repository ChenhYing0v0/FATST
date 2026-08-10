#!/usr/bin/env python3
"""Audit all H5A train/validation artifacts before official-test access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_iscf_bsca_main_v1_hpo import (
    best_training_row,
    file_sha256,
    materialize_jobs,
    metric_map,
)


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
EXPECTED = Counter({"ETTh1": 16, "ECL": 16, "Solar": 16})
FAILURE_PATTERN = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|"
    r"(^|[^A-Za-z0-9_])(nan|inf)([^A-Za-z0-9_]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_hpo_main_ii_h5a.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser.parse_args()


def search_space_hash(config: dict[str, Any]) -> str:
    payload = {
        "jobs": config["jobs"],
        "base_profiles": config.get("base_profiles", {}),
        "hpo_budget": config["hpo_budget"],
        "selection_contract": config["selection_contract"],
        "architecture_invariants": config["architecture_invariants"],
        "base_config": config.get("base_config"),
        "base_config_sha256": config.get("base_config_sha256"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = materialize_jobs(config, args.config)
    ledger = [
        json.loads(line)
        for line in args.ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jobs) == len(ledger) == 48
    assert Counter(job["dataset"] for job in jobs) == EXPECTED
    assert config["authorization"][
        "official_test_H5A_execution_authorized_after_complete_training_manifest"
    ] is True

    config_hash = file_sha256(args.config)
    space_hash = search_space_hash(config)
    ledger_by_trial = {row["trial_id"]: row for row in ledger}
    assert len(ledger_by_trial) == 48
    checkpoint_hashes: set[str] = set()
    best_epochs: list[int] = []
    validation_best: dict[str, tuple[str, float]] = {}

    for job in jobs:
        trial_id = job["trial_id"]
        dataset = job["dataset"]
        directory = args.output_root / "trials" / dataset / trial_id / "seed2021"
        for filename in (
            "checkpoint.pt",
            "training_log.csv",
            "metrics_by_target_horizon.csv",
            "effective_config.json",
            "initialization_contract.json",
            "model_diagnostics.json",
            "environment.json",
        ):
            path = directory / filename
            assert path.is_file() and path.stat().st_size > 0, path
        assert not (directory / "test_audit_metrics_by_target_horizon.csv").exists()

        adapter = json.loads(
            (directory / "effective_config.json").read_text(encoding="utf-8")
        )["adapter"]
        expected = {
            "hpo_config_hash": config_hash,
            "hpo_search_space_hash": space_hash,
            "hpo_trial_id": trial_id,
            "hpo_profile_id": job["profile_id"],
            "seed": 2021,
            "official_test_mode": False,
            "final_evaluation_split": "val",
            "checkpoint_policy": "best-val",
            "evaluation_prefix_mode": "full-crop",
            "validation_horizons": list(HORIZONS),
            "evaluation_horizons": list(HORIZONS),
            "pred_len": 720,
            "seq_len": job["seq_len"],
            "legacy_patch_num": job["patch_num"],
            "legacy_d_model": job["d_model"],
            "legacy_d_ff": job["d_ff"],
            "legacy_dropout": job["dropout"],
            "legacy_layer_norm": job.get("layer_norm", 1),
            "learning_rate": job["learning_rate"],
            "weight_decay": job["weight_decay"],
            "batch_size": job["batch_size"],
            "gradient_accumulation_steps": job["gradient_accumulation_steps"],
            "pcsd_mode_rank": job["mode_rank"],
            "epochs": job.get("max_epochs", config["training"]["max_epochs"]),
            "patience": job.get(
                "early_stopping_patience",
                config["training"]["early_stopping_patience"],
            ),
        }
        for field, value in expected.items():
            assert adapter[field] == value, (trial_id, field, adapter[field], value)

        metrics = metric_map(directory / "metrics_by_target_horizon.csv")
        assert set(metrics) == set(HORIZONS)
        assert all(math.isfinite(value) for pair in metrics.values() for value in pair)
        mean_mse = sum(pair[0] for pair in metrics.values()) / 4
        best = best_training_row(directory / "training_log.csv")
        assert math.isclose(mean_mse, float(best["val_mean_mse"]), rel_tol=0.0, abs_tol=1e-7)

        row = ledger_by_trial[trial_id]
        assert row["status"] == "validation_complete"
        assert row["artifact_completeness_pass"] is True
        assert row["numeric_health_pass"] is True
        assert math.isclose(
            mean_mse,
            float(row["best_validation_mean_mse_4h"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checkpoint_hash = file_sha256(directory / "checkpoint.pt")
        assert checkpoint_hash == row["checkpoint_sha256_before_test"]
        checkpoint_hashes.add(checkpoint_hash)
        best_epochs.append(int(best["epoch"]))
        current = validation_best.get(dataset)
        if current is None or mean_mse < current[1]:
            validation_best[dataset] = (trial_id, mean_mse)

        log = args.output_root / "_logs" / f"train_{trial_id}.log"
        assert log.is_file() and log.stat().st_size > 0
        assert FAILURE_PATTERN.search(log.read_text(encoding="utf-8")) is None

    assert len(checkpoint_hashes) == 48
    assert not (args.output_root / "test_audit").exists()
    print(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "training_complete": "48/48",
                "validation_complete": "48/48",
                "test_complete": "0/48",
                "unique_checkpoint_hashes": len(checkpoint_hashes),
                "config_hash": config_hash,
                "search_space_hash": space_hash,
                "best_epoch_range": [min(best_epochs), max(best_epochs)],
                "validation_best_diagnostic_only": {
                    dataset: {"trial_id": value[0], "mean_mse_4h": value[1]}
                    for dataset, value in sorted(validation_best.items())
                },
                "formal_test_authorized_after_manifest": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
