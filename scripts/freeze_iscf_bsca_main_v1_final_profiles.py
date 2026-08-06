#!/usr/bin/env python3
"""Audit and materialize the terminal ISCF-BSCA-MAIN-v1 profile freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis" / "iscf_bsca_main_v1_hpo_20260731"
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_selected_profiles.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unique_by_trial(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in result and result[trial_id] != row:
            raise ValueError(f"conflicting {label} rows for {trial_id}")
        result[trial_id] = row
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scorecard_path = ROOT / config["selected_scorecard"]["path"]
    if sha256(scorecard_path) != config["selected_scorecard"]["sha256"]:
        raise ValueError("selected scorecard hash mismatch")
    cells = read_csv(scorecard_path)

    manifests: list[dict[str, Any]] = []
    for path in sorted(BASE.glob("*_checkpoint_manifest.csv")):
        manifests.extend(read_csv(path))
    training_ledgers: list[dict[str, Any]] = []
    for path in sorted(BASE.glob("**/trial_ledger.jsonl")):
        training_ledgers.extend(read_jsonl(path))
    test_ledgers: list[dict[str, Any]] = []
    for path in sorted(BASE.glob("**/test_audit_ledger.jsonl")):
        test_ledgers.extend(read_jsonl(path))

    manifest_by_trial = unique_by_trial(manifests, "checkpoint manifest")
    training_by_trial = unique_by_trial(training_ledgers, "training ledger")
    test_by_trial = unique_by_trial(test_ledgers, "test ledger")
    cells_by_key = {
        (row["dataset"], row["trial_id"], int(row["horizon"])): row
        for row in cells
    }

    frozen_cells: list[dict[str, Any]] = []
    frozen_profiles: list[dict[str, Any]] = []
    for dataset, profile in config["profiles"].items():
        trial_id = profile["trial_id"]
        manifest = manifest_by_trial[trial_id]
        training = training_by_trial[trial_id]
        test = test_by_trial[trial_id]
        before_hash = manifest["checkpoint_sha256_before_test"]
        if before_hash != profile["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {trial_id}")
        if test["checkpoint_sha256_after_test"] != before_hash:
            raise ValueError(f"checkpoint mutated during test: {trial_id}")
        if not bool(test["checkpoint_hash_immutable"]):
            raise ValueError(f"checkpoint immutability false: {trial_id}")
        if not bool(test["matrix_complete"]):
            raise ValueError(f"test matrix incomplete: {trial_id}")

        profile_cells = []
        for horizon in HORIZONS:
            row = cells_by_key[(dataset, trial_id, horizon)]
            mse = float(row["test_mse"])
            mae = float(row["test_mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise ValueError(f"non-finite score: {dataset}/{horizon}")
            profile_cells.append((mse, mae))
            frozen_cells.append(
                {
                    "dataset": dataset,
                    "phase": profile["phase"],
                    "trial_id": trial_id,
                    "profile_id": manifest["profile_id"],
                    "seed": config["seed"],
                    "horizon": horizon,
                    "test_mse": mse,
                    "test_mae": mae,
                    "checkpoint_sha256": before_hash,
                    "test_tuned": True,
                    "test_informed": True,
                }
            )
        mean_mse = sum(value[0] for value in profile_cells) / len(HORIZONS)
        mean_mae = sum(value[1] for value in profile_cells) / len(HORIZONS)
        if not math.isclose(mean_mse, float(profile["test_mean_mse_4h"]), abs_tol=1e-12):
            raise ValueError(f"mean MSE mismatch: {dataset}")
        if not math.isclose(mean_mae, float(profile["test_mean_mae_4h"]), abs_tol=1e-12):
            raise ValueError(f"mean MAE mismatch: {dataset}")
        frozen_profiles.append(
            {
                "dataset": dataset,
                "phase": profile["phase"],
                "trial_id": trial_id,
                "profile_id": manifest["profile_id"],
                "seed": config["seed"],
                "best_epoch": manifest["best_epoch"],
                "validation_mean_mse_4h": manifest["validation_mean_mse_4h"],
                "test_mean_mse_4h": mean_mse,
                "test_mean_mae_4h": mean_mae,
                "trainable_parameters": manifest["trainable_parameters"],
                "checkpoint_sha256_before_test": before_hash,
                "checkpoint_sha256_after_test": test["checkpoint_sha256_after_test"],
                "checkpoint_hash_immutable": test["checkpoint_hash_immutable"],
                "training_artifact_dir": manifest["artifact_dir"],
                "test_artifact_dir": test["artifact_dir"],
                "training_protocol_id": training["protocol_id"],
                "test_protocol_id": test["protocol_id"],
                "candidate_version": test["candidate_version"],
                "test_access_date": test["test_access_date"],
                "matrix_complete": test["matrix_complete"],
            }
        )

    if len(frozen_profiles) != 8 or len(frozen_cells) != 32:
        raise ValueError("terminal freeze must contain 8 profiles and 32 cells")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_output = args.output_dir / "selected_main_scorecard_final.csv"
    manifest_output = args.output_dir / "selected_profile_manifest_final.csv"
    write_csv(scorecard_output, frozen_cells)
    write_csv(manifest_output, frozen_profiles)
    status = {
        "protocol_id": config["protocol_id"],
        "candidate_id": config["candidate_id"],
        "hpo_status": "stopped_and_frozen_by_user",
        "freeze_date": config["date"],
        "datasets": len(frozen_profiles),
        "standard_horizon_cells": len(frozen_cells),
        "seed": config["seed"],
        "checkpoint_hashes_unique": len(
            {row["checkpoint_sha256_before_test"] for row in frozen_profiles}
        ),
        "checkpoint_immutability_pass": True,
        "matrix_complete": True,
        "no_further_hpo_authorized": True,
        "scorecard_sha256": sha256(scorecard_output),
        "profile_manifest_sha256": sha256(manifest_output),
        "decision": "terminal_h4n_selected_profiles_frozen_stop_hpo",
    }
    (args.output_dir / "final_hpo_freeze_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
