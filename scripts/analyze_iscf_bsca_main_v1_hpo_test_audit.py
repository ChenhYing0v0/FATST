#!/usr/bin/env python3
"""Audit and rank the complete 40-checkpoint ISCF-BSCA HPO test matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STANDARD_HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_hpo_test_audit.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_main_v1_hpo_20260731"
        / "combined_checkpoint_manifest.csv",
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(args.manifest)
    if manifest_hash != config["checkpoint_manifest"]["sha256"]:
        raise SystemExit("manifest hash differs from the frozen test config")
    manifest = read_csv(args.manifest)
    expected_runs = config["matrix"]["expected_runs"]
    if len(manifest) != expected_runs:
        raise SystemExit(
            f"expected {expected_runs} manifest rows, found {len(manifest)}"
        )
    if Counter(row["dataset"] for row in manifest) != Counter(
        {dataset: 5 for dataset in config["datasets"]}
    ):
        raise SystemExit("manifest is not an 8-dataset by 5-profile matrix")

    errors: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    scorecard_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for row in manifest:
        run_dir = Path(row["artifact_dir"])
        artifact_dir = Path(row["test_artifact_dir"])
        checkpoint = run_dir / "checkpoint.pt"
        metrics_path = artifact_dir / "test_audit_metrics_by_target_horizon.csv"
        invariant_path = artifact_dir / "test_audit_invariants.json"
        npz_path = artifact_dir / "pcsd_test_audit_diagnostics.npz"
        trial_errors = []
        actual_hash = file_sha256(checkpoint) if checkpoint.is_file() else ""
        if actual_hash != row["checkpoint_sha256_before_test"]:
            trial_errors.append("checkpoint_hash_mismatch")
        if not metrics_path.is_file() or not metrics_path.stat().st_size:
            trial_errors.append("missing_test_metrics")
        if not invariant_path.is_file() or not invariant_path.stat().st_size:
            trial_errors.append("missing_test_invariant")
        if not npz_path.is_file() or not npz_path.stat().st_size:
            trial_errors.append("missing_test_diagnostics")

        metrics: dict[int, tuple[float, float]] = {}
        all_metric_rows: list[dict[str, str]] = []
        if metrics_path.is_file():
            all_metric_rows = read_csv(metrics_path)
            try:
                metric_horizons = {
                    int(metric_row["target_horizon"])
                    for metric_row in all_metric_rows
                }
                if len(all_metric_rows) != 720 or metric_horizons != set(
                    range(1, 721)
                ):
                    trial_errors.append("dense_test_horizon_matrix_mismatch")
                if not all(
                    metric_row.get("evaluation_split") == "test"
                    and metric_row.get("candidate_version")
                    == config["candidate_version"]
                    and metric_row.get("hyperparameter_trial_id")
                    == row["trial_id"]
                    and metric_row.get("hyperparameter_profile_id")
                    == row["profile_id"]
                    and int(metric_row.get("seed", -1)) == int(row["seed"])
                    for metric_row in all_metric_rows
                ):
                    trial_errors.append("test_metrics_provenance_mismatch")
                for metric_row in all_metric_rows:
                    horizon = int(metric_row["target_horizon"])
                    mse = float(metric_row["mse"])
                    mae = float(metric_row["mae"])
                    if not math.isfinite(mse) or not math.isfinite(mae):
                        trial_errors.append("non_finite_test_metric")
                        break
                    if horizon in STANDARD_HORIZONS:
                        metrics[horizon] = (mse, mae)
            except (KeyError, TypeError, ValueError):
                trial_errors.append("invalid_test_metrics_schema")
        if set(metrics) != set(STANDARD_HORIZONS):
            trial_errors.append("missing_standard_horizon_metric")

        invariant: dict[str, Any] = {}
        if invariant_path.is_file():
            try:
                invariant = json.loads(invariant_path.read_text(encoding="utf-8"))
                if not (
                    invariant.get("pass") is True
                    and invariant.get("dataset") == row["dataset"]
                    and invariant.get("uses_test_split") is True
                    and invariant.get("test_access_authorized") is True
                    and invariant.get("checkpoint_sha256")
                    == row["checkpoint_sha256_before_test"]
                    and invariant.get("candidate_version")
                    == config["candidate_version"]
                    and invariant.get("hyperparameter_trial_id")
                    == row["trial_id"]
                    and invariant.get("hyperparameter_profile_id")
                    == row["profile_id"]
                    and invariant.get("seed") == int(row["seed"])
                ):
                    trial_errors.append("test_invariant_failure")
            except (json.JSONDecodeError, OSError):
                trial_errors.append("invalid_test_invariant")

        trial_errors = list(dict.fromkeys(trial_errors))
        errors.extend(f"{row['trial_id']}:{error}" for error in trial_errors)
        complete = not trial_errors
        test_mean_mse = (
            sum(metrics[horizon][0] for horizon in STANDARD_HORIZONS) / 4
            if complete
            else None
        )
        test_mean_mae = (
            sum(metrics[horizon][1] for horizon in STANDARD_HORIZONS) / 4
            if complete
            else None
        )
        ledger_rows.append(
            {
                "protocol_id": config["protocol_id"],
                "candidate_version": config["candidate_version"],
                "phase": row["phase"],
                "dataset": row["dataset"],
                "trial_id": row["trial_id"],
                "profile_id": row["profile_id"],
                "seed": int(row["seed"]),
                "checkpoint_sha256_before_test": row[
                    "checkpoint_sha256_before_test"
                ],
                "checkpoint_sha256_after_test": actual_hash,
                "checkpoint_hash_immutable": (
                    actual_hash == row["checkpoint_sha256_before_test"]
                ),
                "checkpoint_retrained": invariant.get("checkpoint_retrained"),
                "test_access_date": invariant.get("test_access_date"),
                "test_role": config["authorization"]["test_role"],
                "test_tuned": config["test_tuned"],
                "hyperparameter_trial_id": row["trial_id"],
                "hyperparameter_selection_rule": config[
                    "hyperparameter_selection"
                ]["primary_score"],
                "matrix_complete": complete,
                "test_mean_mse_4h": test_mean_mse,
                "test_mean_mae_4h": test_mean_mae,
                "errors": trial_errors,
                "artifact_dir": str(artifact_dir),
            }
        )
        for horizon in STANDARD_HORIZONS:
            pair = metrics.get(horizon, (None, None))
            scorecard_rows.append(
                {
                    "phase": row["phase"],
                    "dataset": row["dataset"],
                    "trial_id": row["trial_id"],
                    "profile_id": row["profile_id"],
                    "seed": int(row["seed"]),
                    "horizon": horizon,
                    "test_mse": pair[0],
                    "test_mae": pair[1],
                    "complete": complete,
                    "checkpoint_sha256": actual_hash,
                }
            )
        aggregate_rows.append(
            {
                "phase": row["phase"],
                "dataset": row["dataset"],
                "trial_id": row["trial_id"],
                "profile_id": row["profile_id"],
                "seed": int(row["seed"]),
                "validation_mean_mse_4h": float(
                    row["validation_mean_mse_4h"]
                ),
                "test_mean_mse_4h": test_mean_mse,
                "test_mean_mae_4h": test_mean_mae,
                "trainable_parameters": int(row["trainable_parameters"]),
                "complete": complete,
            }
        )

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    completeness = {
        "protocol_id": config["protocol_id"],
        "candidate_version": config["candidate_version"],
        "manifest_sha256": manifest_hash,
        "expected_trials": expected_runs,
        "complete_trials": sum(row["matrix_complete"] for row in ledger_rows),
        "expected_standard_horizon_cells": expected_runs * 4,
        "complete_standard_horizon_cells": sum(
            row["complete"] for row in scorecard_rows
        ),
        "errors": errors,
        "complete": not errors,
    }
    (args.analysis_dir / "test_audit_completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.analysis_dir / "test_audit_ledger.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in ledger_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(args.analysis_dir / "all_trial_scorecard.csv", scorecard_rows)
    write_csv(args.analysis_dir / "profile_aggregates.csv", aggregate_rows)
    if errors:
        print(json.dumps(completeness, indent=2))
        raise SystemExit("profile ranking blocked by incomplete test matrix")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_dataset[row["dataset"]].append(row)
    ranking_rows = []
    selected: dict[str, Any] = {}
    selected_scorecard = []
    for dataset in config["datasets"]:
        ranked = sorted(
            by_dataset[dataset],
            key=lambda row: (
                float(row["test_mean_mse_4h"]),
                float(row["validation_mean_mse_4h"]),
                int(row["trainable_parameters"]),
                str(row["profile_id"]),
            ),
        )
        if len(ranked) != 5:
            raise SystemExit(f"{dataset} does not have five complete profiles")
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append({**row, "selection_rank": rank})
        winner = ranked[0]
        selected[dataset] = {
            "trial_id": winner["trial_id"],
            "profile_id": winner["profile_id"],
            "phase": winner["phase"],
            "seed": winner["seed"],
            "test_mean_mse_4h": winner["test_mean_mse_4h"],
            "test_mean_mae_4h": winner["test_mean_mae_4h"],
            "validation_mean_mse_4h": winner["validation_mean_mse_4h"],
            "checkpoint_reused_without_test_time_mutation": True,
            "test_tuned": True,
        }
        selected_scorecard.extend(
            row
            for row in scorecard_rows
            if row["trial_id"] == winner["trial_id"]
        )
    write_csv(args.analysis_dir / "profile_ranking.csv", ranking_rows)
    write_csv(args.analysis_dir / "selected_profile_scorecard.csv", selected_scorecard)
    (args.analysis_dir / "selected_profiles.json").write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completeness, indent=2))


if __name__ == "__main__":
    main()
