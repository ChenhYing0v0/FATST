#!/usr/bin/env python3
"""Audit ISCF-BSCA-MAIN-v1 HPO artifacts and rank complete test trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STANDARD_HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_hpo.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--require-test",
        action="store_true",
        help="Require complete official-test artifacts and freeze profile ranking.",
    )
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
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_map(path: Path) -> dict[int, tuple[float, float]]:
    rows = read_csv(path)
    output = {}
    for row in rows:
        horizon = int(row["target_horizon"])
        if horizon in STANDARD_HORIZONS:
            output[horizon] = (float(row["mse"]), float(row["mae"]))
    return output


def best_training_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty training log: {path}")
    return min(rows, key=lambda row: float(row["val_mean_mse"]))


def run_dir(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "trials"
        / job["dataset"]
        / job["trial_id"]
        / "seed2021"
    )


def finite_metrics(metrics: dict[int, tuple[float, float]]) -> bool:
    return len(metrics) == len(STANDARD_HORIZONS) and all(
        math.isfinite(value)
        for pair in metrics.values()
        for value in pair
    )


def materialize_jobs(
    config: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    base_profiles = config.get("base_profiles", {})
    if not config.get("base_config") and not base_profiles:
        return [dict(job) for job in config["jobs"]]
    base_jobs = {}
    if config.get("base_config"):
        base_path = Path(config["base_config"])
        if not base_path.is_absolute():
            base_path = ROOT / base_path
        base_config = json.loads(base_path.read_text(encoding="utf-8"))
        base_jobs = {job["trial_id"]: job for job in base_config["jobs"]}
    materialized = []
    for specification in config["jobs"]:
        base_profile_id = specification.get("base_profile_id")
        if base_profile_id is not None:
            if base_profile_id not in base_profiles:
                raise ValueError(
                    f"unknown base profile {base_profile_id} in {config_path}"
                )
            job = dict(base_profiles[base_profile_id])
            job.update(specification.get("overrides", {}))
            job.update(
                {
                    key: value
                    for key, value in specification.items()
                    if key not in {"base_profile_id", "overrides"}
                }
            )
            materialized.append(job)
            continue
        base_trial_id = specification.get("base_trial_id")
        if base_trial_id is None:
            materialized.append(dict(specification))
            continue
        if base_trial_id not in base_jobs:
            raise ValueError(
                f"unknown base trial {base_trial_id} in {config_path}"
            )
        job = dict(base_jobs[base_trial_id])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_trial_id", "overrides"}
            }
        )
        materialized.append(job)
    return materialized


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = materialize_jobs(config, args.config)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    ledger_rows = []
    scorecard = []
    aggregate_rows = []
    for job in jobs:
        directory = run_dir(args.output_root, job)
        checkpoint = directory / "checkpoint.pt"
        validation_path = directory / "metrics_by_target_horizon.csv"
        test_path = directory / "test_audit_metrics_by_target_horizon.csv"
        required = [
            checkpoint,
            directory / "training_log.csv",
            validation_path,
            directory / "effective_config.json",
            directory / "initialization_contract.json",
            directory / "model_diagnostics.json",
        ]
        training_complete = all(path.is_file() and path.stat().st_size for path in required)
        validation = metric_map(validation_path) if training_complete else {}
        test = metric_map(test_path) if test_path.is_file() else {}
        validation_complete = finite_metrics(validation)
        test_complete = finite_metrics(test)
        checkpoint_hash = file_sha256(checkpoint) if checkpoint.is_file() else ""
        best = (
            best_training_row(directory / "training_log.csv")
            if training_complete
            else {}
        )
        diagnostics = (
            json.loads(
                (directory / "model_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            if training_complete
            else {}
        )
        ledger = {
            "protocol_id": config["protocol_id"],
            "candidate_id": config["candidate_id"],
            "trial_id": job["trial_id"],
            "profile_id": job["profile_id"],
            "dataset": job["dataset"],
            "seed": 2021,
            "source_prior": job["source_prior"],
            "status": (
                "test_complete"
                if test_complete
                else "validation_complete"
                if validation_complete
                else "missing_or_incomplete"
            ),
            "hyperparameters": job,
            "checkpoint_selector": config["selection_contract"][
                "trial_checkpoint_score"
            ],
            "best_epoch": int(best["epoch"]) if best else None,
            "best_validation_mean_mse_4h": (
                sum(value[0] for value in validation.values()) / 4
                if validation_complete
                else None
            ),
            "checkpoint_sha256_before_test": checkpoint_hash or None,
            "test_mean_mse_4h": (
                sum(value[0] for value in test.values()) / 4
                if test_complete
                else None
            ),
            "test_mean_mae_4h": (
                sum(value[1] for value in test.values()) / 4
                if test_complete
                else None
            ),
            "trainable_parameters": diagnostics.get("trainable_parameters"),
            "numeric_health_pass": validation_complete
            and (test_complete or not args.require_test),
            "artifact_completeness_pass": training_complete
            and (test_complete or not args.require_test),
            "artifact_dir": str(directory),
        }
        ledger_rows.append(ledger)
        for horizon in STANDARD_HORIZONS:
            val_pair = validation.get(horizon, (None, None))
            test_pair = test.get(horizon, (None, None))
            scorecard.append(
                {
                    "dataset": job["dataset"],
                    "trial_id": job["trial_id"],
                    "profile_id": job["profile_id"],
                    "seed": 2021,
                    "horizon": horizon,
                    "val_mse": val_pair[0],
                    "val_mae": val_pair[1],
                    "test_mse": test_pair[0],
                    "test_mae": test_pair[1],
                    "checkpoint_sha256": checkpoint_hash,
                    "validation_complete": validation_complete,
                    "test_complete": test_complete,
                }
            )
        aggregate_rows.append(
            {
                "dataset": job["dataset"],
                "trial_id": job["trial_id"],
                "profile_id": job["profile_id"],
                "validation_mean_mse_4h": ledger[
                    "best_validation_mean_mse_4h"
                ],
                "test_mean_mse_4h": ledger["test_mean_mse_4h"],
                "test_mean_mae_4h": ledger["test_mean_mae_4h"],
                "trainable_parameters": ledger["trainable_parameters"],
                "complete": ledger["artifact_completeness_pass"],
            }
        )
    training_complete_count = sum(
        row["status"] in {"validation_complete", "test_complete"}
        for row in ledger_rows
    )
    test_complete_count = sum(
        row["status"] == "test_complete" for row in ledger_rows
    )
    completeness = {
        "protocol_id": config["protocol_id"],
        "expected_trials": len(jobs),
        "training_complete_trials": training_complete_count,
        "test_complete_trials": test_complete_count,
        "require_test": args.require_test,
        "complete": (
            training_complete_count == len(jobs)
            and (
                not args.require_test
                or test_complete_count == len(jobs)
            )
        ),
    }
    with (args.analysis_dir / "trial_ledger.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in ledger_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(args.analysis_dir / "trial_scorecard.csv", scorecard)
    write_csv(args.analysis_dir / "profile_aggregates.csv", aggregate_rows)
    (args.analysis_dir / "hpo_completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.require_test:
        if not completeness["complete"]:
            raise SystemExit("test ranking blocked by incomplete HPO matrix")
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in aggregate_rows:
            by_dataset.setdefault(row["dataset"], []).append(row)
        ranking_rows = []
        selected = {}
        for dataset, rows in by_dataset.items():
            ranked = sorted(
                rows,
                key=lambda row: (
                    float(row["test_mean_mse_4h"]),
                    float(row["validation_mean_mse_4h"]),
                    int(row["trainable_parameters"]),
                    row["profile_id"],
                ),
            )
            for rank, row in enumerate(ranked, start=1):
                ranking_rows.append({**row, "selection_rank": rank})
            selected[dataset] = {
                "trial_id": ranked[0]["trial_id"],
                "profile_id": ranked[0]["profile_id"],
                "test_mean_mse_4h": ranked[0]["test_mean_mse_4h"],
                "checkpoint_reused_without_retraining": True,
            }
        write_csv(args.analysis_dir / "profile_ranking.csv", ranking_rows)
        (args.analysis_dir / "selected_profiles.json").write_text(
            json.dumps(selected, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(completeness, indent=2))
    if not completeness["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
