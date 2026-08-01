#!/usr/bin/env python3
"""Freeze the combined H1/H2 checkpoint manifest before official test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = {
    dataset: index
    for index, dataset in enumerate(
        (
            "ECL",
            "Solar",
            "Weather",
            "ETTm1",
            "ETTm2",
            "ETTh1",
            "ETTh2",
            "Exchange",
        )
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h1-ledger",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_main_v1_hpo_20260731"
        / "h1_artifact_audit"
        / "trial_ledger.jsonl",
    )
    parser.add_argument(
        "--h2-ledger",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_main_v1_hpo_20260731"
        / "h2_artifact_audit"
        / "trial_ledger.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-output-root",
        default=(
            "/home/yingch/exp_outputs/r-2026-fatst/"
            "iscf_bsca_main_v1_hpo/test_audit"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rows = []
    for phase, ledger_path, expected in (
        ("H1", args.h1_ledger, 16),
        ("H2", args.h2_ledger, 24),
    ):
        ledger = read_jsonl(ledger_path)
        if len(ledger) != expected:
            raise ValueError(
                f"{phase} expected {expected} rows, found {len(ledger)}"
            )
        for entry in ledger:
            if not entry["artifact_completeness_pass"]:
                raise ValueError(f"incomplete artifact: {entry['trial_id']}")
            if not entry["numeric_health_pass"]:
                raise ValueError(f"numeric failure: {entry['trial_id']}")
            checkpoint = entry["checkpoint_sha256_before_test"]
            if not checkpoint:
                raise ValueError(f"missing checkpoint hash: {entry['trial_id']}")
            rows.append(
                {
                    "phase": phase,
                    "dataset": entry["dataset"],
                    "trial_id": entry["trial_id"],
                    "profile_id": entry["profile_id"],
                    "seed": entry["seed"],
                    "best_epoch": entry["best_epoch"],
                    "validation_mean_mse_4h": entry[
                        "best_validation_mean_mse_4h"
                    ],
                    "trainable_parameters": entry["trainable_parameters"],
                    "checkpoint_sha256_before_test": checkpoint,
                    "artifact_dir": entry["artifact_dir"],
                    "test_artifact_dir": str(
                        Path(args.test_output_root)
                        / phase.lower()
                        / entry["dataset"]
                        / entry["trial_id"]
                        / "seed2021"
                    ),
                    "source_ledger": str(ledger_path.relative_to(ROOT)),
                }
            )
    if len(rows) != 40 or len({row["trial_id"] for row in rows}) != 40:
        raise ValueError("combined HPO manifest must contain 40 unique trials")
    counts = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    if counts != {dataset: 5 for dataset in DATASET_ORDER}:
        raise ValueError(f"expected five trials per dataset: {counts}")
    rows.sort(
        key=lambda row: (
            DATASET_ORDER[row["dataset"]],
            row["phase"],
            row["trial_id"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "trials": len(rows),
                "datasets": len(counts),
                "trials_per_dataset": sorted(set(counts.values())),
                "manifest_sha256": file_sha256(args.output),
                "test_artifacts_before_launch": "requires_remote_preflight",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
