#!/usr/bin/env python3
"""Freeze the forty H4J checkpoints before direct official-test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BY_DATASET = {
    "ETTh1": 2,
    "ETTh2": 2,
    "ETTm1": 2,
    "ETTm2": 9,
    "Weather": 9,
    "ECL": 4,
    "Solar": 10,
    "Exchange": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_main_v1_hpo_20260731"
        / "h4j_artifact_audit"
        / "trial_ledger.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-output-root",
        default=(
            "/home/yingch/exp_outputs/r-2026-fatst/"
            "iscf_bsca_main_v1_hpo/h4j/test_audit"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    ledger_path = args.ledger.resolve()
    ledger = read_jsonl(ledger_path)
    if len(ledger) != 40:
        raise ValueError(f"expected forty H4J rows, found {len(ledger)}")
    if Counter(entry["dataset"] for entry in ledger) != Counter(
        EXPECTED_BY_DATASET
    ):
        raise ValueError("H4J dataset profile counts differ from the contract")
    rows = []
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
                "phase": "H4J",
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
                    / entry["dataset"]
                    / entry["trial_id"]
                    / "seed2021"
                ),
                "source_ledger": str(ledger_path.relative_to(ROOT)),
            }
        )
    if len({row["trial_id"] for row in rows}) != 40:
        raise ValueError("H4J trial IDs are not unique")
    rows.sort(key=lambda row: (row["dataset"], row["trial_id"]))
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
                "profiles_per_dataset": EXPECTED_BY_DATASET,
                "manifest_sha256": sha256(args.output),
                "test_artifacts_before_launch": "requires_remote_preflight",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
