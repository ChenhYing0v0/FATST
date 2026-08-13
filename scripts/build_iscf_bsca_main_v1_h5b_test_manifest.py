#!/usr/bin/env python3
"""Freeze a complete H5B/H5C/H5D checkpoint block before official-test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("H5B", "H5C", "H5D"), default="H5B"
    )
    parser.add_argument("--expected-trials", type=int, default=36)
    parser.add_argument(
        "--test-output-root",
        default=(
            "/home/yingch/exp_outputs/r-2026-fatst/"
            "iscf_bsca_main_v1_hpo/h5b/test_audit"
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
    if (
        len(ledger) != args.expected_trials
        or {entry["dataset"] for entry in ledger} != {"ETTh1"}
    ):
        raise ValueError(
            f"{args.phase} manifest requires exactly "
            f"{args.expected_trials} ETTh1 rows"
        )

    rows = []
    for entry in ledger:
        if entry["status"] != "validation_complete":
            raise ValueError(f"unexpected trial status: {entry['trial_id']}")
        if not entry["artifact_completeness_pass"] or not entry["numeric_health_pass"]:
            raise ValueError(f"invalid training artifact: {entry['trial_id']}")
        checkpoint = entry["checkpoint_sha256_before_test"]
        if not checkpoint:
            raise ValueError(f"missing checkpoint hash: {entry['trial_id']}")
        rows.append(
            {
                "phase": args.phase,
                "dataset": "ETTh1",
                "trial_id": entry["trial_id"],
                "profile_id": entry["profile_id"],
                "seed": entry["seed"],
                "best_epoch": entry["best_epoch"],
                "validation_mean_mse_4h": entry["best_validation_mean_mse_4h"],
                "trainable_parameters": entry["trainable_parameters"],
                "checkpoint_sha256_before_test": checkpoint,
                "artifact_dir": entry["artifact_dir"],
                "test_artifact_dir": str(
                    Path(args.test_output_root)
                    / "ETTh1"
                    / entry["trial_id"]
                    / "seed2021"
                ),
                "source_ledger": str(ledger_path.relative_to(ROOT)),
            }
        )

    if len({row["trial_id"] for row in rows}) != args.expected_trials:
        raise ValueError(f"{args.phase} trial IDs are not unique")
    if (
        len({row["checkpoint_sha256_before_test"] for row in rows})
        != args.expected_trials
    ):
        raise ValueError(f"{args.phase} checkpoint hashes are not unique")
    rows.sort(key=lambda row: row["trial_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "trials": args.expected_trials,
                "profiles_per_dataset": {"ETTh1": args.expected_trials},
                "unique_checkpoint_hashes": args.expected_trials,
                "manifest_sha256": sha256(args.output),
                "formal_test_authorized": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
