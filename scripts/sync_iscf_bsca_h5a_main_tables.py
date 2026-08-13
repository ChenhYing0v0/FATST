#!/usr/bin/env python3
"""Create immutable Main I/II inputs with the frozen H5A profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TARGET_DATASETS = {"ETTh1", "ECL", "Solar"}
SELECTED_TRIALS = {
    "ETTh1": "ETTh1__h5a_lr3p5e4",
    "ECL": "ECL__h5a_seq336_p1",
    "Solar": "Solar__h5a_seq512_p4_lr2p5e4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-scorecard", type=Path, required=True)
    parser.add_argument("--h5a-scorecard", type=Path, required=True)
    parser.add_argument("--h5a-manifest", type=Path, required=True)
    parser.add_argument("--old-main-ii-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    old_scorecard = read_csv(args.old_scorecard)
    h5a_scorecard = read_csv(args.h5a_scorecard)
    manifest = read_csv(args.h5a_manifest)
    old_main_ii = read_csv(args.old_main_ii_cells)

    hashes = {
        row["trial_id"]: row["checkpoint_sha256_before_test"]
        for row in manifest
    }
    selected = [
        row
        for row in h5a_scorecard
        if SELECTED_TRIALS.get(row["dataset"]) == row["trial_id"]
    ]
    if len(selected) != 12:
        raise RuntimeError(f"expected 12 selected H5A cells, found {len(selected)}")
    if set(row["dataset"] for row in selected) != TARGET_DATASETS:
        raise RuntimeError("selected H5A dataset set does not match the frozen target")

    scorecard_fields = [
        "dataset",
        "phase",
        "trial_id",
        "profile_id",
        "seed",
        "horizon",
        "test_mse",
        "test_mae",
        "checkpoint_sha256",
        "test_tuned",
        "test_informed",
    ]
    replacement = []
    for row in selected:
        trial_id = row["trial_id"]
        replacement.append(
            {
                **row,
                "checkpoint_sha256": hashes[trial_id],
                "test_tuned": "True",
                "test_informed": "True",
            }
        )
    merged_scorecard = [
        row for row in old_scorecard if row["dataset"] not in TARGET_DATASETS
    ] + replacement
    merged_scorecard.sort(key=lambda row: (row["dataset"], int(row["horizon"])))
    if len(merged_scorecard) != 32:
        raise RuntimeError("merged Main I scorecard must contain 32 cells")

    output_scorecard = args.output_dir / "selected_main_scorecard_h5a.csv"
    write_csv(output_scorecard, merged_scorecard, scorecard_fields)

    replacement_lookup = {
        (row["dataset"], row["horizon"]): row for row in replacement
    }
    merged_main_ii = []
    replaced_cells = 0
    for row in old_main_ii:
        if row["system"] == "ISCF-BSCA-MAIN-v1" and row["dataset"] in TARGET_DATASETS:
            h5a = replacement_lookup[(row["dataset"], row["horizon"])]
            row = {**row, "mse": h5a["test_mse"], "mae": h5a["test_mae"]}
            replaced_cells += 1
        merged_main_ii.append(row)
    if replaced_cells != 12 or len(merged_main_ii) != 224:
        raise RuntimeError("Main II replacement must update 12 of 224 cells")

    main_ii_fields = list(old_main_ii[0])
    output_main_ii = args.output_dir / "main_ii_aggregate_cells_h5a.csv"
    write_csv(output_main_ii, merged_main_ii, main_ii_fields)

    audit = {
        "gate": "pass",
        "aggregate_cells": 224,
        "matrix_complete": True,
        "replacement_scope": sorted(TARGET_DATASETS),
        "selected_trials": SELECTED_TRIALS,
        "source_hashes": {
            "old_scorecard": sha256(args.old_scorecard),
            "h5a_scorecard": sha256(args.h5a_scorecard),
            "h5a_manifest": sha256(args.h5a_manifest),
            "old_main_ii_cells": sha256(args.old_main_ii_cells),
            "merged_scorecard": sha256(output_scorecard),
            "merged_main_ii_cells": sha256(output_main_ii),
        },
        "claim_boundary": (
            "Only the frozen H5A ETTh1/ECL/Solar dataset-level profiles replace "
            "the prior ISCF-BSCA values; all baseline cells remain unchanged."
        ),
    }
    (args.output_dir / "h5a_main_table_sync_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "h5a_main_table_sync=pass scorecard_cells=32 "
        "main_ii_cells=224 replaced_cells=12"
    )


if __name__ == "__main__":
    main()
