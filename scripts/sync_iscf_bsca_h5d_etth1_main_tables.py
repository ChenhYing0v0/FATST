#!/usr/bin/env python3
"""Replace the frozen ETTh1 paper row with the author-selected H5D profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TARGET_DATASET = "ETTh1"
SELECTED_TRIAL = "ETTh1__h5d_bs16_lr2p4"
HORIZONS = {"96", "192", "336", "720"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-scorecard", type=Path, required=True)
    parser.add_argument("--h5d-scorecard", type=Path, required=True)
    parser.add_argument("--h5d-manifest", type=Path, required=True)
    parser.add_argument("--old-main-ii-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
    h5d_scorecard = read_csv(args.h5d_scorecard)
    manifest = read_csv(args.h5d_manifest)
    old_main_ii = read_csv(args.old_main_ii_cells)

    selected = [
        row
        for row in h5d_scorecard
        if row["dataset"] == TARGET_DATASET
        and row["trial_id"] == SELECTED_TRIAL
    ]
    if len(selected) != 4 or {row["horizon"] for row in selected} != HORIZONS:
        raise RuntimeError("H5D replacement must contain exactly four ETTh1 rows")

    hashes = {
        row["trial_id"]: row["checkpoint_sha256_before_test"]
        for row in manifest
    }
    if SELECTED_TRIAL not in hashes:
        raise RuntimeError("selected H5D checkpoint is absent from the manifest")

    replacement = [
        {
            **row,
            "checkpoint_sha256": hashes[SELECTED_TRIAL],
            "test_tuned": "True",
            "test_informed": "True",
        }
        for row in selected
    ]
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
    merged_scorecard = [
        row for row in old_scorecard if row["dataset"] != TARGET_DATASET
    ] + replacement
    merged_scorecard.sort(key=lambda row: (row["dataset"], int(row["horizon"])))
    if len(merged_scorecard) != 32:
        raise RuntimeError("merged Main I scorecard must contain 32 cells")

    output_scorecard = args.output_dir / "selected_main_scorecard_h5d_bs16_lr2p4.csv"
    write_csv(output_scorecard, merged_scorecard, scorecard_fields)

    replacement_lookup = {row["horizon"]: row for row in replacement}
    merged_main_ii: list[dict[str, str]] = []
    replaced_cells = 0
    for row in old_main_ii:
        if (
            row["system"] == "ISCF-BSCA-MAIN-v1"
            and row["dataset"] == TARGET_DATASET
        ):
            h5d = replacement_lookup[row["horizon"]]
            row = {**row, "mse": h5d["test_mse"], "mae": h5d["test_mae"]}
            replaced_cells += 1
        merged_main_ii.append(row)
    if replaced_cells != 4 or len(merged_main_ii) != 224:
        raise RuntimeError("Main II replacement must update 4 of 224 cells")

    output_main_ii = args.output_dir / "main_ii_aggregate_cells_h5d_bs16_lr2p4.csv"
    write_csv(output_main_ii, merged_main_ii, list(old_main_ii[0]))

    unchanged_main_i = [
        row for row in old_scorecard if row["dataset"] != TARGET_DATASET
    ]
    unchanged_merged_main_i = [
        row for row in merged_scorecard if row["dataset"] != TARGET_DATASET
    ]
    if unchanged_main_i != unchanged_merged_main_i:
        raise RuntimeError("non-ETTh1 Main I rows changed")
    unchanged_main_ii = [
        row
        for row in old_main_ii
        if not (
            row["system"] == "ISCF-BSCA-MAIN-v1"
            and row["dataset"] == TARGET_DATASET
        )
    ]
    unchanged_merged_main_ii = [
        row
        for row in merged_main_ii
        if not (
            row["system"] == "ISCF-BSCA-MAIN-v1"
            and row["dataset"] == TARGET_DATASET
        )
    ]
    if unchanged_main_ii != unchanged_merged_main_ii:
        raise RuntimeError("non-target Main II rows changed")

    audit = {
        "gate": "pass",
        "aggregate_cells": 224,
        "matrix_complete": True,
        "replacement_scope": [TARGET_DATASET],
        "selected_trial": SELECTED_TRIAL,
        "selection_role": "author_selected_eligible_H5D_profile_after_formal_test",
        "updated_main_i_cells": 4,
        "updated_main_ii_cells": 4,
        "all_baseline_and_non_ETTh1_cells_unchanged": True,
        "source_hashes": {
            "old_scorecard": sha256(args.old_scorecard),
            "h5d_scorecard": sha256(args.h5d_scorecard),
            "h5d_manifest": sha256(args.h5d_manifest),
            "old_main_ii_cells": sha256(args.old_main_ii_cells),
            "merged_scorecard": sha256(output_scorecard),
            "merged_main_ii_cells": sha256(output_main_ii),
        },
        "claim_boundary": (
            "The author-selected eligible H5D ETTh1 profile replaces the H5A "
            "ETTh1 paper row. ECL, Solar, all other datasets, and every baseline "
            "cell remain unchanged. This selection is test-tuned and test-informed."
        ),
    }
    (args.output_dir / "h5d_bs16_lr2p4_main_table_sync_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "h5d_main_table_sync=pass scorecard_cells=32 "
        "main_ii_cells=224 replaced_cells=4"
    )


if __name__ == "__main__":
    main()
