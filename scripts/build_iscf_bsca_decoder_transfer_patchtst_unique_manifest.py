#!/usr/bin/env python3
"""Build the 40-unique-checkpoint PatchTST HPO formal-test manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--v2p1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scorecard = read_csv(args.scorecard)
    if len(scorecard) != 50 or any(row["pass"].lower() != "true" for row in scorecard):
        raise RuntimeError("parent PatchTST HPO scorecard is not complete 50/50")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in scorecard:
        grouped.setdefault(row["checkpoint_sha256"], []).append(row)
    if len(grouped) != 40:
        raise RuntimeError(f"expected 40 unique hashes, observed {len(grouped)}")

    v2p1 = json.loads(args.v2p1_config.read_text())
    selected = {
        (row["dataset"], row["profile_id"]): row["bsca_checkpoint_sha256"]
        for row in v2p1["selected_profiles"]
    }
    old_formal_root = Path(v2p1["artifact_contract"]["remote_output_root"])
    rows: list[dict[str, Any]] = []
    for checkpoint_hash, aliases in sorted(
        grouped.items(), key=lambda item: (item[1][0]["dataset"], item[1][0]["profile_id"])
    ):
        aliases = sorted(aliases, key=lambda row: row["profile_id"])
        representative = aliases[0]
        existing = ""
        for alias in aliases:
            key = (alias["dataset"], alias["profile_id"])
            if selected.get(key) == checkpoint_hash:
                existing = str(
                    old_formal_root
                    / "formal_test"
                    / "patchtst_iscf_bsca"
                    / alias["dataset"]
                    / "seed2021"
                )
                break
        rows.append(
            {
                "dataset": representative["dataset"],
                "representative_profile_id": representative["profile_id"],
                "profile_aliases": ";".join(row["profile_id"] for row in aliases),
                "checkpoint": str(Path(representative["run_dir"]) / "checkpoint.pt"),
                "checkpoint_sha256": checkpoint_hash,
                "existing_formal_artifact_dir": existing,
                "seed": 2021,
            }
        )
    if sum(bool(row["existing_formal_artifact_dir"]) for row in rows) != 5:
        raise RuntimeError("expected exactly five reusable v2.1 formal artifacts")
    write_csv(args.output, rows)
    print("patchtst_unique_manifest=pass rows=40 reusable=5 new=35")


if __name__ == "__main__":
    main()
