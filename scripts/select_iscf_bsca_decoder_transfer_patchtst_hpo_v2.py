#!/usr/bin/env python3
"""Select one validation-only PatchTST decoder profile per dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_patchtst_hpo_v2.json"),
    )
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = read_rows(args.scorecard)
    by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in config["datasets"]}
    profiles = {row["id"]: row for row in config["search_profiles"]}
    for row in rows:
        if row["pass"].lower() != "true":
            continue
        profile = profiles[row["profile_id"]]
        by_dataset[row["dataset"]].append(
            {
                "dataset": row["dataset"],
                "profile_id": row["profile_id"],
                "validation_mean_mse": float(row["validation_mean_mse"]),
                "mode_rank": int(row["mode_rank"]),
                "readout_lr_multiplier": float(row["readout_lr_multiplier"]),
                "readout_weight_decay": float(row["readout_weight_decay"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )

    selected: list[dict[str, Any]] = []
    reference_values = config["reference_profile"]["validation_mean_mse_by_dataset"]
    reference_ranks = config["reference_profile"]["mode_rank_by_dataset"]
    for dataset in config["datasets"]:
        candidates = by_dataset[dataset] + [
            {
                "dataset": dataset,
                "profile_id": config["reference_profile"]["id"],
                "validation_mean_mse": reference_values[dataset],
                "mode_rank": reference_ranks[dataset],
                "readout_lr_multiplier": 1.0,
                "readout_weight_decay": 0.0,
                "checkpoint_sha256": "reused_v1_reference",
            }
        ]
        winner = min(
            candidates,
            key=lambda row: (
                row["validation_mean_mse"],
                row["mode_rank"],
                row["readout_lr_multiplier"],
                row["profile_id"],
            ),
        )
        reference = reference_values[dataset]
        winner["reference_validation_mean_mse"] = reference
        winner["gain_percent_over_reference"] = 100.0 * (reference - winner["validation_mean_mse"]) / reference
        selected.append(winner)

    macro_reference = sum(reference_values.values()) / len(reference_values)
    macro_selected = sum(row["validation_mean_mse"] for row in selected) / len(selected)
    macro_gain = 100.0 * (macro_reference - macro_selected) / macro_reference
    improved_datasets = sum(row["gain_percent_over_reference"] > 0.1 for row in selected)
    gate = config["selection"]
    passed = (
        macro_gain > gate["minimum_macro_validation_mse_gain_percent_over_v1_reference_exclusive"]
        and improved_datasets >= gate["minimum_datasets_with_validation_mse_gain_over_0p1_percent"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "selected_profiles.csv", selected)
    result = {
        "pass": passed,
        "macro_reference_validation_mean_mse": macro_reference,
        "macro_selected_validation_mean_mse": macro_selected,
        "macro_gain_percent": macro_gain,
        "datasets_improved_over_0p1_percent": improved_datasets,
        "formal_test_authorized": False,
        "next": config["next_gate"]["if_pass"] if passed else config["next_gate"]["if_fail"],
    }
    (args.output_dir / "selection_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"patchtst_decoder_hpo_v2_selection={'pass' if passed else 'fail'} "
        f"macro_gain={macro_gain:.4f}% datasets={improved_datasets}/5 test=0"
    )


if __name__ == "__main__":
    main()
