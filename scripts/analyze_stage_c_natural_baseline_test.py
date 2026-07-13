#!/usr/bin/env python3
"""Aggregate frozen A6-LBF natural-profile test metrics across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DATASETS = ("Weather", "ETTm1", "ETTh2")
HORIZONS = (48, 96, 144, 192, 288, 336, 512, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        row
        for dataset in DATASETS
        for row in read_csv(args.input_root / f"{dataset}_natural_baseline_test_metrics.csv")
    ]
    hashes = {row["contract_hash"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    complete = (
        len(rows) == len(DATASETS) * len(HORIZONS) * 3
        and len(hashes) == 1
        and seeds == {2021, 2022, 2023}
        and all(row["evaluation_split"] == "test" for row in rows)
    )
    aggregated: list[dict[str, Any]] = []
    if complete:
        for dataset in DATASETS:
            profile = next(row["profile"] for row in rows if row["dataset"] == dataset)
            for horizon in HORIZONS:
                selected = [
                    row for row in rows
                    if row["dataset"] == dataset and int(row["target_horizon"]) == horizon
                ]
                mse = [float(row["mse"]) for row in selected]
                mae = [float(row["mae"]) for row in selected]
                aggregated.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "target_horizon": horizon,
                        "seed_count": len(selected),
                        "mse_mean": mean(mse),
                        "mse_sample_std": stdev(mse),
                        "mse_cv": stdev(mse) / mean(mse),
                        "mae_mean": mean(mae),
                        "mae_sample_std": stdev(mae),
                        "mae_cv": stdev(mae) / mean(mae),
                    }
                )
    summary = {
        "candidate": "A6-LBF-natural-baseline",
        "complete": complete,
        "decision": "frozen_test_reference_ready" if complete else "analysis_incomplete",
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "seeds": sorted(seeds),
        "expected_rows": 72,
        "observed_rows": len(rows),
        "contract_hash": next(iter(hashes)) if len(hashes) == 1 else None,
        "selection_used_test": False,
        "test_role": "post_freeze_reference_only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "natural_baseline_test_metrics_by_seed.csv", rows)
    if aggregated:
        write_csv(args.output_dir / "natural_baseline_test_metrics_aggregate.csv", aggregated)
    (args.output_dir / "natural_baseline_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"natural_baseline_test_analysis_done complete={complete} rows={len(rows)}")


if __name__ == "__main__":
    main()
