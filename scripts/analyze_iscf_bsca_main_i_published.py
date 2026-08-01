#!/usr/bin/env python3
"""Compare the frozen ISCF-BSCA Main-I row with published Table 6 rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iscf", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> None:
    args = parse_args()
    iscf_rows = read_csv(args.iscf)
    published_rows = read_csv(args.published)
    common_datasets = sorted({row["dataset"] for row in published_rows})
    models = sorted({row["model"] for row in published_rows})
    expected_cells = len(common_datasets) * len(HORIZONS)

    iscf = {
        (row["dataset"], int(row["horizon"])): row
        for row in iscf_rows
        if row["dataset"] in common_datasets
    }
    published = {
        (row["model"], row["dataset"], int(row["horizon"])): row
        for row in published_rows
    }
    if len(iscf) != expected_cells:
        raise ValueError(f"ISCF common matrix incomplete: {len(iscf)}/{expected_cells}")
    if len(published) != len(models) * expected_cells:
        raise ValueError("published matrix incomplete or duplicated")

    pairwise: list[dict[str, object]] = []
    for model in models:
        for dataset in common_datasets:
            for horizon in HORIZONS:
                candidate = iscf[(dataset, horizon)]
                reference = published[(model, dataset, horizon)]
                candidate_mse = float(candidate["test_mse"])
                candidate_mae = float(candidate["test_mae"])
                reference_mse = float(reference["mse"])
                reference_mae = float(reference["mae"])
                pairwise.append(
                    {
                        "reference_model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "iscf_mse": candidate_mse,
                        "reference_mse": reference_mse,
                        "iscf_mse_gain_pct": relative_gain(
                            reference_mse, candidate_mse
                        ),
                        "iscf_mse_better": candidate_mse < reference_mse,
                        "iscf_mae": candidate_mae,
                        "reference_mae": reference_mae,
                        "iscf_mae_gain_pct": relative_gain(
                            reference_mae, candidate_mae
                        ),
                        "iscf_mae_better": candidate_mae < reference_mae,
                    }
                )

    best_pairwise: list[dict[str, object]] = []
    for dataset in common_datasets:
        for horizon in HORIZONS:
            candidate = iscf[(dataset, horizon)]
            references = [
                published[(model, dataset, horizon)] for model in models
            ]
            best_mse_row = min(references, key=lambda row: float(row["mse"]))
            best_mae_row = min(references, key=lambda row: float(row["mae"]))
            candidate_mse = float(candidate["test_mse"])
            candidate_mae = float(candidate["test_mae"])
            best_mse = float(best_mse_row["mse"])
            best_mae = float(best_mae_row["mae"])
            best_pairwise.append(
                {
                    "reference_model": "BestPublishedPerCell",
                    "dataset": dataset,
                    "horizon": horizon,
                    "iscf_mse": candidate_mse,
                    "reference_mse": best_mse,
                    "iscf_mse_gain_pct": relative_gain(best_mse, candidate_mse),
                    "iscf_mse_better": candidate_mse < best_mse,
                    "iscf_mae": candidate_mae,
                    "reference_mae": best_mae,
                    "iscf_mae_gain_pct": relative_gain(best_mae, candidate_mae),
                    "iscf_mae_better": candidate_mae < best_mae,
                    "best_mse_model": best_mse_row["model"],
                    "best_mae_model": best_mae_row["model"],
                }
            )

    summary: list[dict[str, object]] = []
    for model in [*models, "BestPublishedPerCell"]:
        rows = (
            best_pairwise
            if model == "BestPublishedPerCell"
            else [row for row in pairwise if row["reference_model"] == model]
        )
        dataset_mse_wins = 0
        dataset_mae_wins = 0
        horizon_mse_wins = 0
        horizon_mae_wins = 0
        for dataset in common_datasets:
            subset = [row for row in rows if row["dataset"] == dataset]
            dataset_mse_wins += sum(float(row["iscf_mse"]) for row in subset) < sum(
                float(row["reference_mse"]) for row in subset
            )
            dataset_mae_wins += sum(float(row["iscf_mae"]) for row in subset) < sum(
                float(row["reference_mae"]) for row in subset
            )
        for horizon in HORIZONS:
            subset = [row for row in rows if row["horizon"] == horizon]
            horizon_mse_wins += sum(float(row["iscf_mse"]) for row in subset) < sum(
                float(row["reference_mse"]) for row in subset
            )
            horizon_mae_wins += sum(float(row["iscf_mae"]) for row in subset) < sum(
                float(row["reference_mae"]) for row in subset
            )
        iscf_mse = sum(float(row["iscf_mse"]) for row in rows) / len(rows)
        reference_mse = sum(float(row["reference_mse"]) for row in rows) / len(rows)
        iscf_mae = sum(float(row["iscf_mae"]) for row in rows) / len(rows)
        reference_mae = sum(float(row["reference_mae"]) for row in rows) / len(rows)
        summary.append(
            {
                "reference_model": model,
                "cells": len(rows),
                "iscf_mean_mse": iscf_mse,
                "reference_mean_mse": reference_mse,
                "iscf_mse_gain_pct": relative_gain(reference_mse, iscf_mse),
                "mse_cell_wins": sum(row["iscf_mse_better"] for row in rows),
                "mse_dataset_mean_wins": dataset_mse_wins,
                "mse_horizon_mean_wins": horizon_mse_wins,
                "iscf_mean_mae": iscf_mae,
                "reference_mean_mae": reference_mae,
                "iscf_mae_gain_pct": relative_gain(reference_mae, iscf_mae),
                "mae_cell_wins": sum(row["iscf_mae_better"] for row in rows),
                "mae_dataset_mean_wins": dataset_mae_wins,
                "mae_horizon_mean_wins": horizon_mae_wins,
            }
        )

    write_csv(args.output_dir / "iscf_vs_published_pairwise.csv", pairwise)
    write_csv(args.output_dir / "iscf_vs_best_published_per_cell.csv", best_pairwise)
    write_csv(args.output_dir / "iscf_vs_published_summary.csv", summary)
    print(
        f"wrote {len(pairwise)} pairwise rows, {len(best_pairwise)} best-cell rows, "
        f"and {len(summary)} summary rows"
    )


if __name__ == "__main__":
    main()
