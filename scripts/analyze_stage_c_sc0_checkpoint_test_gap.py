#!/usr/bin/env python3
"""Aggregate SC0 validation/test best-versus-last checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("Weather", "ETTm1", "ETTh2")


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


def summarize(rows: list[dict[str, str]], scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "comparisons": len(rows),
        "last_mse_wins": sum(float(row["test_last_vs_best_mse"]) < 0.0 for row in rows),
        "last_mae_wins": sum(float(row["test_last_vs_best_mae"]) < 0.0 for row in rows),
        "mean_validation_last_vs_best_mse": mean(
            float(row["validation_last_vs_best_mse"]) for row in rows
        ),
        "mean_test_last_vs_best_mse": mean(
            float(row["test_last_vs_best_mse"]) for row in rows
        ),
        "mean_test_last_vs_best_mae": mean(
            float(row["test_last_vs_best_mae"]) for row in rows
        ),
        "max_test_last_degradation_mse": max(
            float(row["test_last_vs_best_mse"]) for row in rows
        ),
        "max_test_last_improvement_mse": min(
            float(row["test_last_vs_best_mse"]) for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for dataset in DATASETS:
        rows.extend(
            read_csv(args.input_root / f"{dataset}_checkpoint_test_comparisons.csv")
        )
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 comparisons, found {len(rows)}")
    summaries = [summarize(rows, "all_dense_horizons")]
    full = [row for row in rows if int(row["target_horizon"]) == 720]
    summaries.append(summarize(full, "full720_all_datasets_arms"))
    for dataset in DATASETS:
        selected = [row for row in full if row["dataset"] == dataset]
        summaries.append(summarize(selected, f"full720_{dataset}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sc0_checkpoint_test_comparisons.csv", rows)
    write_csv(args.output_dir / "sc0_checkpoint_test_summary.csv", summaries)
    lines = [
        "# StageC SC0 Best-vs-Last Checkpoint Test Diagnostic",
        "",
        "- `role`: diagnostic-only after carrier calibration; test metrics did not select hyperparameters.",
        "- `source`: the original fixed-20 SC0 seed2021 best/last checkpoints; no retraining.",
        "- `comparison`: `(last - best) / best`; negative means last is better.",
        "",
        "| scope | comparisons | last MSE wins | mean val delta | mean test MSE delta | max test degradation | max test improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scope']} | {row['comparisons']} | {row['last_mse_wins']} | "
            f"{row['mean_validation_last_vs_best_mse']:.2%} | "
            f"{row['mean_test_last_vs_best_mse']:.2%} | "
            f"{row['max_test_last_degradation_mse']:.2%} | "
            f"{row['max_test_last_improvement_mse']:.2%} |"
        )
    lines.extend(
        [
            "",
            "This diagnostic distinguishes validation trajectory overfitting from actual test behavior. It cannot retroactively tune the carrier because test was opened only after the prior calibration decision.",
        ]
    )
    (args.output_dir / "sc0_checkpoint_test_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"stage_c_sc0_checkpoint_test_analysis_done rows={len(rows)}")


if __name__ == "__main__":
    main()
