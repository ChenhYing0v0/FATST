#!/usr/bin/env python3
"""Deep post-hoc audit for the completed SC-D20-CST-v1 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ARMS = (
    "A6_MEASURE_RETRAIN",
    "A6_CST_SPEC",
    "A6_CST_RANDOM",
)
COMPARISONS = (
    ("transfer_spec_vs_a6", "A6_CST_SPEC", "A6_MEASURE_RETRAIN"),
    ("specificity_spec_vs_random", "A6_CST_SPEC", "A6_CST_RANDOM"),
    ("capacity_random_vs_a6", "A6_CST_RANDOM", "A6_MEASURE_RETRAIN"),
)
METRICS = ("mse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d20_cst_step7b.json"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str) -> Path:
    return root / "runs" / arm / dataset / "h720_full" / "seed2021"


def gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def load_split_metrics(
    root: Path,
    datasets: list[str],
    horizons: list[int],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    values: dict[tuple[str, str, str, int], dict[str, float]] = {}
    split_files = {
        "validation": "metrics_by_target_horizon.csv",
        "test": "test_audit_metrics_by_target_horizon.csv",
    }
    for split, filename in split_files.items():
        for arm in ARMS:
            for dataset in datasets:
                rows = read_csv(run_dir(root, arm, dataset) / filename)
                lookup = {int(row["target_horizon"]): row for row in rows}
                for horizon in horizons:
                    row = lookup[horizon]
                    values[(split, arm, dataset, horizon)] = {
                        metric: float(row[metric]) for metric in METRICS
                    }
    return values


def aggregate_standard(
    values: dict[tuple[str, str, str, int], dict[str, float]],
    datasets: list[str],
    horizons: list[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    by_dataset: list[dict[str, Any]] = []
    by_horizon: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for comparison, candidate, reference in COMPARISONS:
            for metric in METRICS:
                gains: dict[tuple[str, int], float] = {}
                for dataset in datasets:
                    for horizon in horizons:
                        candidate_value = values[
                            (split, candidate, dataset, horizon)
                        ][metric]
                        reference_value = values[
                            (split, reference, dataset, horizon)
                        ][metric]
                        value = gain(candidate_value, reference_value)
                        gains[(dataset, horizon)] = value
                        cells.append(
                            {
                                "split": split,
                                "comparison": comparison,
                                "metric": metric,
                                "dataset": dataset,
                                "horizon": horizon,
                                "candidate_value": candidate_value,
                                "reference_value": reference_value,
                                "gain_percent": value,
                            }
                        )
                dataset_means = {
                    dataset: mean(gains[(dataset, horizon)] for horizon in horizons)
                    for dataset in datasets
                }
                horizon_means = {
                    horizon: mean(gains[(dataset, horizon)] for dataset in datasets)
                    for horizon in horizons
                }
                summaries.append(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        "macro_gain_percent": mean(gains.values()),
                        "cell_wins": sum(value > 0.0 for value in gains.values()),
                        "dataset_wins": sum(
                            value > 0.0 for value in dataset_means.values()
                        ),
                        "horizon_wins": sum(
                            value > 0.0 for value in horizon_means.values()
                        ),
                    }
                )
                by_dataset.extend(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        "dataset": dataset,
                        "gain_percent": value,
                        "horizon_wins": sum(
                            gains[(dataset, horizon)] > 0.0 for horizon in horizons
                        ),
                    }
                    for dataset, value in dataset_means.items()
                )
                by_horizon.extend(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        "horizon": horizon,
                        "gain_percent": value,
                        "dataset_wins": sum(
                            gains[(dataset, horizon)] > 0.0 for dataset in datasets
                        ),
                    }
                    for horizon, value in horizon_means.items()
                )
    return cells, summaries, by_dataset, by_horizon


def checkpoint_audit(root: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for dataset in datasets:
            training = read_csv(run_dir(root, arm, dataset) / "training_log.csv")
            scores = [float(row["val_mean_mse"]) for row in training]
            best_index = min(range(len(scores)), key=scores.__getitem__)
            last = training[-1]
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "epochs_ran": len(training),
                    "best_epoch_recomputed": best_index + 1,
                    "best_epoch_reported": int(last["best_epoch_so_far"]),
                    "best_val_mean_mse": scores[best_index],
                    "last_val_mean_mse": scores[-1],
                    "last_vs_best_degradation_percent": 100.0
                    * (scores[-1] / scores[best_index] - 1.0),
                    "stopped_early": len(training) < 20,
                    "stop_triggered": int(last["stop_triggered"]),
                }
            )
    return rows


def load_dense_test(
    root: Path,
    datasets: list[str],
) -> dict[tuple[str, str, int], dict[str, float]]:
    values = {}
    for arm in ARMS:
        for dataset in datasets:
            rows = read_csv(
                run_dir(root, arm, dataset)
                / "test_audit_metrics_by_target_horizon.csv"
            )
            if len(rows) != 720:
                raise ValueError(f"expected 720 dense horizons: {arm}/{dataset}")
            for row in rows:
                horizon = int(row["target_horizon"])
                values[(arm, dataset, horizon)] = {
                    metric: float(row[metric]) for metric in METRICS
                }
    return values


def dense_audit(
    values: dict[tuple[str, str, int], dict[str, float]],
    datasets: list[str],
    bins: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_horizon = []
    bin_rows = []
    for comparison, candidate, reference in COMPARISONS:
        for metric in METRICS:
            horizon_gains = {}
            for horizon in range(1, 721):
                gains = [
                    gain(
                        values[(candidate, dataset, horizon)][metric],
                        values[(reference, dataset, horizon)][metric],
                    )
                    for dataset in datasets
                ]
                horizon_gains[horizon] = mean(gains)
                by_horizon.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "horizon": horizon,
                        "macro_gain_percent": mean(gains),
                        "dataset_wins": sum(value > 0.0 for value in gains),
                    }
                )
            for item in bins:
                start = int(item["start"]) + 1
                end = int(item["end"])
                selected = [horizon_gains[h] for h in range(start, end + 1)]
                bin_rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "bin": item["name"],
                        "start_horizon": start,
                        "end_horizon": end,
                        "macro_gain_percent": mean(selected),
                        "horizon_wins": sum(value > 0.0 for value in selected),
                    }
                )
    return by_horizon, bin_rows


def internal_intensity(
    root: Path,
    standard_by_dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    health = read_csv(root / "analysis" / "internal_health_by_dataset.csv")
    gains = {
        (row["comparison"], row["dataset"]): float(row["gain_percent"])
        for row in standard_by_dataset
        if row["split"] == "test" and row["metric"] == "mse"
    }
    rows = []
    for row in health:
        dataset = row["dataset"]
        spec_contribution = float(row["spec_prediction_contribution_std"])
        random_contribution = float(row["random_prediction_contribution_std"])
        rows.append(
            {
                "dataset": dataset,
                "spec_prediction_contribution_std": spec_contribution,
                "random_prediction_contribution_std": random_contribution,
                "spec_to_random_contribution_ratio": spec_contribution
                / max(random_contribution, 1e-12),
                "spec_random_prediction_nrmse": float(
                    row["spec_random_prediction_nrmse"]
                ),
                "transfer_test_mse_gain_percent": gains[
                    ("transfer_spec_vs_a6", dataset)
                ],
                "specificity_test_mse_gain_percent": gains[
                    ("specificity_spec_vs_random", dataset)
                ],
            }
        )
    return rows


def summary_row(
    summaries: list[dict[str, Any]],
    split: str,
    comparison: str,
    metric: str,
) -> dict[str, Any]:
    return next(
        row
        for row in summaries
        if row["split"] == split
        and row["comparison"] == comparison
        and row["metric"] == metric
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = list(config["datasets"])
    horizons = list(config["matrix"]["horizons"])
    values = load_split_metrics(args.raw_root, datasets, horizons)
    cells, summaries, by_dataset, by_horizon = aggregate_standard(
        values,
        datasets,
        horizons,
    )
    checkpoints = checkpoint_audit(args.raw_root, datasets)
    dense_values = load_dense_test(args.raw_root, datasets)
    dense_horizon, dense_bins = dense_audit(
        dense_values,
        datasets,
        config["diagnostic_protocol"]["future_bins"],
    )
    intensity = internal_intensity(args.raw_root, by_dataset)

    transfer_val = summary_row(
        summaries, "validation", "transfer_spec_vs_a6", "mse"
    )
    transfer_test = summary_row(
        summaries, "test", "transfer_spec_vs_a6", "mse"
    )
    specificity_val = summary_row(
        summaries, "validation", "specificity_spec_vs_random", "mse"
    )
    specificity_test = summary_row(
        summaries, "test", "specificity_spec_vs_random", "mse"
    )
    transfer_reversal = bool(
        transfer_val["macro_gain_percent"] > 0.0
        and transfer_test["macro_gain_percent"] < 0.0
    )
    threshold = config["effectiveness_gates"]["specificity_spec_vs_random"]
    specificity_directional_only = bool(
        specificity_test["macro_gain_percent"] > 0.0
        and specificity_test["macro_gain_percent"]
        < threshold["mse_macro_gain_percent_min"]
    )

    write_csv(args.output_dir / "standard_cells.csv", cells)
    write_csv(args.output_dir / "split_comparison_summary.csv", summaries)
    write_csv(args.output_dir / "comparison_by_dataset.csv", by_dataset)
    write_csv(args.output_dir / "comparison_by_horizon.csv", by_horizon)
    write_csv(args.output_dir / "checkpoint_audit.csv", checkpoints)
    write_csv(args.output_dir / "dense_comparison_by_horizon.csv", dense_horizon)
    write_csv(args.output_dir / "dense_bin_summary.csv", dense_bins)
    write_csv(args.output_dir / "internal_intensity_vs_gain.csv", intensity)

    remote_summary = json.loads(
        (args.raw_root / "analysis" / "step9_summary.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "candidate_version": config["candidate_version"],
        "matrix_complete": remote_summary["matrix_complete"],
        "remote_frozen_decision": remote_summary["decision"],
        "standard_test": {
            "transfer_spec_vs_a6_mse": transfer_test,
            "specificity_spec_vs_random_mse": specificity_test,
        },
        "validation": {
            "transfer_spec_vs_a6_mse": transfer_val,
            "specificity_spec_vs_random_mse": specificity_val,
        },
        "transfer_validation_test_reversal": transfer_reversal,
        "specificity_directional_only": specificity_directional_only,
        "internal_health_all_pass": all(
            remote_summary["internal_health"].values()
        ),
        "deep_failure_attribution": (
            "optimization_or_numeric_pathology(validation_test_mismatch)"
            "+intervention_point_wrong"
        ),
        "exact_design_closed": True,
        "direction_rejection_valid": False,
        "confirmation_authorized": False,
        "rollback": "Contribution1 Step2/4 source-informed redesign",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "deep_audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
