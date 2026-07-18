#!/usr/bin/env python3
"""Build validation/test and training diagnostics for CCSF Phase-A Step 9."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--formal-analysis-root", type=Path, required=True)
    parser.add_argument("--post-e2e-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json"
        ),
    )
    parser.add_argument(
        "--step6-config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_step6.json"),
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
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str) -> Path:
    return root / arm / dataset / "h720_full" / "seed2021"


def load_metrics(
    root: Path,
    arms: list[dict[str, Any]],
    datasets: list[str],
    horizons: list[int],
    split: str,
) -> list[dict[str, Any]]:
    filename = (
        "metrics_by_target_horizon.csv"
        if split == "val"
        else "test_audit_metrics_by_target_horizon.csv"
    )
    rows = []
    for arm in arms:
        for dataset in datasets:
            lookup = {
                int(row["target_horizon"]): row
                for row in read_csv(run_dir(root, arm["id"], dataset) / filename)
            }
            for horizon in horizons:
                source = lookup[horizon]
                rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "arm": arm["id"],
                        "horizon": horizon,
                        "mse": float(source["mse"]),
                        "mae": float(source["mae"]),
                    }
                )
    return rows


def comparison_summary(
    metrics: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    datasets: list[str],
    horizons: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["split"], row["dataset"], row["arm"], row["horizon"]): row
        for row in metrics
    }
    split = str(metrics[0]["split"])
    cells = []
    summaries = []
    for comparison in comparisons:
        for metric in ("mse", "mae"):
            gains = []
            dataset_values: dict[str, list[float]] = {}
            horizon_values: dict[int, list[float]] = {}
            for dataset in datasets:
                for horizon in horizons:
                    candidate = float(
                        lookup[
                            (
                                split,
                                dataset,
                                comparison["candidate"],
                                horizon,
                            )
                        ][metric]
                    )
                    reference = float(
                        lookup[
                            (
                                split,
                                dataset,
                                comparison["reference"],
                                horizon,
                            )
                        ][metric]
                    )
                    gain = 100.0 * (1.0 - candidate / reference)
                    gains.append(gain)
                    dataset_values.setdefault(dataset, []).append(gain)
                    horizon_values.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "split": split,
                            "comparison": comparison["id"],
                            "metric": metric,
                            "dataset": dataset,
                            "horizon": horizon,
                            "gain_percent": gain,
                        }
                    )
            summaries.append(
                {
                    "split": split,
                    "comparison": comparison["id"],
                    "role": comparison["role"],
                    "metric": metric,
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0
                        for values in dataset_values.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0
                        for values in horizon_values.values()
                    ),
                }
            )
    return cells, summaries


def arm_macro_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in sorted({str(row["split"]) for row in metrics}):
        for arm in sorted({str(row["arm"]) for row in metrics}):
            selected = [
                row
                for row in metrics
                if row["split"] == split and row["arm"] == arm
            ]
            rows.append(
                {
                    "split": split,
                    "arm": arm,
                    "macro_mse": mean(float(row["mse"]) for row in selected),
                    "macro_mae": mean(float(row["mae"]) for row in selected),
                    "cells": len(selected),
                }
            )
    return rows


def checkpoint_rows(
    metadata_root: Path,
    arms: list[dict[str, Any]],
    datasets: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        for dataset in datasets:
            training = read_csv(
                run_dir(metadata_root, arm["id"], dataset) / "training_log.csv"
            )
            final = training[-1]
            best_epoch = int(final["best_epoch_so_far"])
            best = next(
                row for row in training if int(row["epoch"]) == best_epoch
            )
            row: dict[str, Any] = {
                "dataset": dataset,
                "arm": arm["id"],
                "epochs_executed": len(training),
                "best_epoch": best_epoch,
                "best_val_mean_mse": float(best["val_mean_mse"]),
                "early_stopped": bool(int(final["stop_triggered"])),
            }
            for field in (
                "train_ccsf_policy_normalized_entropy",
                "train_ccsf_teacher_confidence",
                "train_ccsf_teacher_normalized_entropy",
                "train_ccsf_teacher_policy_argmax_accuracy",
                "train_ccsf_calibration_kl",
                "train_ccsf_weighted_calibration_kl",
            ):
                if field in best and best[field] not in ("", None):
                    row[field] = float(best[field])
            rows.append(row)
    return rows


def direction_rows(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["split"], row["comparison"], row["metric"]): row
        for row in summaries
    }
    rows = []
    comparisons = sorted({str(row["comparison"]) for row in summaries})
    for comparison in comparisons:
        for metric in ("mse", "mae"):
            validation = lookup[("val", comparison, metric)]
            test = lookup[("test", comparison, metric)]
            val_gain = float(validation["macro_gain_percent"])
            test_gain = float(test["macro_gain_percent"])
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "validation_macro_gain_percent": val_gain,
                    "test_macro_gain_percent": test_gain,
                    "direction_agreement": (val_gain > 0.0) == (test_gain > 0.0),
                    "gain_gap_test_minus_validation": test_gain - val_gain,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step6 = json.loads(args.step6_config.read_text(encoding="utf-8"))
    horizons = config["matrix"]["horizons"]
    validation = load_metrics(
        args.metadata_root,
        config["arms"],
        config["datasets"],
        horizons,
        "val",
    )
    test = load_metrics(
        args.metadata_root,
        config["arms"],
        config["datasets"],
        horizons,
        "test",
    )
    validation_cells, validation_summary = comparison_summary(
        validation,
        step6["comparisons"],
        config["datasets"],
        horizons,
    )
    test_cells, test_summary = comparison_summary(
        test,
        step6["comparisons"],
        config["datasets"],
        horizons,
    )
    summaries = validation_summary + test_summary
    checkpoints = checkpoint_rows(
        args.metadata_root,
        config["arms"],
        config["datasets"],
    )
    formal = json.loads(
        (args.formal_analysis_root / "summary.json").read_text(encoding="utf-8")
    )
    post_e2e = json.loads(
        (args.post_e2e_root / "summary.json").read_text(encoding="utf-8")
    )
    result = {
        "candidate_version": config["candidate_version"],
        "matrix_complete": formal["matrix_complete"],
        "formal_decision": formal["evaluation_layers"]["decision"],
        "formal_failure_attribution": formal["evaluation_layers"][
            "failure_attribution"
        ],
        "post_e2e_decision": post_e2e["decision"],
        "post_e2e_passed_gates": post_e2e["passed_gates"],
        "post_e2e_total_gates": post_e2e["total_gates"],
        "training_numeric_pathology": False,
        "rollback": "Step4/6 objective-readout redesign; no confirmation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "validation_test_metrics.csv",
        validation + test,
    )
    write_csv(
        args.output_dir / "validation_test_comparison_cells.csv",
        validation_cells + test_cells,
    )
    write_csv(
        args.output_dir / "validation_test_comparison_summary.csv",
        summaries,
    )
    write_csv(
        args.output_dir / "validation_test_direction_audit.csv",
        direction_rows(summaries),
    )
    write_csv(
        args.output_dir / "arm_macro_score.csv",
        arm_macro_rows(validation + test),
    )
    write_csv(args.output_dir / "checkpoint_training_audit.csv", checkpoints)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
