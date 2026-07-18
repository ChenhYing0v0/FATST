#!/usr/bin/env python3
"""Audit whether CCSF contrast competence emerges after region aggregation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from analyze_stage_c_siff_ccsf_post_e2e import (
    bin_coordinates,
    bin_summary,
    evaluate_bin_policy,
    fit_predict,
    gain_percent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage_c_siff_ccsf_d2_granularity_diagnostic.json"
        ),
    )
    return parser.parse_args()


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


def bins_for_width(width: int, length: int) -> list[dict[str, Any]]:
    if length % width:
        raise ValueError(f"width {width} does not divide domain {length}")
    return [
        {
            "name": f"w{width}_g{start // width}",
            "start": start,
            "end": start + width,
        }
        for start in range(0, length, width)
    ]


def arm_losses(
    arms: np.ndarray,
    targets: np.ndarray,
    bins: list[dict[str, Any]],
) -> np.ndarray:
    values = []
    for entry in bins:
        start, end = int(entry["start"]), int(entry["end"])
        residual = arms[:, :, start:end] - targets[:, None, start:end]
        values.append(np.mean(np.square(residual), axis=-1))
    return np.stack(values, axis=1)


def deterministic_subset(
    values: np.ndarray,
    labels: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] <= maximum:
        return values, labels
    indices = np.linspace(0, values.shape[0] - 1, maximum, dtype=np.int64)
    return values[indices], labels[indices]


def feature_tensors(
    base_logits: np.ndarray,
    contrast: np.ndarray,
    bins: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = bin_coordinates(bins, base_logits.shape[0], base_logits.shape[1])
    base = np.concatenate(
        (coordinate, bin_summary(base_logits, bins)),
        axis=-1,
    )
    full = np.concatenate(
        (base, bin_summary(contrast, bins)),
        axis=-1,
    )
    return base, full


def analyze_dataset(
    dataset: str,
    path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arms = payload["probe_arms"].astype(np.float64)
        targets = payload["probe_targets"].astype(np.float64)
        base_logits = payload["probe_base_logits"].astype(np.float64)
        contrast = payload["probe_contrast_descriptor"].astype(np.float64)
    row_count = arms.shape[0]
    folds = [
        np.arange(row_count) < row_count // 2,
        np.arange(row_count) >= row_count // 2,
    ]
    rows = []
    for width in config["pooling_widths"]:
        bins = bins_for_width(int(width), int(config["domain_length"]))
        losses = arm_losses(arms, targets, bins)
        base, full = feature_tensors(base_logits, contrast, bins)
        bin_count, scope_count = losses.shape[1:]
        for fold, train_rows in enumerate(folds):
            test_rows = ~train_rows
            train_y = np.argmin(losses[train_rows], axis=-1).reshape(-1)
            base_train = base[train_rows].reshape(-1, base.shape[-1])
            full_train = full[train_rows].reshape(-1, full.shape[-1])
            base_train, base_y = deterministic_subset(
                base_train,
                train_y,
                int(config["maximum_training_regions_per_fold"]),
            )
            full_train, full_y = deterministic_subset(
                full_train,
                train_y,
                int(config["maximum_training_regions_per_fold"]),
            )
            base_probability = fit_predict(
                base_train,
                base_y,
                base[test_rows].reshape(-1, base.shape[-1]),
                scope_count,
                config["classifier"],
            ).reshape(-1, bin_count, scope_count)
            full_probability = fit_predict(
                full_train,
                full_y,
                full[test_rows].reshape(-1, full.shape[-1]),
                scope_count,
                config["classifier"],
            ).reshape(-1, bin_count, scope_count)

            generator = np.random.default_rng(21091 + 17 * int(width) + fold)
            contrast_summary = bin_summary(contrast, bins)
            shuffled_train = contrast_summary[train_rows][
                generator.permutation(np.sum(train_rows))
            ]
            shuffled_test = contrast_summary[test_rows][
                generator.permutation(np.sum(test_rows))
            ]
            shuffled_train_x = np.concatenate(
                (base[train_rows], shuffled_train),
                axis=-1,
            ).reshape(-1, full.shape[-1])
            shuffled_train_x, shuffled_y = deterministic_subset(
                shuffled_train_x,
                train_y,
                int(config["maximum_training_regions_per_fold"]),
            )
            shuffled_test_x = np.concatenate(
                (base[test_rows], shuffled_test),
                axis=-1,
            ).reshape(-1, full.shape[-1])
            shuffled_probability = fit_predict(
                shuffled_train_x,
                shuffled_y,
                shuffled_test_x,
                scope_count,
                config["classifier"],
            ).reshape(-1, bin_count, scope_count)

            metrics = {}
            for name, probability in (
                ("base", base_probability),
                ("true_contrast", full_probability),
                ("shuffled_contrast", shuffled_probability),
            ):
                metrics[name] = evaluate_bin_policy(
                    probability,
                    losses[test_rows],
                    arms[test_rows],
                    targets[test_rows],
                    bins,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "width": width,
                    "regions": bin_count,
                    "fold": fold,
                    "train_regions": min(
                        int(np.sum(train_rows) * bin_count),
                        int(config["maximum_training_regions_per_fold"]),
                    ),
                    "true_over_base_expected_gain_percent": gain_percent(
                        metrics["true_contrast"]["expected_arm_mse"],
                        metrics["base"]["expected_arm_mse"],
                    ),
                    "true_over_base_mixture_gain_percent": gain_percent(
                        metrics["true_contrast"]["mixture_mse"],
                        metrics["base"]["mixture_mse"],
                    ),
                    "true_over_shuffled_expected_gain_percent": gain_percent(
                        metrics["true_contrast"]["expected_arm_mse"],
                        metrics["shuffled_contrast"]["expected_arm_mse"],
                    ),
                    "true_over_shuffled_mixture_gain_percent": gain_percent(
                        metrics["true_contrast"]["mixture_mse"],
                        metrics["shuffled_contrast"]["mixture_mse"],
                    ),
                    "true_best_arm_accuracy": metrics["true_contrast"][
                        "best_arm_accuracy"
                    ],
                }
            )
    return rows


def width_summary(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    gates = config["per_width_gate"]
    for width in config["pooling_widths"]:
        selected = [row for row in rows if int(row["width"]) == int(width)]
        dataset_expected = {
            dataset: mean(
                float(row["true_over_shuffled_expected_gain_percent"])
                for row in selected
                if row["dataset"] == dataset
            )
            for dataset in config["datasets"]
        }
        expected_base = mean(
            float(row["true_over_base_expected_gain_percent"])
            for row in selected
        )
        expected_shuffled = mean(
            float(row["true_over_shuffled_expected_gain_percent"])
            for row in selected
        )
        mixture_shuffled = mean(
            float(row["true_over_shuffled_mixture_gain_percent"])
            for row in selected
        )
        positive_datasets = sum(value > 0.0 for value in dataset_expected.values())
        passed = bool(
            expected_base
            >= gates["contrast_over_base_expected_arm_mse_percent_min"]
            and expected_shuffled
            >= gates["contrast_over_shuffled_expected_arm_mse_percent_min"]
            and mixture_shuffled
            >= gates["contrast_over_shuffled_mixture_mse_percent_min"]
            and positive_datasets >= gates["positive_datasets_min"]
        )
        summaries.append(
            {
                "width": width,
                "regions": 720 // int(width),
                "true_over_base_expected_gain_percent": expected_base,
                "true_over_shuffled_expected_gain_percent": expected_shuffled,
                "true_over_shuffled_mixture_gain_percent": mixture_shuffled,
                "positive_datasets": positive_datasets,
                "mean_best_arm_accuracy": mean(
                    float(row["true_best_arm_accuracy"]) for row in selected
                ),
                "per_width_gate_pass": passed,
            }
        )
    lookup = {int(row["width"]): row for row in summaries}
    pointwise = lookup[1]
    native = [lookup[int(width)] for width in config["scope_native_widths"]]
    passing_native = [row for row in native if row["per_width_gate_pass"]]
    best_native = max(
        native,
        key=lambda row: float(
            row["true_over_shuffled_expected_gain_percent"]
        ),
    )
    diagnostic = config["diagnostic_gate"]
    gate_results = {
        "multiple_scope_native_widths_pass": len(passing_native)
        >= diagnostic["scope_native_widths_passing_min"],
        "region_expected_advantage_over_pointwise": (
            float(best_native["true_over_shuffled_expected_gain_percent"])
            - float(pointwise["true_over_shuffled_expected_gain_percent"])
            >= diagnostic[
                "best_region_expected_gain_over_pointwise_margin_percent_min"
            ]
        ),
        "region_mixture_advantage_over_pointwise": (
            float(best_native["true_over_shuffled_mixture_gain_percent"])
            - float(pointwise["true_over_shuffled_mixture_gain_percent"])
            >= diagnostic[
                "best_region_mixture_gain_over_pointwise_margin_percent_min"
            ]
        ),
    }
    passed = all(gate_results.values())
    payload = {
        "diagnostic_id": config["diagnostic_id"],
        "test_derived": True,
        "gate_results": gate_results,
        "scope_native_widths_passing": len(passing_native),
        "best_diagnostic_width": int(best_native["width"]),
        "passed_gates": sum(gate_results.values()),
        "total_gates": len(gate_results),
        "decision": config["decision_map"]["pass"]
        if passed
        else config["decision_map"]["fail"],
        "boundary": config["boundary"],
    }
    return summaries, payload


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for dataset in config["datasets"]:
        path = (
            args.raw_root
            / dataset
            / "h720_full"
            / f"seed{config['seed']}"
            / "pcsd_test_audit_diagnostics.npz"
        )
        rows.extend(analyze_dataset(dataset, path, config))
    summaries, payload = width_summary(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "granularity_by_fold.csv", rows)
    write_csv(args.output_dir / "granularity_summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
