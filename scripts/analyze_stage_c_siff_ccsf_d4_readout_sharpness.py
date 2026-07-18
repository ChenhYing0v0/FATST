#!/usr/bin/env python3
"""Test whether probability softness blocks contrast-to-mixture utility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from analyze_stage_c_siff_ccsf_d2_granularity import (
    arm_losses,
    bins_for_width,
    deterministic_subset,
    feature_tensors,
)
from analyze_stage_c_siff_ccsf_post_e2e import (
    bin_summary,
    fit_predict,
    gain_percent,
    mixture_mse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage_c_siff_ccsf_d4_readout_sharpness_diagnostic.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/"
            "d4_readout_sharpness_diagnostic"
        ),
    )
    return parser.parse_args()


def sharpen(probability: np.ndarray, exponent: float | str) -> np.ndarray:
    if exponent == "hard":
        output = np.zeros_like(probability)
        indices = np.argmax(probability, axis=-1)
        np.put_along_axis(output, indices[..., None], 1.0, axis=-1)
        return output
    powered = np.power(np.clip(probability, 1e-12, 1.0), float(exponent))
    return powered / np.sum(powered, axis=-1, keepdims=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(
    dataset: str, path: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arms = payload["probe_arms"].astype(np.float64)
        targets = payload["probe_targets"].astype(np.float64)
        base_logits = payload["probe_base_logits"].astype(np.float64)
        contrast = payload["probe_contrast_descriptor"].astype(np.float64)
    row_count = arms.shape[0]
    folds = (
        np.arange(row_count) < row_count // 2,
        np.arange(row_count) >= row_count // 2,
    )
    exponents: list[float | str] = list(config["probability_exponents"])
    if config["include_hard_argmax"]:
        exponents.append("hard")
    rows = []
    for width in config["pooling_widths"]:
        bins = bins_for_width(int(width), int(config["domain_length"]))
        losses = arm_losses(arms, targets, bins)
        base_feature, full = feature_tensors(base_logits, contrast, bins)
        contrast_summary = bin_summary(contrast, bins)
        bin_count, arm_count = losses.shape[1:]
        for fold, train_rows in enumerate(folds):
            test_rows = ~train_rows
            labels = np.argmin(losses[train_rows], axis=-1).reshape(-1)
            train_x = full[train_rows].reshape(-1, full.shape[-1])
            train_x, train_y = deterministic_subset(
                train_x,
                labels,
                int(config["maximum_training_regions_per_fold"]),
            )
            true_probability = fit_predict(
                train_x,
                train_y,
                full[test_rows].reshape(-1, full.shape[-1]),
                arm_count,
                config["classifier"],
            ).reshape(-1, bin_count, arm_count)

            generator = np.random.default_rng(31091 + 17 * int(width) + fold)
            shuffled_train = contrast_summary[train_rows][
                generator.permutation(np.sum(train_rows))
            ]
            shuffled_test = contrast_summary[test_rows][
                generator.permutation(np.sum(test_rows))
            ]
            shuffled_train_x = np.concatenate(
                (base_feature[train_rows], shuffled_train), axis=-1
            ).reshape(-1, full.shape[-1])
            shuffled_train_x, shuffled_y = deterministic_subset(
                shuffled_train_x,
                labels,
                int(config["maximum_training_regions_per_fold"]),
            )
            shuffled_probability = fit_predict(
                shuffled_train_x,
                shuffled_y,
                np.concatenate(
                    (base_feature[test_rows], shuffled_test), axis=-1
                ).reshape(-1, full.shape[-1]),
                arm_count,
                config["classifier"],
            ).reshape(-1, bin_count, arm_count)

            for exponent in exponents:
                true_mse = mixture_mse(
                    arms[test_rows],
                    targets[test_rows],
                    sharpen(true_probability, exponent),
                    bins,
                )
                shuffled_mse = mixture_mse(
                    arms[test_rows],
                    targets[test_rows],
                    sharpen(shuffled_probability, exponent),
                    bins,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "width": width,
                        "fold": fold,
                        "exponent": exponent,
                        "true_contrast_mixture_mse": true_mse,
                        "shuffled_contrast_mixture_mse": shuffled_mse,
                        "true_over_shuffled_gain_pct": gain_percent(
                            true_mse, shuffled_mse
                        ),
                    }
                )
    return rows


def summarize(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = {
        (row["dataset"], int(row["width"]), int(row["fold"])): float(
            row["true_contrast_mixture_mse"]
        )
        for row in rows
        if str(row["exponent"]) == "1.0"
    }
    for row in rows:
        row["gain_over_exponent_1_pct"] = gain_percent(
            float(row["true_contrast_mixture_mse"]),
            baseline[(row["dataset"], int(row["width"]), int(row["fold"]))],
        )

    summaries = []
    exponent_values = list(config["probability_exponents"])
    if config["include_hard_argmax"]:
        exponent_values.append("hard")
    for width in config["pooling_widths"]:
        for exponent in exponent_values:
            selected = [
                row
                for row in rows
                if int(row["width"]) == int(width)
                and str(row["exponent"]) == str(exponent)
            ]
            dataset_gains = {
                dataset: mean(
                    float(row["gain_over_exponent_1_pct"])
                    for row in selected
                    if row["dataset"] == dataset
                )
                for dataset in config["datasets"]
            }
            summaries.append(
                {
                    "width": width,
                    "exponent": exponent,
                    "macro_gain_over_exponent_1_pct": mean(
                        dataset_gains.values()
                    ),
                    "positive_datasets": sum(
                        value > 0.0 for value in dataset_gains.values()
                    ),
                    "macro_true_over_shuffled_gain_pct": mean(
                        float(row["true_over_shuffled_gain_pct"])
                        for row in selected
                    ),
                }
            )

    gate = config["diagnostic_gate"]
    eligible = [
        row
        for row in summaries
        if int(row["width"]) in gate["eligible_widths"]
        and str(row["exponent"]) != "1.0"
    ]
    passing = [
        row
        for row in eligible
        if float(row["macro_gain_over_exponent_1_pct"])
        >= gate["minimum_macro_gain_over_exponent_1_pct"]
        and int(row["positive_datasets"])
        >= gate["minimum_positive_datasets"]
        and float(row["macro_true_over_shuffled_gain_pct"])
        >= gate["minimum_true_over_shuffled_gain_pct"]
    ]
    best = max(
        eligible,
        key=lambda row: float(row["macro_gain_over_exponent_1_pct"]),
    )
    passed = bool(passing)
    payload = {
        "diagnostic_id": config["diagnostic_id"],
        "test_derived": True,
        "best_global_diagnostic_arm": best,
        "passing_global_arms": passing,
        "gate_pass": passed,
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
    root = Path(config["raw_root"])
    for dataset in config["datasets"]:
        path = (
            root
            / dataset
            / "h720_full"
            / f"seed{config['seed']}"
            / "pcsd_test_audit_diagnostics.npz"
        )
        rows.extend(analyze_dataset(dataset, path, config))
    summaries, payload = summarize(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "readout_by_fold.csv", rows)
    write_csv(args.output_dir / "readout_summary.csv", summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
