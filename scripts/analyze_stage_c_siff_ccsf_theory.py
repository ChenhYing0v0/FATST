#!/usr/bin/env python3
"""Test whether target-free SIFF arm contrasts predict relative competence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_step5_theory.json"),
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


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def segment_summary(values: np.ndarray) -> np.ndarray:
    """Summarize ``[N,S,L]`` values as ``[N,4*S]`` target-free features."""
    mean = values.mean(axis=-1)
    std = values.std(axis=-1)
    rms = np.sqrt(np.mean(values**2, axis=-1))
    slope = values[..., -1] - values[..., 0]
    return np.concatenate((mean, std, rms, slope), axis=-1)


def feature_sets(
    arms: np.ndarray,
    bins: list[list[int]],
) -> dict[str, np.ndarray]:
    row_count, _scope_count, length = arms.shape
    coordinates = []
    consensus_features = []
    contrast_features = []
    consensus = arms.mean(axis=1, keepdims=True)
    contrast = arms - consensus
    for start_one, end_one in bins:
        start, end = start_one - 1, end_one
        coordinates.append(
            np.tile(
                np.asarray(
                    [
                        0.5 * (start_one + end_one) / length,
                        (end_one - start_one + 1) / length,
                        np.log1p(end_one) / np.log1p(length),
                    ],
                    dtype=np.float64,
                ),
                (row_count, 1),
            )
        )
        consensus_features.append(
            segment_summary(consensus[..., start:end])
        )
        contrast_features.append(segment_summary(contrast[..., start:end]))

    coordinate = np.stack(coordinates, axis=1)
    consensus_summary = np.stack(consensus_features, axis=1)
    contrast_summary = np.stack(contrast_features, axis=1)
    return {
        "coordinate_only": coordinate,
        "consensus_plus_coordinate": np.concatenate(
            (coordinate, consensus_summary),
            axis=-1,
        ),
        "contrast_plus_coordinate": np.concatenate(
            (coordinate, contrast_summary),
            axis=-1,
        ),
        "full_summary": np.concatenate(
            (coordinate, consensus_summary, contrast_summary),
            axis=-1,
        ),
    }


def expand_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
    scope_count: int,
) -> np.ndarray:
    result = np.zeros((probabilities.shape[0], scope_count), dtype=np.float64)
    result[:, classes.astype(int)] = probabilities
    return result


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    scope_count: int,
    classifier: dict[str, Any],
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    model = LogisticRegression(
        C=float(classifier["regularization_c"]),
        max_iter=int(classifier["max_iterations"]),
        class_weight=classifier["class_weight"],
        multi_class="multinomial",
        random_state=0,
    )
    model.fit(scaled_train, train_y)
    return expand_probabilities(
        model.predict_proba(scaled_test),
        model.classes_,
        scope_count,
    )


def evaluate_probabilities(
    probabilities: np.ndarray,
    losses: np.ndarray,
) -> dict[str, float]:
    expected = float(np.mean(np.sum(probabilities * losses, axis=-1)))
    uniform = float(np.mean(losses))
    return {
        "best_arm_accuracy": float(
            np.mean(np.argmax(probabilities, axis=-1) == np.argmin(losses, axis=-1))
        ),
        "expected_arm_mse": expected,
        "allocation_gain_over_uniform_percent": gain_percent(expected, uniform),
    }


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def teacher_geometry(
    dataset: str,
    losses: np.ndarray,
    temperatures: list[float],
) -> list[dict[str, Any]]:
    scope_count = losses.shape[-1]
    centered = losses - losses.mean(axis=-1, keepdims=True)
    standard_scale = np.maximum(losses.std(axis=-1, keepdims=True), 1e-6)
    relative_scale = np.maximum(losses.mean(axis=-1, keepdims=True), 1e-6)
    relative_dispersion = (standard_scale / relative_scale).squeeze(-1)
    teachers = [("pcc_std", 1.0, softmax(-centered / standard_scale))]
    teachers.extend(
        (
            "relative_regret",
            float(temperature),
            softmax(-centered / (relative_scale * float(temperature))),
        )
        for temperature in temperatures
    )
    rows = []
    for teacher, temperature, probability in teachers:
        entropy = -np.sum(
            probability * np.log(np.clip(probability, 1e-12, None)),
            axis=-1,
        ) / np.log(scope_count)
        confidence = 1.0 - entropy
        rows.append(
            {
                "dataset": dataset,
                "teacher": teacher,
                "temperature": temperature,
                "normalized_entropy": float(np.mean(entropy)),
                "confidence_weight_mean": float(np.mean(confidence)),
                "probability_max_mean": float(
                    np.mean(np.max(probability, axis=-1))
                ),
                "relative_error_dispersion_mean": float(
                    np.mean(relative_dispersion)
                ),
                "confidence_dispersion_correlation": float(
                    np.corrcoef(confidence.reshape(-1), relative_dispersion.reshape(-1))[
                        0, 1
                    ]
                ),
            }
        )
    return rows


def analyze_dataset(
    dataset: str,
    path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arms = payload["probe_arms"].astype(np.float64)
        targets = payload["probe_targets"].astype(np.float64)
        saved_losses = payload["arm_row_bin_mse"][: arms.shape[0]].astype(
            np.float64
        )
        saved_policy = payload["policy_row_bin_usage"][: arms.shape[0]].astype(
            np.float64
        )
    if arms.shape[0] != int(config["probe_rows_per_dataset"]):
        raise ValueError(f"unexpected probe rows for {dataset}: {arms.shape}")

    recomputed_losses = []
    for start_one, end_one in config["future_bins"]:
        residual = arms[..., start_one - 1 : end_one] - targets[
            :, None, start_one - 1 : end_one
        ]
        recomputed_losses.append(np.mean(residual**2, axis=-1))
    losses = np.stack(recomputed_losses, axis=1)
    if not np.allclose(losses, saved_losses, rtol=1e-5, atol=1e-6):
        raise ValueError(f"probe/bin loss mismatch for {dataset}")

    features = feature_sets(arms, config["future_bins"])
    row_count, bin_count, scope_count = losses.shape
    fold_masks = [
        np.arange(row_count) < row_count // 2,
        np.arange(row_count) >= row_count // 2,
    ]
    rows: list[dict[str, Any]] = []
    for fold, train_rows in enumerate(fold_masks):
        test_rows = ~train_rows
        train_y = np.argmin(losses[train_rows], axis=-1).reshape(-1)
        test_losses = losses[test_rows].reshape(-1, scope_count)
        test_policy = saved_policy[test_rows].reshape(-1, scope_count)
        for feature_name, values in features.items():
            train_x = values[train_rows].reshape(-1, values.shape[-1])
            test_x = values[test_rows].reshape(-1, values.shape[-1])
            probabilities = fit_predict(
                train_x,
                train_y,
                test_x,
                scope_count,
                config["classifier"],
            )
            metrics = evaluate_probabilities(probabilities, test_losses)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "feature_set": feature_name,
                    **metrics,
                }
            )

        generator = np.random.default_rng(17041 + fold)
        source = features["contrast_plus_coordinate"]
        shuffled_train = source[train_rows].copy()
        shuffled_test = source[test_rows].copy()
        contrast_start = 3
        shuffled_train[..., contrast_start:] = shuffled_train[
            generator.permutation(shuffled_train.shape[0])
        ][..., contrast_start:]
        shuffled_test[..., contrast_start:] = shuffled_test[
            generator.permutation(shuffled_test.shape[0])
        ][..., contrast_start:]
        probabilities = fit_predict(
            shuffled_train.reshape(-1, shuffled_train.shape[-1]),
            train_y,
            shuffled_test.reshape(-1, shuffled_test.shape[-1]),
            scope_count,
            config["classifier"],
        )
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "feature_set": "shuffled_contrast_plus_coordinate",
                **evaluate_probabilities(probabilities, test_losses),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "feature_set": "existing_policy",
                **evaluate_probabilities(test_policy, test_losses),
            }
        )
    return rows


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["fold"], row["feature_set"]): row for row in rows
    }
    comparisons = {
        "contrast_vs_coordinate": (
            "contrast_plus_coordinate",
            "coordinate_only",
        ),
        "contrast_vs_shuffled": (
            "contrast_plus_coordinate",
            "shuffled_contrast_plus_coordinate",
        ),
        "contrast_vs_existing_policy": (
            "contrast_plus_coordinate",
            "existing_policy",
        ),
        "full_vs_contrast": ("full_summary", "contrast_plus_coordinate"),
    }
    result = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for fold in sorted({int(row["fold"]) for row in rows}):
            for name, (candidate, reference) in comparisons.items():
                candidate_row = lookup[(dataset, fold, candidate)]
                reference_row = lookup[(dataset, fold, reference)]
                result.append(
                    {
                        "comparison": name,
                        "dataset": dataset,
                        "fold": fold,
                        "expected_mse_gain_percent": gain_percent(
                            float(candidate_row["expected_arm_mse"]),
                            float(reference_row["expected_arm_mse"]),
                        ),
                        "accuracy_gain_points": float(
                            candidate_row["best_arm_accuracy"]
                        )
                        - float(reference_row["best_arm_accuracy"]),
                    }
                )
    return result


def summarize(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    feature_summary = {}
    for feature in sorted({str(row["feature_set"]) for row in rows}):
        selected = [row for row in rows if row["feature_set"] == feature]
        feature_summary[feature] = {
            "best_arm_accuracy": float(
                np.mean([row["best_arm_accuracy"] for row in selected])
            ),
            "allocation_gain_over_uniform_percent": float(
                np.mean(
                    [row["allocation_gain_over_uniform_percent"] for row in selected]
                )
            ),
        }

    comparison_summary = {}
    for comparison in sorted({str(row["comparison"]) for row in comparisons}):
        selected = [row for row in comparisons if row["comparison"] == comparison]
        comparison_summary[comparison] = {
            "expected_mse_gain_percent": float(
                np.mean([row["expected_mse_gain_percent"] for row in selected])
            ),
            "accuracy_gain_points": float(
                np.mean([row["accuracy_gain_points"] for row in selected])
            ),
            "positive_expected_mse_folds": sum(
                row["expected_mse_gain_percent"] > 0.0 for row in selected
            ),
            "positive_accuracy_folds": sum(
                row["accuracy_gain_points"] > 0.0 for row in selected
            ),
            "folds": len(selected),
        }

    gates = config["diagnostic_gates"]
    versus_coordinate = comparison_summary["contrast_vs_coordinate"]
    versus_shuffled = comparison_summary["contrast_vs_shuffled"]
    versus_policy = comparison_summary["contrast_vs_existing_policy"]
    gate_results = {
        "allocation_gain_over_coordinate_macro": (
            versus_coordinate["expected_mse_gain_percent"]
            >= gates["allocation_gain_over_coordinate_macro_percent_min"]
        ),
        "allocation_gain_over_coordinate_folds": (
            versus_coordinate["positive_expected_mse_folds"]
            >= gates["allocation_gain_over_coordinate_positive_folds_min"]
        ),
        "allocation_gain_over_shuffled_macro": (
            versus_shuffled["expected_mse_gain_percent"]
            >= gates["allocation_gain_over_shuffled_macro_percent_min"]
        ),
        "allocation_gain_over_shuffled_folds": (
            versus_shuffled["positive_expected_mse_folds"]
            >= gates["allocation_gain_over_shuffled_positive_folds_min"]
        ),
        "accuracy_gain_over_existing_policy": (
            versus_policy["accuracy_gain_points"]
            >= gates["best_arm_accuracy_gain_over_existing_policy_points_min"]
        ),
    }
    return {
        "feature_summary": feature_summary,
        "comparison_summary": comparison_summary,
        "gate_results": gate_results,
        "passed_gates": sum(gate_results.values()),
        "total_gates": len(gate_results),
        "decision": (
            "contrast_identifiability_supported"
            if all(gate_results.values())
            else "exact_low_dimensional_contrast_descriptor_not_supported"
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    teacher_rows = []
    for dataset in config["datasets"]:
        path = (
            args.raw_root
            / "siff_equal"
            / dataset
            / "h720_full"
            / f"seed{config['seed']}"
            / "pcsd_test_audit_diagnostics.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(analyze_dataset(dataset, path, config))
        with np.load(path, allow_pickle=False) as payload:
            teacher_rows.extend(
                teacher_geometry(
                    dataset,
                    payload["arm_row_bin_mse"].astype(np.float64),
                    config["relative_regret_temperature_diagnostic_grid"],
                )
            )
    comparisons = comparison_rows(rows)
    summary = summarize(rows, comparisons, config)
    payload = {
        "candidate_version": config["candidate_version"],
        "config": str(args.config),
        "definitions": {
            "contrast_features": (
                "Per-arm mean, standard deviation, RMS, and end-minus-start "
                "slope after subtracting the five-arm pointwise consensus, "
                "plus three fixed future-bin coordinates. No target is used."
            ),
            "expected_arm_mse": (
                "Actual held-out arm MSE averaged under classifier probabilities."
            ),
            "crossfit": (
                "Fit on 128 saved probe rows and evaluate on the other 128, "
                "then reverse the row halves."
            ),
        },
        "boundary": config["decision_boundary"],
        **summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "contrast_predictability_by_fold.csv", rows)
    write_csv(args.output_dir / "contrast_comparisons.csv", comparisons)
    write_csv(args.output_dir / "teacher_geometry.csv", teacher_rows)
    payload["teacher_geometry_macro"] = {
        f"{teacher}_t{temperature}": {
            "normalized_entropy": float(
                np.mean(
                    [
                        row["normalized_entropy"]
                        for row in teacher_rows
                        if row["teacher"] == teacher
                        and row["temperature"] == temperature
                    ]
                )
            ),
            "confidence_weight_mean": float(
                np.mean(
                    [
                        row["confidence_weight_mean"]
                        for row in teacher_rows
                        if row["teacher"] == teacher
                        and row["temperature"] == temperature
                    ]
                )
            ),
            "probability_max_mean": float(
                np.mean(
                    [
                        row["probability_max_mean"]
                        for row in teacher_rows
                        if row["teacher"] == teacher
                        and row["temperature"] == temperature
                    ]
                )
            ),
            "confidence_dispersion_correlation": float(
                np.mean(
                    [
                        row["confidence_dispersion_correlation"]
                        for row in teacher_rows
                        if row["teacher"] == teacher
                        and row["temperature"] == temperature
                    ]
                )
            ),
        }
        for teacher, temperature in sorted(
            {
                (str(row["teacher"]), float(row["temperature"]))
                for row in teacher_rows
            }
        )
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
