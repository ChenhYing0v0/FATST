#!/usr/bin/env python3
"""Diagnose post-E2E CCSF contrast sufficiency and learned allocation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
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
        default=Path(
            "configs/stage_c_siff_ccsf_post_e2e_diagnostic.json"
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


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


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
        random_state=int(classifier["random_state"]),
    )
    model.fit(scaled_train, train_y)
    return expand_probabilities(
        model.predict_proba(scaled_test),
        model.classes_,
        scope_count,
    )


def bin_coordinates(
    bins: list[dict[str, Any]],
    row_count: int,
    length: int,
) -> np.ndarray:
    values = []
    for entry in bins:
        start, end = int(entry["start"]), int(entry["end"])
        values.append(
            np.tile(
                np.asarray(
                    [
                        0.5 * (start + end) / length,
                        (end - start) / length,
                        np.log1p(end) / np.log1p(length),
                    ],
                    dtype=np.float64,
                ),
                (row_count, 1),
            )
        )
    return np.stack(values, axis=1)


def bin_summary(
    values: np.ndarray,
    bins: list[dict[str, Any]],
) -> np.ndarray:
    """Return mean/std summaries for ``[N,T,...]`` as ``[N,K,2*...]``."""
    summaries = []
    for entry in bins:
        start, end = int(entry["start"]), int(entry["end"])
        chunk = values[:, start:end]
        summaries.append(
            np.concatenate(
                (
                    chunk.mean(axis=1).reshape(values.shape[0], -1),
                    chunk.std(axis=1).reshape(values.shape[0], -1),
                ),
                axis=-1,
            )
        )
    return np.stack(summaries, axis=1)


def mixture_mse(
    arms: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: list[dict[str, Any]],
) -> float:
    values = []
    for index, entry in enumerate(bins):
        start, end = int(entry["start"]), int(entry["end"])
        forecast = np.sum(
            arms[:, :, start:end]
            * probabilities[:, index, :, None],
            axis=1,
        )
        values.append(np.mean(np.square(forecast - targets[:, start:end])))
    return float(np.mean(values))


def pointwise_mixture_mse(
    arms: np.ndarray,
    targets: np.ndarray,
    policy: np.ndarray,
    bins: list[dict[str, Any]],
) -> float:
    forecast = np.sum(arms.transpose(0, 2, 1) * policy, axis=-1)
    values = [
        np.mean(
            np.square(
                forecast[:, int(entry["start"]): int(entry["end"])]
                - targets[:, int(entry["start"]): int(entry["end"])]
            )
        )
        for entry in bins
    ]
    return float(np.mean(values))


def evaluate_bin_policy(
    probabilities: np.ndarray,
    losses: np.ndarray,
    arms: np.ndarray,
    targets: np.ndarray,
    bins: list[dict[str, Any]],
) -> dict[str, float]:
    expected = float(np.mean(np.sum(probabilities * losses, axis=-1)))
    return {
        "best_arm_accuracy": float(
            np.mean(
                np.argmax(probabilities, axis=-1)
                == np.argmin(losses, axis=-1)
            )
        ),
        "expected_arm_mse": expected,
        "allocation_gain_over_uniform_percent": gain_percent(
            expected,
            float(np.mean(losses)),
        ),
        "mixture_mse": mixture_mse(
            arms,
            targets,
            probabilities,
            bins,
        ),
    }


def existing_policy_metrics(
    policy: np.ndarray,
    losses: np.ndarray,
    arms: np.ndarray,
    targets: np.ndarray,
    bins: list[dict[str, Any]],
) -> dict[str, float]:
    bin_policy = np.stack(
        [
            policy[:, int(entry["start"]): int(entry["end"])].mean(axis=1)
            for entry in bins
        ],
        axis=1,
    )
    metrics = evaluate_bin_policy(
        bin_policy,
        losses,
        arms,
        targets,
        bins,
    )
    metrics["mixture_mse"] = pointwise_mixture_mse(
        arms,
        targets,
        policy,
        bins,
    )
    clipped = np.clip(policy, 1e-12, 1.0)
    metrics["normalized_entropy"] = float(
        np.mean(
            -np.sum(clipped * np.log(clipped), axis=-1)
            / np.log(policy.shape[-1])
        )
    )
    return metrics


def feature_sets(
    base_logits: np.ndarray,
    contrast: np.ndarray,
    bins: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    coordinates = bin_coordinates(bins, base_logits.shape[0], base_logits.shape[1])
    base_summary = bin_summary(base_logits, bins)
    contrast_summary = bin_summary(contrast, bins)
    return {
        "coordinate_only": coordinates,
        "base_logits_plus_coordinate": np.concatenate(
            (coordinates, base_summary),
            axis=-1,
        ),
        "contrast_plus_coordinate": np.concatenate(
            (coordinates, contrast_summary),
            axis=-1,
        ),
        "base_logits_plus_contrast_plus_coordinate": np.concatenate(
            (coordinates, base_summary, contrast_summary),
            axis=-1,
        ),
    }


def analyze_dataset(
    dataset: str,
    path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as payload:
        arms = payload["probe_arms"].astype(np.float64)
        targets = payload["probe_targets"].astype(np.float64)
        losses = payload["arm_row_bin_mse"][: arms.shape[0]].astype(np.float64)
        final_policy = payload["probe_policy"].astype(np.float64)
        base_policy = payload["probe_base_policy"].astype(np.float64)
        base_logits = payload["probe_base_logits"].astype(np.float64)
        correction = payload["probe_correction_logits"].astype(np.float64)
        contrast = payload["probe_contrast_descriptor"].astype(np.float64)
    bins = config["future_bins"]
    features = feature_sets(base_logits, contrast, bins)
    row_count, bin_count, scope_count = losses.shape
    fold_masks = [
        np.arange(row_count) < row_count // 2,
        np.arange(row_count) >= row_count // 2,
    ]
    rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for fold, train_rows in enumerate(fold_masks):
        test_rows = ~train_rows
        train_y = np.argmin(losses[train_rows], axis=-1).reshape(-1)
        test_losses = losses[test_rows]
        test_arms = arms[test_rows]
        test_targets = targets[test_rows]
        for feature_name, values in features.items():
            probabilities = fit_predict(
                values[train_rows].reshape(-1, values.shape[-1]),
                train_y,
                values[test_rows].reshape(-1, values.shape[-1]),
                scope_count,
                config["classifier"],
            ).reshape(-1, bin_count, scope_count)
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "feature_set": feature_name,
                    **evaluate_bin_policy(
                        probabilities,
                        test_losses,
                        test_arms,
                        test_targets,
                        bins,
                    ),
                }
            )

        generator = np.random.default_rng(19071 + fold)
        base = features["base_logits_plus_coordinate"]
        contrast_only = bin_summary(contrast, bins)
        shuffled_train = contrast_only[train_rows][
            generator.permutation(np.sum(train_rows))
        ]
        shuffled_test = contrast_only[test_rows][
            generator.permutation(np.sum(test_rows))
        ]
        train_x = np.concatenate(
            (base[train_rows], shuffled_train),
            axis=-1,
        )
        test_x = np.concatenate((base[test_rows], shuffled_test), axis=-1)
        probabilities = fit_predict(
            train_x.reshape(-1, train_x.shape[-1]),
            train_y,
            test_x.reshape(-1, test_x.shape[-1]),
            scope_count,
            config["classifier"],
        ).reshape(-1, bin_count, scope_count)
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "feature_set": "shuffled_contrast_plus_base",
                **evaluate_bin_policy(
                    probabilities,
                    test_losses,
                    test_arms,
                    test_targets,
                    bins,
                ),
            }
        )
        for name, policy in (
            ("base_policy", base_policy[test_rows]),
            ("final_policy", final_policy[test_rows]),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "feature_set": name,
                    **existing_policy_metrics(
                        policy,
                        test_losses,
                        test_arms,
                        test_targets,
                        bins,
                    ),
                }
            )

        bin_correction = np.stack(
            [
                correction[
                    test_rows,
                    int(entry["start"]): int(entry["end"]),
                ].mean(axis=1)
                for entry in bins
            ],
            axis=1,
        )
        centered_correction = (
            bin_correction - bin_correction.mean(axis=-1, keepdims=True)
        )
        skill = -test_losses
        centered_skill = skill - skill.mean(axis=-1, keepdims=True)
        numerator = np.sum(centered_correction * centered_skill, axis=-1)
        denominator = np.sqrt(
            np.sum(np.square(centered_correction), axis=-1)
            * np.sum(np.square(centered_skill), axis=-1)
        )
        valid = denominator > 1e-12
        alignment_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "correction_skill_centered_alignment": float(
                    np.mean(numerator[valid] / denominator[valid])
                )
                if np.any(valid)
                else 0.0,
                "correction_rms": float(np.sqrt(np.mean(np.square(correction[test_rows])))),
                "policy_shift_l1": float(
                    np.mean(
                        np.sum(
                            np.abs(
                                final_policy[test_rows]
                                - base_policy[test_rows]
                            ),
                            axis=-1,
                        )
                    )
                ),
            }
        )
    return rows, alignment_rows


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["fold"], row["feature_set"]): row for row in rows
    }
    comparisons = {
        "posthoc_contrast_over_base_features": (
            "base_logits_plus_contrast_plus_coordinate",
            "base_logits_plus_coordinate",
        ),
        "posthoc_contrast_over_shuffled": (
            "base_logits_plus_contrast_plus_coordinate",
            "shuffled_contrast_plus_base",
        ),
        "posthoc_contrast_over_final_policy": (
            "base_logits_plus_contrast_plus_coordinate",
            "final_policy",
        ),
        "final_policy_over_base_policy": ("final_policy", "base_policy"),
    }
    result = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for fold in (0, 1):
            for name, (candidate, reference) in comparisons.items():
                candidate_row = lookup[(dataset, fold, candidate)]
                reference_row = lookup[(dataset, fold, reference)]
                result.append(
                    {
                        "comparison": name,
                        "dataset": dataset,
                        "fold": fold,
                        "expected_arm_mse_gain_percent": gain_percent(
                            float(candidate_row["expected_arm_mse"]),
                            float(reference_row["expected_arm_mse"]),
                        ),
                        "mixture_mse_gain_percent": gain_percent(
                            float(candidate_row["mixture_mse"]),
                            float(reference_row["mixture_mse"]),
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
    alignment_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    summaries = {}
    for comparison in sorted({str(row["comparison"]) for row in comparisons}):
        selected = [row for row in comparisons if row["comparison"] == comparison]
        summaries[comparison] = {
            "expected_arm_mse_gain_percent": mean(
                float(row["expected_arm_mse_gain_percent"]) for row in selected
            ),
            "expected_arm_mse_positive_folds": sum(
                float(row["expected_arm_mse_gain_percent"]) > 0.0
                for row in selected
            ),
            "mixture_mse_gain_percent": mean(
                float(row["mixture_mse_gain_percent"]) for row in selected
            ),
            "mixture_mse_positive_folds": sum(
                float(row["mixture_mse_gain_percent"]) > 0.0
                for row in selected
            ),
            "accuracy_gain_points": mean(
                float(row["accuracy_gain_points"]) for row in selected
            ),
            "folds": len(selected),
        }
    gates = config["diagnostic_gates"]
    over_base = summaries["posthoc_contrast_over_base_features"]
    over_shuffled = summaries["posthoc_contrast_over_shuffled"]
    over_final = summaries["posthoc_contrast_over_final_policy"]
    gate_results = {
        "contrast_over_base_macro": over_base["expected_arm_mse_gain_percent"]
        >= gates["contrast_over_base_expected_arm_mse_macro_percent_min"],
        "contrast_over_base_folds": over_base["expected_arm_mse_positive_folds"]
        >= gates["contrast_over_base_positive_folds_min"],
        "contrast_over_shuffled_macro": over_shuffled[
            "expected_arm_mse_gain_percent"
        ]
        >= gates["contrast_over_shuffled_expected_arm_mse_macro_percent_min"],
        "contrast_over_shuffled_folds": over_shuffled[
            "expected_arm_mse_positive_folds"
        ]
        >= gates["contrast_over_shuffled_positive_folds_min"],
        "posthoc_over_final_expected_macro": over_final[
            "expected_arm_mse_gain_percent"
        ]
        >= gates["posthoc_over_final_policy_expected_arm_mse_macro_percent_min"],
        "posthoc_over_final_expected_folds": over_final[
            "expected_arm_mse_positive_folds"
        ]
        >= gates["posthoc_over_final_policy_positive_folds_min"],
        "posthoc_over_final_mixture_macro": over_final[
            "mixture_mse_gain_percent"
        ]
        >= gates["posthoc_over_final_policy_mixture_mse_macro_percent_min"],
        "posthoc_over_final_mixture_folds": over_final[
            "mixture_mse_positive_folds"
        ]
        >= gates["posthoc_over_final_policy_mixture_positive_folds_min"],
    }
    specificity_pass = all(
        gate_results[key]
        for key in (
            "contrast_over_base_macro",
            "contrast_over_base_folds",
            "contrast_over_shuffled_macro",
            "contrast_over_shuffled_folds",
        )
    )
    policy_gap_pass = all(gate_results.values())
    if policy_gap_pass:
        decision = config["decision_map"]["all_gates_pass"]
    elif specificity_pass:
        decision = config["decision_map"][
            "contrast_specificity_pass_policy_gap_fail"
        ]
    else:
        decision = config["decision_map"]["contrast_specificity_fail"]
    return {
        "comparison_summary": summaries,
        "gate_results": gate_results,
        "passed_gates": sum(gate_results.values()),
        "total_gates": len(gate_results),
        "correction_skill_centered_alignment": mean(
            float(row["correction_skill_centered_alignment"])
            for row in alignment_rows
        ),
        "policy_shift_l1": mean(
            float(row["policy_shift_l1"]) for row in alignment_rows
        ),
        "decision": decision,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        path = (
            args.raw_root
            / dataset
            / "h720_full"
            / f"seed{config['seed']}"
            / "pcsd_test_audit_diagnostics.npz"
        )
        dataset_rows, dataset_alignment = analyze_dataset(
            dataset,
            path,
            config,
        )
        rows.extend(dataset_rows)
        alignment_rows.extend(dataset_alignment)
    comparisons = comparison_rows(rows)
    summary = summarize(rows, comparisons, alignment_rows, config)
    payload = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "test_derived": True,
        "boundary": config["boundary"],
        **summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "policy_and_probe_by_fold.csv", rows)
    write_csv(args.output_dir / "post_e2e_comparisons.csv", comparisons)
    write_csv(args.output_dir / "correction_alignment.csv", alignment_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
