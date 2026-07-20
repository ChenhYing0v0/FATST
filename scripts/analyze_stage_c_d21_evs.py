#!/usr/bin/env python3
"""Analyze D21 validation-to-test evidence-validity identifiability."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class PolicyResult:
    name: str
    mse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_gain(candidate: float, control: float) -> float:
    return (control - candidate) / control


def weighted_selected_mse(
    losses: np.ndarray,
    selection: np.ndarray,
    bin_weights: np.ndarray,
) -> float:
    if losses.ndim != 3:
        raise ValueError("losses must have shape [N, B, S]")
    selected = np.take_along_axis(
        losses,
        selection[..., None],
        axis=2,
    )[..., 0]
    return float(np.average(selected, axis=1, weights=bin_weights).mean())


def _standardize(
    validation_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = validation_features.mean(axis=0, keepdims=True)
    std = validation_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        (validation_features - mean) / std,
        (test_features - mean) / std,
    )


def _ridge_predictor(
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    ones_validation = np.ones((x_validation.shape[0], 1), dtype=np.float64)
    ones_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
    design_validation = np.concatenate([ones_validation, x_validation], axis=1)
    design_test = np.concatenate([ones_test, x_test], axis=1)
    gram = design_validation.T @ design_validation
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        gram + penalty,
        design_validation.T @ y_validation,
    )
    return design_test @ weights


def _tree_predictor(
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    predictions = []
    for output_index in range(y_validation.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            learning_rate=float(config["learning_rate"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=int(config["random_state"]),
        )
        model.fit(x_validation, y_validation[:, output_index])
        predictions.append(model.predict(x_test))
    return np.stack(predictions, axis=1)


def evaluate_policies(
    validation_features: np.ndarray,
    test_features: np.ndarray,
    validation_losses: np.ndarray,
    test_losses: np.ndarray,
    bin_weights: np.ndarray,
    readout_name: str,
    readout_config: dict[str, object],
    permutation_seed: int,
) -> list[PolicyResult]:
    """Fit all controls on validation and return official-test policy MSE."""
    x_validation, x_test = _standardize(
        validation_features.astype(np.float64),
        test_features.astype(np.float64),
    )
    validation_losses = validation_losses.astype(np.float64)
    test_losses = test_losses.astype(np.float64)
    row_count, bin_count, arm_count = validation_losses.shape
    test_row_count = test_losses.shape[0]
    log_risk = np.log(validation_losses + 1e-12)
    centered_risk = log_risk - log_risk.mean(axis=2, keepdims=True)

    if readout_name == "ridge":
        predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        predictor = lambda x_fit, y_fit, x_eval: _ridge_predictor(
            x_fit,
            y_fit,
            x_eval,
            float(readout_config["alpha"]),
        )
    elif readout_name == "hist_gradient_boosting":
        predictor = lambda x_fit, y_fit, x_eval: _tree_predictor(
            x_fit,
            y_fit,
            x_eval,
            readout_config,
        )
    else:
        raise ValueError(f"unsupported readout: {readout_name}")

    validation_weighted = np.average(
        validation_losses,
        axis=1,
        weights=bin_weights,
    )
    global_arm = int(validation_weighted.mean(axis=0).argmin())
    global_selection = np.full(
        (test_row_count, bin_count),
        global_arm,
        dtype=np.int64,
    )

    region_arms = validation_losses.mean(axis=0).argmin(axis=1)
    region_selection = np.broadcast_to(
        region_arms[None, :],
        (test_row_count, bin_count),
    )

    global_targets = np.average(centered_risk, axis=1, weights=bin_weights)
    history_global_scores = predictor(
        x_validation,
        global_targets,
        x_test,
    )
    history_global_arm = history_global_scores.argmin(axis=1)
    history_global_selection = np.repeat(
        history_global_arm[:, None],
        bin_count,
        axis=1,
    )

    region_intercept = (
        centered_risk.mean(axis=0) - global_targets.mean(axis=0)[None, :]
    )
    additive_scores = (
        history_global_scores[:, None, :] + region_intercept[None, :, :]
    )
    additive_selection = additive_scores.argmin(axis=2)

    interaction_targets = centered_risk.reshape(row_count, bin_count * arm_count)
    interaction_scores = predictor(
        x_validation,
        interaction_targets,
        x_test,
    ).reshape(test_row_count, bin_count, arm_count)
    interaction_selection = interaction_scores.argmin(axis=2)

    generator = np.random.default_rng(permutation_seed)
    permuted_features = x_validation[generator.permutation(row_count)]
    permuted_scores = predictor(
        permuted_features,
        interaction_targets,
        x_test,
    ).reshape(test_row_count, bin_count, arm_count)
    permuted_selection = permuted_scores.argmin(axis=2)
    oracle_selection = test_losses.argmin(axis=2)

    selections = {
        "global_fixed": global_selection,
        "region_fixed": region_selection,
        "history_global": history_global_selection,
        "additive_history_region": additive_selection,
        "evs_interaction": interaction_selection,
        "permuted_history": permuted_selection,
        "oracle": oracle_selection,
    }
    results = [
        PolicyResult(
            name=name,
            mse=weighted_selected_mse(test_losses, selection, bin_weights),
        )
        for name, selection in selections.items()
    ]
    if not all(np.isfinite(result.mse) for result in results):
        raise RuntimeError("non-finite D21 policy MSE")
    return results


def _artifact_path(
    root: Path,
    carrier: str,
    arm: str,
    dataset: str,
    seed: int,
    split: str,
) -> Path:
    return root / carrier / arm / dataset / f"seed{seed}" / f"{split}.npz"


def load_matrix(
    root: Path,
    design: dict[str, object],
    carrier: str,
    dataset: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arms = list(design["canonical_arms"])
    seed = int(design["seed"])
    anchor = str(design["descriptor_anchor_arm"])
    loss_parts = []
    common_indices = None
    features = None
    for arm in arms:
        path = _artifact_path(root, carrier, arm, dataset, seed, split)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as payload:
            indices = payload["probe_indices"].astype(np.int64)
            losses = payload["row_bin_mse"].astype(np.float64)
            if common_indices is None:
                common_indices = indices
            elif not np.array_equal(common_indices, indices):
                raise RuntimeError(f"probe index mismatch: {path}")
            loss_parts.append(losses[indices])
            if arm == anchor:
                features = payload["history_features"].astype(np.float64)
    if common_indices is None or features is None:
        raise RuntimeError("missing D21 indices or descriptor anchor")
    matrix = np.stack(loss_parts, axis=2)
    if matrix.shape[0] != features.shape[0]:
        raise RuntimeError("feature/loss probe mismatch")
    return features, matrix, common_indices


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    design = json.loads(args.design.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_weights = np.asarray(
        [entry["end"] - entry["start"] for entry in design["future_bins"]],
        dtype=np.float64,
    )
    readouts = {
        design["readouts"]["primary"]["name"]: design["readouts"]["primary"],
        design["readouts"]["sensitivity"]["name"]: design["readouts"]["sensitivity"],
    }
    policy_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for carrier in design["carriers"]:
        for dataset in design["datasets"]:
            validation_features, validation_losses, validation_indices = load_matrix(
                args.input_root,
                design,
                carrier,
                dataset,
                "val",
            )
            test_features, test_losses, test_indices = load_matrix(
                args.input_root,
                design,
                carrier,
                dataset,
                "test",
            )
            for readout_name, readout_config in readouts.items():
                results = evaluate_policies(
                    validation_features,
                    test_features,
                    validation_losses,
                    test_losses,
                    bin_weights,
                    readout_name,
                    readout_config,
                    int(
                        design["readouts"]["sensitivity"].get(
                            "random_state", 20260720
                        )
                    ),
                )
                by_name = {result.name: result.mse for result in results}
                for result in results:
                    policy_rows.append(
                        {
                            "carrier": carrier,
                            "dataset": dataset,
                            "readout": readout_name,
                            "policy": result.name,
                            "test_mse": result.mse,
                            "validation_probe_rows": validation_indices.shape[0],
                            "test_probe_rows": test_indices.shape[0],
                        }
                    )
                for control in (
                    "global_fixed",
                    "region_fixed",
                    "history_global",
                    "additive_history_region",
                    "permuted_history",
                ):
                    comparison_rows.append(
                        {
                            "carrier": carrier,
                            "dataset": dataset,
                            "readout": readout_name,
                            "comparison": f"evs_interaction_vs_{control}",
                            "candidate_mse": by_name["evs_interaction"],
                            "control_mse": by_name[control],
                            "relative_gain": relative_gain(
                                by_name["evs_interaction"],
                                by_name[control],
                            ),
                        }
                    )
                comparison_rows.append(
                    {
                        "carrier": carrier,
                        "dataset": dataset,
                        "readout": readout_name,
                        "comparison": "oracle_vs_region_fixed",
                        "candidate_mse": by_name["oracle"],
                        "control_mse": by_name["region_fixed"],
                        "relative_gain": relative_gain(
                            by_name["oracle"],
                            by_name["region_fixed"],
                        ),
                    }
                )

    _write_csv(args.output_dir / "policy_test_mse.csv", policy_rows)
    _write_csv(args.output_dir / "comparison_gains.csv", comparison_rows)

    summaries: list[dict[str, object]] = []
    gate_by_readout: dict[str, dict[str, object]] = {}
    gates = design["gates"]
    for readout_name in readouts:
        summary_lookup: dict[tuple[str, str], dict[str, object]] = {}
        for carrier in design["carriers"]:
            comparisons = [
                "evs_interaction_vs_region_fixed",
                "evs_interaction_vs_history_global",
                "evs_interaction_vs_additive_history_region",
                "evs_interaction_vs_permuted_history",
                "oracle_vs_region_fixed",
            ]
            for comparison in comparisons:
                selected = [
                    row
                    for row in comparison_rows
                    if row["carrier"] == carrier
                    and row["readout"] == readout_name
                    and row["comparison"] == comparison
                ]
                gains = np.asarray(
                    [float(row["relative_gain"]) for row in selected]
                )
                summary = {
                    "carrier": carrier,
                    "readout": readout_name,
                    "comparison": comparison,
                    "macro_gain": float(gains.mean()),
                    "positive_datasets": int((gains > 0.0).sum()),
                    "minimum_gain": float(gains.min()),
                }
                summaries.append(summary)
                summary_lookup[(carrier, comparison)] = summary

        neutral_region = summary_lookup[
            ("neutral_raw", "evs_interaction_vs_region_fixed")
        ]
        a6_region = summary_lookup[
            ("a6_natural", "evs_interaction_vs_region_fixed")
        ]
        neutral_history = summary_lookup[
            ("neutral_raw", "evs_interaction_vs_history_global")
        ]
        neutral_additive = summary_lookup[
            ("neutral_raw", "evs_interaction_vs_additive_history_region")
        ]
        neutral_permuted = summary_lookup[
            ("neutral_raw", "evs_interaction_vs_permuted_history")
        ]
        neutral_dataset_rows = [
            row
            for row in comparison_rows
            if row["carrier"] == "neutral_raw"
            and row["readout"] == readout_name
            and row["comparison"] == "evs_interaction_vs_region_fixed"
        ]
        severe_count = sum(
            float(row["relative_gain"])
            < -float(gates["maximum_policy_degradation"])
            for row in neutral_dataset_rows
        )
        checks = {
            "neutral_vs_region_macro": float(neutral_region["macro_gain"])
            >= float(gates["neutral_interaction_vs_region_macro_gain_min"]),
            "neutral_vs_region_datasets": int(
                neutral_region["positive_datasets"]
            )
            >= int(gates["neutral_interaction_vs_region_dataset_required"]),
            "a6_vs_region_macro": float(a6_region["macro_gain"])
            > float(gates["a6_interaction_vs_region_macro_gain_min"]),
            "a6_vs_region_datasets": int(a6_region["positive_datasets"])
            >= int(gates["a6_interaction_vs_region_dataset_required"]),
            "neutral_vs_history_global": float(neutral_history["macro_gain"])
            > 0.0
            and int(neutral_history["positive_datasets"]) >= 3,
            "neutral_vs_additive_macro": float(neutral_additive["macro_gain"])
            >= float(gates["neutral_interaction_vs_additive_macro_gain_min"]),
            "neutral_vs_additive_datasets": int(
                neutral_additive["positive_datasets"]
            )
            >= int(gates["neutral_interaction_vs_additive_dataset_required"]),
            "neutral_vs_permuted_macro": float(neutral_permuted["macro_gain"])
            >= float(gates["neutral_interaction_vs_permuted_macro_gain_min"]),
            "neutral_vs_permuted_datasets": int(
                neutral_permuted["positive_datasets"]
            )
            >= int(gates["neutral_interaction_vs_permuted_dataset_required"]),
            "no_severe_degradation": severe_count
            <= int(gates["maximum_severely_degraded_datasets"]),
        }
        gate_by_readout[readout_name] = {
            "checks": checks,
            "passed": all(checks.values()),
            "severely_degraded_datasets": severe_count,
        }

    _write_csv(args.output_dir / "macro_summary.csv", summaries)
    ridge_pass = bool(gate_by_readout.get("ridge", {}).get("passed", False))
    tree_pass = bool(
        gate_by_readout.get("hist_gradient_boosting", {}).get("passed", False)
    )
    if ridge_pass and tree_pass:
        decision = "problem_gate_passed_robust"
    elif ridge_pass:
        decision = "problem_gate_passed_simple_signal"
    elif tree_pass:
        decision = "problem_partial_pass_nonlinear_identifiability"
    else:
        decision = "exact_descriptor_readout_probe_failed_direction_unresolved"
    gate = {
        "diagnostic_id": design["diagnostic_id"],
        "fit_split": "validation",
        "evaluation_split": "test",
        "test_informed": True,
        "matrix_complete": True,
        "gate_by_readout": gate_by_readout,
        "decision": decision,
        "paper_method_authorized": False,
        "step4_source_design_authorized": ridge_pass or tree_pass,
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# SC-D21-EVS Result",
        "",
        "## Protocol",
        "",
        "- validation only fits the past-only risk probes; official test only evaluates transfer;",
        "- requested horizon and future truth are absent from policy features;",
        "- the result is a problem gate, not a forecasting method result.",
        "",
        "## Macro comparisons",
        "",
        "| Carrier | Readout | Comparison | Macro gain | Positive datasets |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['carrier']} | {row['readout']} | {row['comparison']} "
            f"| {100 * float(row['macro_gain']):+.4f}% "
            f"| {row['positive_datasets']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{decision}`.",
            "",
            "A positive result authorizes only Step4 source-informed method design. "
            "A negative result rejects only this fixed descriptor/readout probe; it "
            "does not by itself reject every representation-level EVS design.",
        ]
    )
    (args.output_dir / "result.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"d21_evs_analysis=pass decision={decision}")


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
