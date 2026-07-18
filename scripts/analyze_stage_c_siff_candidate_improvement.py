#!/usr/bin/env python3
"""Audit SIFF routing calibration and the realizable fusion geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
ARMS = ("siff_equal", "siff_independent_equal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def diagnostics_path(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
) -> Path:
    return (
        root
        / arm
        / dataset
        / "h720_full"
        / f"seed{seed}"
        / "pcsd_test_audit_diagnostics.npz"
    )


def relative_gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


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


def centered_alignment(weights: np.ndarray, losses: np.ndarray) -> float:
    weight_delta = weights - weights.mean(axis=-1, keepdims=True)
    skill_delta = -losses - (-losses).mean(axis=-1, keepdims=True)
    numerator = np.sum(weight_delta * skill_delta, axis=-1)
    denominator = np.sqrt(
        np.sum(weight_delta**2, axis=-1)
        * np.sum(skill_delta**2, axis=-1)
    )
    valid = denominator > 1e-12
    return float(np.mean(numerator[valid] / denominator[valid]))


def routing_statistics(
    arm: str,
    dataset: str,
    losses: np.ndarray,
    usage: np.ndarray,
) -> dict[str, Any]:
    policy_expected = float(np.mean(np.sum(usage * losses, axis=-1)))
    uniform_expected = float(np.mean(losses))
    oracle_expected = float(np.mean(np.min(losses, axis=-1)))
    return {
        "arm": arm,
        "dataset": dataset,
        "policy_best_arm_match_rate": float(
            np.mean(np.argmax(usage, axis=-1) == np.argmin(losses, axis=-1))
        ),
        "policy_skill_centered_alignment": centered_alignment(usage, losses),
        "policy_expected_arm_mse": policy_expected,
        "uniform_expected_arm_mse": uniform_expected,
        "oracle_expected_arm_mse": oracle_expected,
        "policy_allocation_gain_over_uniform_percent": relative_gain(
            policy_expected,
            uniform_expected,
        ),
        "oracle_allocation_gain_over_policy_percent": relative_gain(
            oracle_expected,
            policy_expected,
        ),
    }


def fit_weights(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    arm_count = predictions.shape[-1]

    def objective(weights: np.ndarray) -> float:
        residual = predictions @ weights - targets
        return float(np.mean(residual**2))

    result = minimize(
        objective,
        np.full(arm_count, 1.0 / arm_count),
        method="SLSQP",
        bounds=[(lower, upper)] * arm_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"fusion fit failed: {result.message}")
    return result.x.astype(np.float64)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def fusion_statistics(
    arm: str,
    dataset: str,
    arms: np.ndarray,
    fused: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    fold_masks = [
        np.arange(arms.shape[0]) < arms.shape[0] // 2,
        np.arange(arms.shape[0]) >= arms.shape[0] // 2,
    ]
    for fold, train_mask in enumerate(fold_masks):
        test_mask = ~train_mask
        train_x = np.moveaxis(arms[train_mask], 1, -1).reshape(-1, arms.shape[1])
        train_y = targets[train_mask].reshape(-1)
        test_x = np.moveaxis(arms[test_mask], 1, -1).reshape(-1, arms.shape[1])
        test_y = targets[test_mask].reshape(-1)
        learned = fused[test_mask].reshape(-1)
        uniform_weights = np.full(arms.shape[1], 1.0 / arms.shape[1])
        convex_weights = fit_weights(
            train_x,
            train_y,
            lower=0.0,
            upper=1.0,
        )
        bounded_affine_weights = fit_weights(
            train_x,
            train_y,
            lower=-1.0,
            upper=1.0,
        )
        train_arm_mse = np.mean((train_x - train_y[:, None]) ** 2, axis=0)
        best_arm = int(np.argmin(train_arm_mse))
        learned_mse = mse(learned, test_y)
        uniform_mse = mse(test_x @ uniform_weights, test_y)
        convex_mse = mse(test_x @ convex_weights, test_y)
        affine_mse = mse(test_x @ bounded_affine_weights, test_y)
        best_fixed_mse = mse(test_x[:, best_arm], test_y)
        rows.append(
            {
                "arm": arm,
                "dataset": dataset,
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "learned_fused_mse": learned_mse,
                "uniform_fused_mse": uniform_mse,
                "crossfit_convex_mse": convex_mse,
                "crossfit_bounded_affine_mse": affine_mse,
                "crossfit_best_fixed_mse": best_fixed_mse,
                "convex_gain_over_learned_percent": relative_gain(
                    convex_mse,
                    learned_mse,
                ),
                "affine_gain_over_learned_percent": relative_gain(
                    affine_mse,
                    learned_mse,
                ),
                "affine_gain_over_convex_percent": relative_gain(
                    affine_mse,
                    convex_mse,
                ),
                "uniform_gain_over_learned_percent": relative_gain(
                    uniform_mse,
                    learned_mse,
                ),
                "best_fixed_gain_over_learned_percent": relative_gain(
                    best_fixed_mse,
                    learned_mse,
                ),
                "best_fixed_arm_index": best_arm,
                "convex_weights": json.dumps(convex_weights.tolist()),
                "bounded_affine_weights": json.dumps(
                    bounded_affine_weights.tolist()
                ),
                "affine_negative_weight_count": int(
                    np.sum(bounded_affine_weights < -1e-8)
                ),
                "affine_max_abs_weight": float(
                    np.max(np.abs(bounded_affine_weights))
                ),
            }
        )
    return rows


def summarize(
    routing_rows: list[dict[str, Any]],
    fusion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        route = [row for row in routing_rows if row["arm"] == arm]
        fusion = [row for row in fusion_rows if row["arm"] == arm]
        result[arm] = {
            "datasets": len(route),
            "policy_best_arm_match_rate": float(
                np.mean([row["policy_best_arm_match_rate"] for row in route])
            ),
            "policy_skill_centered_alignment": float(
                np.mean(
                    [row["policy_skill_centered_alignment"] for row in route]
                )
            ),
            "policy_allocation_gain_over_uniform_percent": float(
                np.mean(
                    [
                        row["policy_allocation_gain_over_uniform_percent"]
                        for row in route
                    ]
                )
            ),
            "convex_gain_over_learned_percent": float(
                np.mean([row["convex_gain_over_learned_percent"] for row in fusion])
            ),
            "affine_gain_over_learned_percent": float(
                np.mean([row["affine_gain_over_learned_percent"] for row in fusion])
            ),
            "affine_gain_over_convex_percent": float(
                np.mean([row["affine_gain_over_convex_percent"] for row in fusion])
            ),
            "convex_positive_dataset_folds": sum(
                row["convex_gain_over_learned_percent"] > 0.0 for row in fusion
            ),
            "affine_over_convex_positive_dataset_folds": sum(
                row["affine_gain_over_convex_percent"] > 0.0 for row in fusion
            ),
            "dataset_folds": len(fusion),
        }
    return result


def main() -> None:
    args = parse_args()
    routing_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            path = diagnostics_path(args.raw_root, arm, dataset, args.seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                losses = payload["arm_row_bin_mse"].astype(np.float64)
                usage = payload["policy_row_bin_usage"].astype(np.float64)
                probe_arms = payload["probe_arms"].astype(np.float64)
                probe_fused = payload["probe_fused"].astype(np.float64)
                probe_targets = payload["probe_targets"].astype(np.float64)
            routing_rows.append(
                routing_statistics(arm, dataset, losses, usage)
            )
            fusion_rows.extend(
                fusion_statistics(
                    arm,
                    dataset,
                    probe_arms,
                    probe_fused,
                    probe_targets,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "routing_calibration.csv", routing_rows)
    write_csv(args.output_dir / "probe_fusion_capacity.csv", fusion_rows)
    payload = {
        "raw_root": str(args.raw_root),
        "seed": args.seed,
        "definitions": {
            "policy_best_arm_match_rate": (
                "Fraction of row-bin cells where the most-used arm is the "
                "lowest-MSE arm among the five same-run arms."
            ),
            "policy_skill_centered_alignment": (
                "Mean centered cosine alignment across arms between policy "
                "weights and negative arm MSE; +1 is perfect skill alignment."
            ),
            "policy_allocation_gain_over_uniform_percent": (
                "Gain of policy-weighted expected arm MSE over uniform expected "
                "arm MSE. This is an allocation proxy, not fused-forecast MSE."
            ),
            "crossfit_convex_mse": (
                "Static nonnegative sum-to-one weights fitted on one half of "
                "the saved probe rows and evaluated on the other half."
            ),
            "crossfit_bounded_affine_mse": (
                "Static sum-to-one weights in [-1, 1] fitted and evaluated "
                "with the same two-fold row split."
            ),
        },
        "boundary": (
            "The probe batch has 256 rows per dataset and is diagnostic only. "
            "Static cross-fit stacking does not establish a new model's test "
            "effectiveness; it localizes whether calibration or convex geometry "
            "is a plausible bottleneck."
        ),
        "summary": summarize(routing_rows, fusion_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
