#!/usr/bin/env python3
"""Audit held-out information access for ISCF coalition credit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stage_c_iscf_scc_d0 import (
    gain_percent,
    normalized_positive_credit,
    spearman_last,
    standardized_arm_credit,
    write_csv,
)


REQUIRED_ARRAYS = {
    "probe_arms",
    "probe_fused",
    "probe_targets",
    "probe_direct_policy",
    "scales",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_scc_d0b.json"),
    )
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def normalize_simplex(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(values, 0.0)
    denominator = positive.sum(axis=-1, keepdims=True)
    uniform = np.full_like(positive, 1.0 / positive.shape[-1])
    return np.divide(
        positive,
        denominator,
        out=uniform,
        where=denominator > 0.0,
    )


def build_features(
    arms: np.ndarray,
    policy: np.ndarray,
    fused: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Build target-free features with shape [R,T,D]."""
    arms_rts = arms.transpose(0, 2, 1)
    arm_center = arms_rts - arms_rts.mean(axis=-1, keepdims=True)
    arm_dispersion = np.sqrt(np.mean(np.square(arm_center), axis=-1))
    normalized_arms = arm_center / np.maximum(
        arm_dispersion[..., None],
        epsilon,
    )
    entropy = -np.sum(
        policy.clip(min=1e-12) * np.log(policy.clip(min=1e-12)),
        axis=-1,
        keepdims=True,
    ) / np.log(policy.shape[-1])
    rows, time, _scopes = policy.shape
    position = np.linspace(0.0, 1.0, time, dtype=np.float64)
    position_features = np.stack(
        [
            position,
            np.sin(2.0 * np.pi * position),
            np.cos(2.0 * np.pi * position),
            np.sin(4.0 * np.pi * position),
            np.cos(4.0 * np.pi * position),
        ],
        axis=-1,
    )
    position_features = np.broadcast_to(
        position_features[None, :, :],
        (rows, time, position_features.shape[-1]),
    )
    return np.concatenate(
        [
            normalized_arms,
            policy,
            np.log(np.maximum(arm_dispersion, epsilon))[..., None],
            fused[..., None],
            entropy,
            position_features,
        ],
        axis=-1,
    )


def fit_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    evaluation_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0).clip(min=1e-6)
    target_mean = train_targets.mean(axis=0)
    standardized_train = (train_features - feature_mean) / feature_scale
    standardized_evaluation = (
        evaluation_features - feature_mean
    ) / feature_scale
    gram = standardized_train.T @ standardized_train / len(train_features)
    gram += alpha * np.eye(gram.shape[0], dtype=np.float64)
    cross = (
        standardized_train.T @ (train_targets - target_mean)
        / len(train_features)
    )
    coefficients = np.linalg.solve(gram, cross)
    return target_mean + standardized_evaluation @ coefficients


def shuffle_rows_by_position(
    credit: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = np.empty_like(credit)
    for position in range(credit.shape[1]):
        order = rng.permutation(credit.shape[0])
        shuffled[:, position, :] = credit[order, position, :]
    return shuffled


def forecast_l1(
    credit: np.ndarray,
    arms: np.ndarray,
    targets: np.ndarray,
) -> float:
    forecast = np.sum(credit * arms.transpose(0, 2, 1), axis=-1)
    return float(np.abs(forecast - targets).mean())


def credit_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    correlation = spearman_last(predicted, target)
    top1 = np.argmax(predicted, axis=-1) == np.argmax(target, axis=-1)
    return float(np.median(correlation)), float(np.mean(top1))


def analyze_payload(
    payload: dict[str, np.ndarray],
    config: dict[str, Any],
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    missing = REQUIRED_ARRAYS.difference(payload)
    if missing:
        raise KeyError(f"missing diagnostic arrays: {sorted(missing)}")
    arms = payload["probe_arms"].astype(np.float64)
    fused = payload["probe_fused"].astype(np.float64)
    targets = payload["probe_targets"].astype(np.float64)
    policy = payload["probe_direct_policy"].astype(np.float64)
    scales = tuple(int(value) for value in payload["scales"])
    if scales != tuple(config["coupling_scales"]):
        raise ValueError(f"unexpected scales: {scales}")
    if arms.shape != policy.transpose(0, 2, 1).shape:
        raise ValueError("arms and direct policy shapes are inconsistent")
    if fused.shape != targets.shape or fused.shape != arms.shape[::2]:
        raise ValueError("fused/target shapes do not match arms")
    if not all(
        np.isfinite(value).all()
        for value in (arms, fused, targets, policy)
    ):
        raise ValueError("non-finite diagnostic tensor")

    probe = config["probe"]
    epsilon = float(probe["epsilon"])
    policy_s = policy.transpose(0, 2, 1)
    denominator = np.maximum(1.0 - policy_s, epsilon)
    leave_one_out = (fused[:, None, :] - policy_s * arms) / denominator
    target_s = targets[:, None, :]
    full_absolute_error = np.abs(fused - targets)
    delta_l1 = (
        np.abs(leave_one_out - target_s)
        - full_absolute_error[:, None, :]
    )
    coalition_credit = normalized_positive_credit(delta_l1).transpose(0, 2, 1)
    standalone_credit = standardized_arm_credit(
        np.abs(arms - target_s)
    ).transpose(0, 2, 1)
    features = build_features(arms, policy, fused, epsilon)

    channel_count = int(config["channel_counts"][dataset])
    nominal_train_rows = int(
        np.floor(arms.shape[0] * probe["train_row_fraction"])
    )
    train_rows = nominal_train_rows // channel_count * channel_count
    if train_rows <= 0 or train_rows >= arms.shape[0]:
        raise ValueError("train_row_fraction yields an empty split")
    train_features = features[:train_rows].reshape(-1, features.shape[-1])
    evaluation_features = features[train_rows:].reshape(
        -1,
        features.shape[-1],
    )
    coalition_train = coalition_credit[:train_rows]
    coalition_evaluation = coalition_credit[train_rows:]
    standalone_train = standalone_credit[:train_rows]
    standalone_evaluation = standalone_credit[train_rows:]
    evaluation_arms = arms[train_rows:]
    evaluation_targets = targets[train_rows:]
    evaluation_fused = fused[train_rows:]

    coalition_prediction = normalize_simplex(
        fit_ridge(
            train_features,
            coalition_train.reshape(-1, coalition_train.shape[-1]),
            evaluation_features,
            float(probe["ridge_alpha"]),
        )
    ).reshape(coalition_evaluation.shape)
    standalone_prediction = normalize_simplex(
        fit_ridge(
            train_features,
            standalone_train.reshape(-1, standalone_train.shape[-1]),
            evaluation_features,
            float(probe["ridge_alpha"]),
        )
    ).reshape(standalone_evaluation.shape)

    full_l1 = float(np.abs(evaluation_fused - evaluation_targets).mean())
    predicted_l1 = forecast_l1(
        coalition_prediction,
        evaluation_arms,
        evaluation_targets,
    )
    standalone_predicted_l1 = forecast_l1(
        standalone_prediction,
        evaluation_arms,
        evaluation_targets,
    )
    oracle_l1 = forecast_l1(
        coalition_evaluation,
        evaluation_arms,
        evaluation_targets,
    )
    standalone_oracle_l1 = forecast_l1(
        standalone_evaluation,
        evaluation_arms,
        evaluation_targets,
    )
    prediction_rho, prediction_top1 = credit_metrics(
        coalition_prediction,
        coalition_evaluation,
    )
    standalone_rho, standalone_top1 = credit_metrics(
        standalone_prediction,
        standalone_evaluation,
    )

    rng = np.random.default_rng(int(probe["shuffle_seed"]) + seed)
    shuffled_gains: list[float] = []
    shuffled_rhos: list[float] = []
    shuffled_top1: list[float] = []
    for _ in range(int(probe["shuffle_repetitions"])):
        shuffled_train = shuffle_rows_by_position(coalition_train, rng)
        shuffled_prediction = normalize_simplex(
            fit_ridge(
                train_features,
                shuffled_train.reshape(-1, shuffled_train.shape[-1]),
                evaluation_features,
                float(probe["ridge_alpha"]),
            )
        ).reshape(coalition_evaluation.shape)
        shuffled_l1 = forecast_l1(
            shuffled_prediction,
            evaluation_arms,
            evaluation_targets,
        )
        shuffled_rho, shuffled_match = credit_metrics(
            shuffled_prediction,
            coalition_evaluation,
        )
        shuffled_gains.append(gain_percent(full_l1, shuffled_l1))
        shuffled_rhos.append(shuffled_rho)
        shuffled_top1.append(shuffled_match)

    predicted_gain = gain_percent(full_l1, predicted_l1)
    standalone_predicted_gain = gain_percent(full_l1, standalone_predicted_l1)
    shuffled_gain_p95 = float(np.quantile(shuffled_gains, 0.95))
    shuffled_rho_p95 = float(np.quantile(shuffled_rhos, 0.95))
    shuffled_top1_p95 = float(np.quantile(shuffled_top1, 0.95))
    return {
        "dataset": dataset,
        "seed": seed,
        "channel_count": channel_count,
        "train_probe_rows": train_rows,
        "evaluation_probe_rows": arms.shape[0] - train_rows,
        "feature_dimension": features.shape[-1],
        "full_l1": full_l1,
        "coalition_oracle_gain_l1_percent": gain_percent(full_l1, oracle_l1),
        "standalone_oracle_gain_l1_percent": gain_percent(
            full_l1,
            standalone_oracle_l1,
        ),
        "coalition_predicted_gain_l1_percent": predicted_gain,
        "standalone_predicted_gain_l1_percent": standalone_predicted_gain,
        "coalition_over_standalone_gain_percent": (
            predicted_gain - standalone_predicted_gain
        ),
        "coalition_prediction_spearman_median": prediction_rho,
        "coalition_prediction_top1_fraction": prediction_top1,
        "standalone_prediction_spearman_median": standalone_rho,
        "standalone_prediction_top1_fraction": standalone_top1,
        "shuffle_gain_l1_p95_percent": shuffled_gain_p95,
        "shuffle_spearman_p95": shuffled_rho_p95,
        "shuffle_top1_p95": shuffled_top1_p95,
        "prediction_binding_pass": bool(
            predicted_gain > shuffled_gain_p95
            and prediction_rho > shuffled_rho_p95
            and prediction_top1 > shuffled_top1_p95
        ),
    }


def summarize(
    run_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        selected = [row for row in run_rows if row["dataset"] == dataset]
        gains = [
            float(row["coalition_predicted_gain_l1_percent"])
            for row in selected
        ]
        dataset_rows.append(
            {
                "dataset": dataset,
                "runs": len(selected),
                "median_coalition_predicted_gain_l1_percent": float(
                    np.median(gains)
                ),
                "median_coalition_over_standalone_gain_percent": float(
                    np.median(
                        [
                            row["coalition_over_standalone_gain_percent"]
                            for row in selected
                        ]
                    )
                ),
                "binding_pass_runs": sum(
                    bool(row["prediction_binding_pass"]) for row in selected
                ),
                "all_seeds_positive_gain": bool(
                    len(selected) == len(config["seeds"])
                    and all(gain > 0.0 for gain in gains)
                ),
            }
        )

    gates = config["gates"]
    gains = [
        float(row["coalition_predicted_gain_l1_percent"])
        for row in run_rows
    ]
    advantages = [
        float(row["coalition_over_standalone_gain_percent"])
        for row in run_rows
    ]
    binding_runs = sum(
        bool(row["prediction_binding_pass"]) for row in run_rows
    )
    positive_runs = sum(gain > 0.0 for gain in gains)
    advantage_runs = sum(value > 0.0 for value in advantages)
    stable_datasets = sum(
        bool(row["all_seeds_positive_gain"]) for row in dataset_rows
    )
    checks = {
        "matrix_complete": len(run_rows)
        == len(config["datasets"]) * len(config["seeds"]),
        "prediction_binding": binding_runs
        >= gates["prediction_binding_run_count_min"],
        "realized_gain": (
            float(np.median(gains))
            >= gates["predicted_gain_median_min_percent"]
            and positive_runs >= gates["predicted_positive_gain_run_count_min"]
        ),
        "beyond_standalone": (
            float(np.median(advantages))
            >= gates["coalition_over_standalone_gain_median_min_percent"]
            and advantage_runs
            >= gates["coalition_over_standalone_run_count_min"]
        ),
        "seed_consistency": stable_datasets
        >= gates["all_seed_positive_dataset_count_min"],
    }
    if all(checks.values()):
        decision = config["decision_map"]["all_gates_pass"]
    elif not checks["prediction_binding"] or not checks["realized_gain"]:
        decision = config["decision_map"]["binding_or_gain_fail"]
    elif not checks["beyond_standalone"]:
        decision = config["decision_map"]["standalone_explains"]
    else:
        decision = config["decision_map"]["incomplete_or_unstable"]
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "runs": len(run_rows),
        "median_coalition_predicted_gain_l1_percent": float(np.median(gains)),
        "positive_gain_run_count": positive_runs,
        "prediction_binding_run_count": binding_runs,
        "median_coalition_over_standalone_gain_percent": float(
            np.median(advantages)
        ),
        "coalition_over_standalone_run_count": advantage_runs,
        "all_seed_positive_dataset_count": stable_datasets,
        "checks": checks,
        "decision": decision,
        "method_implementation_authorized": False,
        "forecast_model_training_authorized": False,
        "formal_test_authorized": False,
    }
    return summary, dataset_rows


def synthetic_payload(config: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260722)
    rows = 64
    time = 72
    scopes = len(config["coupling_scales"])
    targets = rng.normal(size=(rows, time))
    arms = targets[:, None, :] + rng.normal(
        scale=np.linspace(0.1, 0.5, scopes)[None, :, None],
        size=(rows, scopes, time),
    )
    logits = rng.normal(size=(rows, time, scopes))
    logits -= logits.max(axis=-1, keepdims=True)
    policy = np.exp(logits)
    policy /= policy.sum(axis=-1, keepdims=True)
    fused = np.sum(policy.transpose(0, 2, 1) * arms, axis=1)
    return {
        "probe_arms": arms.astype(np.float32),
        "probe_fused": fused.astype(np.float32),
        "probe_targets": targets.astype(np.float32),
        "probe_direct_policy": policy.astype(np.float32),
        "scales": np.asarray(config["coupling_scales"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        smoke_config = json.loads(json.dumps(config))
        smoke_config["probe"]["shuffle_repetitions"] = 2
        rows = [
            analyze_payload(
                synthetic_payload(smoke_config),
                smoke_config,
                dataset,
                seed,
            )
            for dataset in smoke_config["datasets"]
            for seed in smoke_config["seeds"]
        ]
        summary, datasets = summarize(rows, smoke_config)
        if (
            summary["runs"] != 15
            or len(datasets) != 5
            or not all(row["feature_dimension"] == 18 for row in rows)
        ):
            raise RuntimeError("synthetic SCC D0B smoke failed")
        print("iscf_scc_d0b_synthetic_smoke=pass")
        return
    if args.validation_root is None or args.output_dir is None:
        raise ValueError("validation-root and output-dir are required")

    run_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            path = (
                args.validation_root
                / dataset
                / f"seed{seed}"
                / "pcsd_validation_diagnostics.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as archive:
                payload = {name: archive[name] for name in archive.files}
            run_rows.append(
                analyze_payload(payload, config, dataset, seed)
            )
    summary, dataset_rows = summarize(run_rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_metrics.csv", run_rows)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_scc_d0b=complete runs={summary['runs']} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
