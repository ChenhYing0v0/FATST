#!/usr/bin/env python3
"""Audit ISCF coalition credit from frozen validation diagnostic tensors."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


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
        default=Path("configs/stage_c_iscf_scc_d0.json"),
    )
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def spearman_last(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise Spearman correlation over the last dimension."""
    if left.shape != right.shape:
        raise ValueError("Spearman inputs must share a shape")
    left_rank = rankdata(left, axis=-1, method="average")
    right_rank = rankdata(right, axis=-1, method="average")
    left_centered = left_rank - left_rank.mean(axis=-1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=-1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=-1)
        * np.sum(np.square(right_centered), axis=-1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def normalized_positive_credit(delta: np.ndarray) -> np.ndarray:
    positive = np.maximum(delta, 0.0)
    denominator = positive.sum(axis=1, keepdims=True)
    uniform = np.full_like(positive, 1.0 / positive.shape[1])
    return np.divide(
        positive,
        denominator,
        out=uniform,
        where=denominator > 0.0,
    )


def standardized_arm_credit(arm_absolute_error: np.ndarray) -> np.ndarray:
    centered = arm_absolute_error - arm_absolute_error.mean(
        axis=1,
        keepdims=True,
    )
    scale = arm_absolute_error.std(axis=1, keepdims=True).clip(min=1e-6)
    logits = -centered / scale
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def gain_percent(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0
    return 100.0 * (1.0 - candidate / reference)


def nonidentity_permutations(
    scope_count: int,
    repetitions: int,
    seed: int,
) -> list[tuple[int, ...]]:
    identity = tuple(range(scope_count))
    values = [
        value
        for value in itertools.permutations(range(scope_count))
        if value != identity
    ]
    if repetitions >= len(values):
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=repetitions, replace=False)
    return [values[int(index)] for index in indices]


def analyze_payload(
    payload: dict[str, np.ndarray],
    config: dict[str, Any],
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = REQUIRED_ARRAYS.difference(payload)
    if missing:
        raise KeyError(f"missing diagnostic arrays: {sorted(missing)}")
    arms = payload["probe_arms"].astype(np.float64)
    fused = payload["probe_fused"].astype(np.float64)
    targets = payload["probe_targets"].astype(np.float64)
    policy = payload["probe_direct_policy"].astype(np.float64).transpose(0, 2, 1)
    scales = tuple(int(value) for value in payload["scales"])
    expected_scales = tuple(config["coupling_scales"])
    if scales != expected_scales:
        raise ValueError(f"unexpected scales: {scales}")
    if arms.shape != policy.shape:
        raise ValueError(
            f"arms and policy must share [R,S,T], got {arms.shape} and {policy.shape}"
        )
    if fused.shape != targets.shape or fused.shape != (
        arms.shape[0],
        arms.shape[2],
    ):
        raise ValueError("fused/target shapes do not match arm rows and time")
    if not all(
        np.isfinite(value).all()
        for value in (arms, fused, targets, policy)
    ):
        raise ValueError("non-finite diagnostic tensor")

    protocol = config["diagnostic_protocol"]
    epsilon = float(protocol["epsilon"])
    reconstructed = np.sum(policy * arms, axis=1)
    reconstruction_gap = float(np.max(np.abs(reconstructed - fused)))
    denominator = np.maximum(1.0 - policy, epsilon)
    leave_one_out = (fused[:, None, :] - policy * arms) / denominator
    target_expanded = targets[:, None, :]
    full_absolute_error = np.abs(fused - targets)
    full_squared_error = np.square(fused - targets)
    arm_absolute_error = np.abs(arms - target_expanded)
    delta_l1 = (
        np.abs(leave_one_out - target_expanded)
        - full_absolute_error[:, None, :]
    )
    delta_mse = (
        np.square(leave_one_out - target_expanded)
        - full_squared_error[:, None, :]
    )
    coalition_credit = normalized_positive_credit(delta_l1)
    arm_credit = standardized_arm_credit(arm_absolute_error)

    coalition_forecast = np.sum(coalition_credit * arms, axis=1)
    arm_credit_forecast = np.sum(arm_credit * arms, axis=1)
    uniform_forecast = arms.mean(axis=1)
    full_l1 = float(full_absolute_error.mean())
    full_mse = float(full_squared_error.mean())
    coalition_l1 = float(np.abs(coalition_forecast - targets).mean())
    coalition_mse = float(np.square(coalition_forecast - targets).mean())
    arm_credit_l1 = float(np.abs(arm_credit_forecast - targets).mean())
    uniform_l1 = float(np.abs(uniform_forecast - targets).mean())

    policy_rts = policy.transpose(0, 2, 1)
    coalition_rts = coalition_credit.transpose(0, 2, 1)
    delta_rts = delta_l1.transpose(0, 2, 1)
    standalone_score_rts = (-arm_absolute_error).transpose(0, 2, 1)
    policy_credit_rho = spearman_last(policy_rts, coalition_rts)
    standalone_credit_rho = spearman_last(standalone_score_rts, delta_rts)
    best_match = np.argmax(delta_rts, axis=-1) == np.argmax(
        standalone_score_rts,
        axis=-1,
    )
    positive_count = np.sum(delta_l1 > 0.0, axis=1)
    policy_entropy = -np.sum(
        policy_rts.clip(min=1e-12) * np.log(policy_rts.clip(min=1e-12)),
        axis=-1,
    ) / np.log(policy.shape[1])

    permutations = nonidentity_permutations(
        arms.shape[1],
        int(protocol["shuffle_repetitions"]),
        int(protocol["shuffle_seed"]) + seed,
    )
    shuffled_gains = []
    for permutation in permutations:
        shuffled = coalition_credit[:, permutation, :]
        forecast = np.sum(shuffled * arms, axis=1)
        shuffled_gains.append(
            gain_percent(full_l1, float(np.abs(forecast - targets).mean()))
        )

    bin_rows: list[dict[str, Any]] = []
    for entry in protocol["future_bins"]:
        start = int(entry["start"])
        end = int(entry["end"])
        for scope_index, scale in enumerate(scales):
            bin_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "bin": entry["name"],
                    "scope_index": scope_index,
                    "scale": scale,
                    "mean_loo_gain_l1": float(
                        delta_l1[:, scope_index, start:end].mean()
                    ),
                    "mean_loo_gain_mse": float(
                        delta_mse[:, scope_index, start:end].mean()
                    ),
                    "mean_coalition_credit": float(
                        coalition_credit[:, scope_index, start:end].mean()
                    ),
                    "mean_policy": float(policy[:, scope_index, start:end].mean()),
                    "positive_fraction": float(
                        np.mean(delta_l1[:, scope_index, start:end] > 0.0)
                    ),
                }
            )

    low_entropy = policy_entropy <= np.quantile(policy_entropy, 0.25)
    high_entropy = policy_entropy >= np.quantile(policy_entropy, 0.75)
    metrics = {
        "dataset": dataset,
        "seed": seed,
        "probe_rows": int(arms.shape[0]),
        "fusion_reconstruction_max_abs": reconstruction_gap,
        "full_l1": full_l1,
        "full_mse": full_mse,
        "coalition_credit_l1": coalition_l1,
        "coalition_credit_mse": coalition_mse,
        "counterfactual_oracle_headroom_l1_percent": gain_percent(
            full_l1,
            coalition_l1,
        ),
        "counterfactual_oracle_headroom_mse_percent": gain_percent(
            full_mse,
            coalition_mse,
        ),
        "arm_credit_gain_l1_percent": gain_percent(full_l1, arm_credit_l1),
        "uniform_gain_l1_percent": gain_percent(full_l1, uniform_l1),
        "positive_contributor_count_median": float(np.median(positive_count)),
        "positive_contributor_count_mean": float(np.mean(positive_count)),
        "policy_credit_spearman_median": float(np.median(policy_credit_rho)),
        "standalone_credit_spearman_median": float(
            np.median(standalone_credit_rho)
        ),
        "standalone_best_match_fraction": float(np.mean(best_match)),
        "policy_entropy_median": float(np.median(policy_entropy)),
        "low_entropy_policy_credit_spearman_median": float(
            np.median(policy_credit_rho[low_entropy])
        ),
        "high_entropy_policy_credit_spearman_median": float(
            np.median(policy_credit_rho[high_entropy])
        ),
        "shuffled_gain_l1_p95_percent": float(
            np.quantile(shuffled_gains, 0.95)
        ),
        "shuffle_specific": bool(
            gain_percent(full_l1, coalition_l1)
            > np.quantile(shuffled_gains, 0.95)
        ),
    }
    return metrics, bin_rows


def summarize(
    run_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = config["datasets"]
    seeds = config["seeds"]
    dataset_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        selected = [row for row in run_rows if row["dataset"] == dataset]
        dataset_rows.append(
            {
                "dataset": dataset,
                "runs": len(selected),
                "median_counterfactual_oracle_headroom_l1_percent": float(
                    np.median(
                        [
                            row["counterfactual_oracle_headroom_l1_percent"]
                            for row in selected
                        ]
                    )
                ),
                "median_positive_contributor_count": float(
                    np.median(
                        [
                            row["positive_contributor_count_median"]
                            for row in selected
                        ]
                    )
                ),
                "median_policy_credit_spearman": float(
                    np.median(
                        [row["policy_credit_spearman_median"] for row in selected]
                    )
                ),
                "median_standalone_credit_spearman": float(
                    np.median(
                        [row["standalone_credit_spearman_median"] for row in selected]
                    )
                ),
                "median_standalone_best_match_fraction": float(
                    np.median(
                        [row["standalone_best_match_fraction"] for row in selected]
                    )
                ),
                "shuffle_specific_runs": sum(
                    bool(row["shuffle_specific"]) for row in selected
                ),
            }
        )
        profiles: dict[int, np.ndarray] = {}
        for seed in seeds:
            values = [
                row["mean_loo_gain_l1"]
                for row in bin_rows
                if row["dataset"] == dataset and int(row["seed"]) == seed
            ]
            profiles[seed] = np.asarray(values, dtype=np.float64)
        correlations = [
            float(spearman_last(profiles[left], profiles[right]))
            for left, right in itertools.combinations(seeds, 2)
        ]
        stability_rows.append(
            {
                "dataset": dataset,
                "seed_pairs": len(correlations),
                "median_credit_seed_spearman": float(np.median(correlations)),
                "minimum_credit_seed_spearman": float(np.min(correlations)),
                "stable": bool(
                    np.median(correlations)
                    >= config["gates"]["credit_seed_stability_spearman_min"]
                ),
            }
        )

    gates = config["gates"]
    median_headroom = float(
        np.median(
            [row["counterfactual_oracle_headroom_l1_percent"] for row in run_rows]
        )
    )
    positive_runs = sum(
        row["positive_contributor_count_median"]
        >= gates["positive_contributor_median_min"]
        for row in run_rows
    )
    median_match = float(
        np.median([row["standalone_best_match_fraction"] for row in run_rows])
    )
    median_standalone_rho = float(
        np.median([row["standalone_credit_spearman_median"] for row in run_rows])
    )
    stable_datasets = sum(bool(row["stable"]) for row in stability_rows)
    shuffle_specific_runs = sum(bool(row["shuffle_specific"]) for row in run_rows)
    reconstruction_max = max(
        float(row["fusion_reconstruction_max_abs"]) for row in run_rows
    )
    checks = {
        "matrix_complete": len(run_rows) == len(datasets) * len(seeds),
        "fusion_reconstruction": reconstruction_max
        <= gates["fusion_reconstruction_max_abs"],
        "headroom": median_headroom
        >= gates["counterfactual_oracle_headroom_median_min_percent"],
        "nondegeneracy": positive_runs
        >= gates["positive_contributor_run_count_min"],
        "distinct_from_standalone": (
            median_match <= gates["standalone_best_match_median_max"]
            or median_standalone_rho
            <= gates["standalone_credit_spearman_median_max"]
        ),
        "seed_stability": stable_datasets
        >= gates["credit_seed_stability_dataset_count_min"],
        "shuffle_specificity": shuffle_specific_runs
        >= gates["shuffle_specific_run_count_min"],
    }
    if all(checks.values()):
        decision = config["decision_map"]["all_gates_pass"]
    elif not checks["headroom"] or not checks["nondegeneracy"]:
        decision = config["decision_map"]["signal_or_nondegeneracy_fail"]
    elif not checks["distinct_from_standalone"]:
        decision = config["decision_map"]["standalone_credit_explains"]
    else:
        decision = config["decision_map"]["stability_or_shuffle_fail"]
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "source_role": config["role"],
        "runs": len(run_rows),
        "median_counterfactual_oracle_headroom_l1_percent": median_headroom,
        "positive_contributor_gate_run_count": positive_runs,
        "median_standalone_best_match_fraction": median_match,
        "median_standalone_credit_spearman": median_standalone_rho,
        "stable_dataset_count": stable_datasets,
        "shuffle_specific_run_count": shuffle_specific_runs,
        "fusion_reconstruction_max_abs": reconstruction_max,
        "checks": checks,
        "decision": decision,
        "method_implementation_authorized": False,
        "remote_training_authorized": False,
        "formal_test_authorized": False,
    }
    return summary, dataset_rows, stability_rows


def synthetic_payload(config: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260722)
    rows = 24
    time = 720
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
        run_rows: list[dict[str, Any]] = []
        bin_rows: list[dict[str, Any]] = []
        for dataset in config["datasets"]:
            for seed in config["seeds"]:
                metrics, bins = analyze_payload(
                    synthetic_payload(config),
                    config,
                    dataset,
                    seed,
                )
                run_rows.append(metrics)
                bin_rows.extend(bins)
        summary, dataset_rows, stability_rows = summarize(
            run_rows,
            bin_rows,
            config,
        )
        if (
            summary["runs"] != 15
            or len(dataset_rows) != 5
            or len(stability_rows) != 5
            or max(
                row["fusion_reconstruction_max_abs"]
                for row in run_rows
            )
            > 1e-5
            or len(bin_rows) != 600
        ):
            raise RuntimeError("synthetic SCC D0 smoke failed")
        print("iscf_scc_d0_synthetic_smoke=pass")
        return
    if args.validation_root is None or args.output_dir is None:
        raise ValueError("validation-root and output-dir are required")

    run_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
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
            metrics, bins = analyze_payload(payload, config, dataset, seed)
            run_rows.append(metrics)
            bin_rows.extend(bins)

    summary, dataset_rows, stability_rows = summarize(
        run_rows,
        bin_rows,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_metrics.csv", run_rows)
    write_csv(args.output_dir / "bin_credit_profiles.csv", bin_rows)
    write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(args.output_dir / "seed_stability.csv", stability_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_scc_d0=complete runs={summary['runs']} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
