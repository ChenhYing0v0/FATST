#!/usr/bin/env python3
"""Audit ISCF-v0 scope relations from existing frozen diagnostic tensors."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SEEDS = (2021, 2022, 2023)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v0_carrier.json"),
    )
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--fcc-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(np.asarray(left, dtype=np.float64))
    right_rank = average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def explained_variance(values: np.ndarray) -> tuple[float, float, float]:
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T / float(centered.shape[1])
    eigenvalues = np.linalg.eigvalsh(gram).clip(min=0.0)[::-1]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    ev1 = float(eigenvalues[0] / total)
    ev2 = float(eigenvalues[:2].sum() / total)
    effective_rank = float(total * total / np.square(eigenvalues).sum())
    return ev1, ev2, effective_rank


def common_energy_ratio(values: np.ndarray) -> float:
    total = float(np.square(values).sum())
    if total <= 0.0:
        return 0.0
    common = values.mean(axis=0)
    return float(values.shape[0] * np.square(common).sum() / total)


def symmetric_nrmse(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.sqrt(0.5 * np.mean(np.square(left) + np.square(right)))
    if denominator <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(left - right))) / denominator)


def normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1.0)
    return float(
        np.mean(-np.sum(values * np.log(values), axis=-1) / np.log(values.shape[-1]))
    )


def null_statistics(
    arms: np.ndarray,
    residuals: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    feature_count = arms.shape[1]
    ev2_values: list[float] = []
    common_values: list[float] = []
    for _ in range(repetitions):
        offsets = rng.integers(1, feature_count, size=arms.shape[0])
        shifted_arms = np.stack(
            [np.roll(row, int(offset)) for row, offset in zip(arms, offsets)]
        )
        shifted_residuals = np.stack(
            [np.roll(row, int(offset)) for row, offset in zip(residuals, offsets)]
        )
        ev2_values.append(explained_variance(shifted_arms)[1])
        common_values.append(common_energy_ratio(shifted_residuals))
    return (
        float(np.quantile(ev2_values, 0.95)),
        float(np.quantile(common_values, 0.95)),
    )


def analyze_payload(
    payload: dict[str, np.ndarray],
    dataset: str,
    seed: int,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "probe_arms",
        "probe_fused",
        "probe_targets",
        "arm_row_bin_mse",
        "fused_row_bin_mse",
        "policy_row_bin_usage",
        "bin_names",
        "scales",
    }
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"missing diagnostic arrays: {sorted(missing)}")

    probe_arms = payload["probe_arms"].astype(np.float64)
    probe_targets = payload["probe_targets"].astype(np.float64)
    arms = probe_arms.transpose(1, 0, 2).reshape(probe_arms.shape[1], -1)
    targets = probe_targets.reshape(-1)
    residuals = arms - targets[None, :]
    scales = payload["scales"].astype(np.int64)
    if tuple(scales.tolist()) != (1, 48, 144, 360, 720):
        raise ValueError(f"unexpected ISCF scopes: {scales.tolist()}")

    ev1, ev2, effective_rank = explained_variance(arms)
    residual_common = common_energy_ratio(residuals)
    rng_seed = 20260721 + 1009 * DATASETS.index(dataset) + seed
    null_ev2, null_common = null_statistics(
        arms,
        residuals,
        repetitions,
        np.random.default_rng(rng_seed),
    )

    pairs: list[dict[str, Any]] = []
    distances: list[float] = []
    log_distances: list[float] = []
    adjacent: list[float] = []
    nonadjacent: list[float] = []
    residual_correlations: list[float] = []
    for left, right in combinations(range(len(scales)), 2):
        distance = symmetric_nrmse(arms[left], arms[right])
        log_distance = float(abs(np.log(scales[left]) - np.log(scales[right])))
        correlation = float(np.corrcoef(residuals[left], residuals[right])[0, 1])
        distances.append(distance)
        log_distances.append(log_distance)
        residual_correlations.append(correlation)
        if right == left + 1:
            adjacent.append(distance)
        else:
            nonadjacent.append(distance)
        pairs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "left_scale": int(scales[left]),
                "right_scale": int(scales[right]),
                "prediction_nrmse": distance,
                "residual_correlation": correlation,
                "log_scale_distance": log_distance,
            }
        )

    arm_losses = payload["arm_row_bin_mse"].astype(np.float64)
    fused_losses = payload["fused_row_bin_mse"].astype(np.float64)
    mean_arm_losses = arm_losses.mean(axis=(0, 1))
    best_fixed = float(mean_arm_losses.min())
    fused = float(fused_losses.mean())
    oracle = float(np.min(arm_losses, axis=-1).mean())
    fused_gain = 100.0 * (best_fixed - fused) / best_fixed
    oracle_headroom = 100.0 * (fused - oracle) / fused

    bin_names = payload["bin_names"].astype(str)
    bin_arm_losses = arm_losses.mean(axis=0)
    bin_rows: list[dict[str, Any]] = []
    best_scopes: list[int] = []
    for bin_index, bin_name in enumerate(bin_names):
        best_index = int(np.argmin(bin_arm_losses[bin_index]))
        best_scopes.append(int(scales[best_index]))
        for scope_index, scale in enumerate(scales):
            bin_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "bin": str(bin_name),
                    "scale": int(scale),
                    "mean_mse": float(bin_arm_losses[bin_index, scope_index]),
                    "is_best": scope_index == best_index,
                }
            )

    policy = payload["policy_row_bin_usage"].astype(np.float64)
    policy_mean_by_bin = policy.mean(axis=0)
    policy_global = policy_mean_by_bin.mean(axis=0, keepdims=True)
    policy_bin_dispersion = float(
        np.sqrt(np.mean(np.square(policy_mean_by_bin - policy_global)))
    )

    metrics = {
        "dataset": dataset,
        "seed": seed,
        "probe_rows": int(probe_arms.shape[0]),
        "centered_scope_ev1": ev1,
        "centered_scope_ev2": ev2,
        "centered_scope_effective_rank": effective_rank,
        "shift_null_ev2_p95": null_ev2,
        "low_dimensional_above_null": ev2 > null_ev2,
        "residual_common_energy_ratio": residual_common,
        "residual_private_energy_ratio": 1.0 - residual_common,
        "shift_null_common_energy_p95": null_common,
        "common_residual_above_null": residual_common > null_common,
        "mean_pairwise_prediction_nrmse": float(np.mean(distances)),
        "mean_pairwise_residual_correlation": float(
            np.mean(residual_correlations)
        ),
        "scale_distance_spearman": spearman(
            np.asarray(log_distances), np.asarray(distances)
        ),
        "adjacent_to_nonadjacent_distance_ratio": float(
            np.mean(adjacent) / np.mean(nonadjacent)
        ),
        "fused_gain_over_best_fixed_arm_percent": fused_gain,
        "oracle_headroom_over_fused_percent": oracle_headroom,
        "unique_best_scopes_across_bins": len(set(best_scopes)),
        "best_scopes_by_bin": ";".join(map(str, best_scopes)),
        "policy_normalized_entropy": normalized_entropy(policy),
        "policy_bin_dispersion": policy_bin_dispersion,
    }
    return metrics, pairs, bin_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def build_summary(
    rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    low_dimensional_count = sum(bool(row["low_dimensional_above_null"]) for row in rows)
    common_count = sum(bool(row["common_residual_above_null"]) for row in rows)
    stable_dataset_count = sum(
        float(row["median_seed_topology_spearman"])
        >= float(gates["topology_seed_rho_min"])
        for row in topology_rows
    )
    low_dimensional = low_dimensional_count >= int(
        gates["low_dimensional_run_count_min"]
    )
    common_specific = (
        common_count >= int(gates["common_residual_run_count_min"])
        and median(rows, "residual_private_energy_ratio")
        >= float(gates["private_energy_median_min"])
    )
    complementarity = (
        median(rows, "oracle_headroom_over_fused_percent")
        >= float(gates["oracle_headroom_median_min_percent"])
        and median(rows, "unique_best_scopes_across_bins")
        >= float(gates["unique_best_scopes_median_min"])
    )
    topology_stable = stable_dataset_count >= int(
        gates["topology_dataset_count_min"]
    )
    if low_dimensional and common_specific and complementarity and topology_stable:
        decision = "function_relation_supported_for_new_step4_problem"
    elif low_dimensional or common_specific or complementarity or topology_stable:
        decision = "function_relation_unresolved_requires_narrow_step4_audit"
    else:
        decision = "stable_function_relation_not_supported"

    order_positive_count = sum(
        float(row["scale_distance_spearman"]) > 0.0 for row in rows
    )
    return {
        "carrier_id": "ISCF-v0",
        "audit_role": "diagnostic_only_test_informed_reuse",
        "run_units": len(rows),
        "all_arrays_finite": all(
            all(
                np.isfinite(float(value))
                for key, value in row.items()
                if key not in {"dataset", "best_scopes_by_bin"}
                and not isinstance(value, bool)
            )
            for row in rows
        ),
        "statistics": {
            "median_centered_scope_ev2": median(rows, "centered_scope_ev2"),
            "median_shift_null_ev2_p95": median(rows, "shift_null_ev2_p95"),
            "low_dimensional_above_null_count": low_dimensional_count,
            "median_residual_common_energy_ratio": median(
                rows, "residual_common_energy_ratio"
            ),
            "median_shift_null_common_energy_p95": median(
                rows, "shift_null_common_energy_p95"
            ),
            "common_residual_above_null_count": common_count,
            "median_residual_private_energy_ratio": median(
                rows, "residual_private_energy_ratio"
            ),
            "median_pairwise_prediction_nrmse": median(
                rows, "mean_pairwise_prediction_nrmse"
            ),
            "median_pairwise_residual_correlation": median(
                rows, "mean_pairwise_residual_correlation"
            ),
            "median_oracle_headroom_percent": median(
                rows, "oracle_headroom_over_fused_percent"
            ),
            "median_fused_gain_over_best_fixed_percent": median(
                rows, "fused_gain_over_best_fixed_arm_percent"
            ),
            "median_unique_best_scopes": median(
                rows, "unique_best_scopes_across_bins"
            ),
            "median_policy_entropy": median(rows, "policy_normalized_entropy"),
            "median_scale_distance_spearman": median(
                rows, "scale_distance_spearman"
            ),
            "positive_scale_order_count": order_positive_count,
            "stable_topology_dataset_count": stable_dataset_count,
        },
        "gates": {
            "low_dimensional_relation": low_dimensional,
            "common_and_private_structure": common_specific,
            "scope_complementarity": complementarity,
            "cross_seed_topology_stability": topology_stable,
        },
        "decision": decision,
        "method_effectiveness_established": False,
        "new_training_or_test_access": False,
    }


def build_dataset_summaries(
    rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    topology = {row["dataset"]: row for row in topology_rows}
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        summaries.append(
            {
                "dataset": dataset,
                "runs": len(selected),
                "median_centered_scope_ev2": median(
                    selected, "centered_scope_ev2"
                ),
                "low_dimensional_above_null_count": sum(
                    bool(row["low_dimensional_above_null"])
                    for row in selected
                ),
                "median_residual_common_energy_ratio": median(
                    selected, "residual_common_energy_ratio"
                ),
                "median_residual_private_energy_ratio": median(
                    selected, "residual_private_energy_ratio"
                ),
                "median_pairwise_prediction_nrmse": median(
                    selected, "mean_pairwise_prediction_nrmse"
                ),
                "median_pairwise_residual_correlation": median(
                    selected, "mean_pairwise_residual_correlation"
                ),
                "median_fused_gain_over_best_fixed_percent": median(
                    selected, "fused_gain_over_best_fixed_arm_percent"
                ),
                "fused_positive_run_count": sum(
                    float(row["fused_gain_over_best_fixed_arm_percent"]) > 0.0
                    for row in selected
                ),
                "median_oracle_headroom_percent": median(
                    selected, "oracle_headroom_over_fused_percent"
                ),
                "median_unique_best_scopes": median(
                    selected, "unique_best_scopes_across_bins"
                ),
                "median_policy_entropy": median(
                    selected, "policy_normalized_entropy"
                ),
                "median_scale_distance_spearman": median(
                    selected, "scale_distance_spearman"
                ),
                "positive_scale_order_run_count": sum(
                    float(row["scale_distance_spearman"]) > 0.0
                    for row in selected
                ),
                "median_seed_topology_spearman": topology[dataset][
                    "median_seed_topology_spearman"
                ],
            }
        )
    return summaries


def synthetic_smoke() -> None:
    rng = np.random.default_rng(20260721)
    rows = 32
    time = 48
    base = rng.normal(size=(rows, time))
    factors = rng.normal(size=(2, rows, time))
    coefficients = np.asarray(
        [[-1.0, 0.2], [-0.5, 0.7], [0.0, 1.0], [0.6, 0.5], [1.0, -0.2]]
    )
    arms = base[:, None, :] + np.einsum("sq,qrt->rst", coefficients, factors)
    targets = base + 0.1 * rng.normal(size=base.shape)
    errors = np.square(arms - targets[:, None, :]).mean(axis=-1)
    payload = {
        "probe_arms": arms.astype(np.float32),
        "probe_fused": arms.mean(axis=1).astype(np.float32),
        "probe_targets": targets.astype(np.float32),
        "arm_row_bin_mse": np.repeat(errors[:, None, :], 8, axis=1).astype(
            np.float32
        ),
        "fused_row_bin_mse": np.repeat(
            np.square(arms.mean(axis=1) - targets).mean(axis=-1)[:, None],
            8,
            axis=1,
        ).astype(np.float32),
        "policy_row_bin_usage": np.full((rows, 8, 5), 0.2, dtype=np.float32),
        "bin_names": np.asarray([f"b{index}" for index in range(8)]),
        "scales": np.asarray([1, 48, 144, 360, 720]),
    }
    metrics, pairs, bins = analyze_payload(payload, "ETTh1", 2021, 8)
    if not metrics["centered_scope_ev2"] > metrics["shift_null_ev2_p95"]:
        raise RuntimeError("synthetic low-dimensional relation was not recovered")
    if len(pairs) != 10 or len(bins) != 40:
        raise RuntimeError("synthetic audit output shape mismatch")
    summary = build_summary(
        [dict(metrics) for _ in range(15)],
        [
            {
                "dataset": dataset,
                "median_seed_topology_spearman": 0.9,
            }
            for dataset in DATASETS
        ],
        {
            "low_dimensional_run_count_min": 12,
            "common_residual_run_count_min": 12,
            "topology_seed_rho_min": 0.5,
            "topology_dataset_count_min": 4,
            "private_energy_median_min": 0.05,
            "oracle_headroom_median_min_percent": 1.0,
            "unique_best_scopes_median_min": 2.0,
        },
    )
    if summary["method_effectiveness_established"] is not False:
        raise RuntimeError("diagnostic-only boundary was not preserved")
    dataset_summaries = build_dataset_summaries(
        [
            {**metrics, "dataset": dataset}
            for dataset in DATASETS
            for _ in SEEDS
        ],
        [
            {
                "dataset": dataset,
                "median_seed_topology_spearman": 0.9,
            }
            for dataset in DATASETS
        ],
    )
    if len(dataset_summaries) != len(DATASETS):
        raise RuntimeError("synthetic dataset aggregation failed")
    print("iscf_v0_function_audit_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    roots = config["artifact_roots"]
    historical_root = args.historical_root or Path(roots["seed2021"])
    fcc_root = args.fcc_root or Path(roots["seed2022_2023"])
    repetitions = int(config["function_audit"]["shuffle_repetitions"])

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    topology_vectors: dict[tuple[str, int], np.ndarray] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            root = historical_root if seed == 2021 else fcc_root
            path = (
                root
                / dataset
                / "h720_full"
                / f"seed{seed}"
                / "pcsd_test_audit_diagnostics.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            metrics, pairs, bins = analyze_payload(
                payload, dataset, seed, repetitions
            )
            rows.append(metrics)
            pair_rows.extend(pairs)
            bin_rows.extend(bins)
            topology_vectors[(dataset, seed)] = np.asarray(
                [row["prediction_nrmse"] for row in pairs], dtype=np.float64
            )

    topology_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        correlations = [
            spearman(
                topology_vectors[(dataset, left)],
                topology_vectors[(dataset, right)],
            )
            for left, right in combinations(SEEDS, 2)
        ]
        topology_rows.append(
            {
                "dataset": dataset,
                "seed2021_2022_spearman": correlations[0],
                "seed2021_2023_spearman": correlations[1],
                "seed2022_2023_spearman": correlations[2],
                "median_seed_topology_spearman": float(
                    np.median(correlations)
                ),
            }
        )

    summary = build_summary(
        rows,
        topology_rows,
        config["function_audit"]["gates"],
    )
    dataset_summaries = build_dataset_summaries(rows, topology_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_function_metrics.csv", rows)
    write_csv(args.output_dir / "pairwise_scope_metrics.csv", pair_rows)
    write_csv(args.output_dir / "bin_scope_specialization.csv", bin_rows)
    write_csv(args.output_dir / "seed_topology_stability.csv", topology_rows)
    write_csv(args.output_dir / "dataset_function_summary.csv", dataset_summaries)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_v0_function_audit=pass runs={len(rows)} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()
