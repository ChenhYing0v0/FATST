#!/usr/bin/env python3
"""Decompose best-arm risk and convex-mixture risk for SIFF-CCSF diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/stage_c_siff_ccsf_d3_mixture_risk_decomposition.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/"
            "d3_mixture_risk_decomposition"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def simplex_oracle_losses(residual: np.ndarray) -> np.ndarray:
    """Return exact active-set simplex-oracle loss per region.

    Args:
        residual: Array with shape [M, L, S].
    """
    if residual.shape[1] == 1:
        point = residual[:, 0, :]
        min_abs = np.min(np.abs(point), axis=1)
        crosses_zero = (np.min(point, axis=1) <= 0.0) & (
            np.max(point, axis=1) >= 0.0
        )
        return np.where(crosses_zero, 0.0, min_abs**2)

    gram = np.einsum("mls,mlt->mst", residual, residual, optimize=True)
    gram /= residual.shape[1]
    arm_count = gram.shape[-1]
    best = np.min(np.diagonal(gram, axis1=1, axis2=2), axis=1)
    ones_cache = {
        size: np.ones((size, 1), dtype=np.float64)
        for size in range(2, arm_count + 1)
    }

    for size in range(2, arm_count + 1):
        for active in itertools.combinations(range(arm_count), size):
            indices = np.asarray(active)
            sub = gram[:, indices][:, :, indices]
            scale = np.maximum(
                np.max(np.abs(sub), axis=(1, 2)),
                np.finfo(np.float64).eps,
            )
            ridge = (1e-10 * scale)[:, None, None]
            regularized = sub + ridge * np.eye(size)[None, :, :]
            rhs = np.broadcast_to(ones_cache[size], (sub.shape[0], size, 1))
            raw = np.linalg.solve(regularized, rhs)[..., 0]
            denominator = np.sum(raw, axis=1)
            valid_denominator = np.abs(denominator) > 1e-12
            weights = np.zeros_like(raw)
            weights[valid_denominator] = (
                raw[valid_denominator]
                / denominator[valid_denominator, None]
            )
            valid = valid_denominator & np.all(weights >= -1e-8, axis=1)
            candidate = np.einsum(
                "mi,mij,mj->m", weights, sub, weights, optimize=True
            )
            best = np.where(valid, np.minimum(best, candidate), best)

    return np.maximum(best, 0.0)


def region_view(array: np.ndarray, width: int) -> np.ndarray:
    sample_count, horizon = array.shape[:2]
    if horizon % width != 0:
        raise ValueError(f"Horizon {horizon} is not divisible by width {width}.")
    region_count = horizon // width
    return array.reshape(sample_count, region_count, width, *array.shape[2:])


def load_dataset_artifact(
    root: Path, dataset: str, seed: int, required_file: str
) -> dict[str, np.ndarray]:
    matches = sorted(
        root.glob(
            f"{dataset}/h720_full/seed{seed}/{required_file}"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one artifact for {dataset}, found {len(matches)}."
        )
    required_keys = (
        "probe_arms",
        "probe_targets",
        "probe_base_policy",
        "probe_policy",
    )
    with np.load(matches[0]) as archive:
        return {
            key: archive[key].astype(np.float64) for key in required_keys
        }


def analyze_dataset(
    arrays: dict[str, np.ndarray], dataset: str, widths: list[int]
) -> list[dict[str, Any]]:
    arms = arrays["probe_arms"]
    target = arrays["probe_targets"]
    base_weights = arrays["probe_base_policy"]
    final_weights = arrays["probe_policy"]
    residual = np.transpose(arms, (0, 2, 1)) - target[:, :, None]
    base_forecast = np.einsum(
        "nts,nst->nt", base_weights, arms, optimize=True
    )
    final_forecast = np.einsum(
        "nts,nst->nt", final_weights, arms, optimize=True
    )

    rows: list[dict[str, Any]] = []
    for width in widths:
        region_residual = region_view(residual, width)
        flattened = region_residual.reshape(-1, width, arms.shape[1])
        per_arm = np.mean(flattened**2, axis=1)
        best_arm = np.min(per_arm, axis=1)
        simplex = simplex_oracle_losses(flattened)
        uniform = np.mean(np.mean(flattened, axis=2) ** 2, axis=1)
        base = np.mean(
            region_view(base_forecast - target, width) ** 2, axis=2
        ).reshape(-1)
        final = np.mean(
            region_view(final_forecast - target, width) ** 2, axis=2
        ).reshape(-1)

        best_mean = float(np.mean(best_arm))
        simplex_mean = float(np.mean(simplex))
        uniform_mean = float(np.mean(uniform))
        base_mean = float(np.mean(base))
        final_mean = float(np.mean(final))
        denominator = uniform_mean - simplex_mean
        cross_term_share = (
            (best_mean - simplex_mean) / denominator
            if denominator > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "width": width,
                "region_count": int(flattened.shape[0]),
                "uniform_mse": uniform_mean,
                "base_policy_mse": base_mean,
                "learned_final_policy_mse": final_mean,
                "best_single_arm_oracle_mse": best_mean,
                "simplex_mixture_oracle_mse": simplex_mean,
                "best_arm_gain_over_uniform_pct": pct_gain(
                    uniform_mean, best_mean
                ),
                "simplex_gain_over_uniform_pct": pct_gain(
                    uniform_mean, simplex_mean
                ),
                "simplex_gain_over_best_arm_pct": pct_gain(
                    best_mean, simplex_mean
                ),
                "simplex_gain_over_learned_final_pct": pct_gain(
                    final_mean, simplex_mean
                ),
                "cross_term_share_of_uniform_to_simplex_gap": cross_term_share,
                "zero_simplex_loss_fraction": float(
                    np.mean(simplex <= 1e-12)
                ),
            }
        )
    return rows


def aggregate(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["problem_gate"]
    eligible = set(gate["eligible_widths"])
    width_rows: list[dict[str, Any]] = []
    pass_count = 0
    for width in config["region_widths"]:
        selected = [row for row in rows if row["width"] == width]
        summary: dict[str, Any] = {
            "width": width,
            "dataset_count": len(selected),
        }
        metric_names = [
            "best_arm_gain_over_uniform_pct",
            "simplex_gain_over_uniform_pct",
            "simplex_gain_over_best_arm_pct",
            "simplex_gain_over_learned_final_pct",
            "cross_term_share_of_uniform_to_simplex_gap",
            "zero_simplex_loss_fraction",
        ]
        for metric in metric_names:
            summary[f"macro_{metric}"] = float(
                np.mean([row[metric] for row in selected])
            )
        dataset_pass = sum(
            row["simplex_gain_over_best_arm_pct"]
            >= gate["minimum_simplex_gain_over_best_arm_pct"]
            and row["simplex_gain_over_learned_final_pct"]
            >= gate["minimum_simplex_gain_over_learned_final_pct"]
            and row["cross_term_share_of_uniform_to_simplex_gap"]
            >= gate["minimum_cross_term_share"]
            for row in selected
        )
        summary["dataset_pass_count"] = dataset_pass
        summary["width_pass"] = (
            width in eligible
            and dataset_pass >= gate["minimum_dataset_pass_count"]
        )
        if summary["width_pass"]:
            pass_count += 1
        width_rows.append(summary)

    problem_supported = pass_count >= gate["minimum_passing_widths"]
    decision_key = "pass" if problem_supported else "fail"
    return {
        "candidate": config["candidate"],
        "current_step": config["current_step"],
        "width_summaries": width_rows,
        "passing_eligible_width_count": pass_count,
        "minimum_passing_widths": gate["minimum_passing_widths"],
        "problem_gate_pass": problem_supported,
        "decision": config["decision_contract"][decision_key],
        "limitations": [
            "Oracle weights use target labels and are not deployable.",
            "A large simplex oracle gap establishes objective mismatch, not descriptor predictability.",
            "Classical forecast-combination and covariance-aware weighting are strong prior-art constraints on any subsequent method claim."
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source = config["source_artifacts"]
    root = Path(source["root"])
    all_rows: list[dict[str, Any]] = []
    for dataset in source["datasets"]:
        arrays = load_dataset_artifact(
            root, dataset, source["seed"], source["required_file"]
        )
        all_rows.extend(
            analyze_dataset(arrays, dataset, config["region_widths"])
        )

    summary = aggregate(all_rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dataset_width_metrics.csv", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
