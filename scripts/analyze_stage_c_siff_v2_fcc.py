#!/usr/bin/env python3
"""Analyze the three-seed SIFF-v2 Final Claim Confirmation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_v2_fcc_v1.json"),
    )
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    horizons: list[int],
    *,
    historical: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariant": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "training": directory / "training_log.csv",
        "initialization": directory / "initialization_contract.json",
        "checkpoint": directory / "checkpoint.pt",
        "diagnostics": directory / "pcsd_test_audit_diagnostics.npz",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
            "seed": seed,
            "source": "historical" if historical else "new",
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    invariant = json.loads(required["invariant"].read_text(encoding="utf-8"))
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    adapter = effective["adapter"]
    metric_lookup = {
        int(row["target_horizon"]): row
        for row in read_csv(required["metrics"])
    }
    selected = []
    for horizon in horizons:
        row = metric_lookup[horizon]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metric: {directory} H{horizon}")
        selected.append(
            {
                "dataset": dataset,
                "arm": arm["id"],
                "horizon": horizon,
                "seed": seed,
                "mse": mse,
                "mae": mae,
                "source": "historical" if historical else "new",
            }
        )

    checkpoint_hash = file_sha256(required["checkpoint"])
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and int(adapter["seed"]) == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["pcc_objective_mode"] == arm["objective_mode"]
        and adapter["validation_horizons"] == [96, 192, 336, 720]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["final_evaluation_split"] == "val"
        and invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("uses_test_split") is True
        and invariant.get("test_access_authorized") is True
        and invariant.get("checkpoint_sha256") == checkpoint_hash
        and (
            historical
            or invariant.get("checkpoint_retrained") is True
        )
    )
    return selected, {
        "dataset": dataset,
        "arm": arm["id"],
        "seed": seed,
        "source": "historical" if historical else "new",
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": checkpoint_hash,
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash", ""
        ),
        "full_prefix_max_abs": float(
            invariant.get("full_prefix_max_abs", float("inf"))
        ),
        "run_dir": str(directory),
    }


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["seed"], row["dataset"], row["arm"], row["horizon"]): row
        for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    all_seeds = [config["historical_seed"], *config["seeds"]]
    for comparison in config["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains: list[float] = []
            dataset_gains: dict[str, list[float]] = {}
            horizon_gains: dict[int, list[float]] = {}
            seed_gains: dict[int, list[float]] = {}
            for seed in all_seeds:
                for dataset in config["datasets"]:
                    for horizon in config["matrix"]["horizons"]:
                        candidate = lookup[
                            (seed, dataset, comparison["candidate"], horizon)
                        ]
                        reference = lookup[
                            (seed, dataset, comparison["reference"], horizon)
                        ]
                        candidate_value = float(candidate[metric])
                        reference_value = float(reference[metric])
                        gain = 100.0 * (1.0 - candidate_value / reference_value)
                        gains.append(gain)
                        dataset_gains.setdefault(dataset, []).append(gain)
                        horizon_gains.setdefault(horizon, []).append(gain)
                        seed_gains.setdefault(seed, []).append(gain)
                        cells.append(
                            {
                                "comparison": comparison["id"],
                                "role": comparison["role"],
                                "metric": metric,
                                "seed": seed,
                                "dataset": dataset,
                                "horizon": horizon,
                                "gain_percent": gain,
                                "candidate_value": candidate_value,
                                "reference_value": reference_value,
                            }
                        )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "role": comparison["role"],
                    "metric": metric,
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0
                        for values in dataset_gains.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0
                        for values in horizon_gains.values()
                    ),
                    "seed_wins": sum(
                        mean(values) > 0.0 for values in seed_gains.values()
                    ),
                    "seed2021_gain_percent": mean(
                        seed_gains[config["historical_seed"]]
                    ),
                    "seed2022_gain_percent": mean(seed_gains[2022]),
                    "seed2023_gain_percent": mean(seed_gains[2023]),
                }
            )
    return cells, summaries


def comparison_gates(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    lookup = {
        (row["comparison"], row["metric"]): row for row in summaries
    }
    gates = config["effectiveness_gates"]
    results = {}
    for comparison in gates["comparison_ids"]:
        mse = lookup[(comparison, "mse")]
        mae = lookup[(comparison, "mae")]
        results[comparison] = bool(
            float(mse["macro_gain_percent"])
            >= gates["mse_macro_gain_percent_min"]
            and float(mae["macro_gain_percent"])
            > gates["mae_macro_gain_percent_min_exclusive"]
            and int(mse["dataset_wins"])
            >= gates["mse_dataset_wins_min"]
            and int(mse["horizon_wins"])
            >= gates["mse_horizon_wins_min"]
            and int(mse["seed_wins"]) >= gates["mse_seed_wins_min"]
        )
    return results


def pairwise_nrmse(probe_arms: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.square(probe_arms)))), 1e-12)
    values = []
    for left, right in combinations(range(probe_arms.shape[1]), 2):
        difference = probe_arms[:, left] - probe_arms[:, right]
        values.append(float(np.sqrt(np.mean(np.square(difference))) / denominator))
    return mean(values)


def normalized_entropy(usage: np.ndarray) -> float:
    clipped = np.clip(usage, 1e-12, 1.0)
    return float(
        (-(clipped * np.log(clipped)).sum(axis=-1) / math.log(usage.shape[-1])).mean()
    )


def internal_health(
    historical_root: Path,
    new_root: Path,
    config: dict[str, Any],
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = []
    all_seeds = [config["historical_seed"], *config["seeds"]]
    for seed in all_seeds:
        root = historical_root if seed == config["historical_seed"] else new_root
        for dataset in config["datasets"]:
            directory = run_dir(root, "siff_equal", dataset, seed)
            diagnostic_path = directory / "pcsd_test_audit_diagnostics.npz"
            with np.load(diagnostic_path) as diagnostics:
                required = {
                    "arm_row_bin_mse",
                    "fused_row_bin_mse",
                    "probe_arms",
                    "probe_fused",
                    "policy_row_bin_usage",
                    "scale_component_contribution",
                }
                missing = required - set(diagnostics.files)
                if missing:
                    raise ValueError(f"{directory} missing diagnostics {sorted(missing)}")
                arm_loss = diagnostics["arm_row_bin_mse"].astype(np.float64)
                fused_loss = diagnostics["fused_row_bin_mse"].astype(np.float64)
                probe_arms = diagnostics["probe_arms"].astype(np.float64)
                probe_fused = diagnostics["probe_fused"].astype(np.float64)
                usage = diagnostics["policy_row_bin_usage"].astype(np.float64)
                components = diagnostics["scale_component_contribution"].astype(
                    np.float64
                )
                arrays = (arm_loss, fused_loss, probe_arms, probe_fused, usage, components)
                denominator = max(
                    float(np.sqrt(np.mean(np.square(probe_fused)))), 1e-12
                )
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "all_finite": all(np.isfinite(array).all() for array in arrays),
                        "oracle_gain_percent": 100.0
                        * (
                            1.0
                            - float(np.mean(np.min(arm_loss, axis=-1)))
                            / float(np.mean(fused_loss))
                        ),
                        "pairwise_arm_nrmse": pairwise_nrmse(probe_arms),
                        "policy_normalized_entropy": normalized_entropy(usage),
                        "nonconstant_component_rms_ratio": float(
                            np.sqrt(np.mean(np.square(components[:, 1])))
                            / denominator
                        ),
                    }
                )

    prefix_gap = max(float(row["full_prefix_max_abs"]) for row in run_rows)
    gates = config["internal_mechanism_health"]
    dataset_oracle = {
        dataset: mean(
            float(row["oracle_gain_percent"])
            for row in rows
            if row["dataset"] == dataset
        )
        for dataset in config["datasets"]
    }
    entropy = mean(float(row["policy_normalized_entropy"]) for row in rows)
    results = {
        "finite": all(bool(row["all_finite"]) for row in rows),
        "prefix_projectivity": prefix_gap
        <= gates["prefix_projectivity_gap_max"],
        "oracle_headroom": sum(value > 0.0 for value in dataset_oracle.values())
        >= gates["oracle_positive_datasets_min"],
        "arm_diversity": mean(float(row["pairwise_arm_nrmse"]) for row in rows)
        >= gates["mean_pairwise_probe_nrmse_min"],
        "policy_entropy": gates["macro_policy_entropy_min"]
        <= entropy
        <= gates["macro_policy_entropy_max"],
        "nonconstant_component_use": mean(
            float(row["nonconstant_component_rms_ratio"]) for row in rows
        )
        >= gates["nonconstant_scale_component_rms_ratio_min"],
    }
    return rows, results


def final_decision(
    comparison_results: dict[str, bool],
    health_results: dict[str, bool],
) -> tuple[str, str]:
    if not comparison_results["siff_over_a6_full"]:
        return (
            "siff_v2_final_claim_not_confirmed_stop_paper_core_rescue",
            "hypothesis_false_or_seed_instability",
        )
    if not comparison_results["ordered_over_independent_equal"]:
        return (
            "performance_pass_attribution_blocked_stop_fcc_promotion",
            "capacity_control_explains",
        )
    if not all(health_results.values()):
        return "design_fault_suspected_no_promotion", "internal_mechanism_health_fail"
    return "passed_core_candidate_pending_modern_baselines", "none"


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for seed in [config["historical_seed"], *config["seeds"]]:
        for dataset in config["datasets"]:
            for arm in config["arms"]:
                value = 0.98 if arm["id"] == "siff_equal" else 1.0
                for horizon in config["matrix"]["horizons"]:
                    metrics.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "arm": arm["id"],
                            "horizon": horizon,
                            "mse": value,
                            "mae": value,
                        }
                    )
    _cells, summaries = comparison_rows(metrics, config)
    results = comparison_gates(summaries, config)
    if not all(results.values()):
        raise RuntimeError("FCC synthetic comparison gate failed")
    print("siff_v2_fcc_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.new_root is None or args.historical_root is None or args.output_dir is None:
        raise ValueError("new-root, historical-root and output-dir are required")

    metrics: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for seed in [config["historical_seed"], *config["seeds"]]:
        historical = seed == config["historical_seed"]
        root = args.historical_root if historical else args.new_root
        for dataset in config["datasets"]:
            for arm in config["arms"]:
                run_metrics, run = load_run(
                    root,
                    arm,
                    dataset,
                    seed,
                    config["matrix"]["horizons"],
                    historical=historical,
                )
                metrics.extend(run_metrics)
                runs.append(run)

    expected_runs = config["matrix"]["effective_runs"]
    expected_cells = config["matrix"]["effective_test_cells"]
    complete = bool(
        len(runs) == expected_runs
        and len(metrics) == expected_cells
        and all(row["status"] == "ok" for row in runs)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", runs)
    if not complete:
        raise RuntimeError("FCC matrix is incomplete or failed protocol audit")

    encoder_initialization_matched = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in runs
                if row["dataset"] == dataset and row["seed"] == seed
            }
        )
        == 1
        for dataset in config["datasets"]
        for seed in [config["historical_seed"], *config["seeds"]]
    )
    unique_checkpoints = len({row["checkpoint_sha256"] for row in runs})
    if not encoder_initialization_matched or unique_checkpoints != expected_runs:
        raise RuntimeError("FCC initialization pairing or checkpoint uniqueness failed")

    cells, summaries = comparison_rows(metrics, config)
    comparisons = comparison_gates(summaries, config)
    health_rows, health = internal_health(
        args.historical_root,
        args.new_root,
        config,
        runs,
    )
    decision, failure = final_decision(comparisons, health)
    summary = {
        "candidate_version": config["candidate_version"],
        "user_authorization": config["authorization"],
        "comparator_change": config["comparator_change"],
        "test_role": config["authorization"]["test_role"],
        "matrix_complete": complete,
        "effective_runs": len(runs),
        "effective_test_cells": len(metrics),
        "checkpoint_retrained": True,
        "checkpoint_hashes_unique": unique_checkpoints == expected_runs,
        "encoder_initialization_matched": encoder_initialization_matched,
        "paper_facing_effectiveness": comparisons["siff_over_a6_full"],
        "matched_mechanism_attribution": comparisons[
            "ordered_over_independent_equal"
        ],
        "internal_mechanism_health": all(health.values()),
        "comparison_gates": comparisons,
        "internal_health_gates": health,
        "failure_attribution": failure,
        "decision": decision,
    }
    write_csv(args.output_dir / "test_metrics_standard_horizons.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "mechanism_health.csv", health_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"siff_v2_fcc_decision={decision}")


if __name__ == "__main__":
    main()
