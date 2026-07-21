#!/usr/bin/env python3
"""Analyze the four-layer SC1-SIFF-v3-TSAF Phase-A matrix."""

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
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_v3_tsaf_step7b.json"),
    )
    parser.add_argument("--seed", type=int, default=2021)
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def arm_directory(
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    new_root: Path,
    reference_root: Path,
) -> Path:
    if arm["source"] == "reused_reference":
        return run_dir(reference_root, arm["source_arm"], dataset, seed)
    return run_dir(new_root, arm["id"], dataset, seed)


def load_run(
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    new_root: Path,
    reference_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = arm_directory(
        arm,
        dataset,
        seed,
        new_root,
        reference_root,
    )
    required = {
        "checkpoint": directory / "checkpoint.pt",
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariants": directory / "test_audit_invariants.json",
        "effective": directory / "effective_config.json",
        "initialization": directory / "initialization_contract.json",
        "training": directory / "training_log.csv",
    }
    if not arm["id"].startswith("a6_"):
        required["diagnostics"] = directory / "pcsd_test_audit_diagnostics.npz"
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
            "source": arm["source"],
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    invariants = json.loads(required["invariants"].read_text(encoding="utf-8"))
    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    adapter = effective["adapter"]
    checkpoint_hash = file_hash(required["checkpoint"])
    expected_hash = None
    if arm["source"] == "reused_reference":
        expected_hash = config["reference_contract"]["checkpoint_sha256"][
            arm["id"]
        ][dataset]
    metric_lookup = {
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
    }
    metrics = []
    for horizon in config["matrix"]["horizons"]:
        row = metric_lookup[horizon]
        mse, mae = float(row["mse"]), float(row["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metric: {directory} H{horizon}")
        metrics.append(
            {
                "dataset": dataset,
                "arm": arm["id"],
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
                "seed": seed,
                "source": arm["source"],
            }
        )
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["pcc_objective_mode"] == arm["objective_mode"]
        and adapter["validation_horizons"]
        == config["training"]["validation_horizons"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["final_evaluation_split"] == "val"
        and invariants.get("pass") is True
        and invariants.get("evaluation_split") == "test"
        and invariants.get("uses_test_split") is True
        and invariants.get("test_access_authorized") is True
        and invariants.get("checkpoint_sha256") == checkpoint_hash
        and (expected_hash is None or checkpoint_hash == expected_hash)
    )
    if not arm["id"].startswith("a6_"):
        protocol_pass = protocol_pass and bool(
            adapter["pcsd_policy_mode"] == arm["policy_mode"]
        )
    return metrics, {
        "dataset": dataset,
        "arm": arm["id"],
        "source": arm["source"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": checkpoint_hash,
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash",
            "",
        ),
        "run_dir": str(directory),
    }


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    cells = []
    summaries = []
    for comparison in config["comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains = []
            dataset_gains: dict[str, list[float]] = {}
            horizon_gains: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[(dataset, comparison["candidate"], horizon)]
                    reference = lookup[(dataset, comparison["reference"], horizon)]
                    candidate_value = float(candidate[metric])
                    reference_value = float(reference[metric])
                    gain = 100.0 * (1.0 - candidate_value / reference_value)
                    gains.append(gain)
                    dataset_gains.setdefault(dataset, []).append(gain)
                    horizon_gains.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "layer": comparison["layer"],
                            "metric": metric,
                            "candidate": comparison["candidate"],
                            "reference": comparison["reference"],
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
                    "layer": comparison["layer"],
                    "metric": metric,
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in dataset_gains.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0 for values in horizon_gains.values()
                    ),
                }
            )
    return cells, summaries


def performance_gates(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = {(row["comparison"], row["metric"]): row for row in summaries}
    primary = config["gates"]["primary_effectiveness"]
    primary_results = {}
    for comparison in primary["comparison_ids"]:
        mse = lookup[(comparison, "mse")]
        mae = lookup[(comparison, "mae")]
        primary_results[comparison] = bool(
            mse["macro_gain_percent"] >= primary["mse_macro_gain_percent_min"]
            and mse["dataset_wins"] >= primary["mse_dataset_wins_min"]
            and mse["horizon_wins"] >= primary["mse_horizon_wins_min"]
            and mse["cell_wins"] >= primary["mse_cell_wins_min"]
            and mae["macro_gain_percent"] >= primary["mae_macro_gain_percent_min"]
        )
    controls = config["gates"]["matched_controls"]
    control_results = {
        comparison: bool(
            lookup[(comparison, "mse")]["macro_gain_percent"]
            > controls["mse_macro_gain_percent_min"]
            and lookup[(comparison, "mse")]["dataset_wins"]
            >= controls["mse_dataset_wins_min"]
        )
        for comparison in controls["comparison_ids"]
    }
    shared = config["gates"]["shared_field"]
    shared_gain = lookup[(shared["comparison_id"], "mse")][
        "macro_gain_percent"
    ]
    return {
        "primary": primary_results,
        "matched_controls": control_results,
        "shared_field_positive": shared_gain
        > shared["mse_macro_gain_percent_min"],
        "shared_field_strict_superiority": shared_gain
        >= shared["strict_superiority_macro_gain_percent_min"],
    }


def pairwise_arm_nrmse(arms: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.square(arms)))), 1e-12)
    return mean(
        float(np.sqrt(np.mean(np.square(arms[:, left] - arms[:, right]))))
        / denominator
        for left, right in combinations(range(arms.shape[1]), 2)
    )


def normalized_entropy(usage: np.ndarray) -> float:
    clipped = np.clip(usage, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1) / math.log(
        usage.shape[-1]
    )
    return float(entropy.mean())


def internal_health_rows(
    new_root: Path,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        candidate_dir = run_dir(new_root, "tsaf", dataset, seed)
        permuted_dir = run_dir(new_root, "tsaf_permuted_scale", dataset, seed)
        with np.load(
            candidate_dir / "pcsd_test_audit_diagnostics.npz"
        ) as candidate, np.load(
            permuted_dir / "pcsd_test_audit_diagnostics.npz"
        ) as permuted:
            arms = candidate["probe_arms"].astype(np.float64)
            fused = candidate["probe_fused"].astype(np.float64)
            usage = candidate["policy_row_bin_usage"].astype(np.float64)
            permuted_usage = permuted["policy_row_bin_usage"].astype(np.float64)
            components = candidate["scale_component_contribution"].astype(
                np.float64
            )
            denominator = max(float(np.sqrt(np.mean(np.square(fused)))), 1e-12)
            finite = all(
                np.isfinite(value).all()
                for value in (arms, fused, usage, permuted_usage, components)
            )
            surface = usage.mean(axis=0)
            permuted_surface = permuted_usage.mean(axis=0)
            ordered_permuted_nrmse = float(
                np.sqrt(np.mean(np.square(surface - permuted_surface)))
                / max(float(np.sqrt(np.mean(np.square(surface)))), 1e-12)
            )
            component_rms = float(
                np.sqrt(np.mean(np.square(components[:, 1]))) / denominator
            )
        invariant = json.loads(
            (candidate_dir / "test_audit_invariants.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "all_finite": finite,
                "arm_prediction_pairwise_nrmse": pairwise_arm_nrmse(arms),
                "target_scale_surface_std": float(surface.std()),
                "ordered_permuted_allocation_nrmse": ordered_permuted_nrmse,
                "allocation_normalized_entropy": normalized_entropy(usage),
                "scale_component_contribution_rms": component_rms,
                "request_invariance_max_abs": float(
                    invariant["full_prefix_max_abs"]
                ),
            }
        )
    return rows


def internal_health_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    gate = config["internal_health"]
    entropy = mean(row["allocation_normalized_entropy"] for row in rows)
    return {
        "all_finite": all(row["all_finite"] for row in rows),
        "arm_prediction_diversity": mean(
            row["arm_prediction_pairwise_nrmse"] for row in rows
        )
        >= gate["arm_prediction_diversity_min"],
        "nonconstant_target_scale_surface": mean(
            row["target_scale_surface_std"] for row in rows
        )
        >= gate["target_scale_surface_std_min"],
        "scale_order_sensitivity": mean(
            row["ordered_permuted_allocation_nrmse"] for row in rows
        )
        >= gate["ordered_permuted_allocation_nrmse_min"],
        "allocation_entropy": gate["allocation_normalized_entropy_min"]
        <= entropy
        <= gate["allocation_normalized_entropy_max"],
        "scale_component_contribution": mean(
            row["scale_component_contribution_rms"] for row in rows
        )
        >= gate["scale_component_contribution_rms_min"],
        "request_invariance": max(
            row["request_invariance_max_abs"] for row in rows
        )
        <= gate["request_invariance_max_abs_max"],
    }


def decision(
    gates: dict[str, Any],
    health: dict[str, bool],
) -> dict[str, Any]:
    effectiveness = all(gates["primary"].values())
    attribution = all(gates["matched_controls"].values()) and gates[
        "shared_field_positive"
    ]
    internal = all(health.values())
    if not effectiveness:
        result = "close_exact_candidate_effectiveness_fail_rollback_step2_or_4"
        failure = "history_free_allocation_hypothesis_or_exact_design_not_supported"
    elif not attribution:
        result = "performance_partial_pass_attribution_blocked"
        failure = "capacity_or_coordinate_control_explains"
    elif not internal:
        result = "design_fault_suspected_return_step7"
        failure = "optimization_or_internal_mechanism_pathology"
    else:
        result = "phase_a_pass_confirmation_required"
        failure = "none"
    return {
        "paper_facing_effectiveness": effectiveness,
        "matched_mechanism_attribution": attribution,
        "internal_mechanism_health": internal,
        "shared_field_strict_superiority": gates[
            "shared_field_strict_superiority"
        ],
        "failure_attribution": failure,
        "decision": result,
        "confirmation_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for dataset in config["datasets"]:
        for arm in config["effective_arms"]:
            for horizon in config["matrix"]["horizons"]:
                value = 0.95 if arm["id"] == "tsaf" else 1.0
                metrics.append(
                    {
                        "dataset": dataset,
                        "arm": arm["id"],
                        "horizon": horizon,
                        "mse": value,
                        "mae": value,
                    }
                )
    _cells, summaries = comparison_rows(metrics, config)
    gates = performance_gates(summaries, config)
    health = {
        "all_finite": True,
        "arm_prediction_diversity": True,
        "nonconstant_target_scale_surface": True,
        "scale_order_sensitivity": True,
        "allocation_entropy": True,
        "scale_component_contribution": True,
        "request_invariance": True,
    }
    result = decision(gates, health)
    if result["decision"] != "phase_a_pass_confirmation_required":
        raise RuntimeError("TSAF four-layer analyzer synthetic smoke failed")
    print("tsaf_four_layer_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.new_root is None or args.reference_root is None or args.output_dir is None:
        raise ValueError("new-root, reference-root, and output-dir are required")

    metrics = []
    runs = []
    for dataset in config["datasets"]:
        for arm in config["effective_arms"]:
            arm_metrics, run = load_run(
                arm,
                dataset,
                args.seed,
                args.new_root,
                args.reference_root,
                config,
            )
            metrics.extend(arm_metrics)
            runs.append(run)
    complete = bool(
        len(runs) == config["matrix"]["effective_runs"]
        and len(metrics) == config["matrix"]["effective_official_test_cells"]
        and all(row["status"] == "ok" for row in runs)
    )
    if not complete:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "run_audit.csv", runs)
        raise RuntimeError("TSAF effective matrix is incomplete")

    cells, summaries = comparison_rows(metrics, config)
    gates = performance_gates(summaries, config)
    health_rows = internal_health_rows(args.new_root, config, args.seed)
    health = internal_health_gate(health_rows, config)
    layers = decision(gates, health)
    summary = {
        "candidate_version": config["candidate_version"],
        "test_access_date": config["authorization"].get(
            "test_access_date",
            "",
        ),
        "user_authorization": config["authorization"],
        "checkpoint_retrained": True,
        "test_role": "primary-mechanism-effectiveness-and-paper-benchmark",
        "matrix_complete": complete,
        "test_informed": True,
        "performance_gates": gates,
        "internal_health": health,
        "evaluation_layers": layers,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", runs)
    write_csv(args.output_dir / "test_metrics_standard_horizons.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "mechanism_health.csv", health_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
