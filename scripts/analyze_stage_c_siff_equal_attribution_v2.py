#!/usr/bin/env python3
"""Analyze the four-layer SIFF_EQUAL attribution matrix."""

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
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_equal_attribution_v2.json"),
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


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    horizons: list[int],
    config: dict[str, Any],
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
    required_for_arm = dict(required)
    if arm["id"].startswith("a6_"):
        required_for_arm.pop("diagnostics")
    missing = [
        name for name, path in required_for_arm.items() if not path.is_file()
    ]
    if missing:
        return [], {
            "dataset": dataset,
            "arm": arm["id"],
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
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
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
                "mse": mse,
                "mae": mae,
                "seed": seed,
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
        and invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("uses_test_split") is True
        and invariant.get("test_access_authorized") is True
        and invariant.get("checkpoint_retrained") is True
        and invariant.get("checkpoint_sha256")
        == file_hash(required["checkpoint"])
    )
    return selected, {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": invariant["checkpoint_sha256"],
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
                    candidate = lookup[
                        (dataset, comparison["candidate"], horizon)
                    ]
                    reference = lookup[
                        (dataset, comparison["reference"], horizon)
                    ]
                    candidate_value = float(candidate[metric])
                    reference_value = float(reference[metric])
                    gain = 100.0 * (1.0 - candidate_value / reference_value)
                    gains.append(gain)
                    dataset_gains.setdefault(dataset, []).append(gain)
                    horizon_gains.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "role": comparison["role"],
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
                }
            )
    return cells, summaries


def hard_gate_results(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    lookup = {
        (row["comparison"], row["metric"]): row for row in summaries
    }
    gates = config["effectiveness_gates"]
    threshold = gates["per_hard_comparison"]
    results = {}
    for comparison in gates["hard_comparison_ids"]:
        row = lookup[(comparison, "mse")]
        passed = bool(
            float(row["macro_gain_percent"])
            >= threshold["mse_macro_gain_percent_min"]
            and int(row["dataset_wins"])
            >= threshold["mse_dataset_wins_min"]
            and int(row["horizon_wins"])
            >= threshold["mse_horizon_wins_min"]
            and int(row["cell_wins"]) >= threshold["mse_cell_wins_min"]
        )
        if comparison in gates["main_mae_comparison_ids"]:
            mae = lookup[(comparison, "mae")]
            passed = passed and bool(
                float(mae["macro_gain_percent"])
                >= gates["main_comparison_mae_macro_gain_percent_min"]
            )
        results[comparison] = passed
    return results


def normalized_pairwise_nrmse(probe_arms: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(np.square(probe_arms)))), 1e-12)
    values = [
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        probe_arms[:, left] - probe_arms[:, right]
                    )
                )
            )
            / denominator
        )
        for left, right in combinations(range(probe_arms.shape[1]), 2)
    ]
    return mean(values)


def normalized_entropy(usage: np.ndarray) -> float:
    scopes = usage.shape[-1]
    clipped = np.clip(usage, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1) / math.log(scopes)
    return float(entropy.mean())


def internal_health_rows(
    raw_root: Path,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        ordered_dir = run_dir(raw_root, "siff_equal", dataset, seed)
        constant_dir = run_dir(
            raw_root,
            "siff_constant_equal",
            dataset,
            seed,
        )
        ordered_path = ordered_dir / "pcsd_test_audit_diagnostics.npz"
        constant_path = constant_dir / "pcsd_test_audit_diagnostics.npz"
        invariant_path = ordered_dir / "test_audit_invariants.json"
        with np.load(ordered_path) as ordered, np.load(constant_path) as constant:
            required = set(config["internal_mechanism_health"]["required_artifacts"])
            missing = sorted(required - set(ordered.files))
            if missing:
                raise ValueError(
                    f"{dataset} SIFF_EQUAL diagnostics missing {missing}"
                )
            arm_loss = ordered["arm_row_bin_mse"].astype(np.float64)
            fused_loss = ordered["fused_row_bin_mse"].astype(np.float64)
            probe_arms = ordered["probe_arms"].astype(np.float64)
            probe_fused = ordered["probe_fused"].astype(np.float64)
            constant_fused = constant["probe_fused"].astype(np.float64)
            usage = ordered["policy_row_bin_usage"].astype(np.float64)
            components = ordered["scale_component_contribution"].astype(
                np.float64
            )
            oracle_gain = 100.0 * (
                1.0 - float(np.mean(np.min(arm_loss, axis=-1)))
                / float(np.mean(fused_loss))
            )
            pairwise_nrmse = normalized_pairwise_nrmse(probe_arms)
            entropy = normalized_entropy(usage)
            denominator = max(
                float(np.sqrt(np.mean(np.square(probe_fused)))),
                1e-12,
            )
            nonconstant_ratio = float(
                np.sqrt(np.mean(np.square(components[:, 1]))) / denominator
            )
            ordered_constant_nrmse = float(
                np.sqrt(np.mean(np.square(probe_fused - constant_fused)))
                / denominator
            )
            finite = all(
                bool(np.isfinite(value).all())
                for value in (
                    arm_loss,
                    fused_loss,
                    probe_arms,
                    probe_fused,
                    usage,
                    components,
                )
            )
        invariant = json.loads(invariant_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "oracle_gain_percent": oracle_gain,
                "pairwise_arm_nrmse": pairwise_nrmse,
                "policy_normalized_entropy": entropy,
                "nonconstant_component_rms_ratio": nonconstant_ratio,
                "ordered_vs_constant_probe_nrmse": ordered_constant_nrmse,
                "prefix_projectivity_gap": float(
                    invariant["full_prefix_max_abs"]
                ),
                "all_finite": finite,
            }
        )
    return rows


def internal_health_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    gates = config["internal_mechanism_health"]
    entropy = mean(float(row["policy_normalized_entropy"]) for row in rows)
    return {
        "finite": all(bool(row["all_finite"]) for row in rows),
        "prefix_projectivity": max(
            float(row["prefix_projectivity_gap"]) for row in rows
        )
        <= gates["prefix_projectivity_gap_max"],
        "oracle_headroom": sum(
            float(row["oracle_gain_percent"]) > 0.0 for row in rows
        )
        >= gates["siff_equal_oracle_positive_datasets_min"],
        "arm_diversity": mean(
            float(row["pairwise_arm_nrmse"]) for row in rows
        )
        >= gates["siff_equal_mean_pairwise_probe_nrmse_min"],
        "policy_entropy": (
            gates["siff_equal_macro_policy_entropy_min"]
            <= entropy
            <= gates["siff_equal_macro_policy_entropy_max"]
        ),
        "nonconstant_component_use": mean(
            float(row["nonconstant_component_rms_ratio"]) for row in rows
        )
        >= gates["nonconstant_scale_component_rms_ratio_min"],
        "ordered_constant_contrast": mean(
            float(row["ordered_vs_constant_probe_nrmse"]) for row in rows
        )
        >= gates["ordered_vs_constant_probe_nrmse_min"],
    }


def decision_from_layers(
    hard_results: dict[str, bool],
    internal_results: dict[str, bool],
    config: dict[str, Any],
) -> dict[str, Any]:
    main_ids = set(config["effectiveness_gates"]["main_mae_comparison_ids"])
    control_ids = set(config["effectiveness_gates"]["hard_comparison_ids"]) - main_ids
    effectiveness = all(hard_results[comparison] for comparison in main_ids)
    attribution = all(hard_results[comparison] for comparison in control_ids)
    internal = all(internal_results.values())
    if not effectiveness:
        decision = "close_exact_candidate_effectiveness_fail"
        failure = "hypothesis_or_implementation_effectiveness_not_supported"
    elif not attribution:
        decision = "partial_pass_attribution_blocked"
        failure = "capacity_order_or_mapping_control_explains"
    elif not internal:
        decision = "design_fault_suspected_return_step4"
        failure = "intervention_or_readout_internal_health_fail"
    else:
        decision = "phase_a_pass_confirmation_required"
        failure = "none"
    return {
        "paper_facing_effectiveness": effectiveness,
        "matched_mechanism_attribution": attribution,
        "internal_mechanism_health": internal,
        "failure_attribution": failure,
        "decision": decision,
        "confirmation_authorized": bool(
            effectiveness and attribution and internal
        ),
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            for horizon in config["matrix"]["horizons"]:
                value = 0.95 if arm["id"] == "siff_equal" else 1.0
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
    hard = hard_gate_results(summaries, config)
    internal = {
        "finite": True,
        "prefix_projectivity": True,
        "oracle_headroom": True,
        "arm_diversity": True,
        "policy_entropy": True,
        "nonconstant_component_use": True,
        "ordered_constant_contrast": True,
    }
    decision = decision_from_layers(hard, internal, config)
    if not all(hard.values()) or decision["decision"] != (
        "phase_a_pass_confirmation_required"
    ):
        raise RuntimeError("four-layer analyzer synthetic smoke failed")
    print("siff_equal_attribution_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")

    metrics: list[dict[str, Any]] = []
    runs = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            arm_metrics, run = load_run(
                args.raw_root,
                arm,
                dataset,
                args.seed,
                config["matrix"]["horizons"],
                config,
            )
            metrics.extend(arm_metrics)
            runs.append(run)
    complete = bool(
        len(runs) == config["matrix"]["phase_a_expected_runs"]
        and len(metrics) == config["matrix"]["phase_a_expected_test_cells"]
        and all(row["status"] == "ok" for row in runs)
    )
    if not complete:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "run_audit.csv", runs)
        raise RuntimeError("SIFF_EQUAL attribution matrix is incomplete")

    cells, summaries = comparison_rows(metrics, config)
    hard = hard_gate_results(summaries, config)
    health_rows = internal_health_rows(args.raw_root, config, args.seed)
    health = internal_health_gate(health_rows, config)
    layers = decision_from_layers(hard, health, config)
    encoder_matched = all(
        len(
            {
                row["encoder_initialization_hash"]
                for row in runs
                if row["dataset"] == dataset
            }
        )
        == 1
        for dataset in config["datasets"]
    )
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
        "encoder_initialization_matched": encoder_matched,
        "hard_comparisons": hard,
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
