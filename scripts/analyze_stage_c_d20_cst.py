#!/usr/bin/env python3
"""Analyze the frozen SC-D20-CST formal diagnostic matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
        default=Path("configs/stage_c_d20_cst_step7b.json"),
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


def nrmse(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(np.square(right))))
    return float(
        np.sqrt(np.mean(np.square(left - right)))
        / max(denominator, 1e-12)
    )


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    horizons: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        "metrics": directory / "test_audit_metrics_by_target_horizon.csv",
        "invariant": directory / "test_audit_invariants.json",
        "diagnostics": directory / "pcsd_test_audit_diagnostics.npz",
        "effective": directory / "effective_config.json",
        "initialization": directory / "initialization_contract.json",
        "model": directory / "model_diagnostics.json",
        "checkpoint": directory / "checkpoint.pt",
        "training": directory / "training_log.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
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
    model = json.loads(required["model"].read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    lookup = {
        int(row["target_horizon"]): row for row in read_csv(required["metrics"])
    }
    selected = []
    for horizon in horizons:
        row = lookup[horizon]
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
        and adapter["history_statistic_mode"]
        == arm["history_statistic_mode"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["validation_horizons"] == [96, 192, 336, 720]
        and invariant.get("pass") is True
        and invariant.get("evaluation_split") == "test"
        and invariant.get("checkpoint_sha256") == file_hash(required["checkpoint"])
    )
    diagnostics = np.load(required["diagnostics"])
    health: dict[str, Any] = {
        "probe_fused": diagnostics["probe_fused"],
    }
    if arm["id"] != "A6_MEASURE_RETRAIN":
        required_keys = {
            "probe_history_summary",
            "probe_history_coefficient",
            "probe_history_prediction_contribution",
        }
        if not required_keys.issubset(diagnostics.files):
            protocol_pass = False
        else:
            summary = diagnostics["probe_history_summary"]
            coefficient = diagnostics["probe_history_coefficient"]
            contribution = diagnostics[
                "probe_history_prediction_contribution"
            ]
            health.update(
                {
                    "summary_std": float(np.std(summary)),
                    "summary_coefficient_std": float(np.std(coefficient)),
                    "summary_prediction_contribution_std": float(
                        np.std(contribution)
                    ),
                    "summary_tensors_finite": bool(
                        np.isfinite(summary).all()
                        and np.isfinite(coefficient).all()
                        and np.isfinite(contribution).all()
                    ),
                }
            )
    training_rows = read_csv(required["training"])
    return selected, {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "checkpoint_sha256": invariant.get("checkpoint_sha256", ""),
        "full_prefix_max_abs": float(invariant.get("full_prefix_max_abs", math.inf)),
        "all_finite": bool(invariant.get("all_finite", False)),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash",
            "",
        ),
        "base_operator_initialization_hash": initialization.get(
            "operator_base_initialization_hash",
            "",
        ),
        "projection_initialization_hash": initialization.get(
            "history_statistic_projection_hash",
            "",
        ),
        "initial_summary_weight_norm": initialization.get(
            "history_statistic_initial_weight_norm",
        ),
        "trained_summary_weight_norm": model.get(
            "history_statistic_weight_norm",
        ),
        "projection_orthogonality_max_abs": model.get(
            "history_statistic_projection_orthogonality_max_abs",
        ),
        "training_epochs": len(training_rows),
        "run_dir": str(directory),
        **health,
    }


def compare(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in config["primary_comparisons"]:
        for metric in config["matrix"]["metrics"]:
            gains = []
            by_dataset: dict[str, list[float]] = {}
            by_horizon: dict[int, list[float]] = {}
            for dataset in config["datasets"]:
                for horizon in config["matrix"]["horizons"]:
                    candidate = lookup[
                        (dataset, comparison["candidate"], horizon)
                    ]
                    reference = lookup[
                        (dataset, comparison["reference"], horizon)
                    ]
                    gain = 100.0 * (
                        1.0 - float(candidate[metric]) / float(reference[metric])
                    )
                    gains.append(gain)
                    by_dataset.setdefault(dataset, []).append(gain)
                    by_horizon.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "role": comparison["role"],
                            "metric": metric,
                            "dataset": dataset,
                            "horizon": horizon,
                            "candidate_value": candidate[metric],
                            "reference_value": reference[metric],
                            "gain_percent": gain,
                        }
                    )
            summaries.append(
                {
                    "comparison": comparison["id"],
                    "role": comparison["role"],
                    "metric": metric,
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0 for values in by_dataset.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0 for values in by_horizon.values()
                    ),
                }
            )
    return cells, summaries


def effectiveness(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    lookup = {
        (row["comparison"], row["metric"]): row for row in summaries
    }
    results = {}
    for comparison, threshold in config["effectiveness_gates"].items():
        mse = lookup[(comparison, "mse")]
        mae = lookup[(comparison, "mae")]
        results[comparison] = bool(
            mse["macro_gain_percent"]
            >= threshold["mse_macro_gain_percent_min"]
            and mse["cell_wins"] >= threshold["mse_cell_wins_min"]
            and mse["dataset_wins"] >= threshold["mse_dataset_wins_min"]
            and mse["horizon_wins"] >= threshold["mse_horizon_wins_min"]
            and mae["macro_gain_percent"]
            >= threshold["mae_macro_gain_percent_min"]
        )
    return results


def internal_health(
    audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    by_run = {
        (row["dataset"], row["arm"]): row
        for row in audits
        if row["status"] == "ok"
    }
    rows = []
    for dataset in config["datasets"]:
        arms = {
            arm["id"]: by_run[(dataset, arm["id"])]
            for arm in config["arms"]
        }
        encoder_hashes = {
            row["encoder_initialization_hash"] for row in arms.values()
        }
        operator_hashes = {
            row["base_operator_initialization_hash"] for row in arms.values()
        }
        spec = arms["A6_CST_SPEC"]
        random = arms["A6_CST_RANDOM"]
        rows.append(
            {
                "dataset": dataset,
                "paired_encoder": len(encoder_hashes) == 1,
                "paired_base_operator": len(operator_hashes) == 1,
                "projection_hashes_different": (
                    spec["projection_initialization_hash"]
                    != random["projection_initialization_hash"]
                ),
                "spec_random_prediction_nrmse": nrmse(
                    spec["probe_fused"],
                    random["probe_fused"],
                ),
                "spec_summary_coefficient_std": spec[
                    "summary_coefficient_std"
                ],
                "random_summary_coefficient_std": random[
                    "summary_coefficient_std"
                ],
                "spec_prediction_contribution_std": spec[
                    "summary_prediction_contribution_std"
                ],
                "random_prediction_contribution_std": random[
                    "summary_prediction_contribution_std"
                ],
                "spec_trained_summary_weight_norm": spec[
                    "trained_summary_weight_norm"
                ],
                "random_trained_summary_weight_norm": random[
                    "trained_summary_weight_norm"
                ],
                "max_projection_orthogonality": max(
                    spec["projection_orthogonality_max_abs"],
                    random["projection_orthogonality_max_abs"],
                ),
                "max_prefix_abs": max(
                    row["full_prefix_max_abs"] for row in arms.values()
                ),
                "all_finite": all(row["all_finite"] for row in arms.values())
                and spec["summary_tensors_finite"]
                and random["summary_tensors_finite"],
            }
        )
    gate = config["internal_health_gates"]
    results = {
        "all_protocols": all(row["status"] == "ok" for row in audits),
        "all_finite": all(row["all_finite"] for row in rows),
        "prefix_projectivity": max(row["max_prefix_abs"] for row in rows)
        <= gate["prefix_max_abs_max"],
        "paired_encoder_initialization": all(
            row["paired_encoder"] for row in rows
        ),
        "paired_base_operator_initialization": all(
            row["paired_base_operator"] for row in rows
        ),
        "projection_hashes_different": all(
            row["projection_hashes_different"] for row in rows
        ),
        "projection_orthogonality": max(
            row["max_projection_orthogonality"] for row in rows
        )
        <= gate["production_projection_orthogonality_max_abs_max"],
        "trained_summary_weights_nonzero": min(
            min(
                row["spec_trained_summary_weight_norm"],
                row["random_trained_summary_weight_norm"],
            )
            for row in rows
        )
        >= gate["trained_summary_weight_norm_min"],
        "summary_coefficients_noncollapsed": min(
            min(
                row["spec_summary_coefficient_std"],
                row["random_summary_coefficient_std"],
            )
            for row in rows
        )
        >= gate["summary_coefficient_std_min"],
        "summary_predictions_noncollapsed": min(
            min(
                row["spec_prediction_contribution_std"],
                row["random_prediction_contribution_std"],
            )
            for row in rows
        )
        >= gate["summary_prediction_contribution_std_min"],
        "spec_random_prediction_diversity": min(
            row["spec_random_prediction_nrmse"] for row in rows
        )
        >= gate["spec_random_prediction_nrmse_min"],
    }
    return rows, results


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    offsets = {
        "A6_MEASURE_RETRAIN": 0.01,
        "A6_CST_SPEC": 0.0,
        "A6_CST_RANDOM": 0.006,
    }
    for dataset_index, dataset in enumerate(config["datasets"]):
        for arm in config["arms"]:
            for horizon in config["matrix"]["horizons"]:
                base = 0.3 + dataset_index * 0.01 + horizon / 10000.0
                value = base + offsets[arm["id"]]
                metrics.append(
                    {
                        "dataset": dataset,
                        "arm": arm["id"],
                        "horizon": horizon,
                        "mse": value,
                        "mae": math.sqrt(value),
                    }
                )
    cells, summaries = compare(metrics, config)
    gates = effectiveness(summaries, config)
    if len(cells) != 80 or len(summaries) != 4 or not all(gates.values()):
        raise RuntimeError("D20 analyzer synthetic smoke failed")
    print("d20_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")
    metrics = []
    audits = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            selected, audit = load_run(
                args.raw_root,
                arm,
                dataset,
                args.seed,
                config["matrix"]["horizons"],
            )
            metrics.extend(selected)
            audits.append(audit)
    missing = [row for row in audits if row["status"] != "ok"]
    if missing:
        raise RuntimeError(
            "D20 matrix incomplete or invalid: "
            + json.dumps(missing, sort_keys=True)
        )
    cells, summaries = compare(metrics, config)
    effectiveness_results = effectiveness(summaries, config)
    health_rows, health_results = internal_health(audits, config)
    transfer = effectiveness_results["transfer_spec_vs_a6"]
    specificity = effectiveness_results["specificity_spec_vs_random"]
    health = all(health_results.values())
    if not health:
        decision = "diagnostic_invalid_for_direction_rejection"
        failure = "optimization_or_numeric_pathology"
        rollback = "Step6/7A"
    elif not transfer:
        decision = "compact_spectrum_transfer_failed"
        failure = "intervention_point_wrong_or_hypothesis_false"
        rollback = "Contribution1 Step2"
    elif not specificity:
        decision = "generic_history_access_explains"
        failure = "capacity_control_explains"
        rollback = "Contribution1 Step2/4"
    else:
        decision = "problem_supported_pending_confirmation"
        failure = "none"
        rollback = "conditional seeds2022/2023 then Step4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_standard_horizons.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    audit_export = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, np.ndarray)
        }
        for row in audits
    ]
    write_csv(args.output_dir / "run_audit.csv", audit_export)
    write_csv(args.output_dir / "internal_health_by_dataset.csv", health_rows)
    payload = {
        "candidate_version": config["candidate_version"],
        "matrix_complete": len(audits) == 15 and len(metrics) == 60,
        "effectiveness": effectiveness_results,
        "internal_health": health_results,
        "decision": decision,
        "failure_attribution": failure,
        "rollback": rollback,
        "confirmation_authorized": False,
    }
    (args.output_dir / "step9_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
