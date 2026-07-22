#!/usr/bin/env python3
"""Analyze the PSA-D1 contemporaneous EQUAL validation control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stage_c_iscf_scc_d0 import gain_percent, write_csv


HORIZONS = (96, 192, 336, 720)
NEW_ARM = "iscf_equal_contemporaneous"
HISTORICAL_ARM = "iscf_equal_historical"
ARMERR_ARM = "iscf_equal_armerr"
SHUFFLED_ARM = "iscf_rscc_shuffled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_psa_d1.json"),
    )
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_specs(
    config: dict[str, Any],
    new_root: Path,
) -> dict[str, dict[str, Any]]:
    references = config["references"]
    return {
        NEW_ARM: {
            "root": new_root / NEW_ARM,
            "probe_root": None,
            "objective": "equal_skill",
            "new": True,
        },
        HISTORICAL_ARM: {
            "root": Path(references["historical_equal"]["root"]),
            "probe_root": Path(
                references["historical_equal"]["probe_root"]
            ),
            "objective": references["historical_equal"]["objective_mode"],
            "new": False,
        },
        ARMERR_ARM: {
            "root": Path(references["equal_armerr"]["root"]),
            "probe_root": None,
            "objective": references["equal_armerr"]["objective_mode"],
            "new": False,
        },
        SHUFFLED_ARM: {
            "root": Path(references["rscc_shuffled"]["root"]),
            "probe_root": None,
            "objective": references["rscc_shuffled"]["objective_mode"],
            "new": False,
        },
    }


def run_dir(spec: dict[str, Any], dataset: str) -> Path:
    return spec["root"] / dataset / "h720_full" / "seed2021"


def probe_dir(spec: dict[str, Any], dataset: str) -> Path:
    if spec["probe_root"] is not None:
        return spec["probe_root"] / dataset / "seed2021"
    return run_dir(spec, dataset)


def audit_runs(
    config: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashes_by_dataset: dict[str, list[str]] = {
        dataset: [] for dataset in config["datasets"]
    }
    for arm, spec in specs.items():
        for dataset in config["datasets"]:
            directory = run_dir(spec, dataset)
            diagnostic = probe_dir(spec, dataset)
            required = [
                "checkpoint.pt",
                "metrics_by_target_horizon.csv",
                "effective_config.json",
                "initialization_contract.json",
            ]
            if spec["new"]:
                required.append("training_log.csv")
            missing = [name for name in required if not (directory / name).is_file()]
            for name in ("pcsd_validation_diagnostics.npz", "trained_invariants.json"):
                if not (diagnostic / name).is_file():
                    missing.append(name)
            effective = json.loads(
                (directory / "effective_config.json").read_text(encoding="utf-8")
            )
            initialization = json.loads(
                (directory / "initialization_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            invariant = json.loads(
                (diagnostic / "trained_invariants.json").read_text(
                    encoding="utf-8"
                )
            )
            initialization_hash = str(
                initialization.get("pcsd_initialization_hash", "")
            )
            hashes_by_dataset[dataset].append(initialization_hash)
            observed_objective = effective["adapter"]["pcc_objective_mode"]
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "is_new_run": bool(spec["new"]),
                    "missing_artifact_count": len(missing),
                    "objective_expected": spec["objective"],
                    "objective_observed": observed_objective,
                    "objective_match": observed_objective == spec["objective"],
                    "initialization_hash": initialization_hash,
                    "checkpoint_sha256": file_sha256(directory / "checkpoint.pt"),
                    "evaluation_split": invariant.get("evaluation_split", "val"),
                    "uses_test_split": bool(invariant.get("uses_test_split", False)),
                    "invariant_pass": bool(invariant.get("pass", True)),
                }
            )
    for row in rows:
        row["dataset_initialization_paired"] = (
            len(set(hashes_by_dataset[row["dataset"]])) == 1
        )
    return rows


def metric_rows(
    config: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, spec in specs.items():
        for dataset in config["datasets"]:
            metrics = read_csv(
                run_dir(spec, dataset) / "metrics_by_target_horizon.csv"
            )
            selected = {
                int(row["target_horizon"]): row
                for row in metrics
                if int(row["target_horizon"]) in HORIZONS
            }
            if tuple(sorted(selected)) != HORIZONS:
                raise ValueError(f"incomplete horizons: {arm} {dataset}")
            for horizon in HORIZONS:
                row = selected[horizon]
                rows.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "evaluation_split": row["evaluation_split"],
                    }
                )
    return rows


def comparison_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (row["arm"], row["dataset"], row["horizon"]): row
        for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        selected: list[dict[str, Any]] = []
        for dataset in config["datasets"]:
            for horizon in HORIZONS:
                candidate = indexed[(comparison["candidate"], dataset, horizon)]
                reference = indexed[(comparison["reference"], dataset, horizon)]
                row = {
                    "comparison": comparison["id"],
                    "candidate": comparison["candidate"],
                    "reference": comparison["reference"],
                    "dataset": dataset,
                    "horizon": horizon,
                    "candidate_mse": candidate["mse"],
                    "reference_mse": reference["mse"],
                    "mse_gain_percent": gain_percent(
                        reference["mse"], candidate["mse"]
                    ),
                    "candidate_mae": candidate["mae"],
                    "reference_mae": reference["mae"],
                    "mae_gain_percent": gain_percent(
                        reference["mae"], candidate["mae"]
                    ),
                }
                cells.append(row)
                selected.append(row)
        dataset_wins = sum(
            np.mean(
                [row["mse_gain_percent"] for row in selected if row["dataset"] == dataset]
            )
            > 0.0
            for dataset in config["datasets"]
        )
        horizon_wins = sum(
            np.mean(
                [row["mse_gain_percent"] for row in selected if row["horizon"] == horizon]
            )
            > 0.0
            for horizon in HORIZONS
        )
        summaries.append(
            {
                "comparison": comparison["id"],
                "candidate": comparison["candidate"],
                "reference": comparison["reference"],
                "macro_mse_gain_percent": float(
                    np.mean([row["mse_gain_percent"] for row in selected])
                ),
                "macro_mae_gain_percent": float(
                    np.mean([row["mae_gain_percent"] for row in selected])
                ),
                "cell_wins": sum(row["mse_gain_percent"] > 0.0 for row in selected),
                "dataset_wins": dataset_wins,
                "horizon_wins": horizon_wins,
            }
        )
    return cells, summaries


def function_drift(
    config: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        with np.load(
            probe_dir(specs[NEW_ARM], dataset) / "pcsd_validation_diagnostics.npz"
        ) as archive:
            new_fused = archive["probe_fused"].astype(np.float64)
            new_policy = archive["probe_direct_policy"].astype(np.float64)
            new_arms = archive["probe_arms"].astype(np.float64)
        with np.load(
            probe_dir(specs[HISTORICAL_ARM], dataset)
            / "pcsd_validation_diagnostics.npz"
        ) as archive:
            old_fused = archive["probe_fused"].astype(np.float64)
            old_policy = archive["probe_direct_policy"].astype(np.float64)
            old_arms = archive["probe_arms"].astype(np.float64)
        if new_fused.shape != old_fused.shape or new_policy.shape != old_policy.shape:
            raise ValueError(f"probe shape mismatch: {dataset}")
        rows.append(
            {
                "dataset": dataset,
                "fused_relative_l1": float(
                    np.abs(new_fused - old_fused).mean()
                    / np.maximum(np.abs(old_fused).mean(), 1e-12)
                ),
                "policy_mean_l1": float(np.abs(new_policy - old_policy).mean()),
                "arms_relative_l1": float(
                    np.abs(new_arms - old_arms).mean()
                    / np.maximum(np.abs(old_arms).mean(), 1e-12)
                ),
            }
        )
    return rows


def training_health(
    config: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        log_rows = read_csv(run_dir(specs[NEW_ARM], dataset) / "training_log.csv")
        gradients = [
            float(row[f"train_pcc_scope_s{scope}_mode_grad_norm"])
            for row in log_rows
            for scope in range(5)
        ]
        rows.append(
            {
                "dataset": dataset,
                "epochs": len(log_rows),
                "minimum_scope_gradient_norm": min(gradients),
                "all_scope_gradients_nonzero": min(gradients) > 0.0,
                "maximum_route_weight": max(
                    float(row["train_pcc_route_weight"]) for row in log_rows
                ),
                "maximum_weighted_route_loss": max(
                    abs(float(row["train_pcc_weighted_route_loss"]))
                    for row in log_rows
                ),
                "all_finite": bool(
                    all(
                        np.isfinite(
                            [
                                float(row["train_loss"]),
                                float(row["val_mean_mse"]),
                                float(row["train_pcc_total_loss"]),
                            ]
                        ).all()
                        for row in log_rows
                    )
                ),
            }
        )
    return rows


def decide(
    comparisons: list[dict[str, Any]],
    functions: list[dict[str, Any]],
    run_audit: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    training: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    indexed = {row["comparison"]: row for row in comparisons}
    gates = config["decision_gates"]
    new_gain = indexed["new_equal_vs_historical"]["macro_mse_gain_percent"]
    armerr_historical = indexed["armerr_vs_historical"][
        "macro_mse_gain_percent"
    ]
    shuffled_historical = indexed["shuffled_vs_historical"][
        "macro_mse_gain_percent"
    ]
    common_gain = 0.5 * (armerr_historical + shuffled_historical)
    recovery_ratio = new_gain / common_gain if common_gain != 0.0 else float("nan")
    protocol_pass = (
        len(run_audit) == config["matrix"]["effective_runs"]
        and len(metrics) == config["matrix"]["effective_validation_cells"]
        and all(
            row["missing_artifact_count"] == 0
            and row["objective_match"]
            and row["dataset_initialization_paired"]
            and row["invariant_pass"]
            and not row["uses_test_split"]
            for row in run_audit
        )
        and all(row["evaluation_split"] == "val" for row in metrics)
    )
    training_pass = all(
        row["all_scope_gradients_nonzero"]
        and row["maximum_route_weight"] == 0.0
        and row["maximum_weighted_route_loss"] == 0.0
        and row["all_finite"]
        for row in training
    )
    function_match = all(
        row["fused_relative_l1"]
        <= gates["function_fused_relative_l1_max"]
        and row["policy_mean_l1"] <= gates["function_policy_mean_l1_max"]
        for row in functions
    )
    new_summary = indexed["new_equal_vs_historical"]
    armerr_new = indexed["armerr_vs_new_equal"]
    shuffled_new = indexed["shuffled_vs_new_equal"]
    checks = {
        "protocol_complete_no_test": protocol_pass,
        "equal_training_health": training_pass,
        "new_equal_function_matches_historical": function_match,
        "run_drift_new_equal_gain": new_gain
        >= gates["run_drift_new_equal_gain_min_percent"],
        "run_drift_recovery_ratio": recovery_ratio
        >= gates["run_drift_recovery_ratio_min"],
        "run_drift_dataset_horizon": (
            new_summary["dataset_wins"] >= gates["dataset_wins_min"]
            and new_summary["horizon_wins"] >= gates["horizon_wins_min"]
        ),
        "run_drift_controls_explained": (
            armerr_new["macro_mse_gain_percent"]
            < gates["control_residual_gain_max_percent"]
            and shuffled_new["macro_mse_gain_percent"]
            < gates["control_residual_gain_max_percent"]
        ),
        "coadaptation_new_equal_stable": abs(new_gain)
        < gates["coadaptation_new_equal_abs_gain_max_percent"],
        "coadaptation_control_gain": (
            armerr_new["macro_mse_gain_percent"]
            >= gates["coadaptation_control_gain_min_percent"]
            and shuffled_new["macro_mse_gain_percent"]
            >= gates["coadaptation_control_gain_min_percent"]
        ),
        "coadaptation_dataset_horizon": (
            armerr_new["dataset_wins"] >= gates["dataset_wins_min"]
            and armerr_new["horizon_wins"] >= gates["horizon_wins_min"]
            and shuffled_new["dataset_wins"] >= gates["dataset_wins_min"]
            and shuffled_new["horizon_wins"] >= gates["horizon_wins_min"]
        ),
    }
    run_drift_pass = all(
        checks[name]
        for name in (
            "protocol_complete_no_test",
            "equal_training_health",
            "run_drift_new_equal_gain",
            "run_drift_recovery_ratio",
            "run_drift_dataset_horizon",
            "run_drift_controls_explained",
        )
    )
    coadaptation_pass = all(
        checks[name]
        for name in (
            "protocol_complete_no_test",
            "equal_training_health",
            "new_equal_function_matches_historical",
            "coadaptation_new_equal_stable",
            "coadaptation_control_gain",
            "coadaptation_dataset_horizon",
        )
    )
    if not protocol_pass or not training_pass:
        decision = "diagnostic_invalid_for_attribution"
        failure = "optimization_or_numeric_pathology"
    elif run_drift_pass:
        decision = "contemporaneous_run_drift_explains"
        failure = "capacity_control_explains"
    elif coadaptation_pass:
        decision = "joint_training_route_regularization_supported_as_carrier_clue"
        failure = "none_control_clue_only"
    else:
        decision = "h2_h3_unresolved"
        failure = "attribution_incomplete"
    return {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "runs": len(run_audit),
        "validation_cells": len(metrics),
        "new_equal_vs_historical_mse_gain_percent": new_gain,
        "common_control_vs_historical_mse_gain_percent": common_gain,
        "new_equal_recovery_ratio": recovery_ratio,
        "armerr_vs_new_equal_mse_gain_percent": armerr_new[
            "macro_mse_gain_percent"
        ],
        "shuffled_vs_new_equal_mse_gain_percent": shuffled_new[
            "macro_mse_gain_percent"
        ],
        "checks": checks,
        "failure_attribution": failure,
        "decision": decision,
        "formal_test_access_authorized": False,
        "confirmation_seeds_authorized": False,
        "method_promotion_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    arms = (NEW_ARM, HISTORICAL_ARM, ARMERR_ARM, SHUFFLED_ARM)
    def build_metrics(offsets: dict[str, float]) -> list[dict[str, Any]]:
        return [
            {
                "arm": arm,
                "dataset": dataset,
                "horizon": horizon,
                "mse": 1.0 + offsets[arm],
                "mae": 0.5 + offsets[arm],
                "evaluation_split": "val",
            }
            for arm in arms
            for dataset in config["datasets"]
            for horizon in HORIZONS
        ]

    run_audit = [
        {
            "missing_artifact_count": 0,
            "objective_match": True,
            "dataset_initialization_paired": True,
            "invariant_pass": True,
            "uses_test_split": False,
        }
        for _ in range(20)
    ]
    functions = [
        {"fused_relative_l1": 0.001, "policy_mean_l1": 0.001}
        for _ in config["datasets"]
    ]
    training = [
        {
            "all_scope_gradients_nonzero": True,
            "maximum_route_weight": 0.0,
            "maximum_weighted_route_loss": 0.0,
            "all_finite": True,
        }
        for _ in config["datasets"]
    ]
    drift_metrics = build_metrics(
        {
            NEW_ARM: -0.006,
            HISTORICAL_ARM: 0.0,
            ARMERR_ARM: -0.0065,
            SHUFFLED_ARM: -0.0065,
        }
    )
    cells, summaries = comparison_rows(drift_metrics, config)
    drift_decision = decide(
        summaries,
        functions,
        run_audit,
        drift_metrics,
        training,
        config,
    )
    coadaptation_metrics = build_metrics(
        {
            NEW_ARM: 0.0,
            HISTORICAL_ARM: 0.0,
            ARMERR_ARM: -0.0065,
            SHUFFLED_ARM: -0.0065,
        }
    )
    _, coadaptation_summaries = comparison_rows(coadaptation_metrics, config)
    coadaptation_decision = decide(
        coadaptation_summaries,
        functions,
        run_audit,
        coadaptation_metrics,
        training,
        config,
    )
    if (
        len(drift_metrics) != 80
        or len(cells) != 100
        or len(summaries) != 5
    ):
        raise RuntimeError("synthetic PSA-D1 comparison smoke failed")
    if drift_decision["decision"] != "contemporaneous_run_drift_explains":
        raise RuntimeError("synthetic PSA-D1 drift decision failed")
    if coadaptation_decision["decision"] != (
        "joint_training_route_regularization_supported_as_carrier_clue"
    ):
        raise RuntimeError("synthetic PSA-D1 co-adaptation decision failed")
    print(
        "iscf_psa_d1_synthetic_smoke=pass comparisons=5 cells=100 "
        "decisions=drift,coadaptation"
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.new_root is None or args.output_dir is None:
        raise ValueError("new-root and output-dir are required")
    specs = arm_specs(config, args.new_root)
    runs = audit_runs(config, specs)
    metrics = metric_rows(config, specs)
    cells, comparisons = comparison_rows(metrics, config)
    functions = function_drift(config, specs)
    training = training_health(config, specs)
    decision = decide(
        comparisons,
        functions,
        runs,
        metrics,
        training,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", runs)
    write_csv(args.output_dir / "validation_metrics.csv", metrics)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", comparisons)
    write_csv(args.output_dir / "function_drift.csv", functions)
    write_csv(args.output_dir / "training_health.csv", training)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_psa_d1=complete runs={decision['runs']} "
        f"decision={decision['decision']}"
    )


if __name__ == "__main__":
    main()
