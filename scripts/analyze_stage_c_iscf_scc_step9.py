#!/usr/bin/env python3
"""Analyze the ISCF-SCC Step7B matched validation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_stage_c_iscf_scc_d0 import (
    gain_percent,
    normalized_positive_credit,
    spearman_last,
    write_csv,
)


HORIZONS = (96, 192, 336, 720)
NEW_ARMS = ("iscf_fused", "iscf_armerr", "iscf_scc", "iscf_scc_shuffled")
ALL_ARMS = ("iscf_equal", *NEW_ARMS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_scc_step7b.json"),
    )
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--parent-probe-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_dir(
    arm: str,
    dataset: str,
    new_root: Path,
    parent_root: Path,
) -> Path:
    if arm == "iscf_equal":
        return parent_root / dataset / "h720_full" / "seed2021"
    return new_root / arm / dataset / "h720_full" / "seed2021"


def probe_dir(
    arm: str,
    dataset: str,
    new_root: Path,
    parent_probe_root: Path,
) -> Path:
    if arm == "iscf_equal":
        return parent_probe_root / dataset / "seed2021"
    return new_root / arm / dataset / "h720_full" / "seed2021"


def objective_by_arm(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(arm["id"]): str(arm["objective_mode"])
        for arm in config["arms"]
    }


def audit_runs(
    config: dict[str, Any],
    new_root: Path,
    parent_root: Path,
    parent_probe_root: Path,
) -> list[dict[str, Any]]:
    required = config["artifact_schema"]["required"]
    objectives = objective_by_arm(config)
    rows: list[dict[str, Any]] = []
    initialization_by_dataset: dict[str, list[str]] = {
        dataset: [] for dataset in config["datasets"]
    }
    for arm in ALL_ARMS:
        for dataset in config["datasets"]:
            directory = run_dir(arm, dataset, new_root, parent_root)
            diagnostic_directory = probe_dir(
                arm,
                dataset,
                new_root,
                parent_probe_root,
            )
            missing = [
                name
                for name in required
                if not (
                    diagnostic_directory / name
                    if name in {
                        "pcsd_validation_diagnostics.npz",
                        "trained_invariants.json",
                    }
                    else directory / name
                ).is_file()
            ]
            effective = json.loads(
                (directory / "effective_config.json").read_text(encoding="utf-8")
            )
            invariant = json.loads(
                (diagnostic_directory / "trained_invariants.json").read_text(
                    encoding="utf-8"
                )
            )
            initialization = json.loads(
                (directory / "initialization_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            initialization_hash = str(
                initialization.get("pcsd_initialization_hash", "")
            )
            initialization_by_dataset[dataset].append(initialization_hash)
            expected_objective = objectives[arm]
            observed_objective = effective["adapter"]["pcc_objective_mode"]
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "missing_artifact_count": len(missing),
                    "objective_expected": expected_objective,
                    "objective_observed": observed_objective,
                    "objective_match": observed_objective == expected_objective,
                    "initialization_hash": initialization_hash,
                    "checkpoint_sha256": file_sha256(directory / "checkpoint.pt"),
                    "evaluation_split": invariant.get(
                        "evaluation_split",
                        "historical_parent_validation",
                    ),
                    "uses_test_split": invariant.get("uses_test_split", False),
                    "invariant_pass": invariant.get("pass", arm == "iscf_equal"),
                }
            )
    for row in rows:
        row["dataset_initialization_paired"] = (
            len(set(initialization_by_dataset[row["dataset"]])) == 1
        )
    return rows


def metric_cells(
    config: dict[str, Any],
    new_root: Path,
    parent_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ALL_ARMS:
        for dataset in config["datasets"]:
            directory = run_dir(arm, dataset, new_root, parent_root)
            metrics = read_csv(directory / "metrics_by_target_horizon.csv")
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
    metric_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (row["arm"], row["dataset"], row["horizon"]): row
        for row in metric_rows
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        comparison_cells: list[dict[str, Any]] = []
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
                        reference["mse"],
                        candidate["mse"],
                    ),
                    "candidate_mae": candidate["mae"],
                    "reference_mae": reference["mae"],
                    "mae_gain_percent": gain_percent(
                        reference["mae"],
                        candidate["mae"],
                    ),
                }
                cells.append(row)
                comparison_cells.append(row)
        dataset_wins = sum(
            np.mean(
                [
                    row["mse_gain_percent"]
                    for row in comparison_cells
                    if row["dataset"] == dataset
                ]
            )
            > 0.0
            for dataset in config["datasets"]
        )
        horizon_wins = sum(
            np.mean(
                [
                    row["mse_gain_percent"]
                    for row in comparison_cells
                    if row["horizon"] == horizon
                ]
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
                    np.mean([row["mse_gain_percent"] for row in comparison_cells])
                ),
                "macro_mae_gain_percent": float(
                    np.mean([row["mae_gain_percent"] for row in comparison_cells])
                ),
                "cell_wins": sum(
                    row["mse_gain_percent"] > 0.0 for row in comparison_cells
                ),
                "dataset_wins": dataset_wins,
                "horizon_wins": horizon_wins,
            }
        )
    return cells, summaries


def internal_health(
    config: dict[str, Any],
    new_root: Path,
    parent_probe_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ALL_ARMS:
        for dataset in config["datasets"]:
            directory = probe_dir(arm, dataset, new_root, parent_probe_root)
            with np.load(directory / "pcsd_validation_diagnostics.npz") as archive:
                arms = archive["probe_arms"].astype(np.float64)
                fused = archive["probe_fused"].astype(np.float64)
                target = archive["probe_targets"].astype(np.float64)
                policy = archive["probe_direct_policy"].astype(
                    np.float64
                ).transpose(0, 2, 1)
                usage = archive["policy_row_bin_usage"].mean(axis=(0, 1))
            leave_one_out = (
                fused[:, None, :] - policy * arms
            ) / np.maximum(1.0 - policy, 1e-6)
            full_error = np.abs(fused - target)
            delta = np.abs(leave_one_out - target[:, None, :]) - full_error[:, None, :]
            credit = normalized_positive_credit(delta)
            credit_forecast = np.sum(credit * arms, axis=1)
            policy_rts = policy.transpose(0, 2, 1)
            credit_rts = credit.transpose(0, 2, 1)
            policy_entropy = -np.sum(
                policy_rts.clip(min=1e-12)
                * np.log(policy_rts.clip(min=1e-12)),
                axis=-1,
            ) / np.log(policy.shape[1])
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "coalition_oracle_headroom_l1_percent": gain_percent(
                        float(full_error.mean()),
                        float(np.abs(credit_forecast - target).mean()),
                    ),
                    "policy_credit_spearman_median": float(
                        np.median(spearman_last(policy_rts, credit_rts))
                    ),
                    "policy_entropy_median": float(np.median(policy_entropy)),
                    "minimum_scope_usage": float(usage.min()),
                    "nonzero_usage_scope_count": int(np.sum(usage > 1e-6)),
                    "positive_contributor_count_mean": float(
                        np.mean(np.sum(delta > 0.0, axis=1))
                    ),
                }
            )
    return rows


def training_health(
    config: dict[str, Any],
    new_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    objectives = objective_by_arm(config)
    for arm in NEW_ARMS:
        for dataset in config["datasets"]:
            directory = run_dir(arm, dataset, new_root, new_root)
            log_rows = read_csv(directory / "training_log.csv")
            gradient_values = [
                float(row[f"train_pcc_scope_s{scope}_mode_grad_norm"])
                for row in log_rows
                for scope in range(5)
            ]
            best = min(log_rows, key=lambda row: float(row["val_mean_mse"]))
            last = log_rows[-1]
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "objective": objectives[arm],
                    "epochs": len(log_rows),
                    "best_epoch": int(best["epoch"]),
                    "best_val_mean_mse": float(best["val_mean_mse"]),
                    "minimum_scope_gradient_norm": min(gradient_values),
                    "all_scope_gradients_nonzero": min(gradient_values) > 0.0,
                    "final_credit_argmax_accuracy": float(
                        last["train_pcc_credit_argmax_accuracy"]
                    ),
                    "final_credit_policy_kl": float(
                        last["train_pcc_credit_policy_kl"]
                    ),
                    "final_credit_entropy": float(
                        last["train_pcc_credit_normalized_entropy"]
                    ),
                    "final_policy_entropy": float(
                        last["train_pcc_policy_normalized_entropy"]
                    ),
                    "final_positive_contributor_count": float(
                        last["train_pcc_coalition_positive_count"]
                    ),
                }
            )
    return rows


def decision_summary(
    run_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    comparison_summaries: list[dict[str, Any]],
    internal_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {row["comparison"]: row for row in comparison_summaries}
    primary = comparisons[config["validation_gates"]["primary_comparison"]]
    controls = [
        comparisons[name]
        for name in (
            "objective_vs_fused",
            "credit_vs_armerr",
            "binding_vs_shuffled",
        )
    ]
    gates = config["validation_gates"]
    checks = {
        "artifact_matrix_complete": len(run_rows) == 25
        and all(row["missing_artifact_count"] == 0 for row in run_rows),
        "validation_matrix_complete": len(metric_rows) == 100,
        "protocol_and_initialization": all(
            row["objective_match"]
            and row["dataset_initialization_paired"]
            and not row["uses_test_split"]
            for row in run_rows
        ),
        "effectiveness_vs_equal": (
            primary["macro_mse_gain_percent"]
            >= gates["macro_mse_gain_percent_min"]
            and primary["macro_mae_gain_percent"]
            > gates["macro_mae_gain_percent_min"]
            and primary["dataset_wins"] >= gates["dataset_wins_min"]
            and primary["horizon_wins"] >= gates["horizon_wins_min"]
        ),
        "matched_controls": all(
            row["macro_mse_gain_percent"]
            >= gates["matched_control_macro_mse_gain_percent_min"]
            for row in controls
        ),
        "numeric_and_gradient_health": all(
            row["all_scope_gradients_nonzero"] for row in training_rows
        )
        and all(
            np.isfinite(
                [
                    row["coalition_oracle_headroom_l1_percent"],
                    row["policy_credit_spearman_median"],
                    row["policy_entropy_median"],
                ]
            ).all()
            for row in internal_rows
        ),
    }
    equal_headroom = float(
        np.median(
            [
                row["coalition_oracle_headroom_l1_percent"]
                for row in internal_rows
                if row["arm"] == "iscf_equal"
            ]
        )
    )
    scc_headroom = float(
        np.median(
            [
                row["coalition_oracle_headroom_l1_percent"]
                for row in internal_rows
                if row["arm"] == "iscf_scc"
            ]
        )
    )
    if checks["effectiveness_vs_equal"] and checks["matched_controls"]:
        decision = "scc_v0_validation_pass_request_formal_test_design"
        failure = "none"
    else:
        decision = "scc_v0_failed_return_step5_reliability_preserving_design"
        failure = "intervention_point_wrong"
    return {
        "candidate_version": config["candidate_version"],
        "runs": len(run_rows),
        "validation_cells": len(metric_rows),
        "checks": checks,
        "primary_macro_mse_gain_percent": primary["macro_mse_gain_percent"],
        "primary_macro_mae_gain_percent": primary["macro_mae_gain_percent"],
        "matched_control_macro_mse_gains_percent": {
            row["comparison"]: row["macro_mse_gain_percent"] for row in controls
        },
        "equal_median_coalition_oracle_headroom_l1_percent": equal_headroom,
        "scc_median_coalition_oracle_headroom_l1_percent": scc_headroom,
        "failure_attribution": failure,
        "decision": decision,
        "formal_test_authorized": False,
        "seed_confirmation_authorized": False,
    }


def synthetic_smoke(config: dict[str, Any]) -> None:
    metrics = []
    for arm_index, arm in enumerate(ALL_ARMS):
        for dataset in config["datasets"]:
            for horizon in HORIZONS:
                metrics.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": 1.0 + 0.01 * arm_index,
                        "mae": 0.5 + 0.01 * arm_index,
                        "evaluation_split": "val",
                    }
                )
    cells, summaries = comparison_rows(metrics, config)
    if len(metrics) != 100 or len(cells) != 80 or len(summaries) != 4:
        raise RuntimeError("synthetic SCC Step9 smoke failed")
    print("iscf_scc_step9_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    required = (
        args.new_root,
        args.parent_root,
        args.parent_probe_root,
        args.output_dir,
    )
    if any(value is None for value in required):
        raise ValueError("all artifact roots and output-dir are required")

    run_rows = audit_runs(
        config,
        args.new_root,
        args.parent_root,
        args.parent_probe_root,
    )
    metric_rows = metric_cells(config, args.new_root, args.parent_root)
    cells, summaries = comparison_rows(metric_rows, config)
    internal_rows = internal_health(
        config,
        args.new_root,
        args.parent_probe_root,
    )
    training_rows = training_health(config, args.new_root)
    decision = decision_summary(
        run_rows,
        metric_rows,
        summaries,
        internal_rows,
        training_rows,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_audit.csv", run_rows)
    write_csv(args.output_dir / "validation_metrics.csv", metric_rows)
    write_csv(args.output_dir / "comparison_cells.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "internal_health.csv", internal_rows)
    write_csv(args.output_dir / "training_health.csv", training_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_scc_step9=complete runs={decision['runs']} "
        f"decision={decision['decision']}"
    )


if __name__ == "__main__":
    main()
