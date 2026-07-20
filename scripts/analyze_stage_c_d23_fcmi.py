#!/usr/bin/env python3
"""Analyze the frozen SC-D23-FCMI formal matrix."""

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
        default=Path("configs/stage_c_d23_fcmi_step7b.json"),
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{seed}"


def metric_rows(
    path: Path,
    dataset: str,
    arm: str,
    split: str,
    horizons: list[int],
) -> list[dict[str, Any]]:
    lookup = {
        int(row["target_horizon"]): row for row in read_csv(path)
    }
    rows = []
    for horizon in horizons:
        source = lookup[horizon]
        mse, mae = float(source["mse"]), float(source["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(
                f"non-finite {split} metric: {dataset}/{arm}/H{horizon}"
            )
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "split": split,
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
            }
        )
    return rows


def load_run(
    root: Path,
    arm: dict[str, Any],
    dataset: str,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(root, arm["id"], dataset, seed)
    required = {
        "validation_metrics": directory / "metrics_by_target_horizon.csv",
        "validation_invariant": directory / "trained_invariants.json",
        "validation_diagnostics": (
            directory / "pcsd_validation_diagnostics.npz"
        ),
        "effective": directory / "effective_config.json",
        "initialization": directory / "initialization_contract.json",
        "model": directory / "model_diagnostics.json",
        "checkpoint": directory / "checkpoint.pt",
        "training": directory / "training_log.csv",
    }
    formal_test = arm["evaluation_role"] == "formal_test"
    if formal_test:
        required.update(
            {
                "test_metrics": (
                    directory / "test_audit_metrics_by_target_horizon.csv"
                ),
                "test_invariant": directory / "test_audit_invariants.json",
                "test_diagnostics": (
                    directory / "pcsd_test_audit_diagnostics.npz"
                ),
            }
        )
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [], [], {
            "dataset": dataset,
            "arm": arm["id"],
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    initialization = json.loads(
        required["initialization"].read_text(encoding="utf-8")
    )
    model = json.loads(required["model"].read_text(encoding="utf-8"))
    validation_invariant = json.loads(
        required["validation_invariant"].read_text(encoding="utf-8")
    )
    expected_dense_rank = (
        config["fcmi_contract"]["dense_ranks"][dataset]
        if arm["id"] == "DENSE_DUAL_MATCHED"
        else 0
    )
    protocol_pass = bool(
        adapter["dataset"] == dataset
        and adapter["seed"] == seed
        and adapter["readout_mode"] == arm["readout_mode"]
        and adapter["pcc_objective_mode"] == arm["objective_mode"]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["validation_horizons"] == [96, 192, 336, 720]
        and adapter["fcmi_dense_rank"] == expected_dense_rank
        and validation_invariant.get("pass") is True
        and validation_invariant.get("evaluation_split") == "val"
        and validation_invariant.get("checkpoint_sha256")
        == file_hash(required["checkpoint"])
    )
    test_rows: list[dict[str, Any]] = []
    test_invariant: dict[str, Any] = {}
    if formal_test:
        test_invariant = json.loads(
            required["test_invariant"].read_text(encoding="utf-8")
        )
        protocol_pass = protocol_pass and bool(
            test_invariant.get("pass") is True
            and test_invariant.get("evaluation_split") == "test"
            and test_invariant.get("checkpoint_sha256")
            == file_hash(required["checkpoint"])
        )
        test_rows = metric_rows(
            required["test_metrics"],
            dataset,
            arm["id"],
            "test",
            config["matrix"]["horizons"],
        )
    validation_rows = metric_rows(
        required["validation_metrics"],
        dataset,
        arm["id"],
        "val",
        config["matrix"]["horizons"],
    )

    diagnostic_path = (
        required["test_diagnostics"]
        if formal_test
        else required["validation_diagnostics"]
    )
    diagnostics = np.load(diagnostic_path)
    health: dict[str, Any] = {}
    if arm["id"] == "FCMI":
        health = {
            "context_coordinate_std": float(
                np.mean(diagnostics["probe_fcmi_context_coordinate_std"])
            ),
            "main_rms": float(
                np.mean(diagnostics["probe_fcmi_main_rms"])
            ),
            "interaction_rms": float(
                np.mean(diagnostics["probe_fcmi_interaction_rms"])
            ),
            "attention_target_dispersion": float(
                np.mean(
                    diagnostics[
                        "probe_fcmi_attention_target_dispersion"
                    ]
                )
            ),
            "interaction_prediction_std": float(
                np.std(
                    diagnostics[
                        "probe_fcmi_interaction_prediction_contribution"
                    ]
                )
            ),
        }
    elif arm["id"] == "DENSE_DUAL_MATCHED":
        health = {
            "dense_prediction_residual_std": float(
                np.std(diagnostics["probe_fcmi_dense_residual"])
            ),
            "dense_coefficient_weight_norm": float(
                model["fcmi_dense_coefficient_weight_norm"]
            ),
        }
    return test_rows, validation_rows, {
        "dataset": dataset,
        "arm": arm["id"],
        "status": "ok" if protocol_pass else "audit_fail",
        "protocol_pass": protocol_pass,
        "evaluation_role": arm["evaluation_role"],
        "checkpoint_sha256": file_hash(required["checkpoint"]),
        "full_prefix_max_abs": float(
            (
                test_invariant
                if formal_test
                else validation_invariant
            ).get("full_prefix_max_abs", math.inf)
        ),
        "all_finite": bool(
            (
                test_invariant
                if formal_test
                else validation_invariant
            ).get("all_finite", False)
        ),
        "encoder_initialization_hash": initialization.get(
            "encoder_initialization_hash",
            "",
        ),
        "fcmi_common_initialization_hash": initialization.get(
            "fcmi_common_initialization_hash",
            "",
        ),
        "fcmi_main_initialization_hash": initialization.get(
            "fcmi_main_initialization_hash",
            "",
        ),
        "fcmi_interaction_initialization_hash": initialization.get(
            "fcmi_interaction_initialization_hash",
            "",
        ),
        "active_forward_parameters": model.get(
            "active_forward_parameters",
            0,
        ),
        "training_epochs": len(read_csv(required["training"])),
        "run_dir": str(directory),
        **health,
    }


def compare(
    metrics: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {
        (row["dataset"], row["arm"], row["horizon"]): row
        for row in metrics
    }
    cells: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for comparison in comparisons:
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
                        1.0
                        - float(candidate[metric])
                        / float(reference[metric])
                    )
                    gains.append(gain)
                    by_dataset.setdefault(dataset, []).append(gain)
                    by_horizon.setdefault(horizon, []).append(gain)
                    cells.append(
                        {
                            "comparison": comparison["id"],
                            "layer": comparison["layer"],
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
                    "layer": comparison["layer"],
                    "metric": metric,
                    "macro_gain_percent": mean(gains),
                    "cell_wins": sum(value > 0.0 for value in gains),
                    "dataset_wins": sum(
                        mean(values) > 0.0
                        for values in by_dataset.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0.0
                        for values in by_horizon.values()
                    ),
                }
            )
    return cells, summaries


def comparison_passes(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, bool]:
    lookup = {
        (row["comparison"], row["metric"]): row for row in summaries
    }
    passes = {}
    for comparison, gate in config["comparison_gates"].items():
        mse = lookup[(comparison, "mse")]
        mae = lookup[(comparison, "mae")]
        passes[comparison] = bool(
            mse["macro_gain_percent"]
            >= gate["mse_macro_gain_percent_min"]
            and mse["cell_wins"] >= gate["mse_cell_wins_min"]
            and mse["dataset_wins"] >= gate["mse_dataset_wins_min"]
            and mse["horizon_wins"] >= gate["mse_horizon_wins_min"]
            and mae["macro_gain_percent"]
            >= gate["mae_macro_gain_percent_min"]
        )
    return passes


def health_passes(
    audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    gates = config["internal_health_gates"]
    rows = []
    passes: dict[str, bool] = {}
    for dataset in config["datasets"]:
        dataset_rows = [
            row for row in audits if row["dataset"] == dataset
        ]
        fcmi = next(row for row in dataset_rows if row["arm"] == "FCMI")
        dense = next(
            row
            for row in dataset_rows
            if row["arm"] == "DENSE_DUAL_MATCHED"
        )
        a6 = next(
            row for row in dataset_rows if row["arm"] == "A6_MEASURE"
        )
        dual = [
            row
            for row in dataset_rows
            if row["arm"]
            in {
                "STANDARD_DUAL_MATCHED",
                "GENERIC_DUAL_MATCHED",
                "FCMI",
                "FCMI_ORDER_SHUFFLED",
                "TARGET_SHUFFLED_QUERY",
                "DENSE_DUAL_MATCHED",
            }
        ]
        dense_gap = abs(
            dense["active_forward_parameters"]
            - a6["active_forward_parameters"]
        ) / a6["active_forward_parameters"]
        row = {
            "dataset": dataset,
            "protocol": all(item["protocol_pass"] for item in dataset_rows),
            "finite": all(item["all_finite"] for item in dataset_rows),
            "prefix_max_abs": max(
                item["full_prefix_max_abs"] for item in dataset_rows
            ),
            "encoder_hash_count": len(
                {item["encoder_initialization_hash"] for item in dual}
            ),
            "common_hash_count": len(
                {item["fcmi_common_initialization_hash"] for item in dual}
            ),
            "main_hash_count": len(
                {item["fcmi_main_initialization_hash"] for item in dual}
            ),
            "interaction_hash_count": len(
                {
                    item["fcmi_interaction_initialization_hash"]
                    for item in dual
                }
            ),
            "dense_a6_parameter_relative_gap": dense_gap,
            "context_coordinate_std": fcmi["context_coordinate_std"],
            "main_rms": fcmi["main_rms"],
            "interaction_rms": fcmi["interaction_rms"],
            "attention_target_dispersion": (
                fcmi["attention_target_dispersion"]
            ),
            "interaction_prediction_std": (
                fcmi["interaction_prediction_std"]
            ),
            "dense_coefficient_weight_norm": (
                dense["dense_coefficient_weight_norm"]
            ),
            "dense_prediction_residual_std": (
                dense["dense_prediction_residual_std"]
            ),
        }
        passed = bool(
            row["protocol"]
            and row["finite"]
            and row["prefix_max_abs"] <= gates["prefix_max_abs_max"]
            and row["encoder_hash_count"] == 1
            and row["common_hash_count"] == 1
            and row["main_hash_count"] == 1
            and row["interaction_hash_count"] == 1
            and dense_gap
            <= config["fcmi_contract"][
                "dense_a6_active_parameter_relative_gap_max"
            ]
            and row["context_coordinate_std"]
            >= gates["fcmi_context_coordinate_std_min"]
            and row["main_rms"] >= gates["fcmi_main_rms_min"]
            and row["interaction_rms"]
            >= gates["fcmi_interaction_rms_min"]
            and row["attention_target_dispersion"]
            >= gates["fcmi_attention_target_dispersion_min"]
            and row["interaction_prediction_std"]
            >= gates["fcmi_interaction_prediction_std_min"]
            and row["dense_coefficient_weight_norm"]
            >= gates["dense_trained_coefficient_weight_norm_min"]
            and row["dense_prediction_residual_std"]
            >= gates["dense_prediction_residual_std_min"]
        )
        row["pass"] = passed
        rows.append(row)
        passes[dataset] = passed
    return passes, rows


def decide(
    comparison_results: dict[str, bool],
    health_results: dict[str, bool],
    protocol_valid: bool,
) -> str:
    if not protocol_valid:
        return "numeric_or_protocol_invalid"
    if not comparison_results["effectiveness_fcmi_vs_a6"]:
        return "fails_a6_internal_valid"
    attribution_ids = {
        "decomposition_fcmi_vs_standard_dual",
        "interaction_fcmi_vs_generic_dual",
        "order_fcmi_vs_order_shuffled",
        "capacity_fcmi_vs_dense_dual",
        "target_coordinate_fcmi_vs_target_shuffle",
    }
    if not all(comparison_results[item] for item in attribution_ids):
        return "beats_a6_but_attribution_fails"
    if not all(health_results.values()):
        return "attribution_pass_internal_fails"
    return "all_seed2021_gates_pass"


def synthetic_smoke(config: dict[str, Any]) -> None:
    summaries = []
    for comparison in config["comparison_gates"]:
        summaries.extend(
            [
                {
                    "comparison": comparison,
                    "metric": "mse",
                    "macro_gain_percent": 1.0,
                    "cell_wins": 15,
                    "dataset_wins": 4,
                    "horizon_wins": 4,
                },
                {
                    "comparison": comparison,
                    "metric": "mae",
                    "macro_gain_percent": 0.5,
                    "cell_wins": 15,
                    "dataset_wins": 4,
                    "horizon_wins": 4,
                },
            ]
        )
    comparisons = comparison_passes(summaries, config)
    health = {dataset: True for dataset in config["datasets"]}
    if (
        not all(comparisons.values())
        or decide(comparisons, health, True)
        != "all_seed2021_gates_pass"
        or decide(comparisons, health, False)
        != "numeric_or_protocol_invalid"
    ):
        raise RuntimeError("D23 FCMI analyzer synthetic smoke failed")
    print("d23_fcmi_analyzer_synthetic_smoke=pass")


def write_report(
    path: Path,
    decision: str,
    comparisons: dict[str, bool],
    health: dict[str, bool],
) -> None:
    lines = [
        "# SC-D23-FCMI Step9/10 Formal Audit",
        "",
        f"Decision=`{decision}`。",
        "",
        "## Four evidence layers",
        "",
        f"- paper-facing effectiveness: "
        f"`{comparisons['effectiveness_fcmi_vs_a6']}`；",
        "- matched attribution: "
        + json.dumps(comparisons, ensure_ascii=False, sort_keys=True)
        + "；",
        "- internal health: "
        + json.dumps(health, ensure_ascii=False, sort_keys=True)
        + "；",
        "- failure attribution: 见machine decision与冻结decision map。",
        "",
        "该报告只允许由完整五dataset矩阵生成；不得选择性删除负dataset、"
        "horizon、control或seed。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    if args.raw_root is None or args.output_dir is None:
        raise ValueError("raw-root and output-dir are required")

    test_metrics: list[dict[str, Any]] = []
    validation_metrics: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            test_rows, validation_rows, audit = load_run(
                args.raw_root,
                arm,
                dataset,
                args.seed,
                config,
            )
            test_metrics.extend(test_rows)
            validation_metrics.extend(validation_rows)
            audits.append(audit)
    missing = [row for row in audits if row["status"] != "ok"]
    if missing:
        raise RuntimeError(f"incomplete or invalid D23 matrix: {missing}")

    test_cells, test_summaries = compare(
        test_metrics,
        config["test_comparisons"],
        config,
    )
    validation_cells, validation_summaries = compare(
        validation_metrics,
        config["validation_comparisons"],
        config,
    )
    summaries = test_summaries + validation_summaries
    cells = test_cells + validation_cells
    comparison_results = comparison_passes(summaries, config)
    health_results, health_rows = health_passes(audits, config)
    protocol_valid = bool(
        len(test_metrics) == config["matrix"]["official_test_cells"]
        and len(validation_metrics)
        == config["matrix"]["validation_cells"]
        and all(row["protocol_pass"] for row in audits)
    )
    decision = decide(
        comparison_results,
        health_results,
        protocol_valid,
    )
    payload = {
        "candidate_version": config["candidate_version"],
        "seed": args.seed,
        "decision": decision,
        "decision_consequence": config["decision_map"][decision],
        "protocol_valid": protocol_valid,
        "comparison_passes": comparison_results,
        "internal_health_passes": health_results,
        "matrix": config["matrix"],
        "test_access": {
            "test_informed": config["test_informed"],
            "authorization": config["authorization"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_comparisons.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "run_audit.csv", audits)
    write_csv(args.output_dir / "internal_health.csv", health_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "step9_10_report.md",
        decision,
        comparison_results,
        health_results,
    )
    print(f"d23_fcmi_analysis_done decision={decision}")


if __name__ == "__main__":
    main()
