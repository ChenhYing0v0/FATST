#!/usr/bin/env python3
"""Aggregate the frozen D22-C target-access matrix and apply its problem gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relative_gain(primary: float, control: float) -> float:
    if not math.isfinite(primary) or not math.isfinite(control) or control <= 0:
        raise ValueError(f"invalid metric pair primary={primary} control={control}")
    return (control - primary) / control


def load_matrix(
    input_root: Path,
    config: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, float]],
    list[dict[str, Any]],
    list[str],
]:
    metrics: dict[tuple[str, str, str, str], dict[str, float]] = {}
    parameter_rows = []
    errors = []
    for dataset in config["datasets"]:
        metadata_path = input_root / dataset / "metadata.json"
        if not metadata_path.exists():
            errors.append(f"missing metadata: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("candidate_version") != config["candidate_version"]:
            errors.append(f"candidate version mismatch: {dataset}")
        if not metadata.get("matrix_complete", False):
            errors.append(f"incomplete arm matrix: {dataset}")
        for arm in config["arms"]:
            arm_root = input_root / dataset / arm
            summary_path = arm_root / "summary.json"
            metric_path = arm_root / "metrics.csv"
            if not summary_path.exists() or not metric_path.exists():
                errors.append(f"missing arm artifacts: {dataset}/{arm}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            parameter_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "parameter_count": int(summary["parameter_count"]),
                    "best_epoch": int(summary["best_epoch"]),
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "attention_entropy_test": float(
                        summary["test_health"]["attention_entropy"]
                    ),
                    "attention_target_dispersion_test": float(
                        summary["test_health"]["attention_target_dispersion"]
                    ),
                    "prediction_coordinate_dispersion_test": float(
                        summary["test_health"]["prediction_coordinate_dispersion"]
                    ),
                }
            )
            for row in read_csv(metric_path):
                key = (dataset, arm, row["split"], row["region"])
                if key in metrics:
                    errors.append(f"duplicate metric row: {key}")
                    continue
                metrics[key] = {
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                }
    return metrics, parameter_rows, errors


def parameter_audit(
    parameter_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parameter_rows:
        grouped[row["dataset"]].append(row)
    audit_rows = []
    maximum_gap = 0.0
    for dataset in config["datasets"]:
        rows = grouped.get(dataset, [])
        counts = [int(row["parameter_count"]) for row in rows]
        if not counts:
            continue
        gap = (max(counts) - min(counts)) / max(counts)
        maximum_gap = max(maximum_gap, gap)
        audit_rows.append(
            {
                "dataset": dataset,
                "minimum_parameters": min(counts),
                "maximum_parameters": max(counts),
                "relative_gap": gap,
                "arm_count": len(rows),
                "exact_match": int(gap == 0.0),
            }
        )
    return audit_rows, maximum_gap


def comparison_rows(
    metrics: dict[tuple[str, str, str, str], dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = config["primary_arm"]
    controls = [arm for arm in config["arms"] if arm != primary]
    rows = []
    for split in ("validation", "test"):
        for control in controls:
            for dataset in config["datasets"]:
                for horizon in config["paper_facing_horizons"]:
                    region = f"prefix_{int(horizon)}"
                    primary_metric = metrics.get((dataset, primary, split, region))
                    control_metric = metrics.get((dataset, control, split, region))
                    if primary_metric is None or control_metric is None:
                        continue
                    rows.append(
                        {
                            "split": split,
                            "control": control,
                            "dataset": dataset,
                            "horizon": int(horizon),
                            "ordered_mse": primary_metric["mse"],
                            "control_mse": control_metric["mse"],
                            "mse_gain": relative_gain(
                                primary_metric["mse"],
                                control_metric["mse"],
                            ),
                            "ordered_mae": primary_metric["mae"],
                            "control_mae": control_metric["mae"],
                            "mae_gain": relative_gain(
                                primary_metric["mae"],
                                control_metric["mae"],
                            ),
                        }
                    )
    return rows


def bin_rows(
    metrics: dict[tuple[str, str, str, str], dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    primary = config["primary_arm"]
    controls = [arm for arm in config["arms"] if arm != primary]
    rows = []
    for control in controls:
        for dataset in config["datasets"]:
            for entry in config["coordinate_bins"]:
                region = f"bin_{entry['name']}"
                primary_metric = metrics.get((dataset, primary, "test", region))
                control_metric = metrics.get((dataset, control, "test", region))
                if primary_metric is None or control_metric is None:
                    continue
                rows.append(
                    {
                        "control": control,
                        "dataset": dataset,
                        "bin": entry["name"],
                        "start": int(entry["start"]),
                        "end": int(entry["end"]),
                        "mse_gain": relative_gain(
                            primary_metric["mse"],
                            control_metric["mse"],
                        ),
                        "mae_gain": relative_gain(
                            primary_metric["mae"],
                            control_metric["mae"],
                        ),
                    }
                )
    return rows


def aggregate_comparisons(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    aggregate_rows = []
    controls = [
        arm for arm in config["arms"] if arm != config["primary_arm"]
    ]
    for split in ("validation", "test"):
        for control in controls:
            selected = [
                row
                for row in rows
                if row["split"] == split and row["control"] == control
            ]
            expected = len(config["datasets"]) * len(
                config["paper_facing_horizons"]
            )
            dataset_means = {
                dataset: sum(
                    row["mse_gain"]
                    for row in selected
                    if row["dataset"] == dataset
                )
                / len(config["paper_facing_horizons"])
                for dataset in config["datasets"]
            }
            horizon_means = {
                int(horizon): sum(
                    row["mse_gain"]
                    for row in selected
                    if row["horizon"] == int(horizon)
                )
                / len(config["datasets"])
                for horizon in config["paper_facing_horizons"]
            }
            aggregate_rows.append(
                {
                    "split": split,
                    "control": control,
                    "cell_count": len(selected),
                    "expected_cell_count": expected,
                    "macro_mse_gain": sum(row["mse_gain"] for row in selected)
                    / len(selected),
                    "macro_mae_gain": sum(row["mae_gain"] for row in selected)
                    / len(selected),
                    "positive_cells": sum(row["mse_gain"] > 0 for row in selected),
                    "positive_datasets": sum(
                        value > 0 for value in dataset_means.values()
                    ),
                    "positive_horizons": sum(
                        value > 0 for value in horizon_means.values()
                    ),
                    "minimum_cell_gain": min(row["mse_gain"] for row in selected),
                    "maximum_cell_gain": max(row["mse_gain"] for row in selected),
                }
            )
    return aggregate_rows


def apply_gate(
    aggregate_rows: list[dict[str, Any]],
    maximum_parameter_gap: float,
    load_errors: list[str],
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    gates = config["gates"]
    results = []
    pathology = bool(load_errors)
    if maximum_parameter_gap > float(gates["max_parameter_relative_gap"]):
        pathology = True
        load_errors.append(
            f"parameter gap {maximum_parameter_gap:.6f} exceeds contract"
        )
    test_rows = {
        row["control"]: row
        for row in aggregate_rows
        if row["split"] == "test"
    }
    validation_rows = {
        row["control"]: row
        for row in aggregate_rows
        if row["split"] == "validation"
    }
    expected_cells = len(config["datasets"]) * len(
        config["paper_facing_horizons"]
    )
    key_controls = {"GLOBAL_COMPRESSED", "GENERIC_MATCHED"}
    for control in [
        arm for arm in config["arms"] if arm != config["primary_arm"]
    ]:
        test = test_rows.get(control)
        validation = validation_rows.get(control)
        if test is None or validation is None:
            pathology = True
            load_errors.append(f"missing aggregate comparison: {control}")
            continue
        required_mse = float(gates["ordered_over_each_control_macro_mse_min"])
        if control == "GLOBAL_COMPRESSED":
            required_mse = float(gates["ordered_over_global_macro_mse_min"])
        elif control == "GENERIC_MATCHED":
            required_mse = float(gates["ordered_over_generic_macro_mse_min"])
        conditions = {
            "matrix_complete": int(test["cell_count"]) == expected_cells,
            "macro_mse": float(test["macro_mse_gain"]) >= required_mse,
            "macro_mae": float(test["macro_mae_gain"])
            >= float(gates["ordered_over_each_control_macro_mae_min"]),
            "no_severe_degradation": float(test["minimum_cell_gain"])
            > -float(gates["severe_cell_degradation_threshold"]),
        }
        if control in key_controls:
            conditions.update(
                {
                    "positive_cells": int(test["positive_cells"])
                    >= int(gates["key_control_positive_cells_min"]),
                    "positive_datasets": int(test["positive_datasets"])
                    >= int(gates["key_control_positive_datasets_min"]),
                    "positive_horizons": int(test["positive_horizons"])
                    >= int(gates["key_control_positive_horizons_min"]),
                    "validation_test_same_sign": (
                        float(validation["macro_mse_gain"]) > 0
                    )
                    == (float(test["macro_mse_gain"]) > 0),
                }
            )
        results.append(
            {
                "control": control,
                "required_macro_mse_gain": required_mse,
                "observed_macro_mse_gain": float(test["macro_mse_gain"]),
                "observed_macro_mae_gain": float(test["macro_mae_gain"]),
                "conditions": conditions,
                "passed": all(conditions.values()),
            }
        )
    if pathology:
        return (
            "target_access_unresolved_protocol_invalid",
            results,
            "optimization_or_numeric_pathology",
        )
    if results and all(result["passed"] for result in results):
        return (
            "target_coordinate_information_access_supported",
            results,
            "none",
        )
    shuffle_controls = {"ORDER_SHUFFLED", "TARGET_SHUFFLED_QUERY"}
    shuffle_pass = all(
        result["passed"]
        for result in results
        if result["control"] in shuffle_controls
    )
    generic_pass = next(
        (
            result["passed"]
            for result in results
            if result["control"] == "GENERIC_MATCHED"
        ),
        False,
    )
    if not shuffle_pass:
        attribution = "hypothesis_false_exact_target_order_access_protocol"
    elif not generic_pass:
        attribution = "capacity_control_explains_or_intervention_point_wrong"
    else:
        attribution = "readout_or_head_design_wrong_or_split_instability"
    return "target_coordinate_information_access_not_supported", results, attribution


def format_percent(value: float) -> str:
    return f"{100.0 * value:+.4f}%"


def write_report(
    path: Path,
    decision: str,
    attribution: str,
    aggregate_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    bin_metrics: list[dict[str, Any]],
    parameter_gap: float,
    errors: list[str],
    config: dict[str, Any],
) -> None:
    test_rows = [
        row for row in aggregate_rows if row["split"] == "test"
    ]
    validation_rows = {
        row["control"]: row
        for row in aggregate_rows
        if row["split"] == "validation"
    }
    lines = [
        "# SC-D22-HFA D22-C target-coordinate information-access audit",
        "",
        "## 1. Decision",
        "",
        f"- `decision`: `{decision}`",
        f"- `failure_attribution`: `{attribution}`",
        f"- `candidate_version`: `{config['candidate_version']}`",
        "- `role`: `diagnostic_only_raw_history_primary`",
        "- `test_role`: one-shot `test_informed` problem gate；validation只选择checkpoint。",
        "- ordered patch memory仍是诊断载体，不是paper contribution。",
        "",
        "## 2. Complete 20-cell scorecard",
        "",
        "| Control | Val MSE gain | Test MSE gain | Test MAE gain | Positive cells | Datasets | Horizons | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    gate_lookup = {row["control"]: row for row in gate_rows}
    for row in test_rows:
        val = validation_rows[row["control"]]
        gate = gate_lookup[row["control"]]
        lines.append(
            f"| `{row['control']}` | {format_percent(val['macro_mse_gain'])} | "
            f"{format_percent(row['macro_mse_gain'])} | "
            f"{format_percent(row['macro_mae_gain'])} | "
            f"{row['positive_cells']}/20 | {row['positive_datasets']}/5 | "
            f"{row['positive_horizons']}/4 | "
            f"{'pass' if gate['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "所有gain定义为`(control - ORDERED_TARGET_ACCESS) / control`；正值表示ordered更好。",
            "",
            "## 3. Coordinate-bin audit",
            "",
            "| Control | Bin | Macro MSE gain | Positive datasets |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    grouped_bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bin_metrics:
        grouped_bins[(row["control"], row["bin"])].append(row)
    for (control, bin_name), rows in grouped_bins.items():
        macro = sum(row["mse_gain"] for row in rows) / len(rows)
        positive = sum(row["mse_gain"] > 0 for row in rows)
        lines.append(
            f"| `{control}` | `{bin_name}` | {format_percent(macro)} | "
            f"{positive}/{len(rows)} |"
        )
    lines.extend(
        [
            "",
            "## 4. Static and protocol integrity",
            "",
            f"- maximum trainable-parameter relative gap: `{parameter_gap:.8f}`；",
            f"- matrix/load errors: `{len(errors)}`；",
        ]
    )
    for error in errors:
        lines.append(f"- [Protocol Error] {error}")
    lines.extend(
        [
            "",
            "## 5. Failure attribution and rollback",
            "",
        ]
    )
    if decision == "target_coordinate_information_access_supported":
        lines.extend(
            [
                "[Strong Evidence] raw-history primary在matched capacity下支持target-coordinate-specific retrieval necessity。",
                "返回Step4进行source-informed `lead-time-conditioned evidence operator`设计；本诊断arm不能直接升级为method。",
                "Contribution 2继续保持open，只有首个E2E operator暴露真实瓶颈后才允许定义。",
            ]
        )
    elif decision == "target_coordinate_information_access_not_supported":
        lines.extend(
            [
                "[Fact] exact D22-C v1 problem gate失败；不得通过seed、width、readout或representation rescue重开同一协议。",
                f"[Failure Attribution] `{attribution}`。",
                "该结论关闭exact neutral query-to-patch diagnostic，不以frozen replacement否定任意未来E2E设计。",
                "依照用户2026-07-20方向约束，rollback到joint Step2/3，在deterministic-MSE fixed-past边界内寻找新的、可证伪的问题；不自动pivot任务。",
            ]
        )
    else:
        lines.extend(
            [
                "[Fact] protocol或numeric pathology使本轮不能作方向判断。",
                "只允许修复一次构造/数值问题并重跑完全相同的冻结矩阵；不得据此升级或拒绝method方向。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics, parameter_rows, errors = load_matrix(args.input_root, config)
    parameter_audit_rows, maximum_gap = parameter_audit(parameter_rows, config)
    comparisons = comparison_rows(metrics, config)
    bins = bin_rows(metrics, config)
    aggregates = aggregate_comparisons(comparisons, config)
    decision, gate_rows, attribution = apply_gate(
        aggregates,
        maximum_gap,
        errors,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_comparisons.csv", comparisons)
    write_csv(args.output_dir / "coordinate_bin_comparisons.csv", bins)
    write_csv(args.output_dir / "aggregate_comparisons.csv", aggregates)
    write_csv(args.output_dir / "parameter_audit.csv", parameter_audit_rows)
    write_csv(args.output_dir / "internal_health.csv", parameter_rows)
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "decision": decision,
        "failure_attribution": attribution,
        "maximum_parameter_relative_gap": maximum_gap,
        "matrix_complete": not errors,
        "errors": errors,
        "gate_results": gate_rows,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "d22c_result_report.md",
        decision,
        attribution,
        aggregates,
        gate_rows,
        bins,
        maximum_gap,
        errors,
        config,
    )
    print(
        f"d22c_decision={decision} failure_attribution={attribution} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
