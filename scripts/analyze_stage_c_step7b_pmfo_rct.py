#!/usr/bin/env python3
"""Analyze the StageC Step 7B PMFO-RCT architecture-only screening matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTm1", "ETTh2", "Weather")
ARMS = (
    "a6",
    "dense_mlp_matched",
    "pmfo_no_transition",
    "pmfo_no_conservation",
    "pmfo_rct",
)
CONTROL_ARMS = (
    "dense_mlp_matched",
    "pmfo_no_transition",
    "pmfo_no_conservation",
)
READOUTS = {
    "a6": "learned-basis-forecast-operator",
    "dense_mlp_matched": "dense-mlp-matched",
    "pmfo_no_transition": "pmfo-rct-no-transition",
    "pmfo_no_conservation": "pmfo-rct-no-conservation",
    "pmfo_rct": "pmfo-rct",
}
STANDARD_HORIZONS = (48, 96, 192, 336, 720)
SEED = 2021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def run_dir(root: Path, arm: str, dataset: str) -> Path:
    return root / arm / dataset / "h720_full" / f"seed{SEED}"


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value}")
    return number


def load_run(root: Path, arm: str, dataset: str) -> dict[str, Any]:
    directory = run_dir(root, arm, dataset)
    required = {
        "metrics": directory / "metrics_by_target_horizon.csv",
        "training": directory / "training_log.csv",
        "config": directory / "effective_config.json",
        "diagnostics": directory / "model_diagnostics.json",
        "invariants": directory / "trained_invariants.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "missing",
            "missing": ",".join(missing),
            "run_dir": str(directory),
        }

    metrics = read_csv(required["metrics"])
    by_horizon = {int(row["target_horizon"]): row for row in metrics}
    if sorted(by_horizon) != list(range(1, 721)):
        return {
            "dataset": dataset,
            "arm": arm,
            "status": "incomplete_dense_horizons",
            "dense_horizon_count": len(by_horizon),
            "run_dir": str(directory),
        }
    mse_values = [finite_float(by_horizon[horizon]["mse"]) for horizon in range(1, 721)]
    mae_values = [finite_float(by_horizon[horizon]["mae"]) for horizon in range(1, 721)]
    training = read_csv(required["training"])
    if not training:
        raise ValueError(f"empty training log: {required['training']}")
    for row in training:
        for field in ("train_loss", "val_mean_mse", "lr"):
            finite_float(row[field])
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    diagnostics = json.loads(required["diagnostics"].read_text(encoding="utf-8"))
    invariants = json.loads(required["invariants"].read_text(encoding="utf-8"))
    adapter = config["adapter"]
    protocol_ok = (
        adapter["dataset"] == dataset
        and adapter["mode"] == "unified"
        and adapter["pred_len"] == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == [720]
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["pred_loss_mode"] == "full"
        and adapter["protocol_class"] == "method_screening"
        and adapter["readout_mode"] == READOUTS[arm]
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == "test"
    )
    result: dict[str, Any] = {
        "dataset": dataset,
        "arm": arm,
        "status": "ok" if protocol_ok else "protocol_mismatch",
        "readout_mode": adapter["readout_mode"],
        "profile_hash": adapter["profile_hash"],
        "dense_horizon_count": len(by_horizon),
        "dense_mse_auc": mean(mse_values),
        "dense_mae_auc": mean(mae_values),
        "h720_mse": mse_values[-1],
        "h720_mae": mae_values[-1],
        "epochs_ran": len(training),
        "best_epoch": int(training[-1]["best_epoch_so_far"]),
        "best_val_mse": min(float(row["val_mean_mse"]) for row in training),
        "invariant_pass": bool(invariants["pass"]),
        "full_prefix_max_abs": float(invariants["full_prefix_max_abs"]),
        "total_parameters": diagnostics.get("total_parameters", ""),
        "pmfo_decoder_parameters": diagnostics.get(
            "pmfo_decoder_parameters",
            "",
        ),
        "run_dir": str(directory),
    }
    for horizon in STANDARD_HORIZONS:
        result[f"h{horizon}_mse"] = finite_float(by_horizon[horizon]["mse"])
        result[f"h{horizon}_mae"] = finite_float(by_horizon[horizon]["mae"])
    return result


def relative_improvement(candidate: float, reference: float) -> float:
    return (1.0 - candidate / reference) * 100.0


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["arm"]): row
        for row in summary
        if row["status"] == "ok"
    }
    rows = []
    for dataset in DATASETS:
        for candidate_arm in ARMS[1:]:
            candidate = lookup.get((dataset, candidate_arm))
            baseline = lookup.get((dataset, "a6"))
            if candidate is None or baseline is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_arm": candidate_arm,
                    "reference_arm": "a6",
                    "candidate_dense_mse_auc": candidate["dense_mse_auc"],
                    "reference_dense_mse_auc": baseline["dense_mse_auc"],
                    "dense_mse_improvement_pct": relative_improvement(
                        candidate["dense_mse_auc"],
                        baseline["dense_mse_auc"],
                    ),
                    "dense_mae_improvement_pct": relative_improvement(
                        candidate["dense_mae_auc"],
                        baseline["dense_mae_auc"],
                    ),
                }
            )
    return rows


def decide_gate(summary: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(DATASETS) * len(ARMS)
    valid = [row for row in summary if row["status"] == "ok"]
    if len(valid) != expected:
        return {
            "decision": "analysis_pending",
            "complete_runs": len(valid),
            "expected_runs": expected,
            "failure_attribution": "incomplete_artifacts_or_protocol_mismatch",
        }
    lookup = {(row["dataset"], row["arm"]): row for row in valid}
    dataset_rows = []
    for dataset in DATASETS:
        pmfo = lookup[(dataset, "pmfo_rct")]
        a6 = lookup[(dataset, "a6")]
        controls = [lookup[(dataset, arm)] for arm in CONTROL_ARMS]
        best_control = min(controls, key=lambda row: row["dense_mse_auc"])
        dataset_rows.append(
            {
                "dataset": dataset,
                "pmfo_vs_a6_improvement_pct": relative_improvement(
                    pmfo["dense_mse_auc"],
                    a6["dense_mse_auc"],
                ),
                "best_control": best_control["arm"],
                "pmfo_vs_best_control_improvement_pct": relative_improvement(
                    pmfo["dense_mse_auc"],
                    best_control["dense_mse_auc"],
                ),
                "pmfo_invariant_pass": pmfo["invariant_pass"],
            }
        )
    macro_a6 = mean(row["pmfo_vs_a6_improvement_pct"] for row in dataset_rows)
    macro_control = mean(
        row["pmfo_vs_best_control_improvement_pct"] for row in dataset_rows
    )
    minimum_dataset = min(
        row["pmfo_vs_a6_improvement_pct"] for row in dataset_rows
    )
    invariants_pass = all(row["pmfo_invariant_pass"] for row in dataset_rows)
    numeric_pathology = any(
        lookup[(dataset, "pmfo_rct")]["dense_mse_auc"]
        > 2.0 * lookup[(dataset, "a6")]["dense_mse_auc"]
        for dataset in DATASETS
    )
    gates = {
        "macro_vs_a6_ge_1pct": macro_a6 >= 1.0,
        "each_dataset_not_worse_0_5pct": minimum_dataset >= -0.5,
        "macro_vs_best_control_ge_0_5pct": macro_control >= 0.5,
        "trained_invariants_pass": invariants_pass,
        "no_numeric_pathology": not numeric_pathology,
    }
    if all(gates.values()):
        decision = "partial_pass"
        attribution = "single_seed_architecture_signal_requires_three_seed_confirmation"
    elif numeric_pathology or not invariants_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
        attribution = "optimization_numeric_or_implementation_pathology"
    elif (
        lookup[("ETTh2", "pmfo_rct")]["dense_mse_auc"]
        > 1.005 * lookup[("ETTh2", "a6")]["dense_mse_auc"]
        and all(
            lookup[(dataset, "pmfo_rct")]["dense_mse_auc"]
            <= 1.005 * lookup[(dataset, "a6")]["dense_mse_auc"]
            for dataset in ("ETTm1", "Weather")
        )
    ):
        decision = "rollback_interface_audit"
        attribution = "intervention_point_wrong_possible_on_etth2"
    elif any(
        row["best_control"] == "pmfo_no_transition"
        and row["pmfo_vs_best_control_improvement_pct"] < 0.5
        for row in dataset_rows
    ):
        decision = "rollback_step4"
        attribution = "recursive_transition_not_supported_or_control_explains"
    else:
        decision = "rollback_step4"
        attribution = "capacity_or_structural_control_explains"
    return {
        "decision": decision,
        "complete_runs": len(valid),
        "expected_runs": expected,
        "macro_pmfo_vs_a6_improvement_pct": macro_a6,
        "minimum_dataset_pmfo_vs_a6_improvement_pct": minimum_dataset,
        "macro_pmfo_vs_best_control_improvement_pct": macro_control,
        "dataset_results": dataset_rows,
        "gates": gates,
        "failure_attribution": attribution,
    }


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(
    output_dir: Path,
    summary: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    valid = [row for row in summary if row["status"] == "ok"]
    fields = [
        "dataset",
        "arm",
        "dense_mse_auc",
        "dense_mae_auc",
        "h48_mse",
        "h96_mse",
        "h192_mse",
        "h336_mse",
        "h720_mse",
        "epochs_ran",
        "invariant_pass",
    ]
    lines = [
        "# StageC Step 7B PMFO-RCT Screening Report",
        "",
        "## Scope",
        "",
        "三数据集、五arms、seed2021；训练保持frozen full-H720 pointwise L1，",
        "best checkpoint由H720 validation MSE选择；test一次生成H720后聚合H1..720 MSE/MAE。",
        "`dense_mse_auc`定义为720个prefix MSE的算术平均，对应uniform horizon measure。",
        "",
        "## Run Summary",
        "",
        *markdown_table(valid, fields),
        "",
        "## Gate",
        "",
        f"- decision: `{gate['decision']}`；",
        f"- complete: `{gate['complete_runs']}/{gate['expected_runs']}`；",
        f"- failure attribution: `{gate['failure_attribution']}`；",
    ]
    if gate["decision"] != "analysis_pending":
        lines.extend(
            [
                f"- macro PMFO vs A6: `{gate['macro_pmfo_vs_a6_improvement_pct']:.4f}%`；",
                f"- worst dataset PMFO vs A6: `{gate['minimum_dataset_pmfo_vs_a6_improvement_pct']:.4f}%`；",
                f"- macro PMFO vs per-dataset best control: `{gate['macro_pmfo_vs_best_control_improvement_pct']:.4f}%`；",
                "",
                "[Boundary] 单seed只能形成`partial_pass`或rollback，不能形成effectiveness claim。",
            ]
        )
    (output_dir / "step7b_screening_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        load_run(args.raw_root, arm, dataset)
        for dataset in DATASETS
        for arm in ARMS
    ]
    comparisons = comparison_rows(summary)
    gate = decide_gate(summary)
    write_csv(args.output_dir / "run_summary.csv", summary)
    if comparisons:
        write_csv(args.output_dir / "comparisons.csv", comparisons)
    (args.output_dir / "step7b_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render_report(args.output_dir, summary, gate)
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
