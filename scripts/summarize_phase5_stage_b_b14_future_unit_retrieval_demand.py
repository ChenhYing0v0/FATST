from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("ETTh2", "ETTm1", "Weather")
UNIT_SIZES = (180, 240)
CSV_FILES = (
    "b14_future_unit_retrieval_batches.csv",
    "b14_future_unit_retrieval_pairs.csv",
    "b14_future_unit_retrieval_profiles.csv",
    "b14_future_unit_retrieval_summary.csv",
    "b14_future_unit_retrieval_bootstrap.csv",
    "b14_history_patch_evidence_audit.csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(input_root: Path, name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        rows.extend(read_csv(input_root / dataset / "seed2021" / name))
    return rows


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def decision(summary_rows: list[dict[str, str]]) -> tuple[str, dict[str, bool]]:
    if len(summary_rows) != len(DATASETS) * len(UNIT_SIZES):
        return "diagnostic_invalid_for_direction_rejection", {}
    expected_settings = {
        (dataset, unit_size) for dataset in DATASETS for unit_size in UNIT_SIZES
    }
    returned_settings = {
        (row["dataset"], int(row["unit_size"])) for row in summary_rows
    }
    if returned_settings != expected_settings:
        return "diagnostic_invalid_for_direction_rejection", {}
    evidence_valid = all(truth(row["audit_pass"]) for row in summary_rows)
    finite = all(truth(row["finite_profiles"]) for row in summary_rows)
    mass_valid = all(
        float(row["max_patch_mass_conservation_error"]) <= 1e-6
        for row in summary_rows
    )
    dataset_support = {
        dataset: all(
            truth(row["retrieval_demand_mismatch_support"])
            for row in summary_rows
            if row["dataset"] == dataset
        )
        and sum(row["dataset"] == dataset for row in summary_rows) == len(UNIT_SIZES)
        for dataset in DATASETS
    }
    if not evidence_valid or not finite or not mass_valid:
        return "diagnostic_invalid_for_direction_rejection", dataset_support
    if sum(dataset_support.values()) >= 2:
        return "partial_pass_retrieval_demand_mismatch", dataset_support
    setting_support = sum(
        truth(row["retrieval_demand_mismatch_support"]) for row in summary_rows
    )
    sensitivity_specific = sum(
        float(row["mean_sensitivity_cosine"]) < 0.80 for row in summary_rows
    )
    if setting_support:
        return "dataset_or_unit_size_specific_mismatch", dataset_support
    if sensitivity_specific >= 3:
        return "current_a6_sensitivity_already_unit_specific", dataset_support
    return "retrieval_demand_problem_not_supported", dataset_support


def report(
    path: Path,
    summary_rows: list[dict[str, str]],
    final_decision: str,
    dataset_support: dict[str, bool],
) -> None:
    lines = [
        "# Phase5 StageB B14-FURD Step 3 Cross-Dataset Report",
        "",
        f"[Decision] `{final_decision}`。",
        "",
        "## Gate Results",
        "",
        "| Dataset | U | dCos mean | dCos p05 | dJS mean | dJS p05 | sensitivity cos | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['unit_size']} | "
            f"{float(row['delta_cosine_mean']):.4f} | "
            f"{float(row['delta_cosine_p05']):.4f} | "
            f"{float(row['delta_js_mean']):.4f} | "
            f"{float(row['delta_js_p05']):.4f} | "
            f"{float(row['mean_sensitivity_cosine']):.4f} | "
            f"{'yes' if truth(row['retrieval_demand_mismatch_support']) else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Dataset Gate",
            "",
            "Dataset-level support requires both U180 and U240 to pass. Overall "
            "support requires at least two datasets.",
            "",
            *[
                f"- {dataset}: {'support' if supported else 'no support'}"
                for dataset, supported in dataset_support.items()
            ],
            "",
            "## Failure Attribution Boundary",
            "",
            "该诊断只判断 accepted A6 是否存在 error-conditioned demand 与 existing "
            "sensitivity 的 patch-level",
            "mismatch。正结果只允许进入 parameter-matched B14-B probe；负结果回滚 "
            "Step 2，不允许通过实现",
            "cross-attention 来替代 problem evidence。任何 evidence-contract、non-finite "
            "或 mass-conservation",
            "问题只能标记为 `diagnostic_invalid_for_direction_rejection`。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[dict[str, str]]] = {}
    for name in CSV_FILES:
        combined[name] = collect(args.input_root, name)
        write_csv(args.output_dir / name, combined[name])

    summary_rows = combined["b14_future_unit_retrieval_summary.csv"]
    summary_rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), int(row["unit_size"])))
    final_decision, dataset_support = decision(summary_rows)
    per_dataset = {}
    for dataset in DATASETS:
        decision_path = (
            args.input_root
            / dataset
            / "seed2021"
            / "b14_future_unit_retrieval_decision.json"
        )
        per_dataset[dataset] = json.loads(decision_path.read_text())
        if per_dataset[dataset].get("dataset") != dataset:
            raise ValueError(
                f"dataset identity mismatch for {decision_path}: "
                f"{per_dataset[dataset].get('dataset')}"
            )
    (args.output_dir / "b14_future_unit_retrieval_decision.json").write_text(
        json.dumps(
            {
                "decision": final_decision,
                "dataset_support": dataset_support,
                "per_dataset": per_dataset,
                "setting_count": len(summary_rows),
                "support_setting_count": sum(
                    truth(row["retrieval_demand_mismatch_support"])
                    for row in summary_rows
                ),
                "max_mass_conservation_error": float(
                    np.max(
                        [
                            float(row["max_patch_mass_conservation_error"])
                            for row in summary_rows
                        ]
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report(
        args.output_dir / "b14_future_unit_retrieval_report.md",
        summary_rows,
        final_decision,
        dataset_support,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
