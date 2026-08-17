#!/usr/bin/env python3
"""Build the accuracy, parameter, and one-epoch efficiency table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SYSTEMS = ("ISCF-BSCA", "TimeAlign", "QDF")
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
HORIZONS = {"96", "192", "336", "720"}
EXPECTED_MODELS = {"ISCF-BSCA": 1, "TimeAlign": 4, "QDF": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-i-data", type=Path, required=True)
    parser.add_argument("--parameter-data", type=Path, required=True)
    parser.add_argument("--epoch-cycle-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_table(rows: list[dict[str, Any]]) -> str:
    display = {
        "ISCF-BSCA": r"\textbf{ISCF-BSCA (ours)}",
        "TimeAlign": "TimeAlign",
        "QDF": "QDF",
    }
    body = []
    for row in rows:
        model_count = f'{int(row["model_count"])}'
        mse = f'{float(row["main_i_mse"]):.3f}'
        mae = f'{float(row["main_i_mae"]):.3f}'
        params = f'{float(row["total_parameters_million"]):.3f}'
        epoch = f'{float(row["one_epoch_cycle_seconds"]):.1f}'
        if row["system"] == "ISCF-BSCA":
            model_count = rf"\textbf{{{model_count}}}"
            mse = rf"\textbf{{{mse}}}"
            mae = rf"\textbf{{{mae}}}"
            params = rf"\textbf{{{params}}}"
        if row["system"] == "QDF":
            epoch = rf"\textbf{{{epoch}}}"
        body.append(
            " & ".join(
                [
                    display[str(row["system"])],
                    model_count,
                    mse,
                    mae,
                    params,
                    epoch,
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Accuracy and training-resource cost for serving four forecasting horizons. Accuracy is the macro average of the 28 Main I dataset--horizon cells. Parameter count and one-epoch cycle time are macro-averaged over seven datasets; horizon-specific systems sum their four native models. Lower is better.}",
            r"\label{tab:iscf-bsca-efficiency}",
            r"\small",
            r"\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"System & Models & MSE & MAE & Params (M) & 1-Epoch (s) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.98\linewidth}{\footnotesize ISCF-BSCA counts one unified model and one training epoch. TimeAlign and QDF count the sum of four independently trained models for $H\in\{96,192,336,720\}$. For each checkpoint, 1-Epoch is the median logged duration of a native training epoch plus its scheduled validation pass; no test evaluation is included.}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    main_rows = read_csv(args.main_i_data)
    parameter_rows = read_csv(args.parameter_data)
    epoch_rows = read_csv(args.epoch_cycle_data)

    dataset_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for dataset in DATASETS:
            accuracy = [
                row
                for row in main_rows
                if row["model"] == system
                and row["dataset"] == dataset
                and row["horizon"] in HORIZONS
            ]
            if len(accuracy) != 4:
                raise RuntimeError(f"incomplete Main I cells: {system}/{dataset}")
            parameter = [
                row
                for row in parameter_rows
                if row["system"] == system and row["dataset"] == dataset
            ]
            if len(parameter) != 1:
                raise RuntimeError(f"invalid parameter unit: {system}/{dataset}")
            cycles = [
                row
                for row in epoch_rows
                if row["system"] == system and row["dataset"] == dataset
            ]
            if len(cycles) != EXPECTED_MODELS[system]:
                raise RuntimeError(f"incomplete epoch cycles: {system}/{dataset}")
            model_count = int(parameter[0]["trained_model_count"])
            if model_count != EXPECTED_MODELS[system]:
                raise RuntimeError(f"model-count mismatch: {system}/{dataset}")
            dataset_rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "model_count": model_count,
                    "main_i_mse": statistics.fmean(
                        float(row["mse"]) for row in accuracy
                    ),
                    "main_i_mae": statistics.fmean(
                        float(row["mae"]) for row in accuracy
                    ),
                    "total_parameters": int(parameter[0]["total_stored_parameters"]),
                    "total_parameters_million": int(
                        parameter[0]["total_stored_parameters"]
                    )
                    / 1e6,
                    "one_epoch_cycle_seconds": sum(
                        float(row["epoch_cycle_median_seconds"]) for row in cycles
                    ),
                    "timing_checkpoint_count": len(cycles),
                }
            )

    system_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        selected = [row for row in dataset_rows if row["system"] == system]
        system_rows.append(
            {
                "system": system,
                "model_count": EXPECTED_MODELS[system],
                "main_i_mse": statistics.fmean(
                    float(row["main_i_mse"]) for row in selected
                ),
                "main_i_mae": statistics.fmean(
                    float(row["main_i_mae"]) for row in selected
                ),
                "total_parameters_million": statistics.fmean(
                    float(row["total_parameters_million"]) for row in selected
                ),
                "one_epoch_cycle_seconds": statistics.fmean(
                    float(row["one_epoch_cycle_seconds"]) for row in selected
                ),
            }
        )

    primary = next(row for row in system_rows if row["system"] == "ISCF-BSCA")
    comparisons = {}
    for baseline in ("TimeAlign", "QDF"):
        row = next(item for item in system_rows if item["system"] == baseline)
        comparisons[baseline] = {
            "mse_improvement_percent": (
                (row["main_i_mse"] - primary["main_i_mse"])
                / row["main_i_mse"]
                * 100.0
            ),
            "mae_improvement_percent": (
                (row["main_i_mae"] - primary["main_i_mae"])
                / row["main_i_mae"]
                * 100.0
            ),
            "parameter_reduction_percent": (
                (row["total_parameters_million"] - primary["total_parameters_million"])
                / row["total_parameters_million"]
                * 100.0
            ),
            "iscf_epoch_cycle_ratio": (
                primary["one_epoch_cycle_seconds"]
                / row["one_epoch_cycle_seconds"]
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "efficiency_dataset_results.csv", dataset_rows)
    write_csv(args.output_dir / "efficiency_system_macro_results.csv", system_rows)
    table_dir = args.output_dir / "table"
    table_dir.mkdir(exist_ok=True)
    fragment = latex_table(system_rows)
    (table_dir / "table_iscf_bsca_efficiency.tex").write_text(
        fragment, encoding="utf-8"
    )
    standalone = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[letterpaper,margin=0.55in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            fragment,
            r"\end{document}",
            "",
        ]
    )
    (table_dir / "table_iscf_bsca_efficiency_standalone.tex").write_text(
        standalone, encoding="utf-8"
    )
    summary = {
        "protocol_id": "ISCF-BSCA-EFFICIENCY-ACCURACY-PARAMS-EPOCH-20260817",
        "status": "complete_log_reuse_no_new_training_no_test_access",
        "systems": system_rows,
        "comparisons": comparisons,
        "input_sha256": {
            "main_i_data": sha256(args.main_i_data),
            "parameter_data": sha256(args.parameter_data),
            "epoch_cycle_data": sha256(args.epoch_cycle_data),
        },
    }
    (args.output_dir / "efficiency_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("efficiency_accuracy_params_epoch=pass systems=3 datasets=7")


if __name__ == "__main__":
    main()
