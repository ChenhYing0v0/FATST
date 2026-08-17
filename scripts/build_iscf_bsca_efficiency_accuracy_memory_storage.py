#!/usr/bin/env python3
"""Build the Main-I accuracy, peak-memory, and checkpoint-storage table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SYSTEMS = ("ISCF-BSCA", "TimeAlign", "QDF", "AMD", "SimpleTM")
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
HORIZONS = {"96", "192", "336", "720"}
EXPECTED_MODELS = {
    "ISCF-BSCA": 1,
    "TimeAlign": 4,
    "QDF": 4,
    "AMD": 4,
    "SimpleTM": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-i-data", type=Path, required=True)
    parser.add_argument("--profiler-data", type=Path, required=True)
    parser.add_argument("--additional-units-dir", type=Path)
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


def lower_is_better(value: float, values: list[float]) -> str:
    ordered = sorted(set(values))
    formatted = f"{value:.3f}"
    if value == ordered[0]:
        return rf"\textbf{{{formatted}}}"
    if len(ordered) > 1 and value == ordered[1]:
        return rf"\underline{{{formatted}}}"
    return formatted


def latex_table(rows: list[dict[str, Any]]) -> str:
    display = {
        "ISCF-BSCA": r"\textbf{ISCF-BSCA (ours)}",
        "TimeAlign": "TimeAlign",
        "QDF": "QDF",
        "AMD": "AMD",
        "SimpleTM": "SimpleTM",
    }
    metric_values = {
        key: [float(row[key]) for row in rows]
        for key in (
            "main_i_mse",
            "main_i_mae",
            "peak_inference_memory_mib",
            "checkpoint_storage_mib",
        )
    }
    body = []
    for row in rows:
        fields = [display[str(row["system"])]]
        for key in (
            "main_i_mse",
            "main_i_mae",
            "peak_inference_memory_mib",
            "checkpoint_storage_mib",
        ):
            fields.append(
                lower_is_better(float(row[key]), metric_values[key])
            )
        body.append(" & ".join(fields) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Accuracy, peak inference memory, and checkpoint storage for serving four forecasting horizons. Accuracy is the macro average of the 28 Main I dataset--horizon cells. Memory and storage are macro-averaged over seven datasets. Lower is better.}",
            r"\label{tab:iscf-bsca-efficiency}",
            r"\small",
            r"\setlength{\tabcolsep}{7pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"System & MSE & MAE & Peak memory (MiB) & Checkpoints (MiB) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.98\linewidth}{\footnotesize ISCF-BSCA stores and serves one unified checkpoint. TimeAlign, QDF, AMD, and SimpleTM sum the four native checkpoints for $H\in\{96,192,336,720\}$. Peak memory is measured on an exclusive RTX 3090 in FP32 with batch size 1 and all service checkpoints resident: one $H=720$ forward plus prefix views for ISCF-BSCA, and four sequential native forwards for each horizon-specific family. SimpleTM resource cost uses repeat 0 as one deployable, non-ensemble instance per horizon, whereas its frozen Main I accuracy follows the table's multi-run summary. Synthetic standardized inputs are used; no test loader or labels are accessed. Best results are bold and second-best results are underlined.}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    main_rows = read_csv(args.main_i_data)
    profiler_rows = read_csv(args.profiler_data)
    if args.additional_units_dir is not None:
        for path in sorted(args.additional_units_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("gate") != "pass":
                raise RuntimeError(f"failed additional profiler unit: {path}")
            profiler_rows.append({key: str(value) for key, value in payload.items()})

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
            service = [
                row
                for row in profiler_rows
                if row["system"] == system and row["dataset"] == dataset
            ]
            if len(service) != 1:
                raise RuntimeError(f"invalid profiler unit: {system}/{dataset}")
            service_row = service[0]
            model_count = int(service_row["trained_model_count"])
            if model_count != EXPECTED_MODELS[system]:
                raise RuntimeError(f"model-count mismatch: {system}/{dataset}")
            checkpoint_bytes = int(service_row["actual_checkpoint_bytes"])
            peak_memory_bytes = int(service_row["peak_inference_memory_bytes"])
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
                    "checkpoint_storage_bytes": checkpoint_bytes,
                    "checkpoint_storage_mib": checkpoint_bytes / (1024**2),
                    "peak_inference_memory_bytes": peak_memory_bytes,
                    "peak_inference_memory_mib": peak_memory_bytes / (1024**2),
                    "incremental_activation_peak_mib": int(
                        service_row["incremental_activation_peak_bytes"]
                    )
                    / (1024**2),
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
                "peak_inference_memory_mib": statistics.fmean(
                    float(row["peak_inference_memory_mib"]) for row in selected
                ),
                "checkpoint_storage_mib": statistics.fmean(
                    float(row["checkpoint_storage_mib"]) for row in selected
                ),
            }
        )

    primary = next(row for row in system_rows if row["system"] == "ISCF-BSCA")
    comparisons = {}
    for baseline in ("TimeAlign", "QDF", "AMD", "SimpleTM"):
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
            "peak_memory_reduction_percent": (
                (row["peak_inference_memory_mib"] - primary["peak_inference_memory_mib"])
                / row["peak_inference_memory_mib"]
                * 100.0
            ),
            "checkpoint_storage_reduction_percent": (
                (row["checkpoint_storage_mib"] - primary["checkpoint_storage_mib"])
                / row["checkpoint_storage_mib"]
                * 100.0
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
        "protocol_id": "ISCF-BSCA-EFFICIENCY-ACCURACY-MEMORY-STORAGE-20260817",
        "status": "complete_profiler_reuse_plus_additional_peak_measurement_no_new_training_no_test_access",
        "systems": system_rows,
        "comparisons": comparisons,
        "input_sha256": {
            "main_i_data": sha256(args.main_i_data),
            "profiler_data": sha256(args.profiler_data),
        },
    }
    (args.output_dir / "efficiency_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("efficiency_accuracy_memory_storage=pass systems=5 datasets=7")


if __name__ == "__main__":
    main()
