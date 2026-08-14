#!/usr/bin/env python3
"""Aggregate the frozen 35-unit ISCF-BSCA efficiency matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


SYSTEMS = (
    "ISCF-BSCA",
    "TimeAlign",
    "QDF",
    "DLinear-H720-prefix",
    "PatchTST-H720-prefix",
)
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
CHPC = {
    "ISCF-BSCA": "Architectural",
    "TimeAlign": "No",
    "QDF": "No",
    "DLinear-H720-prefix": "Service protocol",
    "PatchTST-H720-prefix": "Service protocol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_seconds(path: Path) -> float:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = [float(row["epoch_seconds"]) for row in rows]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        values = [
            float(value)
            for value in re.findall(
                r"^Epoch:\s*\d+\s+cost time:\s*([0-9.eE+-]+)\s*$",
                text,
                flags=re.MULTILINE,
            )
        ]
    if not values:
        raise RuntimeError(f"no native epoch timing found: {path}")
    return sum(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[dict[str, Any]]) -> str:
    display = {
        "ISCF-BSCA": r"\textbf{ISCF-BSCA}",
        "TimeAlign": "TimeAlign",
        "QDF": "QDF",
        "DLinear-H720-prefix": r"DLinear-$H720$-prefix",
        "PatchTST-H720-prefix": r"PatchTST-$H720$-prefix",
    }
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    display[str(row["system"])],
                    f'{float(row["trained_model_count"]):.0f}',
                    f'{float(row["parameters_million"]):.3f}',
                    f'{float(row["checkpoint_mib"]):.2f}',
                    f'{float(row["training_gpu_hours"]):.3f}',
                    f'{float(row["single_request_latency_ms"]):.3f}',
                    f'{float(row["all_horizon_service_latency_ms"]):.3f}',
                    f'{float(row["peak_inference_memory_mib"]):.1f}',
                    str(row["chpc_guarantee"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Deployment efficiency for serving four requested horizons. "
            r"Values are macro-averaged over seven datasets on one exclusive RTX 3090 "
            r"with FP32 and batch size 1.}",
            r"\label{tab:iscf-bsca-efficiency}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrl}",
            r"\toprule",
            r"System & Models & Params (M) & Ckpt. (MiB) & Train (GPU h) "
            r"& Single (ms) & All-$H$ (ms) & Peak (MiB) & CHPC \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize Models, parameters, checkpoints and "
            r"training time count the complete deployed service per dataset. Single is "
            r"the arithmetic mean over requests $H\in\{96,192,336,720\}$. All-$H$ uses "
            r"one $H=720$ forward plus prefix views for one-model services and four "
            r"sequential native forwards for horizon-specific families. Inputs are "
            r"synthetic standardized tensors; no test loader or labels are accessed.}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = manifest["rows"]
    unit_rows: list[dict[str, Any]] = []
    raw_units: dict[tuple[str, str], dict[str, Any]] = {}
    for system in SYSTEMS:
        for dataset in DATASETS:
            path = args.unit_dir / f"{system}__{dataset}.json"
            unit = json.loads(path.read_text(encoding="utf-8"))
            if unit["gate"] != "pass":
                raise RuntimeError(f"profiler gate failed: {path}")
            expected = [
                row
                for row in manifest_rows
                if row["system"] == system and row["dataset"] == dataset
            ]
            if unit["checkpoint_sha256"] != [
                row["checkpoint_sha256"] for row in expected
            ]:
                raise RuntimeError(f"manifest mismatch: {path}")
            train_seconds = sum(
                training_seconds(Path(str(row["training_log_path"])))
                for row in expected
            )
            raw_units[(system, dataset)] = unit
            unit_rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "trained_model_count": unit["trained_model_count"],
                    "total_stored_parameters": unit["total_stored_parameters"],
                    "actual_checkpoint_bytes": unit["actual_checkpoint_bytes"],
                    "training_seconds": train_seconds,
                    "training_gpu_hours": train_seconds / 3600.0,
                    "single_request_latency_ms": unit[
                        "single_request_latency_ms"
                    ],
                    "all_horizon_service_latency_ms": unit[
                        "all_horizon_service"
                    ]["latency_ms"],
                    "all_horizon_p95_iteration_latency_ms": unit[
                        "all_horizon_service"
                    ]["p95_iteration_latency_ms"],
                    "all_horizon_round_cv": unit["all_horizon_service"][
                        "round_cv"
                    ],
                    "peak_inference_memory_bytes": unit[
                        "peak_inference_memory_bytes"
                    ],
                    "incremental_activation_peak_bytes": unit[
                        "incremental_activation_peak_bytes"
                    ],
                    "checkpoint_sha256": ";".join(unit["checkpoint_sha256"]),
                }
            )
    if len(unit_rows) != 35:
        raise RuntimeError(f"expected 35 units, got {len(unit_rows)}")

    aggregate_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        selected = [row for row in unit_rows if row["system"] == system]
        mean = lambda key: statistics.fmean(float(row[key]) for row in selected)
        aggregate_rows.append(
            {
                "system": system,
                "trained_model_count": mean("trained_model_count"),
                "parameters_million": mean("total_stored_parameters") / 1e6,
                "checkpoint_mib": mean("actual_checkpoint_bytes") / (1024**2),
                "training_gpu_hours": mean("training_gpu_hours"),
                "training_gpu_hours_all_seven_datasets": sum(
                    float(row["training_gpu_hours"]) for row in selected
                ),
                "single_request_latency_ms": mean("single_request_latency_ms"),
                "all_horizon_service_latency_ms": mean(
                    "all_horizon_service_latency_ms"
                ),
                "all_horizon_p95_iteration_latency_ms": mean(
                    "all_horizon_p95_iteration_latency_ms"
                ),
                "max_unit_round_cv": max(
                    float(row["all_horizon_round_cv"]) for row in selected
                ),
                "peak_inference_memory_mib": mean(
                    "peak_inference_memory_bytes"
                )
                / (1024**2),
                "incremental_activation_peak_mib": mean(
                    "incremental_activation_peak_bytes"
                )
                / (1024**2),
                "chpc_guarantee": CHPC[system],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "efficiency_35_service_units.csv", unit_rows)
    write_csv(args.output_dir / "efficiency_system_macro_means.csv", aggregate_rows)
    table_dir = args.output_dir / "table"
    table_dir.mkdir(exist_ok=True)
    fragment = latex_table(aggregate_rows)
    (table_dir / "table_iscf_bsca_efficiency.tex").write_text(
        fragment, encoding="utf-8"
    )
    standalone = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[letterpaper,margin=0.45in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
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
        "gate": "efficiency_complete",
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": sha256(args.manifest),
        "service_units": len(unit_rows),
        "checkpoint_objects": len(manifest_rows),
        "test_loader_or_labels_accessed": False,
        "systems": aggregate_rows,
    }
    (args.output_dir / "efficiency_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("efficiency_aggregation=pass units=35 systems=5")


if __name__ == "__main__":
    main()
