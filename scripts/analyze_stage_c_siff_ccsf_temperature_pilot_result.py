#!/usr/bin/env python3
"""Audit the completed CCSF validation pilot and freeze its shared temperature."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_temperature_pilot_v1.json"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_temperature_pilot_retry1_result_20260718/raw"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_ccsf_temperature_pilot_retry1_result_20260718/audit"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def temperature_tag(value: float) -> str:
    return str(value).replace("0.", "").replace(".", "")


def run_dir(root: Path, temperature: float, dataset: str, seed: int) -> Path:
    return root / f"tau{temperature_tag(temperature)}" / dataset / "h720_full" / f"seed{seed}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_path = args.raw_root / "_analysis_seed2021" / "selected_temperature.json"
    cell_path = args.raw_root / "_analysis_seed2021" / "temperature_cell_metrics.csv"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    cells = read_csv(cell_path)
    expected_horizons = set(config["training"]["validation_horizons"])

    run_rows = []
    metric_rows = []
    for temperature in config["temperatures"]:
        for dataset in config["datasets"]:
            directory = run_dir(
                args.raw_root,
                float(temperature),
                dataset,
                int(config["seed"]),
            )
            required = [
                directory / "effective_config.json",
                directory / "training_log.csv",
                directory / "metrics_by_target_horizon.csv",
                directory / "environment.json",
                directory / "initialization_contract.json",
                directory / "model_diagnostics.json",
            ]
            effective = json.loads(required[0].read_text(encoding="utf-8"))
            adapter = effective["adapter"]
            training_text = required[1].read_text(encoding="utf-8").lower()
            metrics = read_csv(required[2])
            horizons = {int(row["target_horizon"]) for row in metrics}
            finite = all(
                math.isfinite(float(row[metric]))
                for row in metrics
                for metric in ("mse", "mae")
            ) and not any(token in training_text for token in (",nan", ",inf", "-inf"))
            protocol = bool(
                adapter["final_evaluation_split"] == "val"
                and adapter["official_test_mode"] is False
                and adapter["checkpoint_policy"] == "best-val"
                and float(adapter["ccsf_calibration_temperature"])
                == float(temperature)
                and all(row["evaluation_split"] == "val" for row in metrics)
                and all(int(row["official_test_mode"]) == 0 for row in metrics)
                and all(row["checkpoint_policy"] == "best-val" for row in metrics)
            )
            complete = bool(
                all(path.is_file() and path.stat().st_size > 0 for path in required)
                and len(metrics) == 4
                and horizons == expected_horizons
            )
            run_rows.append(
                {
                    "temperature": temperature,
                    "dataset": dataset,
                    "complete": complete,
                    "finite": finite,
                    "protocol_pass": protocol,
                    "evaluation_split": adapter["final_evaluation_split"],
                    "official_test_mode": adapter["official_test_mode"],
                    "checkpoint_policy": adapter["checkpoint_policy"],
                }
            )
            for row in metrics:
                metric_rows.append(
                    {
                        "temperature": float(temperature),
                        "dataset": dataset,
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[float(row["temperature"])].append(row)
    comparison_rows = []
    for temperature in config["temperatures"]:
        selected_cells = grouped[float(temperature)]
        comparison_rows.append(
            {
                "temperature": temperature,
                "validation_cells": len(selected_cells),
                "macro_mse": mean([row["mse"] for row in selected_cells]),
                "macro_mae": mean([row["mae"] for row in selected_cells]),
            }
        )
    selected_temperature = float(selected["selected_temperature"])
    selected_summary = next(
        row for row in comparison_rows if float(row["temperature"]) == selected_temperature
    )
    for row in comparison_rows:
        row["selected"] = float(row["temperature"]) == selected_temperature
        row["selected_gain_mse_percent"] = gain(
            float(row["macro_mse"]),
            float(selected_summary["macro_mse"]),
        )

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_key[(row["dataset"], row["target_horizon"])].append(row)
    winner_rows = []
    for (dataset, horizon), rows in sorted(by_key.items()):
        best = min(rows, key=lambda row: row["mse"])
        selected_row = next(
            row for row in rows if row["temperature"] == selected_temperature
        )
        winner_rows.append(
            {
                "dataset": dataset,
                "target_horizon": horizon,
                "best_temperature": best["temperature"],
                "selected_temperature": selected_temperature,
                "selected_is_cell_best": best["temperature"] == selected_temperature,
                "selected_mse": selected_row["mse"],
                "cell_best_mse": best["mse"],
            }
        )

    aggregation_rows = []
    for aggregation, keys in (
        ("dataset", config["datasets"]),
        ("horizon", config["training"]["validation_horizons"]),
    ):
        for key in keys:
            rows_by_temperature = []
            for temperature in config["temperatures"]:
                rows = [
                    row
                    for row in metric_rows
                    if row["temperature"] == float(temperature)
                    and (
                        row["dataset"] == key
                        if aggregation == "dataset"
                        else row["target_horizon"] == key
                    )
                ]
                rows_by_temperature.append(
                    (float(temperature), mean([row["mse"] for row in rows]))
                )
            best_temperature, best_mse = min(
                rows_by_temperature,
                key=lambda item: item[1],
            )
            selected_mse = next(
                value
                for temperature, value in rows_by_temperature
                if temperature == selected_temperature
            )
            aggregation_rows.append(
                {
                    "aggregation": aggregation,
                    "unit": key,
                    "best_temperature": best_temperature,
                    "selected_temperature": selected_temperature,
                    "selected_is_unit_best": best_temperature == selected_temperature,
                    "selected_mse": selected_mse,
                    "unit_best_mse": best_mse,
                }
            )

    checkpoint_hashes = {row["checkpoint_sha256"] for row in cells}
    categories = {
        "runs_complete": len(run_rows) == 15 and all(row["complete"] for row in run_rows),
        "validation_cells_complete": len(metric_rows) == len(cells) == 60,
        "finite_metrics_and_training": all(row["finite"] for row in run_rows),
        "validation_only_protocol": all(row["protocol_pass"] for row in run_rows),
        "checkpoint_hashes_complete": len(checkpoint_hashes) == 15,
        "selection_recomputed": math.isclose(
            float(selected["selection_score_macro_mse"]),
            float(selected_summary["macro_mse"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "selected_temperature_is_macro_best": selected_temperature
        == min(comparison_rows, key=lambda row: row["macro_mse"])["temperature"],
        "pilot_checkpoint_not_reused": selected["pilot_checkpoints_reused"] is False,
        "formal_test_remains_unauthorized": bool(
            selected["test_accessed"] is False
            and selected["formal_phase_a_authorized"] is False
            and selected["formal_test_access_authorized"] is False
        ),
    }
    result = {
        "pilot_id": config["pilot_id"],
        "formal_candidate_version": selected["formal_candidate_version"],
        "selected_temperature": selected_temperature,
        "selected_macro_mse": selected_summary["macro_mse"],
        "selected_macro_mae": selected_summary["macro_mae"],
        "gain_vs_tau005_percent": gain(
            next(row["macro_mse"] for row in comparison_rows if row["temperature"] == 0.05),
            selected_summary["macro_mse"],
        ),
        "gain_vs_tau01_percent": gain(
            next(row["macro_mse"] for row in comparison_rows if row["temperature"] == 0.1),
            selected_summary["macro_mse"],
        ),
        "cell_wins": sum(row["selected_is_cell_best"] for row in winner_rows),
        "dataset_wins": sum(
            row["selected_is_unit_best"]
            for row in aggregation_rows
            if row["aggregation"] == "dataset"
        ),
        "horizon_wins": sum(
            row["selected_is_unit_best"]
            for row in aggregation_rows
            if row["aggregation"] == "horizon"
        ),
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "overall_pass": all(categories.values()),
        "decision": "freeze_tau025_formal_candidate_prelaunch_next"
        if all(categories.values())
        else "pilot_audit_fail_do_not_freeze_candidate",
        "formal_phase_a_authorized": False,
        "formal_test_access_authorized": False,
        "config_sha256": file_hash(args.config),
        "selected_artifact_sha256": file_hash(selected_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "run_protocol_audit.csv", run_rows)
    write_csv(args.output_dir / "temperature_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "cell_winners.csv", winner_rows)
    write_csv(args.output_dir / "aggregation_stability.csv", aggregation_rows)
    (args.output_dir / "pilot_result_gate.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selected_temperature.json").write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not result["overall_pass"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
