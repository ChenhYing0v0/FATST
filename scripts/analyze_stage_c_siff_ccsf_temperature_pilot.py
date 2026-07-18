#!/usr/bin/env python3
"""Select the frozen shared CCSF temperature from validation-only runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_temperature_pilot_v1.json"),
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temperature_tag(value: float) -> str:
    return str(value).replace("0.", "").replace(".", "")


def run_dir(root: Path, temperature: float, dataset: str, seed: int) -> Path:
    return (
        root
        / f"tau{temperature_tag(temperature)}"
        / dataset
        / "h720_full"
        / f"seed{seed}"
    )


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
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    scores = {0.05: 1.0, 0.1: 0.9, 0.25: 0.9}
    rows = []
    for temperature in config["temperatures"]:
        for dataset in config["datasets"]:
            for horizon in config["training"]["validation_horizons"]:
                rows.append(
                    {
                        "temperature": temperature,
                        "dataset": dataset,
                        "target_horizon": horizon,
                        "mse": scores[float(temperature)],
                        "mae": scores[float(temperature)] + 0.1,
                        "evaluation_split": "val",
                        "checkpoint_policy": "best-val",
                        "checkpoint_sha256": "synthetic",
                        "metrics_path": "synthetic",
                    }
                )
    return rows


def load_cells(
    config: dict[str, Any],
    raw_root: Path,
) -> list[dict[str, Any]]:
    expected_horizons = set(config["training"]["validation_horizons"])
    rows = []
    for temperature in config["temperatures"]:
        for dataset in config["datasets"]:
            directory = run_dir(
                raw_root,
                float(temperature),
                dataset,
                int(config["seed"]),
            )
            metrics_path = directory / "metrics_by_target_horizon.csv"
            checkpoint_path = directory / "checkpoint.pt"
            if not metrics_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"incomplete pilot run: {directory}"
                )
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                metrics = list(csv.DictReader(handle))
            horizons = {int(row["target_horizon"]) for row in metrics}
            if horizons != expected_horizons or len(metrics) != len(expected_horizons):
                raise ValueError(
                    f"invalid validation horizon matrix: {metrics_path}"
                )
            checkpoint_sha256 = file_hash(checkpoint_path)
            for metric in metrics:
                split = metric.get("evaluation_split", "val")
                checkpoint_policy = metric.get(
                    "checkpoint_policy",
                    "best-val",
                )
                if split != "val" or checkpoint_policy != "best-val":
                    raise ValueError(
                        f"pilot must use validation/best-val only: {metrics_path}"
                    )
                mse = float(metric["mse"])
                mae = float(metric["mae"])
                if not math.isfinite(mse) or not math.isfinite(mae):
                    raise ValueError(f"non-finite pilot metric: {metrics_path}")
                rows.append(
                    {
                        "temperature": float(temperature),
                        "dataset": dataset,
                        "target_horizon": int(metric["target_horizon"]),
                        "mse": mse,
                        "mae": mae,
                        "evaluation_split": split,
                        "checkpoint_policy": checkpoint_policy,
                        "checkpoint_sha256": checkpoint_sha256,
                        "metrics_path": str(metrics_path),
                    }
                )
    return rows


def summarize(
    cells: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    expected_cells = len(config["datasets"]) * len(
        config["training"]["validation_horizons"]
    )
    for temperature in config["temperatures"]:
        selected = [
            row
            for row in cells
            if math.isclose(
                float(row["temperature"]),
                float(temperature),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(selected) != expected_cells:
            raise ValueError(
                f"temperature {temperature} has {len(selected)} cells, "
                f"expected {expected_cells}"
            )
        rows.append(
            {
                "temperature": float(temperature),
                "validation_cells": len(selected),
                "macro_mse": sum(float(row["mse"]) for row in selected)
                / len(selected),
                "macro_mae": sum(float(row["mae"]) for row in selected)
                / len(selected),
                "dataset_count": len({row["dataset"] for row in selected}),
                "horizon_count": len(
                    {int(row["target_horizon"]) for row in selected}
                ),
            }
        )
    return rows


def select_temperature(
    summary: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    best_mse = min(float(row["macro_mse"]) for row in summary)
    tied = [
        row
        for row in summary
        if float(row["macro_mse"]) <= best_mse + tolerance
    ]
    return max(tied, key=lambda row: float(row["temperature"]))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        cells = synthetic_cells(config)
    else:
        if args.raw_root is None:
            raise ValueError("--raw-root is required outside synthetic smoke")
        cells = load_cells(config, args.raw_root)
    expected_cells = int(config["matrix"]["expected_validation_cells"])
    if len(cells) != expected_cells:
        raise ValueError(
            f"pilot has {len(cells)} cells, expected {expected_cells}"
        )
    summary = summarize(cells, config)
    selected = select_temperature(
        summary,
        float(config["selection"]["tie_tolerance"]),
    )
    temperature = float(selected["temperature"])
    tag = temperature_tag(temperature)
    payload = {
        "pilot_id": config["pilot_id"],
        "pilot_complete": True,
        "validation_only": True,
        "test_accessed": False,
        "selected_temperature": temperature,
        "selection_score_macro_mse": float(selected["macro_mse"]),
        "secondary_macro_mae": float(selected["macro_mae"]),
        "tie_break": config["selection"]["tie_break"],
        "formal_candidate_version": config["selection"][
            "formal_candidate_version_template"
        ].format(temperature_tag=tag),
        "pilot_checkpoints_reused": False,
        "formal_phase_a_authorized": False,
        "formal_test_access_authorized": False,
        "config_sha256": file_hash(args.config),
        "profile_sha256": config["profiles"]["sha256"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "temperature_cell_metrics.csv", cells)
    write_csv(args.output_dir / "temperature_summary.csv", summary)
    (args.output_dir / "selected_temperature.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
