#!/usr/bin/env python3
"""Audit and aggregate the complete frozen Main II H720-prefix matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SYSTEMS = (
    "ISCF-BSCA-MAIN-v1",
    "TimeAlign",
    "QDF",
    "AMD",
    "SimpleTM",
    "iTransformer",
    "PatchTST",
    "DLinear",
)
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
EXACT_ANCHORS = {
    "ISCF-BSCA-MAIN-v1",
    "TimeAlign",
    "QDF",
    "AMD",
    "SimpleTM",
}
MAIN_I_NAMES = {"ISCF-BSCA-MAIN-v1": "ISCF-BSCA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_tolerance(reference: float, absolute_floor: float) -> float:
    """Allow the frozen floor or four float32 ULPs, whichever is larger."""
    return max(
        absolute_floor,
        4.0 * abs(float(np.spacing(np.float32(reference)))),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--main-i-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def metric_paths(root: Path) -> list[Path]:
    paths = [
        root
        / "formal_test_reused"
        / "ISCF-BSCA-MAIN-v1"
        / "prefix_metrics.csv"
    ]
    paths.extend(
        sorted(
            path
            for path in (root / "formal_test_reused").glob("*/prefix_metrics.csv")
            if path.parent.name != "ISCF-BSCA-MAIN-v1"
        )
    )
    paths.extend(sorted((root / "formal_test_new").glob("*/prefix/prefix_metrics.csv")))
    return paths


def main() -> None:
    args = parse_args()
    paths = metric_paths(args.results_root)
    if len(paths) != 64:
        raise RuntimeError(f"expected 64 metric files, found {len(paths)}")
    raw_rows: list[dict[str, object]] = []
    input_hashes = []
    for path in paths:
        rows = read_csv(path)
        for row in rows:
            row["horizon"] = int(row["horizon"])
            row["repeat"] = int(row["repeat"])
            row["mse"] = float(row["mse"])
            row["mae"] = float(row["mae"])
            row["source_metric_file"] = str(path)
            raw_rows.append(row)
        input_hashes.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
    if len(raw_rows) != 280:
        raise RuntimeError(f"expected 280 raw prefix rows, found {len(raw_rows)}")

    by_checkpoint: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["system"]),
            str(row["dataset"]),
            int(row["repeat"]),
            str(row["checkpoint_sha256"]),
        )
        by_checkpoint[key].append(row)
    if len(by_checkpoint) != 70:
        raise RuntimeError(f"expected 70 checkpoint evaluations, found {len(by_checkpoint)}")
    for key, rows in by_checkpoint.items():
        if {int(row["horizon"]) for row in rows} != set(HORIZONS):
            raise RuntimeError(f"incomplete checkpoint prefix surface: {key}")
        if not all(str(row["prefix_identity"]).lower() == "true" for row in rows):
            raise RuntimeError(f"prefix identity failure: {key}")

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["system"]), str(row["dataset"]), int(row["horizon"]))].append(row)
    expected_keys = {
        (system, dataset, horizon)
        for system in SYSTEMS
        for dataset in DATASETS
        for horizon in HORIZONS
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise RuntimeError(f"aggregate key mismatch: missing={missing} extra={extra}")

    aggregate_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                rows = grouped[(system, dataset, horizon)]
                expected_repeats = 3 if system == "SimpleTM" else 1
                if len(rows) != expected_repeats:
                    raise RuntimeError(
                        f"repeat count mismatch: {system}/{dataset}/H{horizon}"
                    )
                aggregate_rows.append(
                    {
                        "system": system,
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": sum(float(row["mse"]) for row in rows) / len(rows),
                        "mae": sum(float(row["mae"]) for row in rows) / len(rows),
                        "checkpoint_repetitions": len(rows),
                        "prefix_identity": True,
                        "matrix_complete": True,
                        "system_role": "source_native_H720_one_model_all_horizons",
                    }
                )
    if len(aggregate_rows) != 224:
        raise RuntimeError("aggregate matrix is not 224 cells")

    main_i = {
        (row["model"], row["dataset"], int(row["horizon"])): row
        for row in read_csv(args.main_i_table)
        if row["horizon"] in {str(horizon) for horizon in HORIZONS}
    }
    continuity_rows: list[dict[str, object]] = []
    exact_failures = []
    for row in aggregate_rows:
        if int(row["horizon"]) != 720:
            continue
        system = str(row["system"])
        main_i_name = MAIN_I_NAMES.get(system, system)
        anchor = main_i[(main_i_name, str(row["dataset"]), 720)]
        mse_delta = float(row["mse"]) - float(anchor["mse"])
        mae_delta = float(row["mae"]) - float(anchor["mae"])
        exact_required = system in EXACT_ANCHORS
        mse_tolerance = numeric_tolerance(
            float(anchor["mse"]), args.exact_tolerance
        )
        mae_tolerance = numeric_tolerance(
            float(anchor["mae"]), args.exact_tolerance
        )
        passed = (
            abs(mse_delta) <= mse_tolerance
            and abs(mae_delta) <= mae_tolerance
            if exact_required
            else True
        )
        continuity = {
            "system": system,
            "dataset": row["dataset"],
            "main_ii_mse": row["mse"],
            "main_i_h720_mse": float(anchor["mse"]),
            "mse_delta": mse_delta,
            "main_ii_mae": row["mae"],
            "main_i_h720_mae": float(anchor["mae"]),
            "mae_delta": mae_delta,
            "mse_tolerance": mse_tolerance,
            "mae_tolerance": mae_tolerance,
            "numeric_tolerance_rule": "max(1e-8, four_float32_ULPs_of_anchor)",
            "exact_required": exact_required,
            "pass": passed,
            "reference_role": (
                "exact_local_anchor" if exact_required else "published_three_run_mean"
            ),
        }
        continuity_rows.append(continuity)
        if exact_required and not passed:
            exact_failures.append(continuity)
    if exact_failures:
        raise RuntimeError(f"exact H720 continuity failures: {exact_failures}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_checkpoint_prefix_metrics.csv", raw_rows)
    write_csv(args.output_dir / "main_ii_aggregate_cells.csv", aggregate_rows)
    write_csv(args.output_dir / "h720_main_i_continuity.csv", continuity_rows)
    summary = {
        "gate": "pass",
        "checkpoint_evaluations": len(by_checkpoint),
        "raw_prefix_rows": len(raw_rows),
        "aggregate_cells": len(aggregate_rows),
        "metric_scalars": len(aggregate_rows) * 2,
        "exact_h720_anchors": sum(row["exact_required"] for row in continuity_rows),
        "exact_h720_anchor_failures": 0,
        "published_reference_deviations": sum(
            not row["exact_required"] for row in continuity_rows
        ),
        "matrix_complete": True,
        "negative_results_retained": True,
        "input_files": input_hashes,
    }
    (args.output_dir / "main_ii_result_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_result_audit=pass checkpoints=70 raw_rows=280 "
        "aggregate_cells=224 exact_h720_anchors=35"
    )


if __name__ == "__main__":
    main()
