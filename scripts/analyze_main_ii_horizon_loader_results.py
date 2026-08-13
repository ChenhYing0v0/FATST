#!/usr/bin/env python3
"""Audit and aggregate Main II H720 checkpoints on fixed-H official loaders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ISCF_SYSTEM = "ISCF-BSCA-MAIN-v1"
EXTERNAL_SYSTEMS = (
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
EXPECTED_DROP_LAST = {
    "TimeAlign": False,
    "QDF": True,
    "AMD": False,
    "SimpleTM": True,
    "iTransformer": True,
    "PatchTST": True,
    "DLinear": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def numeric_tolerance(reference: float, floor: float) -> float:
    return max(floor, 4.0 * abs(float(np.spacing(np.float32(reference)))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--iscf-scorecard", type=Path, required=True)
    parser.add_argument("--previous-main-ii-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.job_manifest.read_text(encoding="utf-8"))
    if manifest.get("gate") != "pass" or manifest.get("formal_evaluations") != 252:
        raise RuntimeError("formal job manifest gate has not passed")

    metric_paths = sorted((args.results_root / "formal").glob("*/prefix_metrics.csv"))
    if len(metric_paths) != 252:
        raise RuntimeError(f"expected 252 metric files, found {len(metric_paths)}")

    raw_rows: list[dict[str, object]] = []
    input_files: list[dict[str, object]] = []
    for path in metric_paths:
        rows = read_csv(path)
        if len(rows) != 1:
            raise RuntimeError(f"expected one metric row: {path}")
        row: dict[str, object] = dict(rows[0])
        row.update(
            {
                "repeat": int(row["repeat"]),
                "horizon": int(row["horizon"]),
                "loader_horizon": int(row["loader_horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "origin_count": int(row["origin_count"]),
                "loader_drop_last": as_bool(row["loader_drop_last"]),
                "input_only_inference": as_bool(row["input_only_inference"]),
                "source_metric_file": str(path),
            }
        )
        system = str(row["system"])
        if system not in EXTERNAL_SYSTEMS:
            raise RuntimeError(f"unexpected external system: {system}")
        if int(row["horizon"]) != int(row["loader_horizon"]):
            raise RuntimeError(f"loader/requested horizon mismatch: {path}")
        if bool(row["loader_drop_last"]) != EXPECTED_DROP_LAST[system]:
            raise RuntimeError(f"drop_last mismatch: {path}")
        if not bool(row["input_only_inference"]):
            raise RuntimeError(f"input-only inference gate failed: {path}")
        if not math.isfinite(float(row["mse"])) or not math.isfinite(float(row["mae"])):
            raise RuntimeError(f"non-finite metric: {path}")
        raw_rows.append(row)
        input_files.append({"path": str(path), "sha256": sha256(path), "rows": 1})

    expected_raw_keys = {
        (system, dataset, repeat, horizon)
        for system in EXTERNAL_SYSTEMS
        for dataset in DATASETS
        for repeat in range(3 if system == "SimpleTM" else 1)
        for horizon in HORIZONS
    }
    observed_raw_keys = {
        (str(row["system"]), str(row["dataset"]), int(row["repeat"]), int(row["horizon"]))
        for row in raw_rows
    }
    if observed_raw_keys != expected_raw_keys:
        raise RuntimeError(
            "formal identity mismatch: "
            f"missing={sorted(expected_raw_keys - observed_raw_keys)} "
            f"extra={sorted(observed_raw_keys - expected_raw_keys)}"
        )

    by_checkpoint: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["system"]),
            str(row["dataset"]),
            int(row["repeat"]),
            str(row["checkpoint_sha256"]),
        )
        by_checkpoint[key].append(row)
    if len(by_checkpoint) != 63 or len({key[3] for key in by_checkpoint}) != 63:
        raise RuntimeError("expected 63 unique checkpoint objects")

    origin_rows: list[dict[str, object]] = []
    for key, rows in sorted(by_checkpoint.items()):
        by_horizon = {int(row["horizon"]): row for row in rows}
        if set(by_horizon) != set(HORIZONS):
            raise RuntimeError(f"incomplete checkpoint horizon surface: {key}")
        counts = [int(by_horizon[horizon]["origin_count"]) for horizon in HORIZONS]
        monotonic = all(left >= right for left, right in zip(counts, counts[1:]))
        if not monotonic:
            raise RuntimeError(f"origin-count monotonicity failed: {key} {counts}")
        for horizon, count in zip(HORIZONS, counts):
            origin_rows.append(
                {
                    "system": key[0],
                    "dataset": key[1],
                    "repeat": key[2],
                    "checkpoint_sha256": key[3],
                    "horizon": horizon,
                    "origin_count": count,
                    "drop_last": EXPECTED_DROP_LAST[key[0]],
                    "non_increasing_as_h_grows": monotonic,
                }
            )

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["system"]), str(row["dataset"]), int(row["horizon"]))].append(row)
    external_rows: list[dict[str, object]] = []
    for system in EXTERNAL_SYSTEMS:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                rows = grouped[(system, dataset, horizon)]
                expected_repeats = 3 if system == "SimpleTM" else 1
                if len(rows) != expected_repeats:
                    raise RuntimeError(f"repeat count mismatch: {system}/{dataset}/H{horizon}")
                external_rows.append(
                    {
                        "system": system,
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse": sum(float(row["mse"]) for row in rows) / len(rows),
                        "mae": sum(float(row["mae"]) for row in rows) / len(rows),
                        "checkpoint_repetitions": len(rows),
                        "prefix_identity": False,
                        "matrix_complete": True,
                        "system_role": "source_native_H720_model_on_official_fixed_H_loader",
                    }
                )
    if len(external_rows) != 196:
        raise RuntimeError("external aggregate matrix is not 196 cells")

    iscf_rows = []
    for row in read_csv(args.iscf_scorecard):
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        if dataset not in DATASETS or horizon not in HORIZONS:
            continue
        iscf_rows.append(
            {
                "system": ISCF_SYSTEM,
                "dataset": dataset,
                "horizon": horizon,
                "mse": float(row["test_mse"]),
                "mae": float(row["test_mae"]),
                "checkpoint_repetitions": 1,
                "prefix_identity": True,
                "matrix_complete": True,
                "system_role": "frozen_current_ISCF_fixed_H_formal_scorecard",
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )
    if len(iscf_rows) != 28:
        raise RuntimeError(f"expected 28 current ISCF cells, found {len(iscf_rows)}")
    final_rows = sorted(
        [*iscf_rows, *external_rows],
        key=lambda row: (str(row["system"]), str(row["dataset"]), int(row["horizon"])),
    )
    if len(final_rows) != 224:
        raise RuntimeError("final Main II matrix is not 224 cells")

    previous = {
        (row["system"], row["dataset"], int(row["horizon"])): row
        for row in read_csv(args.previous_main_ii_cells)
    }
    continuity_rows: list[dict[str, object]] = []
    failures = []
    for row in external_rows:
        if int(row["horizon"]) != 720:
            continue
        key = (str(row["system"]), str(row["dataset"]), 720)
        anchor = previous[key]
        mse_delta = float(row["mse"]) - float(anchor["mse"])
        mae_delta = float(row["mae"]) - float(anchor["mae"])
        mse_tolerance = numeric_tolerance(float(anchor["mse"]), args.exact_tolerance)
        mae_tolerance = numeric_tolerance(float(anchor["mae"]), args.exact_tolerance)
        passed = abs(mse_delta) <= mse_tolerance and abs(mae_delta) <= mae_tolerance
        audit = {
            "system": key[0],
            "dataset": key[1],
            "new_h720_mse": row["mse"],
            "previous_same_checkpoint_h720_mse": float(anchor["mse"]),
            "mse_delta": mse_delta,
            "mse_tolerance": mse_tolerance,
            "new_h720_mae": row["mae"],
            "previous_same_checkpoint_h720_mae": float(anchor["mae"]),
            "mae_delta": mae_delta,
            "mae_tolerance": mae_tolerance,
            "pass": passed,
        }
        continuity_rows.append(audit)
        if not passed:
            failures.append(audit)
    if failures:
        raise RuntimeError(f"H720 continuity failures: {failures}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_external_horizon_loader_metrics.csv", raw_rows)
    write_csv(args.output_dir / "origin_count_audit.csv", origin_rows)
    write_csv(args.output_dir / "external_aggregate_cells.csv", external_rows)
    write_csv(args.output_dir / "main_ii_aggregate_cells.csv", final_rows)
    write_csv(args.output_dir / "h720_same_checkpoint_continuity.csv", continuity_rows)
    summary = {
        "gate": "pass",
        "protocol": "horizon_specific_loader_H720_checkpoint",
        "checkpoint_objects": len(by_checkpoint),
        "unique_checkpoint_hashes": len({key[3] for key in by_checkpoint}),
        "formal_external_evaluations": len(raw_rows),
        "external_aggregate_cells": len(external_rows),
        "iscf_reused_cells": len(iscf_rows),
        "aggregate_cells": len(final_rows),
        "metric_scalars": len(final_rows) * 2,
        "h720_continuity_checks": len(continuity_rows),
        "h720_continuity_failures": 0,
        "origin_monotonicity_failures": 0,
        "matrix_complete": True,
        "negative_results_retained": True,
        "job_manifest_sha256": sha256(args.job_manifest),
        "iscf_scorecard_sha256": sha256(args.iscf_scorecard),
        "previous_main_ii_cells_sha256": sha256(args.previous_main_ii_cells),
        "input_files": input_files,
    }
    (args.output_dir / "main_ii_result_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_horizon_loader_audit=pass checkpoints=63 evaluations=252 "
        "external_cells=196 final_cells=224 h720_continuity=49"
    )


if __name__ == "__main__":
    main()
