#!/usr/bin/env python3
"""Audit and aggregate the official AMD and SimpleTM Main I reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = config["evaluation_contract"]["datasets"]
    horizons = config["evaluation_contract"]["horizons"]
    expected_simpletm = config["baselines"]["SimpleTM"][
        "native_repeats_by_dataset_horizon"
    ]

    raw_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    for baseline in ("AMD", "SimpleTM"):
        for dataset in datasets:
            unit = args.output_root / "runs" / f"{baseline}__{dataset}"
            completion_path = unit / "complete.json"
            metrics_path = unit / "metrics.csv"
            log_path = unit / "run.log"
            if not completion_path.is_file() or not metrics_path.is_file() or not log_path.is_file():
                raise FileNotFoundError(f"incomplete formal unit: {baseline}:{dataset}")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if not completion.get("complete") or completion["formal_metric_rows"] <= 0:
                raise RuntimeError(f"invalid completion record: {completion_path}")
            rows = read_csv(metrics_path)
            raw_rows.extend(rows)
            for row in rows:
                checkpoint = Path(row["checkpoint"])
                actual = sha256_file(checkpoint)
                if actual != row["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
                if actual in checkpoint_hashes:
                    raise RuntimeError(f"duplicate checkpoint hash: {actual}")
                checkpoint_hashes.add(actual)
                artifact_rows.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "horizon": int(row["horizon"]),
                        "repeat": int(row["repeat"]),
                        "role": "checkpoint",
                        "path": str(checkpoint),
                        "sha256": actual,
                    }
                )
            for role, path in (
                ("metrics", metrics_path),
                ("completion", completion_path),
                ("log", log_path),
            ):
                artifact_rows.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "horizon": "all",
                        "repeat": "all",
                        "role": role,
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                )

    amd_rows = [row for row in raw_rows if row["baseline"] == "AMD"]
    simpletm_rows = [row for row in raw_rows if row["baseline"] == "SimpleTM"]
    if len(amd_rows) != 28 or len(simpletm_rows) != 82:
        raise RuntimeError(
            f"metric row count mismatch: AMD={len(amd_rows)} SimpleTM={len(simpletm_rows)}"
        )

    cell_rows: list[dict[str, Any]] = []
    for baseline, rows in (("AMD", amd_rows), ("SimpleTM", simpletm_rows)):
        for dataset in datasets:
            for horizon in horizons:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and int(row["horizon"]) == horizon
                ]
                expected = 1 if baseline == "AMD" else expected_simpletm[dataset][str(horizon)]
                if len(selected) != expected:
                    raise RuntimeError(
                        f"{baseline} {dataset} H{horizon}: expected {expected} repeats, "
                        f"found {len(selected)}"
                    )
                cell_rows.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "horizon": horizon,
                        "repeat_count": len(selected),
                        "mse": mean(float(row["mse"]) for row in selected),
                        "mae": mean(float(row["mae"]) for row in selected),
                        "source_role": "local_official_native_reproduction",
                    }
                )
    if len(cell_rows) != 56:
        raise RuntimeError(f"expected 56 table cells, found {len(cell_rows)}")

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_dir / "cell_metrics.csv", cell_rows)
    write_csv(args.audit_dir / "artifact_manifest.csv", artifact_rows)
    summary = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "matrix_complete": True,
        "table_cells": len(cell_rows),
        "raw_metric_rows": len(raw_rows),
        "AMD_metric_rows": len(amd_rows),
        "SimpleTM_metric_rows": len(simpletm_rows),
        "unique_checkpoint_hashes": len(checkpoint_hashes),
        "macro": {
            baseline: {
                "mse": mean(
                    float(row["mse"])
                    for row in cell_rows
                    if row["baseline"] == baseline
                ),
                "mae": mean(
                    float(row["mae"])
                    for row in cell_rows
                    if row["baseline"] == baseline
                ),
            }
            for baseline in ("AMD", "SimpleTM")
        },
    }
    (args.audit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
