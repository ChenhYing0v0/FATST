#!/usr/bin/env python3
"""Audit ISCF-BSCA-v1 validation and frozen official-test artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = ("Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2")
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage_c_iscf_bsca_v1.json"))
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path("analysis/stage_c_siff_equal_attribution_step9_20260718/raw/siff_independent_equal"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, dataset: str, candidate: bool) -> Path:
    prefix = root / "iscf_bsca_v1" if candidate else root
    return prefix / dataset / "h720_full" / "seed2021"


def selected_metrics(directory: Path, split: str) -> dict[int, dict[str, str]]:
    name = "metrics_by_target_horizon.csv" if split == "val" else "test_audit_metrics_by_target_horizon.csv"
    rows = read_rows(directory / name)
    selected = {int(row["target_horizon"]): row for row in rows if int(row["target_horizon"]) in HORIZONS}
    if tuple(sorted(selected)) != HORIZONS:
        raise ValueError(f"incomplete {split} horizons in {directory}")
    return selected


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate = run_dir(args.candidate_root, dataset, True)
        reference = run_dir(args.reference_root, dataset, False)
        required_candidate = (
            "checkpoint.pt", "training_log.csv", "metrics_by_target_horizon.csv",
            "effective_config.json", "initialization_contract.json", "model_diagnostics.json",
            "pcsd_validation_diagnostics.npz", "trained_invariants.json",
            "test_audit_metrics_by_target_horizon.csv", "test_audit_invariants.json",
            "pcsd_test_audit_diagnostics.npz",
        )
        missing = [name for name in required_candidate if not (candidate / name).is_file()]
        invariant = json.loads((candidate / "test_audit_invariants.json").read_text(encoding="utf-8"))
        effective = json.loads((candidate / "effective_config.json").read_text(encoding="utf-8"))
        audit.append({
            "dataset": dataset,
            "missing_count": len(missing),
            "checkpoint_sha256": sha256(candidate / "checkpoint.pt"),
            "objective": effective["adapter"]["pcc_objective_mode"],
            "test_split": invariant.get("evaluation_split"),
            "uses_test_split": invariant.get("uses_test_split"),
            "test_invariant_pass": invariant.get("pass"),
            "checkpoint_retrained_for_candidate": invariant.get("checkpoint_retrained"),
        })
        for split in ("val", "test"):
            c_metrics = selected_metrics(candidate, split)
            r_metrics = selected_metrics(reference, split)
            for horizon in HORIZONS:
                c_row, r_row = c_metrics[horizon], r_metrics[horizon]
                cells.append({
                    "split": split, "dataset": dataset, "horizon": horizon,
                    "candidate_mse": float(c_row["mse"]), "reference_mse": float(r_row["mse"]),
                    "mse_gain_percent": gain(float(r_row["mse"]), float(c_row["mse"])),
                    "candidate_mae": float(c_row["mae"]), "reference_mae": float(r_row["mae"]),
                    "mae_gain_percent": gain(float(r_row["mae"]), float(c_row["mae"])),
                })

    test = [row for row in cells if row["split"] == "test"]
    macro_mse = float(np.mean([row["mse_gain_percent"] for row in test]))
    macro_mae = float(np.mean([row["mae_gain_percent"] for row in test]))
    dataset_wins = int(
        sum(
            bool(
                np.mean(
                    [
                        row["mse_gain_percent"]
                        for row in test
                        if row["dataset"] == dataset
                    ]
                )
                > 0
            )
            for dataset in DATASETS
        )
    )
    horizon_wins = int(
        sum(
            bool(
                np.mean(
                    [
                        row["mse_gain_percent"]
                        for row in test
                        if row["horizon"] == horizon
                    ]
                )
                > 0
            )
            for horizon in HORIZONS
        )
    )
    gates = config["gates"]
    artifacts_pass = all(
        row["missing_count"] == 0
        and row["test_invariant_pass"]
        and row["checkpoint_retrained_for_candidate"]
        for row in audit
    )
    performance_pass = (
        macro_mse >= gates["macro_mse_gain_percent_min"]
        and macro_mae > gates["macro_mae_gain_percent_min_exclusive"]
        and dataset_wins >= gates["dataset_mse_wins_min"]
        and horizon_wins >= gates["horizon_mse_wins_min"]
    )
    summary = {
        "candidate_version": config["candidate_version"],
        "test_informed": True,
        "matrix_complete": len(test) == 20 and len(audit) == 5,
        "macro_test_mse_gain_percent": macro_mse,
        "macro_test_mae_gain_percent": macro_mae,
        "test_cell_wins": int(sum(row["mse_gain_percent"] > 0 for row in test)),
        "dataset_mse_wins": int(dataset_wins),
        "horizon_mse_wins": int(horizon_wins),
        "artifact_and_nonmutation_pass": artifacts_pass,
        "performance_gate_pass": bool(performance_pass),
        "decision": "performance_partial_pass_pending_confirmation_seed" if performance_pass and artifacts_pass else "exact_bsca_v1_not_supported_or_invalid",
    }
    write_rows(args.output_dir / "run_audit.csv", audit)
    write_rows(args.output_dir / "comparison_cells.csv", cells)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
