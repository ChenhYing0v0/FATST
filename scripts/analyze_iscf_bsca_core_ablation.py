#!/usr/bin/env python3
"""Aggregate the complete 100-cell ISCF-BSCA Core-Ablation scorecard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_core_ablation_protocol.json"),
    )
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm_id: str, dataset: str, seed: int) -> Path:
    if arm_id == "full_iscf_bsca":
        return root / dataset / "h720_full" / f"seed{seed}"
    return root / arm_id / dataset / "h720_full" / f"seed{seed}"


def mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("cannot aggregate empty or non-finite values")
    return sum(values) / len(values)


def main(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    manifest_hashes = {
        (row["arm_id"], row["dataset"], int(row["seed"])): row["checkpoint_sha256"]
        for row in manifest["rows"]
    }
    arms = {arm["id"]: arm for arm in config["arms"]}
    horizons = [int(value) for value in config["matrix"]["horizons"]]
    seed = int(config["seeds"][0])
    cells: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for arm_id in config["table"]["row_order"]:
        arm = arms[arm_id]
        for dataset in config["table"]["dataset_order"]:
            root = args.full_root if arm_id == "full_iscf_bsca" else args.new_root
            directory = run_dir(root, arm_id, dataset, seed)
            metrics_path = directory / "test_audit_metrics_by_target_horizon.csv"
            invariants_path = directory / "test_audit_invariants.json"
            checkpoint_path = directory / "checkpoint.pt"
            if not metrics_path.is_file() or not invariants_path.is_file():
                raise RuntimeError(f"missing formal-test artifacts: {directory}")
            invariants = load_json(invariants_path)
            if invariants.get("pass") is not True or invariants.get("uses_test_split") is not True:
                raise RuntimeError(f"formal-test invariant failure: {directory}")
            checkpoint_hash = sha256(checkpoint_path)
            if invariants.get("checkpoint_sha256") != checkpoint_hash:
                raise RuntimeError(f"test invariant checkpoint hash mismatch: {directory}")
            if arm_id != "full_iscf_bsca":
                expected_hash = manifest_hashes[(arm_id, dataset, seed)]
                if checkpoint_hash != expected_hash:
                    raise RuntimeError(f"checkpoint mutated after manifest freeze: {directory}")
            checkpoint_rows.append(
                {
                    "arm_id": arm_id,
                    "dataset": dataset,
                    "seed": seed,
                    "source": arm["source"],
                    "checkpoint_sha256": checkpoint_hash,
                    "test_access_date": invariants.get("test_access_date"),
                    "test_role": config["authorization"]["test_role"],
                }
            )
            by_horizon = {
                int(row["target_horizon"]): row for row in read_csv(metrics_path)
            }
            for horizon in horizons:
                row = by_horizon[horizon]
                mse, mae = float(row["mse"]), float(row["mae"])
                if not math.isfinite(mse) or not math.isfinite(mae):
                    raise RuntimeError(f"non-finite metric in {metrics_path}")
                cells.append(
                    {
                        "arm_id": arm_id,
                        "table_label": arm["table_label"],
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed": seed,
                        "mse": mse,
                        "mae": mae,
                        "source": arm["source"],
                        "checkpoint_sha256": checkpoint_hash,
                        "test_role": config["authorization"]["test_role"],
                        "test_tuned": False,
                    }
                )

    expected_cells = int(config["matrix"]["effective_test_cells"])
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} cells, found {len(cells)}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["arm_id"], cell["dataset"])].append(cell)
    dataset_rows = []
    for arm_id in config["table"]["row_order"]:
        for dataset in config["table"]["dataset_order"]:
            rows = grouped[(arm_id, dataset)]
            dataset_rows.append(
                {
                    "arm_id": arm_id,
                    "table_label": arms[arm_id]["table_label"],
                    "dataset": dataset,
                    "mean_mse": mean([row["mse"] for row in rows]),
                    "mean_mae": mean([row["mae"] for row in rows]),
                    "horizon_count": len(rows),
                }
            )

    overall_rows = []
    for arm_id in config["table"]["row_order"]:
        rows = [row for row in dataset_rows if row["arm_id"] == arm_id]
        overall_rows.append(
            {
                "arm_id": arm_id,
                "table_label": arms[arm_id]["table_label"],
                "macro_mse": mean([row["mean_mse"] for row in rows]),
                "macro_mae": mean([row["mean_mae"] for row in rows]),
                "dataset_count": len(rows),
            }
        )

    full_overall = next(row for row in overall_rows if row["arm_id"] == "full_iscf_bsca")
    full_cells = [row for row in cells if row["arm_id"] == "full_iscf_bsca"]
    gate_rows = []
    gates = config["gates"]
    for control_id in config["table"]["row_order"][1:]:
        control_overall = next(row for row in overall_rows if row["arm_id"] == control_id)
        control_cells = [row for row in cells if row["arm_id"] == control_id]
        control_map = {(row["dataset"], row["horizon"]): row for row in control_cells}
        dataset_mse_wins = sum(
            next(
                row["mean_mse"]
                for row in dataset_rows
                if row["arm_id"] == "full_iscf_bsca" and row["dataset"] == dataset
            )
            < next(
                row["mean_mse"]
                for row in dataset_rows
                if row["arm_id"] == control_id and row["dataset"] == dataset
            )
            for dataset in config["table"]["dataset_order"]
        )
        horizon_mse_wins = 0
        for horizon in horizons:
            full_value = mean([row["mse"] for row in full_cells if row["horizon"] == horizon])
            control_value = mean([row["mse"] for row in control_cells if row["horizon"] == horizon])
            horizon_mse_wins += full_value < control_value
        cell_mse_wins = sum(
            row["mse"] < control_map[(row["dataset"], row["horizon"])]["mse"]
            for row in full_cells
        )
        cell_mae_wins = sum(
            row["mae"] < control_map[(row["dataset"], row["horizon"])]["mae"]
            for row in full_cells
        )
        mse_gain = 100.0 * (
            control_overall["macro_mse"] - full_overall["macro_mse"]
        ) / control_overall["macro_mse"]
        mae_gain = 100.0 * (
            control_overall["macro_mae"] - full_overall["macro_mae"]
        ) / control_overall["macro_mae"]
        passed = bool(
            mse_gain > gates["per_control_macro_mse_gain_percent_min_exclusive"]
            and mae_gain > gates["per_control_macro_mae_gain_percent_min_exclusive"]
            and dataset_mse_wins >= gates["per_control_dataset_mse_wins_min"]
            and horizon_mse_wins >= gates["per_control_horizon_mse_wins_min"]
        )
        gate_rows.append(
            {
                "control_id": control_id,
                "control_label": arms[control_id]["table_label"],
                "full_vs_control_macro_mse_gain_percent": mse_gain,
                "full_vs_control_macro_mae_gain_percent": mae_gain,
                "dataset_mse_wins": dataset_mse_wins,
                "horizon_mse_wins": horizon_mse_wins,
                "cell_mse_wins": cell_mse_wins,
                "cell_mae_wins": cell_mae_wins,
                "gate_pass": passed,
            }
        )

    all_pass = all(row["gate_pass"] for row in gate_rows)
    decision = (
        "passed_core_candidate_matched_attribution"
        if all_pass
        else "performance_partial_pass_or_failed_control; limit claims to passing controls"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "core_ablation_100_cells.csv", cells)
    write_csv(args.output_dir / "core_ablation_dataset_means.csv", dataset_rows)
    write_csv(args.output_dir / "core_ablation_overall_means.csv", overall_rows)
    write_csv(args.output_dir / "core_ablation_control_gates.csv", gate_rows)
    write_csv(args.output_dir / "core_ablation_checkpoint_manifest.csv", checkpoint_rows)
    summary = {
        "candidate_version": config["candidate_version"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol_sha256": sha256(args.config),
        "training_manifest_sha256": sha256(args.manifest),
        "matrix_complete": len(cells) == expected_cells,
        "cell_count": len(cells),
        "checkpoint_count": len(checkpoint_rows),
        "all_four_controls_pass": all_pass,
        "decision": decision,
        "gate_rows": gate_rows,
    }
    (args.output_dir / "core_ablation_result_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"core_ablation_analysis=pass cells={len(cells)} "
        f"all_controls_pass={str(all_pass).lower()} decision={decision}"
    )


if __name__ == "__main__":
    main(parse_args())
