#!/usr/bin/env python3
"""Analyze StageC SC0-DAP-R2 Phase A validation-only patch screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_dataset_profile_calibration_r2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, dataset: str, profile: str, seed: int) -> Path:
    return root / f"SC0DAP_R2A_{profile}" / dataset / "h720_full" / f"seed{seed}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def collect(
    raw_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    diagnostics = []
    seed = int(config["common"]["screen_seed"])
    expected_hash = config_hash(config_path)
    profiles = config["phase_a_patch_screen"]["profiles"]
    for dataset in config["datasets"]:
        for profile, specification in profiles.items():
            directory = run_dir(raw_root, dataset, profile, seed)
            required = (
                directory / "effective_config.json",
                directory / "model_diagnostics.json",
                directory / "training_log.csv",
                directory / "metrics_by_target_horizon.csv",
            )
            if not all(path.exists() for path in required):
                diagnostics.append(
                    {"dataset": dataset, "profile": profile, "status": "missing", "run_dir": str(directory)}
                )
                continue
            effective = json.loads(required[0].read_text(encoding="utf-8"))
            model_info = json.loads(required[1].read_text(encoding="utf-8"))
            training = read_csv(required[2])
            adapter = effective["adapter"]
            official = effective["official_args"]
            config_ok = (
                adapter.get("protocol_profile") == config["protocol_profile"]
                and adapter.get("profile_hash") == expected_hash
                and adapter.get("final_evaluation_split") == "val"
                and int(official["patch_num"]) == int(specification["patch_num"])
                and int(official["d_model"]) == int(specification["d_model"])
                and int(official["d_ff"]) == int(specification["d_ff"])
            )
            training_ok = bool(training) and all(
                finite(row.get(field))
                for row in training
                for field in ("train_loss", "val_mean_mse", "lr")
            )
            diagnostics.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "status": "ok" if config_ok and training_ok else "mismatch",
                    "config_ok": int(config_ok),
                    "training_ok": int(training_ok),
                    "patch_num": official["patch_num"],
                    "d_model": official["d_model"],
                    "d_ff": official["d_ff"],
                    "active_forward_parameters": model_info["active_forward_parameters"],
                    "trained_epochs": len(training),
                    "best_epoch": training[-1].get("best_epoch_so_far", ""),
                    "run_dir": str(directory),
                }
            )
            for row in read_csv(required[3]):
                if row.get("evaluation_split") != "val":
                    raise ValueError(f"non-validation metric found in {directory}")
                metrics.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return metrics, diagnostics


def select(
    metrics: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    winners = {}
    profiles = list(config["phase_a_patch_screen"]["profiles"])
    horizons = [int(value) for value in config["common"]["evaluation_horizons"]]
    for dataset in config["datasets"]:
        values = {
            profile: {
                horizon: next(
                    float(row["mse"])
                    for row in metrics
                    if row["dataset"] == dataset
                    and row["profile"] == profile
                    and int(row["target_horizon"]) == horizon
                )
                for horizon in horizons
            }
            for profile in profiles
        }
        candidate_rows = []
        for profile in profiles:
            regrets = {
                horizon: values[profile][horizon]
                / min(values[other][horizon] for other in profiles)
                - 1.0
                for horizon in horizons
            }
            candidate_rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "macro_dense_regret": mean(regrets.values()),
                    "max_dense_regret": max(regrets.values()),
                    "h720_regret": regrets[720],
                    "h720_val_mse": values[profile][720],
                    **{f"h{horizon}_regret": regrets[horizon] for horizon in horizons},
                }
            )
        candidate_rows.sort(
            key=lambda row: (
                float(row["macro_dense_regret"]),
                float(row["max_dense_regret"]),
                float(row["h720_regret"]),
                str(row["profile"]),
            )
        )
        winner = str(candidate_rows[0]["profile"])
        winners[dataset] = winner
        for row in candidate_rows:
            row["selected"] = int(row["profile"] == winner)
        rows.extend(candidate_rows)
    return rows, winners


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics, diagnostics = collect(args.raw_root, args.config, config)
    expected_runs = len(config["datasets"]) * len(config["phase_a_patch_screen"]["profiles"])
    complete = (
        len(diagnostics) == expected_runs
        and all(row["status"] == "ok" for row in diagnostics)
        and len(metrics) == expected_runs * len(config["common"]["evaluation_horizons"])
    )
    selections, winners = select(metrics, config) if complete else ([], {})
    summary = {
        "candidate": "SC0-DAP-R2A",
        "complete": complete,
        "decision": "phase_a_patch_selected" if complete else "analysis_incomplete",
        "expected_runs": expected_runs,
        "profile_hash": config_hash(args.config),
        "selected_profiles": winners,
        "parameter_count_used_for_selection": False,
        "test_metrics_used_for_selection": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "r2a_validation_horizon_metrics.csv", metrics)
    write_csv(args.output_dir / "r2a_run_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "r2a_patch_selection.csv", selections)
    (args.output_dir / "r2a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"stage_c_dap_r2a_analysis_done complete={complete} winners={winners}")


if __name__ == "__main__":
    main()
