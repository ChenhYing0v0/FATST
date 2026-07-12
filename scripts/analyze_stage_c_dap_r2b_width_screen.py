#!/usr/bin/env python3
"""Analyze StageC SC0-DAP-R2 Phase B natural width screen."""

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
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--phase-a-summary", type=Path, required=True)
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_name(patch_num: int, width_name: str, width: dict[str, int]) -> str:
    return f"r2b_p{patch_num}_d{width['d_model']}_ff{width['d_ff']}_{width_name}"


def location(
    phase_a_root: Path,
    phase_b_root: Path,
    dataset: str,
    patch_num: int,
    width_name: str,
    width: dict[str, int],
    seed: int,
) -> Path:
    if width_name == "medium":
        run_name = f"SC0DAP_R2A_r2a_p{patch_num}_d64_ff128"
        return phase_a_root / run_name / dataset / "h720_full" / f"seed{seed}"
    run_name = f"SC0DAP_R2B_{profile_name(patch_num, width_name, width)}"
    return phase_b_root / run_name / dataset / "h720_full" / f"seed{seed}"


def collect(
    phase_a_root: Path,
    phase_b_root: Path,
    phase_a_summary: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    metrics = []
    diagnostics = []
    seed = int(config["common"]["screen_seed"])
    expected_hash = file_hash(config_path)
    phase_a_profiles = config["phase_a_patch_screen"]["profiles"]
    widths = config["phase_b_width_screen"]["widths"]
    selected_patch = {
        dataset: int(phase_a_profiles[profile]["patch_num"])
        for dataset, profile in phase_a_summary["selected_profiles"].items()
    }
    for dataset in config["datasets"]:
        patch_num = selected_patch[dataset]
        for width_name, width in widths.items():
            profile = profile_name(patch_num, width_name, width)
            directory = location(
                phase_a_root, phase_b_root, dataset, patch_num,
                width_name, width, seed,
            )
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
                adapter.get("profile_hash") == expected_hash
                and adapter.get("final_evaluation_split") == "val"
                and int(official["patch_num"]) == patch_num
                and int(official["d_model"]) == int(width["d_model"])
                and int(official["d_ff"]) == int(width["d_ff"])
            )
            training_ok = bool(training) and all(
                math.isfinite(float(row[field]))
                for row in training
                for field in ("train_loss", "val_mean_mse", "lr")
            )
            diagnostics.append(
                {
                    "dataset": dataset, "profile": profile,
                    "status": "ok" if config_ok and training_ok else "mismatch",
                    "patch_num": patch_num, "d_model": official["d_model"], "d_ff": official["d_ff"],
                    "active_forward_parameters": model_info["active_forward_parameters"],
                    "trained_epochs": len(training), "best_epoch": training[-1].get("best_epoch_so_far", ""),
                    "reused_phase_a": int(width_name == "medium"), "run_dir": str(directory),
                }
            )
            for row in read_csv(required[3]):
                if row.get("evaluation_split") != "val":
                    raise ValueError(f"non-validation metric in {directory}")
                metrics.append(
                    {"dataset": dataset, "profile": profile, "target_horizon": int(row["target_horizon"]),
                     "mse": float(row["mse"]), "mae": float(row["mae"])}
                )
    return metrics, diagnostics, selected_patch


def select(metrics: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    horizons = [int(value) for value in config["common"]["evaluation_horizons"]]
    rows = []
    winners = {}
    for dataset in config["datasets"]:
        profiles = sorted({str(row["profile"]) for row in metrics if row["dataset"] == dataset})
        values = {
            profile: {
                horizon: next(float(row["mse"]) for row in metrics
                              if row["dataset"] == dataset and row["profile"] == profile
                              and int(row["target_horizon"]) == horizon)
                for horizon in horizons
            }
            for profile in profiles
        }
        candidates = []
        for profile in profiles:
            regrets = {
                horizon: values[profile][horizon] / min(values[p][horizon] for p in profiles) - 1.0
                for horizon in horizons
            }
            candidates.append(
                {"dataset": dataset, "profile": profile,
                 "macro_dense_regret": mean(regrets.values()), "max_dense_regret": max(regrets.values()),
                 "h720_regret": regrets[720], "h720_val_mse": values[profile][720],
                 **{f"h{horizon}_regret": regrets[horizon] for horizon in horizons}}
            )
        candidates.sort(key=lambda row: (
            float(row["macro_dense_regret"]), float(row["max_dense_regret"]),
            float(row["h720_regret"]), str(row["profile"])))
        winners[dataset] = str(candidates[0]["profile"])
        for row in candidates:
            row["selected"] = int(row["profile"] == winners[dataset])
        rows.extend(candidates)
    return rows, winners


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    phase_a_summary = json.loads(args.phase_a_summary.read_text(encoding="utf-8"))
    metrics, diagnostics, selected_patch = collect(
        args.phase_a_root, args.phase_b_root, phase_a_summary, args.config, config
    )
    expected_runs = len(config["datasets"]) * len(config["phase_b_width_screen"]["widths"])
    complete = (len(diagnostics) == expected_runs and all(row["status"] == "ok" for row in diagnostics)
                and len(metrics) == expected_runs * len(config["common"]["evaluation_horizons"]))
    selections, winners = select(metrics, config) if complete else ([], {})
    summary = {
        "candidate": "SC0-DAP-R2B", "complete": complete,
        "decision": "phase_b_width_selected" if complete else "analysis_incomplete",
        "expected_profiles": expected_runs, "new_remote_runs": 6,
        "profile_hash": file_hash(args.config), "selected_patch_num": selected_patch,
        "selected_profiles": winners, "parameter_count_used_for_selection": False,
        "test_metrics_used_for_selection": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "r2b_validation_horizon_metrics.csv", metrics)
    write_csv(args.output_dir / "r2b_run_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "r2b_width_selection.csv", selections)
    (args.output_dir / "r2b_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"stage_c_dap_r2b_analysis_done complete={complete} winners={winners}")


if __name__ == "__main__":
    main()
