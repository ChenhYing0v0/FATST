#!/usr/bin/env python3
"""Analyze selected-only multi-seed stability for StageC SC0-DAP-R2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_dataset_profile_calibration_r2.json"
PROFILE_RE = re.compile(r"r2b_p(?P<patch>\d+)_d(?P<model>\d+)_ff(?P<ff>\d+)_(?P<width>\w+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--r2b-summary", type=Path, required=True)
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


def parse_profile(profile: str) -> dict[str, Any]:
    match = PROFILE_RE.fullmatch(profile)
    if match is None:
        raise ValueError(f"invalid selected profile: {profile}")
    return {
        "patch_num": int(match["patch"]),
        "d_model": int(match["model"]),
        "d_ff": int(match["ff"]),
        "width": match["width"],
    }


def run_location(
    phase_a_root: Path,
    phase_b_root: Path,
    confirmation_root: Path,
    dataset: str,
    profile: str,
    seed: int,
) -> Path:
    parsed = parse_profile(profile)
    if seed == 2021 and parsed["width"] == "medium":
        run_name = (
            f"SC0DAP_R2A_r2a_p{parsed['patch_num']}_"
            f"d{parsed['d_model']}_ff{parsed['d_ff']}"
        )
        root = phase_a_root
    elif seed == 2021:
        run_name = f"SC0DAP_R2B_{profile}"
        root = phase_b_root
    else:
        run_name = f"SC0DAP_R2C_{profile}"
        root = confirmation_root
    return root / run_name / dataset / "h720_full" / f"seed{seed}"


def collect(
    args: argparse.Namespace,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    current_hash = file_hash(args.config)
    selection_hash = str(summary["profile_hash"])
    seeds = [int(config["common"]["screen_seed"])] + [
        int(seed) for seed in config["common"]["confirmation_seeds"]
    ]
    for dataset in config["datasets"]:
        profile = summary["selected_profiles"][dataset]
        parsed = parse_profile(profile)
        for seed in seeds:
            directory = run_location(
                args.phase_a_root,
                args.phase_b_root,
                args.confirmation_root,
                dataset,
                profile,
                seed,
            )
            required = (
                directory / "effective_config.json",
                directory / "model_diagnostics.json",
                directory / "training_log.csv",
                directory / "metrics_by_target_horizon.csv",
            )
            if not all(path.exists() for path in required):
                diagnostics.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "seed": seed,
                        "status": "missing",
                        "run_dir": str(directory),
                    }
                )
                continue
            effective = json.loads(required[0].read_text(encoding="utf-8"))
            model_info = json.loads(required[1].read_text(encoding="utf-8"))
            training = read_csv(required[2])
            adapter = effective["adapter"]
            official = effective["official_args"]
            expected_hash = selection_hash if seed == 2021 else current_hash
            config_ok = (
                adapter.get("profile_hash") == expected_hash
                and adapter.get("final_evaluation_split") == "val"
                and int(official["patch_num"]) == parsed["patch_num"]
                and int(official["d_model"]) == parsed["d_model"]
                and int(official["d_ff"]) == parsed["d_ff"]
                and int(official["seed"]) == seed
            )
            training_ok = bool(training) and all(
                math.isfinite(float(row[field]))
                for row in training
                for field in ("train_loss", "val_mean_mse", "lr")
            )
            status = "ok" if config_ok and training_ok else "mismatch"
            diagnostics.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "seed": seed,
                    "status": status,
                    "patch_num": parsed["patch_num"],
                    "d_model": parsed["d_model"],
                    "d_ff": parsed["d_ff"],
                    "active_forward_parameters": model_info["active_forward_parameters"],
                    "trained_epochs": len(training),
                    "best_epoch": training[-1].get("best_epoch_so_far", ""),
                    "reused_selection_run": int(seed == 2021),
                    "run_dir": str(directory),
                }
            )
            for row in read_csv(required[3]):
                if row.get("evaluation_split") != "val":
                    raise ValueError(f"non-validation metric in {directory}")
                metrics.append(
                    {
                        "dataset": dataset,
                        "profile": profile,
                        "seed": seed,
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                    }
                )
    return metrics, diagnostics


def stability_rows(
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    policy = config["phase_c_stability_confirmation"]
    mean_limit = float(policy["dataset_mean_mse_cv_max"])
    max_limit = float(policy["dataset_max_mse_cv_max"])
    horizon_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        profile = next(row["profile"] for row in metrics if row["dataset"] == dataset)
        cvs = []
        for horizon in config["common"]["evaluation_horizons"]:
            values = [
                float(row["mse"])
                for row in metrics
                if row["dataset"] == dataset
                and int(row["target_horizon"]) == int(horizon)
            ]
            cv = stdev(values) / mean(values)
            cvs.append(cv)
            horizon_rows.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "target_horizon": horizon,
                    "seed_count": len(values),
                    "mean_mse": mean(values),
                    "sample_std_mse": stdev(values),
                    "mse_cv": cv,
                    "min_mse": min(values),
                    "max_mse": max(values),
                }
            )
        mean_cv = mean(cvs)
        max_cv = max(cvs)
        passed = mean_cv <= mean_limit and max_cv <= max_limit
        dataset_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "mean_dense_mse_cv": mean_cv,
                "max_dense_mse_cv": max_cv,
                "mean_cv_limit": mean_limit,
                "max_cv_limit": max_limit,
                "stability_pass": int(passed),
            }
        )
    return horizon_rows, dataset_rows, all(
        bool(row["stability_pass"]) for row in dataset_rows
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = json.loads(args.r2b_summary.read_text(encoding="utf-8"))
    metrics, diagnostics = collect(args, config, summary)
    seed_count = 1 + len(config["common"]["confirmation_seeds"])
    expected_runs = len(config["datasets"]) * seed_count
    expected_metrics = expected_runs * len(config["common"]["evaluation_horizons"])
    complete = (
        len(diagnostics) == expected_runs
        and all(row["status"] == "ok" for row in diagnostics)
        and len(metrics) == expected_metrics
    )
    horizon_stability: list[dict[str, Any]] = []
    dataset_stability: list[dict[str, Any]] = []
    stable = False
    if complete:
        horizon_stability, dataset_stability, stable = stability_rows(metrics, config)
    decision = (
        "dataset_profiles_stable_and_frozen"
        if complete and stable
        else "protocol_audit_required" if complete else "analysis_incomplete"
    )
    output = {
        "candidate": "SC0-DAP-R2C",
        "complete": complete,
        "stability_pass": stable,
        "decision": decision,
        "expected_profile_seed_instances": expected_runs,
        "new_remote_runs": len(config["datasets"]) * len(config["common"]["confirmation_seeds"]),
        "profile_hash": file_hash(args.config),
        "selected_profiles": summary["selected_profiles"],
        "parameter_count_used_for_gate": False,
        "test_metrics_used_for_gate": False,
        "relative_winner_reconfirmed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "r2c_validation_metrics.csv", metrics)
    write_csv(args.output_dir / "r2c_run_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "r2c_horizon_stability.csv", horizon_stability)
    write_csv(args.output_dir / "r2c_dataset_stability.csv", dataset_stability)
    (args.output_dir / "r2c_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"stage_c_dap_r2c_analysis_done complete={complete} stable={stable}")


if __name__ == "__main__":
    main()
