#!/usr/bin/env python3
"""Analyze validation-only ETTh1/ETTm2 natural-profile extension phases."""

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


HORIZON_FIELDS = (48, 96, 144, 192, 288, 336, 512, 720)
PROFILE_RE = re.compile(r"r2b_p(\d+)_d(\d+)_ff(\d+)_(narrow|medium|wide)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b", "c"], required=True)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--phase-c-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase-a-summary", type=Path)
    parser.add_argument("--phase-b-summary", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def phase_a_dir(root: Path, dataset: str, profile: str) -> Path:
    return root / f"SC0FIVE_R2A_{profile}" / dataset / "h720_full" / "seed2021"


def phase_b_profile(patch_num: int, width_name: str, width: dict[str, int]) -> str:
    return f"r2b_p{patch_num}_d{width['d_model']}_ff{width['d_ff']}_{width_name}"


def phase_b_dir(root: Path, dataset: str, profile: str) -> Path:
    return root / f"SC0FIVE_R2B_{profile}" / dataset / "h720_full" / "seed2021"


def phase_c_dir(root: Path, dataset: str, profile: str, seed: int) -> Path:
    return root / f"SC0FIVE_R2C_{profile}" / dataset / "h720_full" / f"seed{seed}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_run(
    directory: Path,
    config: dict[str, Any],
    config_path: Path,
    expected: dict[str, int],
    seed: int,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    required = {
        "effective": directory / "effective_config.json",
        "diagnostics": directory / "model_diagnostics.json",
        "training": directory / "training_log.csv",
        "metrics": directory / "metrics_by_target_horizon.csv",
    }
    if not all(path.exists() for path in required.values()):
        return {}, {"status": "missing", "run_dir": str(directory)}
    effective = json.loads(required["effective"].read_text(encoding="utf-8"))
    diagnostics = json.loads(required["diagnostics"].read_text(encoding="utf-8"))
    training = read_csv(required["training"])
    adapter = effective["adapter"]
    official = effective["official_args"]
    config_ok = (
        adapter.get("protocol_profile") == config["protocol_profile"]
        and adapter.get("profile_hash") == file_hash(config_path)
        and adapter.get("final_evaluation_split") == "val"
        and int(adapter.get("seed")) == seed
        and int(official["patch_num"]) == expected["patch_num"]
        and int(official["d_model"]) == expected["d_model"]
        and int(official["d_ff"]) == expected["d_ff"]
    )
    training_ok = bool(training) and all(
        finite(row.get(field))
        for row in training
        for field in ("train_loss", "val_mean_mse", "lr")
    )
    values: dict[int, dict[str, float]] = {}
    for row in read_csv(required["metrics"]):
        if row.get("evaluation_split") != "val":
            raise ValueError(f"non-validation metric at {directory}")
        horizon = int(row["target_horizon"])
        values[horizon] = {"mse": float(row["mse"]), "mae": float(row["mae"])}
    horizons_ok = set(values) == set(HORIZON_FIELDS)
    return values, {
        "status": "ok" if config_ok and training_ok and horizons_ok else "mismatch",
        "config_ok": int(config_ok),
        "training_ok": int(training_ok),
        "horizons_ok": int(horizons_ok),
        "active_forward_parameters": diagnostics["active_forward_parameters"],
        "trained_epochs": len(training),
        "run_dir": str(directory),
    }


def selection_rows(
    values: dict[str, dict[str, dict[int, dict[str, float]]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    winners: dict[str, str] = {}
    for dataset, profiles in values.items():
        candidates = []
        for profile, horizon_values in profiles.items():
            regrets = {
                horizon: horizon_values[horizon]["mse"]
                / min(other[horizon]["mse"] for other in profiles.values())
                - 1.0
                for horizon in HORIZON_FIELDS
            }
            candidates.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "macro_dense_regret": mean(regrets.values()),
                    "max_dense_regret": max(regrets.values()),
                    "h720_regret": regrets[720],
                    "h720_val_mse": horizon_values[720]["mse"],
                    **{f"h{horizon}_regret": regrets[horizon] for horizon in HORIZON_FIELDS},
                }
            )
        candidates.sort(
            key=lambda row: (
                float(row["macro_dense_regret"]),
                float(row["max_dense_regret"]),
                float(row["h720_regret"]),
                str(row["profile"]),
            )
        )
        winners[dataset] = str(candidates[0]["profile"])
        for row in candidates:
            row["selected"] = int(row["profile"] == winners[dataset])
        rows.extend(candidates)
    return rows, winners


def analyze_phase_a(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    profiles = config["phase_a_patch_screen"]["profiles"]
    values: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    diagnostics = []
    for dataset in config["datasets"]:
        values[dataset] = {}
        for profile, specification in profiles.items():
            run_values, diagnostic = validate_run(
                phase_a_dir(args.phase_a_root, dataset, profile),
                config,
                args.config,
                specification,
                2021,
            )
            diagnostics.append({"dataset": dataset, "profile": profile, **diagnostic})
            if run_values:
                values[dataset][profile] = run_values
    complete = len(diagnostics) == 6 and all(row["status"] == "ok" for row in diagnostics)
    selections, winners = selection_rows(values) if complete else ([], {})
    summary = {
        "phase": "a",
        "complete": complete,
        "decision": "phase_a_patch_selected" if complete else "analysis_incomplete",
        "profile_hash": file_hash(args.config),
        "selected_profiles": winners,
        "test_metrics_used_for_selection": False,
        "parameter_count_used_for_selection": False,
    }
    return diagnostics, selections, summary


def analyze_phase_b(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.phase_a_summary is None:
        raise ValueError("phase B requires --phase-a-summary")
    phase_a = json.loads(args.phase_a_summary.read_text(encoding="utf-8"))
    a_profiles = config["phase_a_patch_screen"]["profiles"]
    widths = config["phase_b_width_screen"]["widths"]
    values: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    diagnostics = []
    for dataset in config["datasets"]:
        a_profile = phase_a["selected_profiles"][dataset]
        patch_num = int(a_profiles[a_profile]["patch_num"])
        values[dataset] = {}
        for width_name, width in widths.items():
            profile = phase_b_profile(patch_num, width_name, width)
            expected = {"patch_num": patch_num, **width}
            if width_name == "medium":
                directory = phase_a_dir(args.phase_a_root, dataset, a_profile)
            else:
                directory = phase_b_dir(args.phase_b_root, dataset, profile)
            run_values, diagnostic = validate_run(
                directory, config, args.config, expected, 2021
            )
            diagnostics.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "reused_phase_a": int(width_name == "medium"),
                    **diagnostic,
                }
            )
            if run_values:
                values[dataset][profile] = run_values
    complete = len(diagnostics) == 6 and all(row["status"] == "ok" for row in diagnostics)
    selections, winners = selection_rows(values) if complete else ([], {})
    summary = {
        "phase": "b",
        "complete": complete,
        "decision": "phase_b_width_selected" if complete else "analysis_incomplete",
        "profile_hash": file_hash(args.config),
        "selected_profiles": winners,
        "test_metrics_used_for_selection": False,
        "parameter_count_used_for_selection": False,
    }
    return diagnostics, selections, summary


def profile_specification(profile: str) -> dict[str, int]:
    match = PROFILE_RE.fullmatch(profile)
    if match is None:
        raise ValueError(f"invalid profile: {profile}")
    patch_num, d_model, d_ff, _width = match.groups()
    return {
        "patch_num": int(patch_num),
        "patch_len": 720 // int(patch_num),
        "d_model": int(d_model),
        "d_ff": int(d_ff),
    }


def analyze_phase_c(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.phase_a_summary is None or args.phase_b_summary is None:
        raise ValueError("phase C requires phase A and B summaries")
    phase_a = json.loads(args.phase_a_summary.read_text(encoding="utf-8"))
    phase_b = json.loads(args.phase_b_summary.read_text(encoding="utf-8"))
    diagnostics = []
    stability_rows = []
    selected_profiles = phase_b["selected_profiles"]
    policy = config["phase_c_stability_confirmation"]
    all_stable = True
    for dataset in config["datasets"]:
        profile = selected_profiles[dataset]
        specification = profile_specification(profile)
        per_seed: dict[int, dict[int, dict[str, float]]] = {}
        for seed in (2021, 2022, 2023):
            if seed == 2021 and profile.endswith("_medium"):
                a_profile = phase_a["selected_profiles"][dataset]
                directory = phase_a_dir(args.phase_a_root, dataset, a_profile)
            elif seed == 2021:
                directory = phase_b_dir(args.phase_b_root, dataset, profile)
            else:
                directory = phase_c_dir(args.phase_c_root, dataset, profile, seed)
            run_values, diagnostic = validate_run(
                directory, config, args.config, specification, seed
            )
            diagnostics.append(
                {"dataset": dataset, "profile": profile, "seed": seed, **diagnostic}
            )
            if run_values:
                per_seed[seed] = run_values
        if len(per_seed) != 3:
            all_stable = False
            continue
        horizon_cvs = {}
        for horizon in HORIZON_FIELDS:
            values = [per_seed[seed][horizon]["mse"] for seed in (2021, 2022, 2023)]
            horizon_cvs[horizon] = stdev(values) / mean(values)
        mean_cv = mean(horizon_cvs.values())
        max_cv = max(horizon_cvs.values())
        stable = (
            mean_cv <= float(policy["dataset_mean_mse_cv_max"])
            and max_cv <= float(policy["dataset_max_mse_cv_max"])
        )
        all_stable = all_stable and stable
        stability_rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "mean_dense_mse_cv": mean_cv,
                "max_dense_mse_cv": max_cv,
                "three_seed_mean_validation_mse_h720": mean(
                    [per_seed[seed][720]["mse"] for seed in (2021, 2022, 2023)]
                ),
                "stable": int(stable),
                **{f"h{horizon}_mse_cv": horizon_cvs[horizon] for horizon in HORIZON_FIELDS},
            }
        )
    complete = len(diagnostics) == 6 and all(row["status"] == "ok" for row in diagnostics)
    decision = (
        "dataset_profiles_stable_and_ready_to_freeze"
        if complete and all_stable
        else "stability_gate_failed_or_incomplete"
    )
    summary = {
        "phase": "c",
        "complete": complete,
        "all_stable": all_stable,
        "decision": decision,
        "profile_hash": file_hash(args.config),
        "selected_profiles": selected_profiles,
        "selected_profile_specs": {
            dataset: profile_specification(profile)
            for dataset, profile in selected_profiles.items()
        },
        "test_metrics_used_for_gate": False,
        "parameter_count_used_for_gate": False,
    }
    return diagnostics, stability_rows, summary


def synthetic_smoke() -> None:
    values = {
        "ETTh1": {
            "a": {h: {"mse": 1.0, "mae": 1.0} for h in HORIZON_FIELDS},
            "b": {h: {"mse": 1.1, "mae": 1.0} for h in HORIZON_FIELDS},
        }
    }
    rows, winners = selection_rows(values)
    if winners != {"ETTh1": "a"} or len(rows) != 2:
        raise RuntimeError("selection smoke failed")
    specification = profile_specification("r2b_p24_d32_ff64_narrow")
    if specification != {"patch_num": 24, "patch_len": 30, "d_model": 32, "d_ff": 64}:
        raise RuntimeError("profile parse smoke failed")
    print("stage_c_five_profile_extension_analyzer_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.phase == "a":
        diagnostics, results, summary = analyze_phase_a(args, config)
        result_name = "phase_a_patch_selection.csv"
    elif args.phase == "b":
        diagnostics, results, summary = analyze_phase_b(args, config)
        result_name = "phase_b_width_selection.csv"
    else:
        diagnostics, results, summary = analyze_phase_c(args, config)
        result_name = "phase_c_stability.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"phase_{args.phase}_run_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / result_name, results)
    (args.output_dir / f"phase_{args.phase}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"stage_c_five_profile_extension_phase_{args.phase}_done "
        f"decision={summary['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
