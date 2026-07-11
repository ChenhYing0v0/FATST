#!/usr/bin/env python3
"""Analyze multi-seed validation-only StageC SC0-R1 carrier calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_mechanism_control_r1.json"


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def profile_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, dataset: str, arm: str, seed: int) -> Path:
    return (
        root
        / f"SC0R1_{arm}_validation_only"
        / dataset
        / "h720_full"
        / f"seed{seed}"
    )


def collect(
    raw_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    expected_hash = profile_hash(config_path)
    common = config["common"]
    for seed in common["seeds"]:
        for dataset in config["datasets"]:
            for arm_name, arm in config["arms"].items():
                directory = run_dir(raw_root, dataset, arm_name, int(seed))
                required = (
                    directory / "effective_config.json",
                    directory / "model_diagnostics.json",
                    directory / "training_log.csv",
                    directory / "metrics_by_target_horizon.csv",
                )
                if not all(path.exists() for path in required):
                    diagnostics.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "arm": arm_name,
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
                last_training = training[-1] if training else {}
                config_ok = (
                    adapter.get("protocol_class") == "mechanism_control"
                    and adapter.get("protocol_profile") == config["protocol_profile"]
                    and adapter.get("profile_hash") == expected_hash
                    and adapter.get("final_evaluation_split") == "val"
                    and adapter.get("checkpoint_policy") == "best-val"
                    and bool(adapter.get("enable_early_stopping"))
                    and int(adapter.get("patience"))
                    == int(common["early_stopping_patience"])
                    and float(adapter.get("early_stopping_min_delta"))
                    == float(common["early_stopping_min_delta"])
                    and int(adapter.get("seed")) == int(seed)
                    and int(official["patch_num"]) == int(arm["patch_num"])
                    and int(official["d_model"]) == int(arm["d_model"])
                    and int(official["d_ff"]) == int(arm["d_ff"])
                )
                parameter_ok = (
                    int(model_info["active_forward_parameters"])
                    == int(arm["active_forward_parameters"])
                    and int(model_info["unused_proj_x_parameters"])
                    == int(arm["unused_proj_x_parameters"])
                )
                training_ok = (
                    bool(training)
                    and len(training) <= int(common["max_epochs"])
                    and all(
                        finite(row.get(field))
                        for row in training
                        for field in ("train_loss", "val_mean_mse", "lr")
                    )
                    and int(last_training.get("best_epoch_so_far", 0)) > 0
                )
                diagnostics.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "arm": arm_name,
                        "status": (
                            "ok" if config_ok and parameter_ok and training_ok else "mismatch"
                        ),
                        "config_ok": int(config_ok),
                        "parameter_ok": int(parameter_ok),
                        "training_ok": int(training_ok),
                        "trained_epochs": len(training),
                        "best_epoch": last_training.get("best_epoch_so_far", ""),
                        "stopped_early": last_training.get("stop_triggered", ""),
                        "active_forward_parameters": model_info[
                            "active_forward_parameters"
                        ],
                        "mean_epoch_seconds": mean(
                            float(row["epoch_seconds"]) for row in training
                        ),
                        "run_dir": str(directory),
                    }
                )
                for row in read_csv(required[3]):
                    if row.get("evaluation_split") != "val":
                        raise ValueError(f"forbidden non-validation metric: {directory}")
                    if row.get("protocol_class") != "mechanism_control":
                        raise ValueError(f"wrong protocol class: {directory}")
                    metrics.append(
                        {
                            "seed": seed,
                            "dataset": dataset,
                            "arm": arm_name,
                            "target_horizon": int(row["target_horizon"]),
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                        }
                    )
    return metrics, diagnostics


def efficiency(diagnostics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        arm: mean(
            float(row["mean_epoch_seconds"])
            for row in diagnostics
            if row["arm"] == arm and row["status"] == "ok"
        )
        for arm in {str(row["arm"]) for row in diagnostics}
        if any(row["arm"] == arm and row["status"] == "ok" for row in diagnostics)
    }


def choose(
    rows: list[dict[str, Any]],
    tie_delta: float,
    arm_efficiency: dict[str, float],
    config: dict[str, Any],
) -> str:
    ordered = sorted(rows, key=lambda row: float(row["macro_regret"]))
    best_score = float(ordered[0]["macro_regret"])
    tied = [
        row
        for row in ordered
        if float(row["macro_regret"]) - best_score < tie_delta
    ]
    return str(
        min(
            tied,
            key=lambda row: (
                arm_efficiency.get(str(row["arm"]), math.inf),
                int(config["arms"][str(row["arm"])]["active_forward_parameters"]),
            ),
        )["arm"]
    )


def regret_rows(
    values: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for arm in config["arms"]:
        regrets = {
            dataset: values[dataset][arm] / min(values[dataset].values()) - 1.0
            for dataset in config["datasets"]
        }
        rows.append(
            {
                "arm": arm,
                "macro_regret": mean(regrets.values()),
                "max_dataset_regret": max(regrets.values()),
                **{f"{dataset}_val_mse": values[dataset][arm] for dataset in config["datasets"]},
                **{f"{dataset}_regret": regrets[dataset] for dataset in config["datasets"]},
            }
        )
    return rows


def select(
    metrics: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    horizon = int(config["common"]["pred_len"])
    full = [row for row in metrics if int(row["target_horizon"]) == horizon]
    arm_efficiency = efficiency(diagnostics)
    tie_delta = float(config["gates"]["tie_macro_score_delta"])
    seed_rows: list[dict[str, Any]] = []
    seed_winners: dict[int, str] = {}
    seed_dataset_regrets: dict[tuple[int, str, str], float] = {}

    for seed in config["common"]["seeds"]:
        values = {
            dataset: {
                arm: next(
                    float(row["mse"])
                    for row in full
                    if int(row["seed"]) == int(seed)
                    and row["dataset"] == dataset
                    and row["arm"] == arm
                )
                for arm in config["arms"]
            }
            for dataset in config["datasets"]
        }
        current = regret_rows(values, config)
        winner = choose(current, tie_delta, arm_efficiency, config)
        seed_winners[int(seed)] = winner
        for row in current:
            row["seed"] = seed
            row["selected"] = int(row["arm"] == winner)
            for dataset in config["datasets"]:
                seed_dataset_regrets[(int(seed), dataset, str(row["arm"]))] = float(
                    row[f"{dataset}_regret"]
                )
        seed_rows.extend(current)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_winners: dict[str, str] = {}
    for selector, aggregate in (("pooled_mean", mean), ("median_seed", median)):
        values = {
            dataset: {
                arm: aggregate(
                    float(row["mse"])
                    for row in full
                    if row["dataset"] == dataset and row["arm"] == arm
                )
                for arm in config["arms"]
            }
            for dataset in config["datasets"]
        }
        current = regret_rows(values, config)
        winner = choose(current, tie_delta, arm_efficiency, config)
        aggregate_winners[selector] = winner
        for row in current:
            row["selector"] = selector
            row["selected"] = int(row["arm"] == winner)
        aggregate_rows.extend(current)

    selected_arm = aggregate_winners.get("pooled_mean", "")
    selected_pooled = next(
        (row for row in aggregate_rows if row["selector"] == "pooled_mean" and row["arm"] == selected_arm),
        {},
    )
    winner_count = sum(winner == selected_arm for winner in seed_winners.values())
    max_seed_regret = max(
        (
            regret
            for (seed, dataset, arm), regret in seed_dataset_regrets.items()
            if arm == selected_arm
        ),
        default=math.inf,
    )
    gate = {
        "selected_arm": selected_arm,
        "aggregate_winners": aggregate_winners,
        "seed_winners": seed_winners,
        "selected_seed_winner_count": winner_count,
        "selected_pooled_max_dataset_regret": selected_pooled.get(
            "max_dataset_regret", math.inf
        ),
        "selected_max_seed_dataset_regret": max_seed_regret,
    }
    return seed_rows, aggregate_rows, gate


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# StageC SC0-R1 Multi-Seed Carrier Calibration Report",
        "",
        f"- `decision`: `{summary['decision']}`",
        f"- `selected_arm`: `{summary['selected_arm']}`",
        f"- `profile_hash`: `{summary['profile_hash']}`",
        f"- `test_metrics_used_for_selection`: `{str(summary['test_metrics_used_for_selection']).lower()}`",
        f"- `seed_winners`: `{summary['seed_winners']}`",
        f"- `aggregate_winners`: `{summary['aggregate_winners']}`",
        f"- `selected_seed_winner_count`: `{summary['selected_seed_winner_count']}`",
        f"- `selected_pooled_max_dataset_regret`: `{summary['selected_pooled_max_dataset_regret']:.6f}`",
        f"- `selected_max_seed_dataset_regret`: `{summary['selected_max_seed_dataset_regret']:.6f}`",
        "",
        "A pass freezes one validation-selected global carrier profile. A failure returns StageC to Step 2/3 and does not authorize dataset-specific presets.",
    ]
    (output_dir / "sc0_r1_carrier_calibration_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metrics, diagnostics = collect(args.raw_root, args.config, config)
    expected_runs = (
        len(config["datasets"])
        * len(config["arms"])
        * len(config["common"]["seeds"])
    )
    complete = (
        len(diagnostics) == expected_runs
        and all(row["status"] == "ok" for row in diagnostics)
        and len(metrics)
        == expected_runs * len(config["common"]["evaluation_horizons"])
    )
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    gate: dict[str, Any] = {
        "selected_arm": "",
        "aggregate_winners": {},
        "seed_winners": {},
        "selected_seed_winner_count": 0,
        "selected_pooled_max_dataset_regret": math.inf,
        "selected_max_seed_dataset_regret": math.inf,
    }
    if complete:
        seed_rows, aggregate_rows, gate = select(metrics, diagnostics, config)
    gates = config["gates"]
    passed = (
        complete
        and len(set(gate["aggregate_winners"].values())) == 1
        and gate["selected_seed_winner_count"]
        >= int(gates["required_seed_winner_count"])
        and gate["selected_pooled_max_dataset_regret"]
        <= float(gates["max_pooled_per_dataset_regret"])
        and gate["selected_max_seed_dataset_regret"]
        <= float(gates["max_seed_per_dataset_regret"])
    )
    summary = {
        "candidate": config["candidate"],
        "complete": complete,
        "decision": (
            "global_profile_selected_and_frozen"
            if passed
            else ("common_profile_not_stable" if complete else "analysis_incomplete")
        ),
        "expected_runs": expected_runs,
        "profile_hash": profile_hash(args.config),
        "test_metrics_used_for_selection": False,
        **gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sc0_r1_validation_horizon_metrics.csv", metrics)
    write_csv(args.output_dir / "sc0_r1_run_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "sc0_r1_seed_selection.csv", seed_rows)
    write_csv(args.output_dir / "sc0_r1_aggregate_selection.csv", aggregate_rows)
    (args.output_dir / "sc0_r1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir, summary)
    print(
        f"stage_c_sc0_r1_analysis_done decision={summary['decision']} "
        f"selected={summary['selected_arm']}"
    )


if __name__ == "__main__":
    main()
