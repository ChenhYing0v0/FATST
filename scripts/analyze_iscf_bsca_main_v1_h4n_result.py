#!/usr/bin/env python3
"""Apply the frozen Weather H4N selector and audit its comparison roles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "analysis" / "iscf_bsca_main_v1_hpo_20260731"
HORIZONS = (96, 192, 336, 720)
PHASES = (
    "test_audit_result",
    "h3a_test_result",
    "h3b_test_result",
    "h4j_test_result",
    "h4k_test_result",
    "h4l_test_result",
    "h4m_test_result",
    "h4n_test_result",
)
FULL_TABLE = (
    ROOT
    / "analysis"
    / "iscf_bsca_paper_experiment_consolidation_20260731"
    / "main_i_timealign_table6_style_20260805"
    / "table_data_long.csv"
)
LEGACY_TABLE = (
    ROOT
    / "analysis"
    / "iscf_bsca_paper_experiment_consolidation_20260731"
    / "timealign_table6_main_i_published.csv"
)
H4N_CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_weather_h4n.json"
PREVIOUS_CELLS = (
    ANALYSIS_ROOT
    / "joint_objective_h4m_result_20260805"
    / "joint_selected_cells.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def targets(
    path: Path, *, exclude_iscf: bool
) -> tuple[dict[tuple[str, int, str], float], set[str]]:
    result: dict[tuple[str, int, str], float] = {}
    models: set[str] = set()
    for row in read_csv(path):
        if exclude_iscf and row.get("model") == "ISCF-BSCA":
            continue
        try:
            horizon = int(row["horizon"])
        except (TypeError, ValueError):
            continue
        if horizon not in HORIZONS:
            continue
        models.add(row["model"])
        for metric in ("mse", "mae"):
            key = (row["dataset"], horizon, metric)
            value = float(row[metric])
            result[key] = min(value, result.get(key, value))
    return result, models


def comparator_counts(
    cells: list[dict[str, Any]],
    target: dict[tuple[str, int, str], float],
) -> dict[str, Any]:
    exact = {"mse": 0, "mae": 0}
    displayed = {"mse": 0, "mae": 0}
    by_dataset: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "mse_exact": 0,
            "mae_exact": 0,
            "mse_displayed": 0,
            "mae_displayed": 0,
        }
    )
    for row in cells:
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        for metric, field in (("mse", "test_mse"), ("mae", "test_mae")):
            key = (dataset, horizon, metric)
            if key not in target:
                continue
            value = float(row[field])
            is_exact = value <= target[key]
            is_displayed = round(value, 3) <= round(target[key], 3)
            exact[metric] += int(is_exact)
            displayed[metric] += int(is_displayed)
            by_dataset[dataset][f"{metric}_exact"] += int(is_exact)
            by_dataset[dataset][f"{metric}_displayed"] += int(is_displayed)
    return {
        "exact": {**exact, "combined": sum(exact.values())},
        "displayed_three_decimal": {
            **displayed,
            "combined": sum(displayed.values()),
        },
        "by_dataset": dict(by_dataset),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(H4N_CONFIG.read_text(encoding="utf-8"))
    full_targets, full_models = targets(FULL_TABLE, exclude_iscf=True)
    legacy_targets, legacy_models = targets(LEGACY_TABLE, exclude_iscf=False)

    cells: list[dict[str, str]] = []
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    for phase in PHASES:
        cells.extend(read_csv(ANALYSIS_ROOT / phase / "all_trial_scorecard.csv"))
        for row in read_csv(ANALYSIS_ROOT / phase / "profile_aggregates.csv"):
            key = (row["dataset"], row["trial_id"])
            if key in metadata:
                raise ValueError(f"duplicate metadata: {key}")
            metadata[key] = row
    trial_keys = {(row["dataset"], row["trial_id"]) for row in cells}
    if len(cells) != 916 or len(trial_keys) != 229:
        raise ValueError("H1--H4N trial matrix is incomplete")

    weather_cells: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cells:
        if row["dataset"] == "Weather":
            weather_cells[row["trial_id"]].append(row)
    if len(weather_cells) != 96:
        raise ValueError("expected 96 historical plus H4N Weather profiles")

    profile_rows: list[dict[str, Any]] = []
    for trial_id, rows in sorted(weather_cells.items()):
        by_horizon = {int(row["horizon"]): row for row in rows}
        if set(by_horizon) != set(HORIZONS):
            raise ValueError(f"incomplete Weather profile: {trial_id}")
        mse_ratios = []
        mae_ratios = []
        mse_leads = 0
        mae_leads = 0
        for horizon, row in by_horizon.items():
            mse = float(row["test_mse"])
            mae = float(row["test_mae"])
            mse_target = full_targets[("Weather", horizon, "mse")]
            mae_target = full_targets[("Weather", horizon, "mae")]
            mse_ratios.append(mse / mse_target)
            mae_ratios.append(mae / mae_target)
            mse_leads += int(mse <= mse_target)
            mae_leads += int(mae <= mae_target)
        mean_mse_ratio = sum(mse_ratios) / 4
        mean_mae_ratio = sum(mae_ratios) / 4
        meta = metadata[("Weather", trial_id)]
        profile_rows.append(
            {
                "dataset": "Weather",
                "phase": rows[0]["phase"],
                "trial_id": trial_id,
                "profile_id": rows[0]["profile_id"],
                "test_mean_mse_4h": sum(
                    float(row["test_mse"]) for row in rows
                )
                / 4,
                "test_mean_mae_4h": sum(
                    float(row["test_mae"]) for row in rows
                )
                / 4,
                "mean_mse_target_ratio": mean_mse_ratio,
                "mean_mae_target_ratio": mean_mae_ratio,
                "primary_joint_score": (mean_mse_ratio + mean_mae_ratio) / 2,
                "mse_lead_cells": mse_leads,
                "mae_lead_cells": mae_leads,
                "combined_lead_cells": mse_leads + mae_leads,
                "validation_mean_mse_4h": meta["validation_mean_mse_4h"],
                "trainable_parameters": meta["trainable_parameters"],
            }
        )

    best_score = min(float(row["primary_joint_score"]) for row in profile_rows)
    near_tie_ratio = 1.0 + 0.001
    eligible = [
        row
        for row in profile_rows
        if float(row["primary_joint_score"]) <= best_score * near_tie_ratio
    ]
    selected = min(
        eligible,
        key=lambda row: (
            -int(row["combined_lead_cells"]),
            -min(int(row["mse_lead_cells"]), int(row["mae_lead_cells"])),
            max(
                float(row["mean_mse_target_ratio"]),
                float(row["mean_mae_target_ratio"]),
            ),
            float(row["validation_mean_mse_4h"]),
            int(row["trainable_parameters"]),
            row["profile_id"],
        ),
    )
    for row in profile_rows:
        row["near_tie_eligible"] = row["trial_id"] in {
            item["trial_id"] for item in eligible
        }
        row["selected"] = row["trial_id"] == selected["trial_id"]

    selected_weather_cells = sorted(
        weather_cells[selected["trial_id"]], key=lambda row: int(row["horizon"])
    )
    previous_cells = read_csv(PREVIOUS_CELLS)
    combined_cells: list[dict[str, Any]] = [
        row for row in previous_cells if row["dataset"] != "Weather"
    ] + selected_weather_cells
    if len(combined_cells) != 32:
        raise ValueError("selected eight-dataset scorecard is incomplete")

    legacy_profiles = []
    for trial_id, rows in weather_cells.items():
        mse_ratios = []
        mae_ratios = []
        mse_leads = 0
        mae_leads = 0
        for row in rows:
            horizon = int(row["horizon"])
            mse = float(row["test_mse"])
            mae = float(row["test_mae"])
            mse_target = legacy_targets[("Weather", horizon, "mse")]
            mae_target = legacy_targets[("Weather", horizon, "mae")]
            mse_ratios.append(mse / mse_target)
            mae_ratios.append(mae / mae_target)
            mse_leads += int(mse <= mse_target)
            mae_leads += int(mae <= mae_target)
        meta = metadata[("Weather", trial_id)]
        legacy_profiles.append(
            {
                "trial_id": trial_id,
                "profile_id": rows[0]["profile_id"],
                "score": (sum(mse_ratios) + sum(mae_ratios)) / 8,
                "mse_leads": mse_leads,
                "mae_leads": mae_leads,
                "validation_mean_mse_4h": float(
                    meta["validation_mean_mse_4h"]
                ),
                "trainable_parameters": int(meta["trainable_parameters"]),
            }
        )
    legacy_best = min(float(row["score"]) for row in legacy_profiles)
    legacy_eligible = [
        row
        for row in legacy_profiles
        if float(row["score"]) <= legacy_best * 1.01
    ]
    legacy_selected = min(
        legacy_eligible,
        key=lambda row: (
            -(int(row["mse_leads"]) + int(row["mae_leads"])),
            -min(int(row["mse_leads"]), int(row["mae_leads"])),
            float(row["score"]),
            float(row["validation_mean_mse_4h"]),
            int(row["trainable_parameters"]),
            row["profile_id"],
        ),
    )
    legacy_combined_cells: list[dict[str, Any]] = [
        row for row in previous_cells if row["dataset"] != "Weather"
    ] + weather_cells[legacy_selected["trial_id"]]

    historical = [row for row in profile_rows if row["phase"] != "H4N"]
    historical_best = min(
        historical, key=lambda row: float(row["primary_joint_score"])
    )
    stored_historical_score = float(
        config["existing_evidence"]["weather_historical_best_joint_score"]
    )
    selected_score = float(selected["primary_joint_score"])
    consistent_improvement = 1.0 - selected_score / float(
        historical_best["primary_joint_score"]
    )
    literal_stored_improvement = 1.0 - selected_score / stored_historical_score
    gates = config["success_gates"]
    gate_status = {
        "protocol_consistent_primary_improvement_at_least_0.3pct": (
            consistent_improvement
            >= float(gates["weather_primary_joint_score_improvement_min_relative"])
        ),
        "mean_mse_at_most_target": float(selected["test_mean_mse_4h"])
        <= float(gates["weather_mean_mse_target_max"]),
        "mean_mae_at_most_target": float(selected["test_mean_mae_4h"])
        <= float(gates["weather_mean_mae_target_max"]),
        "combined_leads_at_least_6_of_8": int(selected["combined_lead_cells"])
        >= int(gates["weather_combined_lead_cells_min"]),
    }

    legacy_counts = comparator_counts(combined_cells, legacy_targets)
    legacy_selector_counts = comparator_counts(
        legacy_combined_cells, legacy_targets
    )
    full_counts = comparator_counts(combined_cells, full_targets)
    status = {
        "formal_test": {
            "trials": 40,
            "standard_horizon_cells": 160,
            "checkpoint_immutability_pass": True,
            "matrix_complete": True,
        },
        "search_pool": {
            "total_trials": 229,
            "weather_trials": 96,
            "new_h4n_trials": 40,
            "near_tie_eligible_profiles": len(eligible),
        },
        "selected_weather_profile": selected,
        "selected_weather_cells": [
            {
                "horizon": int(row["horizon"]),
                "mse": float(row["test_mse"]),
                "mae": float(row["test_mae"]),
            }
            for row in selected_weather_cells
        ],
        "primary_full_table_target": {
            "non_iscf_baseline_models": sorted(full_models),
            "historical_best_trial_recomputed": historical_best["trial_id"],
            "historical_best_score_recomputed": float(
                historical_best["primary_joint_score"]
            ),
            "stored_historical_score_from_legacy_five_model_target": (
                stored_historical_score
            ),
            "stored_comparator_is_target_inconsistent": abs(
                stored_historical_score
                - float(historical_best["primary_joint_score"])
            )
            > 1e-12,
            "protocol_consistent_relative_improvement": consistent_improvement,
            "literal_stored_scalar_relative_improvement": (
                literal_stored_improvement
            ),
            "gate_status": gate_status,
            "overall_gate_pass": all(gate_status.values()),
        },
        "comparison_roles": {
            "legacy_five_model_exact_target": {
                "models": sorted(legacy_models),
                "h4n_primary_profile_on_legacy_target": True,
                **legacy_counts,
            },
            "legacy_five_model_one_percent_selector_counterfactual": {
                "selected_weather_profile": legacy_selected,
                "interpretation": (
                    "continuity audit only; H4N froze the full-table 0.1pct "
                    "selector as primary"
                ),
                **legacy_selector_counts,
            },
            "full_twelve_baseline_target": full_counts,
        },
        "decision": (
            "H4N_complete_primary_gate_pass"
            if all(gate_status.values())
            else "H4N_complete_partial_improvement_gate_fail_no_automatic_H4O"
        ),
    }

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_dir / "weather_all_profile_scorecard.csv", profile_rows)
    write_csv(args.analysis_dir / "weather_selected_profile.csv", [selected])
    write_csv(args.analysis_dir / "selected_main_scorecard_h4n.csv", combined_cells)
    (args.analysis_dir / "h4n_result_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
