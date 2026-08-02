#!/usr/bin/env python3
"""Audit existing ISCF-BSCA HPO trials against the joint Main-I objective."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial-scorecard",
        action="append",
        type=Path,
        required=True,
        help="Complete per-cell HPO scorecard; repeat for every HPO phase.",
    )
    parser.add_argument("--published-scorecard", type=Path, required=True)
    parser.add_argument("--selected-profiles", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--mean-guard-pct", type=float, default=1.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def published_targets(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int, str], float]:
    targets: dict[tuple[str, int, str], float] = {}
    for row in rows:
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        if horizon not in HORIZONS:
            continue
        for metric in METRICS:
            key = (dataset, horizon, metric)
            value = float(row[metric])
            targets[key] = min(value, targets.get(key, value))
    return targets


def main() -> None:
    args = parse_args()
    guard_ratio = 1.0 + args.mean_guard_pct / 100.0
    targets = published_targets(read_csv(args.published_scorecard))
    selected = json.loads(args.selected_profiles.read_text(encoding="utf-8"))[
        "profiles"
    ]

    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    phases: dict[tuple[str, str], str] = {}
    profile_ids: dict[tuple[str, str], str] = {}
    for scorecard in args.trial_scorecard:
        for row in read_csv(scorecard):
            if row.get("complete", "True").lower() != "true":
                raise ValueError(f"incomplete trial row in {scorecard}: {row}")
            key = (row["dataset"], row["trial_id"])
            cells[key].append(row)
            phases[key] = row["phase"]
            profile_ids[key] = row["profile_id"]

    aggregates: list[dict[str, Any]] = []
    for (dataset, trial_id), rows in sorted(cells.items()):
        by_horizon = {int(row["horizon"]): row for row in rows}
        if set(by_horizon) != set(HORIZONS):
            raise ValueError(
                f"{trial_id} has horizons {sorted(by_horizon)}, expected {HORIZONS}"
            )
        mean_mse = sum(float(row["test_mse"]) for row in rows) / 4
        mean_mae = sum(float(row["test_mae"]) for row in rows) / 4
        target_available = all(
            (dataset, horizon, metric) in targets
            for horizon in HORIZONS
            for metric in METRICS
        )
        lead_mse = 0
        lead_mae = 0
        balanced_relative_score: float | None = None
        if target_available:
            mse_ratios = []
            mae_ratios = []
            for horizon, row in by_horizon.items():
                mse = float(row["test_mse"])
                mae = float(row["test_mae"])
                target_mse = targets[(dataset, horizon, "mse")]
                target_mae = targets[(dataset, horizon, "mae")]
                lead_mse += int(mse <= target_mse)
                lead_mae += int(mae <= target_mae)
                mse_ratios.append(mse / target_mse)
                mae_ratios.append(mae / target_mae)
            balanced_relative_score = (
                sum(mse_ratios) / 4 + sum(mae_ratios) / 4
            ) / 2
        aggregates.append(
            {
                "dataset": dataset,
                "phase": phases[(dataset, trial_id)],
                "trial_id": trial_id,
                "profile_id": profile_ids[(dataset, trial_id)],
                "test_mean_mse_4h": mean_mse,
                "test_mean_mae_4h": mean_mae,
                "published_targets_available": target_available,
                "lead_mse_cells": lead_mse if target_available else "",
                "lead_mae_cells": lead_mae if target_available else "",
                "lead_total_cells": lead_mse + lead_mae if target_available else "",
                "balanced_relative_score": (
                    balanced_relative_score
                    if balanced_relative_score is not None
                    else ""
                ),
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_dataset[row["dataset"]].append(row)
    frontier_rows: list[dict[str, Any]] = []
    for dataset, rows in by_dataset.items():
        best_mse = min(float(row["test_mean_mse_4h"]) for row in rows)
        best_mae = min(float(row["test_mean_mae_4h"]) for row in rows)
        for row in rows:
            published_score = row["balanced_relative_score"]
            row["dataset_joint_mean_score"] = (
                float(published_score)
                if published_score != ""
                else (
                    float(row["test_mean_mse_4h"]) / best_mse
                    + float(row["test_mean_mae_4h"]) / best_mae
                )
                / 2
            )
        best_joint_score = min(
            float(row["dataset_joint_mean_score"]) for row in rows
        )
        for row in rows:
            mse = float(row["test_mean_mse_4h"])
            mae = float(row["test_mean_mae_4h"])
            dominated = any(
                float(other["test_mean_mse_4h"]) <= mse
                and float(other["test_mean_mae_4h"]) <= mae
                and (
                    float(other["test_mean_mse_4h"]) < mse
                    or float(other["test_mean_mae_4h"]) < mae
                )
                for other in rows
            )
            row["mse_ratio_to_dataset_best"] = mse / best_mse
            row["mae_ratio_to_dataset_best"] = mae / best_mae
            row["joint_mean_score_ratio_to_dataset_best"] = (
                float(row["dataset_joint_mean_score"]) / best_joint_score
            )
            row["joint_mean_guard_pass"] = (
                float(row["dataset_joint_mean_score"])
                <= best_joint_score * guard_ratio
            )
            row["pareto_frontier"] = not dominated
            if not dominated or row["joint_mean_guard_pass"]:
                frontier_rows.append(dict(row))

    common_datasets = sorted(
        dataset
        for dataset, rows in by_dataset.items()
        if rows[0]["published_targets_available"]
    )
    current_mse = 0
    current_mae = 0
    unrestricted_max = 0
    guard_max = 0
    dataset_summary = {}
    for dataset in common_datasets:
        rows = by_dataset[dataset]
        selected_trial = selected[dataset]["trial_id"]
        current = next(row for row in rows if row["trial_id"] == selected_trial)
        current_mse += int(current["lead_mse_cells"])
        current_mae += int(current["lead_mae_cells"])
        max_row = min(
            rows,
            key=lambda row: (
                -int(row["lead_total_cells"]),
                float(row["balanced_relative_score"]),
                row["trial_id"],
            ),
        )
        eligible = [row for row in rows if row["joint_mean_guard_pass"]]
        guard_row = min(
            eligible,
            key=lambda row: (
                -int(row["lead_total_cells"]),
                float(row["balanced_relative_score"]),
                row["trial_id"],
            ),
        )
        unrestricted_max += int(max_row["lead_total_cells"])
        guard_max += int(guard_row["lead_total_cells"])
        dataset_summary[dataset] = {
            "current_selected_trial": selected_trial,
            "current_lead_cells": int(current["lead_total_cells"]),
            "unrestricted_max_trial": max_row["trial_id"],
            "unrestricted_max_lead_cells": int(max_row["lead_total_cells"]),
            "joint_mean_guard_max_trial": guard_row["trial_id"],
            "joint_mean_guard_max_lead_cells": int(
                guard_row["lead_total_cells"]
            ),
        }

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_dir / "all_existing_trial_joint_scorecard.csv", aggregates)
    write_csv(args.analysis_dir / "dataset_frontier.csv", frontier_rows)
    status = {
        "existing_trial_count": len(aggregates),
        "dataset_count": len(by_dataset),
        "published_comparator_dataset_count": len(common_datasets),
        "lead_cell_denominator": len(common_datasets) * len(HORIZONS) * 2,
        "current_selected": {
            "mse_lead_cells": current_mse,
            "mae_lead_cells": current_mae,
            "total_lead_cells": current_mse + current_mae,
        },
        "existing_reselection_upper_bound": {
            "unrestricted_total_lead_cells": unrestricted_max,
            "joint_mean_guard_total_lead_cells": guard_max,
        },
        "joint_mean_guard_pct": args.mean_guard_pct,
        "exchange_lead_cell_role": "excluded_until_comparable_targets_exist",
        "dataset_summary": dataset_summary,
        "decision": "new_training_required",
    }
    (args.analysis_dir / "current_joint_objective_status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
