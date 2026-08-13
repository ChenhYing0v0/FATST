#!/usr/bin/env python3
"""Apply the frozen ETTh1 H5C selector after the complete formal test."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5c-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)


def load_targets(
    path: Path,
    system_field: str,
) -> dict[tuple[int, str], dict[str, Decimal]]:
    values: dict[tuple[int, str], list[Decimal]] = defaultdict(list)
    for row in read_csv(path):
        if row["dataset"] != "ETTh1" or row["horizon"] == "Avg.":
            continue
        if row[system_field] in {"ISCF-BSCA", "ISCF-BSCA-MAIN-v1"}:
            continue
        for metric in METRICS:
            values[(int(row["horizon"]), metric)].append(
                rounded(float(row[metric]))
            )
    expected = {(horizon, metric) for horizon in HORIZONS for metric in METRICS}
    if set(values) != expected:
        raise ValueError(f"incomplete comparison surface: {path}")
    return {
        key: {
            "best": distinct[0],
            "top2": distinct[min(1, len(distinct) - 1)],
        }
        for key, items in values.items()
        for distinct in [sorted(set(items))]
    }


def summarize(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    main_i: dict[tuple[int, str], dict[str, Decimal]],
    main_ii: dict[tuple[int, str], dict[str, Decimal]],
    current_mse: float,
    current_mae: float,
) -> dict[str, Any]:
    by_horizon = {int(row["horizon"]): row for row in rows}
    if len(by_horizon) != 4 or set(by_horizon) != set(HORIZONS):
        raise ValueError(f"incomplete profile: {metadata['trial_id']}")
    counts = {
        "main_i_best": 0,
        "main_i_top2": 0,
        "main_ii_best": 0,
        "main_ii_top2": 0,
        "main_ii_best_mse": 0,
        "main_ii_best_mae": 0,
    }
    best_ids = []
    for horizon in HORIZONS:
        for metric in METRICS:
            value = rounded(float(by_horizon[horizon][f"test_{metric}"]))
            counts["main_i_best"] += int(
                value <= main_i[(horizon, metric)]["best"]
            )
            counts["main_i_top2"] += int(
                value <= main_i[(horizon, metric)]["top2"]
            )
            is_main_ii_best = value <= main_ii[(horizon, metric)]["best"]
            counts["main_ii_best"] += int(is_main_ii_best)
            counts["main_ii_top2"] += int(
                value <= main_ii[(horizon, metric)]["top2"]
            )
            counts[f"main_ii_best_{metric}"] += int(is_main_ii_best)
            if is_main_ii_best:
                best_ids.append(f"H{horizon}_{metric.upper()}")

    mean_mse = sum(float(by_horizon[h]["test_mse"]) for h in HORIZONS) / 4
    mean_mae = sum(float(by_horizon[h]["test_mae"]) for h in HORIZONS) / 4
    if not (
        math.isclose(
            mean_mse,
            float(metadata["test_mean_mse_4h"]),
            rel_tol=0.0,
            abs_tol=2e-7,
        )
        and math.isclose(
            mean_mae,
            float(metadata["test_mean_mae_4h"]),
            rel_tol=0.0,
            abs_tol=2e-7,
        )
    ):
        raise ValueError(f"stored mean mismatch: {metadata['trial_id']}")
    mse_guard = mean_mse <= current_mse * 1.002
    mae_guard = mean_mae <= current_mae * 1.002
    return {
        "phase": metadata["phase"],
        "dataset": "ETTh1",
        "trial_id": metadata["trial_id"],
        "profile_id": metadata["profile_id"],
        "seed": int(metadata["seed"]),
        "test_mean_mse_4h": mean_mse,
        "test_mean_mae_4h": mean_mae,
        "mean_mse_ratio_to_h5b": mean_mse / current_mse,
        "mean_mae_ratio_to_h5b": mean_mae / current_mae,
        **counts,
        "main_ii_best_cell_ids": ";".join(best_ids),
        "mean_mse_guard_pass": mse_guard,
        "mean_mae_guard_pass": mae_guard,
        "eligible": mse_guard and mae_guard,
        "validation_mean_mse_4h": float(metadata["validation_mean_mse_4h"]),
        "trainable_parameters": int(metadata["trainable_parameters"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(
        (
            ROOT / "configs/iscf_bsca_main_v1_hpo_etth1_h5c_test_audit.json"
        ).read_text(encoding="utf-8")
    )
    completeness = json.loads(
        (args.h5c_analysis_dir / "test_audit_completeness.json").read_text(
            encoding="utf-8"
        )
    )
    if not (
        completeness.get("complete") is True
        and completeness.get("complete_trials") == 54
        and completeness.get("complete_standard_horizon_cells") == 216
        and completeness.get("errors") == []
    ):
        raise ValueError("H5C complete formal-test gate has not passed")

    surfaces = config["frozen_comparison_surfaces"]
    main_i = load_targets(ROOT / surfaces["main_i_table_data"], "model")
    main_ii = load_targets(ROOT / surfaces["main_ii_table_data"], "system")

    h5b_dir = (
        ROOT
        / "analysis/iscf_bsca_main_v1_hpo_20260731/"
        "h5b_formal_test_result_20260813/frozen_selector"
    )
    h5b_result = json.loads(
        (h5b_dir / "h5b_selection_result.json").read_text(encoding="utf-8")
    )
    current_metadata = h5b_result["selected_profile"]
    current_cells = read_csv(h5b_dir / "selected_profile_scorecard.csv")
    current_mse = float(current_metadata["test_mean_mse_4h"])
    current_mae = float(current_metadata["test_mean_mae_4h"])
    current = summarize(
        current_cells,
        current_metadata,
        main_i,
        main_ii,
        current_mse,
        current_mae,
    )
    if current["main_ii_best"] != 4:
        raise ValueError("H5B current Main II best-cell count drifted")

    cells_by_trial: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.h5c_analysis_dir / "all_trial_scorecard.csv"):
        cells_by_trial[row["trial_id"]].append(row)
    metadata_by_trial = {
        row["trial_id"]: row
        for row in read_csv(args.h5c_analysis_dir / "profile_aggregates.csv")
    }
    if len(cells_by_trial) != 54 or len(metadata_by_trial) != 54:
        raise ValueError("H5C profile pool is incomplete")

    candidate_rows = [
        summarize(
            rows,
            metadata_by_trial[trial_id],
            main_i,
            main_ii,
            current_mse,
            current_mae,
        )
        for trial_id, rows in cells_by_trial.items()
    ]
    eligible = [row for row in candidate_rows if row["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -int(row["main_ii_best"]),
            -int(row["main_i_best"]),
            -int(row["main_ii_top2"]),
            float(row["test_mean_mse_4h"]),
            float(row["test_mean_mae_4h"]),
            float(row["validation_mean_mse_4h"]),
            int(row["trainable_parameters"]),
            str(row["profile_id"]),
        ),
    )
    if not ranked:
        raise ValueError("H5C has no profile passing both aggregate guards")
    candidate_winner = ranked[0]
    winner = (
        candidate_winner
        if candidate_winner["main_ii_best"] > current["main_ii_best"]
        else current
    )
    all_rows = [current, *candidate_rows]
    for rank, row in enumerate(ranked, start=1):
        row["eligible_rank"] = rank
    for row in all_rows:
        row.setdefault("eligible_rank", 0 if row is current else "")
        row["selected"] = row["trial_id"] == winner["trial_id"]
    all_rows.sort(
        key=lambda row: (
            not bool(row["selected"]),
            row["eligible_rank"] if row["eligible_rank"] != "" else 10_000,
            row["trial_id"],
        )
    )

    selected_cells = (
        current_cells
        if winner["phase"] == "H5B"
        else cells_by_trial[winner["trial_id"]]
    )
    candidate_cells = cells_by_trial[candidate_winner["trial_id"]]

    def format_cells(
        cells: list[dict[str, Any]], profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "dataset": "ETTh1",
                "phase": profile["phase"],
                "trial_id": profile["trial_id"],
                "profile_id": profile["profile_id"],
                "seed": profile["seed"],
                "horizon": int(row["horizon"]),
                "test_mse": float(row["test_mse"]),
                "test_mae": float(row["test_mae"]),
            }
            for row in sorted(cells, key=lambda item: int(item["horizon"]))
        ]

    output_cells = format_cells(selected_cells, winner)
    output_candidate_cells = format_cells(candidate_cells, candidate_winner)
    gate_pass = winner["main_ii_best"] >= 5 and winner["eligible"]
    result = {
        "protocol_id": config["protocol_id"],
        "candidate_version": config["candidate_version"],
        "formal_test_complete": True,
        "h5c_profiles": 54,
        "h5c_eligible_profiles": len(eligible),
        "h5c_max_main_i_best_cells": max(
            int(row["main_i_best"]) for row in candidate_rows
        ),
        "h5c_max_main_ii_best_cells": max(
            int(row["main_ii_best"]) for row in candidate_rows
        ),
        "h5c_max_main_ii_top2_cells": max(
            int(row["main_ii_top2"]) for row in candidate_rows
        ),
        "current_h5b_profile": current,
        "best_h5c_profile": candidate_winner,
        "selected_profile": winner,
        "minimum_Main_II_best_cells": 5,
        "stretch_Main_II_best_cells": 6,
        "gate_pass": gate_pass,
        "automatic_table_mutation_authorized": False,
        "decision": (
            "H5C_success_gate_pass_selection_frozen_table_mutation_not_authorized"
            if gate_pass
            else "H5C_no_eligible_best_cell_improvement_retain_H5B_profile"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_profile_ranking.csv", all_rows)
    write_csv(args.output_dir / "selected_profile_scorecard.csv", output_cells)
    write_csv(
        args.output_dir / "best_h5c_profile_scorecard.csv",
        output_candidate_cells,
    )
    (args.output_dir / "h5c_selection_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
