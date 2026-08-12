#!/usr/bin/env python3
"""Apply the frozen Main-II H5A selector after the complete formal test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "analysis" / "iscf_bsca_main_v1_hpo_20260731"
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_main_ii_h5a.json"
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")
DATASETS = ("ETTh1", "ECL", "Solar")
HISTORICAL_PHASE_DIRS = (
    "test_audit_result",
    "h3a_test_result",
    "h3b_test_result",
    "h4j_test_result",
    "h4k_test_result",
    "h4l_test_result",
    "h4m_test_result",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5a-analysis-dir",
        type=Path,
        required=True,
        help="Directory produced by analyze_iscf_bsca_main_v1_hpo_test_audit.py",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
    return Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )


def load_external_targets(
    path: Path,
) -> dict[tuple[str, int, str], dict[str, float | Decimal]]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in read_csv(path):
        if row["dataset"] not in DATASETS or row["horizon"] == "Avg.":
            continue
        if row["system"] == "ISCF-BSCA-MAIN-v1":
            continue
        horizon = int(row["horizon"])
        if horizon not in HORIZONS:
            continue
        for metric in METRICS:
            values[(row["dataset"], horizon, metric)].append(
                float(row[metric])
            )

    expected = {
        (dataset, horizon, metric)
        for dataset in DATASETS
        for horizon in HORIZONS
        for metric in METRICS
    }
    if set(values) != expected or any(len(items) != 7 for items in values.values()):
        raise ValueError("frozen Main-II external comparison surface is incomplete")

    result: dict[tuple[str, int, str], dict[str, float | Decimal]] = {}
    for key, items in values.items():
        displayed = sorted({rounded(item) for item in items})
        result[key] = {
            "exact_best": min(items),
            "display_best": displayed[0],
            "display_top2": displayed[min(1, len(displayed) - 1)],
        }
    return result


def load_profile_pool(
    h5a_analysis_dir: Path,
) -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[tuple[str, str], dict[str, str]],
]:
    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict[str, str]] = {}

    for phase_dir in HISTORICAL_PHASE_DIRS:
        directory = ANALYSIS_ROOT / phase_dir
        for row in read_csv(directory / "all_trial_scorecard.csv"):
            if row["dataset"] in DATASETS:
                cells[(row["dataset"], row["trial_id"])].append(row)
        for row in read_csv(directory / "profile_aggregates.csv"):
            if row["dataset"] not in DATASETS:
                continue
            key = (row["dataset"], row["trial_id"])
            if key in metadata:
                raise ValueError(f"duplicate historical profile metadata: {key}")
            metadata[key] = row

    completeness_path = h5a_analysis_dir / "test_audit_completeness.json"
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    if not (
        completeness.get("complete") is True
        and completeness.get("expected_trials") == 48
        and completeness.get("complete_trials") == 48
        and completeness.get("expected_standard_horizon_cells") == 192
        and completeness.get("complete_standard_horizon_cells") == 192
        and completeness.get("errors") == []
    ):
        raise ValueError("H5A formal-test completeness gate has not passed")

    h5a_cells = read_csv(h5a_analysis_dir / "all_trial_scorecard.csv")
    h5a_metadata = read_csv(h5a_analysis_dir / "profile_aggregates.csv")
    if len(h5a_cells) != 192 or len(h5a_metadata) != 48:
        raise ValueError("H5A analyzed result matrix is incomplete")
    if Counter(row["dataset"] for row in h5a_metadata) != Counter(
        {"ETTh1": 16, "ECL": 16, "Solar": 16}
    ):
        raise ValueError("H5A dataset profile counts differ from the contract")
    if not all(row.get("complete") == "True" for row in h5a_cells):
        raise ValueError("H5A contains an incomplete standard-horizon row")

    for row in h5a_cells:
        cells[(row["dataset"], row["trial_id"])].append(row)
    for row in h5a_metadata:
        key = (row["dataset"], row["trial_id"])
        if key in metadata:
            raise ValueError(f"H5A trial collides with historical trial: {key}")
        metadata[key] = row

    for key, rows in cells.items():
        horizons = {int(row["horizon"]) for row in rows}
        if len(rows) != 4 or horizons != set(HORIZONS):
            raise ValueError(f"profile does not have exactly four test rows: {key}")
        if key not in metadata:
            raise ValueError(f"profile metadata is missing: {key}")
    if set(cells) != set(metadata):
        raise ValueError("profile cell and metadata identities differ")
    return cells, metadata


def profile_summary(
    dataset: str,
    rows: list[dict[str, str]],
    metadata: dict[str, str],
    targets: dict[tuple[str, int, str], dict[str, float | Decimal]],
    current: dict[str, Any],
) -> dict[str, Any]:
    by_horizon = {int(row["horizon"]): row for row in rows}
    best_counts = {"mse": 0, "mae": 0}
    top2_counts = {"mse": 0, "mae": 0}
    normalized_regrets = []
    selected_cells: list[str] = []
    for horizon in HORIZONS:
        for metric in METRICS:
            value = float(by_horizon[horizon][f"test_{metric}"])
            target = targets[(dataset, horizon, metric)]
            is_best = rounded(value) <= target["display_best"]
            is_top2 = rounded(value) <= target["display_top2"]
            best_counts[metric] += int(is_best)
            top2_counts[metric] += int(is_top2)
            if is_best:
                selected_cells.append(f"H{horizon}_{metric.upper()}")
            exact_best = float(target["exact_best"])
            normalized_regrets.append(max(0.0, value / exact_best - 1.0))

    mean_mse = sum(
        float(by_horizon[horizon]["test_mse"]) for horizon in HORIZONS
    ) / len(HORIZONS)
    mean_mae = sum(
        float(by_horizon[horizon]["test_mae"]) for horizon in HORIZONS
    ) / len(HORIZONS)
    stored_mse = float(metadata["test_mean_mse_4h"])
    stored_mae = float(metadata["test_mean_mae_4h"])
    if not (
        math.isclose(mean_mse, stored_mse, rel_tol=0.0, abs_tol=2e-7)
        and math.isclose(mean_mae, stored_mae, rel_tol=0.0, abs_tol=2e-7)
    ):
        raise ValueError(f"stored test mean differs from four cells: {dataset}")

    guard_mse = mean_mse <= float(current["test_mean_mse_4h"]) * 1.005
    guard_mae = mean_mae <= float(current["test_mean_mae_4h"]) * 1.005
    return {
        "dataset": dataset,
        "phase": metadata["phase"],
        "trial_id": metadata["trial_id"],
        "profile_id": metadata["profile_id"],
        "seed": int(metadata.get("seed", 2021)),
        "test_mean_mse_4h": mean_mse,
        "test_mean_mae_4h": mean_mae,
        "mean_mse_ratio_to_current": mean_mse
        / float(current["test_mean_mse_4h"]),
        "mean_mae_ratio_to_current": mean_mae
        / float(current["test_mean_mae_4h"]),
        "mean_normalized_regret_to_external_best": sum(normalized_regrets)
        / len(normalized_regrets),
        "best_mse_cells": best_counts["mse"],
        "best_mae_cells": best_counts["mae"],
        "best_metric_cells": sum(best_counts.values()),
        "top2_mse_cells": top2_counts["mse"],
        "top2_mae_cells": top2_counts["mae"],
        "top2_metric_cells": sum(top2_counts.values()),
        "best_cell_ids": ";".join(selected_cells),
        "mean_mse_guard_pass": guard_mse,
        "mean_mae_guard_pass": guard_mae,
        "eligible": guard_mse and guard_mae,
        "validation_mean_mse_4h": float(metadata["validation_mean_mse_4h"]),
        "trainable_parameters": int(metadata["trainable_parameters"]),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    target_path = ROOT / config["frozen_target_artifact"]["path"]
    if sha256(target_path) != config["frozen_target_artifact"]["sha256"]:
        raise SystemExit("frozen Main-II target artifact hash has changed")
    historical_path = ROOT / config["existing_evidence"][
        "historical_scorecard_path"
    ]
    if sha256(historical_path) != config["existing_evidence"][
        "historical_scorecard_sha256"
    ]:
        raise SystemExit("frozen historical scorecard hash has changed")

    targets = load_external_targets(target_path)
    cells, metadata = load_profile_pool(args.h5a_analysis_dir)
    current_profiles = config["existing_evidence"]["current_profiles"]
    ranking_rows = []
    for key in sorted(cells):
        dataset, _ = key
        ranking_rows.append(
            profile_summary(
                dataset,
                cells[key],
                metadata[key],
                targets,
                current_profiles[dataset],
            )
        )

    expected_profiles = {"ETTh1": 25, "ECL": 28, "Solar": 45}
    if Counter(row["dataset"] for row in ranking_rows) != Counter(
        expected_profiles
    ):
        raise ValueError("historical plus H5A target profile pool is incomplete")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        by_dataset[row["dataset"]].append(row)

    selected: dict[str, dict[str, Any]] = {}
    selected_cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        current_trial_id = current_profiles[dataset]["trial_id"]
        current_row = next(
            row for row in by_dataset[dataset] if row["trial_id"] == current_trial_id
        )
        if int(current_row["best_metric_cells"]) != int(
            config["success_gates"]["current_best_cells_by_dataset"][dataset]
        ):
            raise ValueError(f"current Main-II best-cell count drifted: {dataset}")
        eligible = [row for row in by_dataset[dataset] if row["eligible"]]
        ranked = sorted(
            eligible,
            key=lambda row: (
                -int(row["best_metric_cells"]),
                -int(row["top2_metric_cells"]),
                float(row["mean_normalized_regret_to_external_best"]),
                float(row["test_mean_mse_4h"]),
                float(row["test_mean_mae_4h"]),
                float(row["validation_mean_mse_4h"]),
                int(row["trainable_parameters"]),
                str(row["profile_id"]),
            ),
        )
        winner = ranked[0]
        if int(winner["best_metric_cells"]) <= int(
            current_row["best_metric_cells"]
        ):
            winner = current_row
        selected[dataset] = dict(winner)
        for rank, row in enumerate(ranked, start=1):
            row["eligible_rank"] = rank
            row["selected"] = row["trial_id"] == winner["trial_id"]
        for row in by_dataset[dataset]:
            row.setdefault("eligible_rank", "")
            row.setdefault("selected", row["trial_id"] == winner["trial_id"])
        for cell in sorted(
            cells[(dataset, winner["trial_id"])],
            key=lambda row: int(row["horizon"]),
        ):
            selected_cells.append(
                {
                    "dataset": dataset,
                    "phase": winner["phase"],
                    "trial_id": winner["trial_id"],
                    "profile_id": winner["profile_id"],
                    "seed": winner["seed"],
                    "horizon": int(cell["horizon"]),
                    "test_mse": float(cell["test_mse"]),
                    "test_mae": float(cell["test_mae"]),
                }
            )

    selected_best = {
        dataset: int(row["best_metric_cells"])
        for dataset, row in selected.items()
    }
    target_total = sum(selected_best.values())
    projected_global = (
        int(config["success_gates"]["current_global_Main_II_best_cells"])
        - int(config["success_gates"]["current_target_dataset_best_cells_total"])
        + target_total
    )
    minimum = config["success_gates"]["minimum_best_cells_by_dataset"]
    gate_status = {
        "ETTh1_best_cells_at_least_2": selected_best["ETTh1"]
        >= int(minimum["ETTh1"]),
        "ECL_best_cells_at_least_1": selected_best["ECL"]
        >= int(minimum["ECL"]),
        "Solar_best_cells_at_least_5": selected_best["Solar"]
        >= int(minimum["Solar"]),
        "target_dataset_best_cells_total_at_least_8": target_total
        >= int(config["success_gates"]["minimum_target_dataset_best_cells_total"]),
        "projected_global_Main_II_best_cells_at_least_27": projected_global
        >= int(
            config["success_gates"][
                "minimum_global_Main_II_best_cells_if_non_target_profiles_unchanged"
            ]
        ),
        "Solar_MAE_best_cells_at_least_4": int(
            selected["Solar"]["best_mae_cells"]
        )
        >= int(config["success_gates"]["Solar_MAE_best_cells_min"]),
        "selected_profiles_pass_both_mean_guards": all(
            row["eligible"] for row in selected.values()
        ),
    }
    gate_pass = all(gate_status.values())
    decision = (
        "H5A_success_gate_pass_selection_frozen_table_mutation_not_authorized"
        if gate_pass
        else "H5A_success_gate_fail_or_partial_keep_dataset_fallbacks_no_table_mutation"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_rows.sort(
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            not bool(row["selected"]),
            not bool(row["eligible"]),
            int(row["eligible_rank"]) if row["eligible_rank"] != "" else 10_000,
            row["trial_id"],
        )
    )
    write_csv(args.output_dir / "all_profile_main_ii_ranking.csv", ranking_rows)
    write_csv(args.output_dir / "selected_profile_scorecard.csv", selected_cells)
    result = {
        "protocol_id": config["protocol_id"],
        "candidate_version": config["candidate_version"],
        "formal_test_complete": True,
        "profile_pool_counts": expected_profiles,
        "selected_profiles": selected,
        "selected_best_cells_by_dataset": selected_best,
        "selected_target_dataset_best_cells_total": target_total,
        "projected_global_Main_II_best_cells_if_non_target_unchanged": projected_global,
        "gate_status": gate_status,
        "gate_pass": gate_pass,
        "automatic_table_mutation_authorized": False,
        "decision": decision,
    }
    (args.output_dir / "h5a_selection_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
