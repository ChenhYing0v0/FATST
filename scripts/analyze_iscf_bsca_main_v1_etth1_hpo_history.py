#!/usr/bin/env python3
"""Consolidate all completed ETTh1 HPO trials before H5D design."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from analyze_iscf_bsca_main_v1_hpo import materialize_jobs


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")
CONFIGS = (
    ("H1", "configs/iscf_bsca_main_v1_hpo.json"),
    ("H2", "configs/iscf_bsca_main_v1_hpo_h2.json"),
    ("H4J", "configs/iscf_bsca_main_v1_hpo_joint_h4j.json"),
    ("H4K", "configs/iscf_bsca_main_v1_hpo_targeted_h4k.json"),
    ("H5A", "configs/iscf_bsca_main_v1_hpo_main_ii_h5a.json"),
    ("H5B", "configs/iscf_bsca_main_v1_hpo_etth1_h5b.json"),
    ("H5C", "configs/iscf_bsca_main_v1_hpo_etth1_h5c.json"),
)
RESULTS = (
    "analysis/iscf_bsca_main_v1_hpo_20260731/test_audit_result/"
    "all_trial_scorecard.csv",
    "analysis/iscf_bsca_main_v1_hpo_20260731/h4j_test_result/"
    "all_trial_scorecard.csv",
    "analysis/iscf_bsca_main_v1_hpo_20260731/h4k_test_result/"
    "all_trial_scorecard.csv",
    "analysis/iscf_bsca_main_v1_hpo_20260731/"
    "h5a_formal_test_result_20260813/all_trial_scorecard.csv",
    "analysis/iscf_bsca_main_v1_hpo_20260731/"
    "h5b_formal_test_result_20260813/all_trial_scorecard.csv",
    "analysis/iscf_bsca_main_v1_hpo_20260731/"
    "h5c_formal_test_result_20260813/all_trial_scorecard.csv",
)
MAIN_II = (
    "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
    "main_ii_h720_prefix_20260808/formal_results_20260813/table/"
    "table_data_long.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float) -> Decimal:
    return Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )


def comparison_thresholds() -> dict[tuple[int, str], Decimal]:
    values: dict[tuple[int, str], list[Decimal]] = defaultdict(list)
    for row in read_csv(ROOT / MAIN_II):
        if (
            row["dataset"] != "ETTh1"
            or row["horizon"] == "Avg."
            or row["system"] == "ISCF-BSCA-MAIN-v1"
        ):
            continue
        for metric in METRICS:
            values[(int(row["horizon"]), metric)].append(
                rounded(float(row[metric]))
            )
    expected = {
        (horizon, metric) for horizon in HORIZONS for metric in METRICS
    }
    if set(values) != expected:
        raise ValueError("incomplete frozen Main II ETTh1 comparison surface")
    return {key: min(items) for key, items in values.items()}


def main() -> None:
    args = parse_args()
    jobs: dict[str, dict[str, Any]] = {}
    phase_counts: Counter[str] = Counter()
    for phase, relative in CONFIGS:
        path = ROOT / relative
        config = json.loads(path.read_text(encoding="utf-8"))
        for materialized in materialize_jobs(config, path):
            if materialized["dataset"] != "ETTh1":
                continue
            trial_id = materialized["trial_id"]
            if trial_id in jobs:
                raise ValueError(f"duplicate historical trial: {trial_id}")
            job = dict(materialized)
            job["phase"] = phase
            job["max_epochs"] = job.get(
                "max_epochs", config["training"]["max_epochs"]
            )
            job["early_stopping_patience"] = job.get(
                "early_stopping_patience",
                config["training"]["early_stopping_patience"],
            )
            jobs[trial_id] = job
            phase_counts[phase] += 1

    cells: dict[tuple[str, int], dict[str, str]] = {}
    for relative in RESULTS:
        for row in read_csv(ROOT / relative):
            if row["dataset"] != "ETTh1":
                continue
            key = (row["trial_id"], int(row["horizon"]))
            if key in cells:
                raise ValueError(f"duplicate result cell: {key}")
            cells[key] = row

    thresholds = comparison_thresholds()
    rows: list[dict[str, Any]] = []
    for trial_id, job in jobs.items():
        trial_cells = {horizon: cells[(trial_id, horizon)] for horizon in HORIZONS}
        row: dict[str, Any] = {
            "phase": job["phase"],
            "trial_id": trial_id,
            "profile_id": job["profile_id"],
            "source_prior": job["source_prior"],
            "seq_len": job["seq_len"],
            "patch_num": job["patch_num"],
            "patch_len": job["seq_len"] / job["patch_num"],
            "d_model": job["d_model"],
            "d_ff": job["d_ff"],
            "dropout": job["dropout"],
            "learning_rate": job["learning_rate"],
            "weight_decay": job["weight_decay"],
            "batch_size": job["batch_size"],
            "gradient_accumulation_steps": job[
                "gradient_accumulation_steps"
            ],
            "mode_rank": job["mode_rank"],
            "layer_norm": job.get("layer_norm", 1),
            "max_epochs": job["max_epochs"],
            "early_stopping_patience": job["early_stopping_patience"],
        }
        best_cells = 0
        for horizon in HORIZONS:
            for metric in METRICS:
                value = float(trial_cells[horizon][f"test_{metric}"])
                row[f"h{horizon}_{metric}"] = value
                best_cells += int(
                    rounded(value) <= thresholds[(horizon, metric)]
                )
        row["mean_mse_4h"] = sum(
            float(trial_cells[horizon]["test_mse"])
            for horizon in HORIZONS
        ) / len(HORIZONS)
        row["mean_mae_4h"] = sum(
            float(trial_cells[horizon]["test_mae"])
            for horizon in HORIZONS
        ) / len(HORIZONS)
        row["main_ii_best_cells"] = best_cells
        row["h192_normalized_gap"] = sum(
            row[f"h192_{metric}"] / float(thresholds[(192, metric)]) - 1
            for metric in METRICS
        ) / len(METRICS)
        row["h336_normalized_gap"] = sum(
            row[f"h336_{metric}"] / float(thresholds[(336, metric)]) - 1
            for metric in METRICS
        ) / len(METRICS)
        rows.append(row)

    if len(rows) != 115:
        raise ValueError(f"expected 115 ETTh1 profiles, got {len(rows)}")
    rows.sort(key=lambda item: (item["phase"], item["trial_id"]))

    h5c_groups: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "H5C":
            grouped[str(row["source_prior"])].append(row)
    for group, items in sorted(grouped.items()):
        h5c_groups.append(
            {
                "source_prior": group,
                "profiles": len(items),
                "minimum_mean_mse_4h": min(
                    float(item["mean_mse_4h"]) for item in items
                ),
                "minimum_mean_mae_4h": min(
                    float(item["mean_mae_4h"]) for item in items
                ),
                "minimum_h192_normalized_gap": min(
                    float(item["h192_normalized_gap"]) for item in items
                ),
                "minimum_h336_normalized_gap": min(
                    float(item["h336_normalized_gap"]) for item in items
                ),
                "maximum_main_ii_best_cells": max(
                    int(item["main_ii_best_cells"]) for item in items
                ),
            }
        )

    current = next(
        row for row in rows if row["trial_id"] == "ETTh1__h5b_seq640_p20"
    )
    best_mean = min(rows, key=lambda item: float(item["mean_mse_4h"]))
    best_h192 = min(
        rows, key=lambda item: float(item["h192_normalized_gap"])
    )
    best_h336 = min(
        rows, key=lambda item: float(item["h336_normalized_gap"])
    )
    best_count_distribution = {
        phase: dict(sorted(Counter(
            int(row["main_ii_best_cells"])
            for row in rows
            if row["phase"] == phase
        ).items()))
        for phase in phase_counts
    }
    summary = {
        "profiles": len(rows),
        "profiles_by_phase": dict(phase_counts),
        "complete_standard_cells": len(cells),
        "main_ii_thresholds_after_three_decimal_rounding": {
            f"H{horizon}_{metric.upper()}": str(thresholds[(horizon, metric)])
            for horizon in HORIZONS
            for metric in METRICS
        },
        "current_h5b": current,
        "best_mean_mse_profile": best_mean,
        "best_h192_joint_gap_profile": best_h192,
        "best_h336_joint_gap_profile": best_h336,
        "best_cell_count_distribution_by_phase": best_count_distribution,
        "design_decision": (
            "H5D_focus_batch_lr_dropout0_and_"
            "p19_p21_geometry_rank_dropout0_interactions"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_etth1_profiles.csv", rows)
    write_csv(args.output_dir / "h5c_group_summary.csv", h5c_groups)
    (args.output_dir / "history_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
