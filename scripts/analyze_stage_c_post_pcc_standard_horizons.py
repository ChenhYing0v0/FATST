#!/usr/bin/env python3
"""Re-evaluate the SIFF/MCCA screen on paper-facing standard horizons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192, 336, 720)
NEW_ARMS = (
    "pcsd_mcca",
    "siff_equal",
    "siff_pcc",
    "siff_mcca",
    "siff_const_mcca",
    "siff_permuted_mcca",
    "pcsd_q1_wide_mcca",
    "independent_scope_matched_mcca",
    "dense_siff_matched",
    "pcsd_pointwise_mcca",
    "pcsd_uniform_balanced_ot",
)
PCSD_REFERENCES = ("a6", "pcsd_direct", "dense_matched")
PCC_REFERENCES = ("equal_skill", "pcc_transport_full")
PAIRWISE_EFFECTS = {
    "architecture_equal": ("siff_equal", "equal_skill"),
    "architecture_pcc": ("siff_pcc", "pcc_transport_full"),
    "architecture_mcca": ("siff_mcca", "pcsd_mcca"),
    "mcca_pcsd": ("pcsd_mcca", "pcc_transport_full"),
    "mcca_siff": ("siff_mcca", "siff_pcc"),
    "joint_over_a6": ("siff_mcca", "a6"),
    "ordered_over_constant": ("siff_mcca", "siff_const_mcca"),
    "ordered_over_permuted": ("siff_mcca", "siff_permuted_mcca"),
    "ordered_over_q1_width": ("siff_mcca", "pcsd_q1_wide_mcca"),
    "ordered_over_independent": (
        "siff_mcca",
        "independent_scope_matched_mcca",
    ),
    "ordered_over_dense": ("siff_mcca", "dense_siff_matched"),
    "transport_over_pointwise": ("pcsd_mcca", "pcsd_pointwise_mcca"),
    "capability_over_uniform_ot": (
        "pcsd_mcca",
        "pcsd_uniform_balanced_ot",
    ),
}
COMPOSITE_EFFECTS = {
    "architecture_main_effect": (
        "architecture_equal",
        "architecture_pcc",
        "architecture_mcca",
    ),
    "mcca_main_effect": ("mcca_pcsd", "mcca_siff"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def arm_root(artifact_root: Path, arm: str) -> Path:
    if arm in PCSD_REFERENCES:
        return artifact_root / "references" / "pcsd_cf"
    if arm in PCC_REFERENCES:
        return artifact_root / "references" / "pcc"
    return artifact_root


def metric_path(
    artifact_root: Path,
    arm: str,
    dataset: str,
    seed: int,
) -> Path:
    return (
        arm_root(artifact_root, arm)
        / arm
        / dataset
        / "h720_full"
        / f"seed{seed}"
        / "metrics_by_target_horizon.csv"
    )


def gain_percent(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def metric_table(
    artifact_root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    arms = (*NEW_ARMS, *PCSD_REFERENCES, *PCC_REFERENCES)
    for dataset in DATASETS:
        for arm in arms:
            path = metric_path(artifact_root, arm, dataset, seed)
            raw = read_csv(path)
            by_horizon = {
                int(row["target_horizon"]): row
                for row in raw
            }
            if sorted(by_horizon) != list(range(1, 721)):
                raise ValueError(f"incomplete horizon curve: {path}")
            for horizon in HORIZONS:
                mse = float(by_horizon[horizon]["mse"])
                mae = float(by_horizon[horizon]["mae"])
                if not math.isfinite(mse) or not math.isfinite(mae):
                    raise ValueError(f"non-finite metric: {path}, H={horizon}")
                row = {
                    "dataset": dataset,
                    "arm": arm,
                    "horizon": horizon,
                    "mse": mse,
                    "mae": mae,
                }
                rows.append(row)
                lookup[(dataset, arm, horizon)] = {
                    "mse": mse,
                    "mae": mae,
                }
    return rows, lookup


def pairwise_effect_rows(
    lookup: dict[tuple[str, str, int], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effect, (candidate, reference) in PAIRWISE_EFFECTS.items():
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate_row = lookup[(dataset, candidate, horizon)]
                reference_row = lookup[(dataset, reference, horizon)]
                rows.append(
                    {
                        "effect": effect,
                        "candidate": candidate,
                        "reference": reference,
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse_gain_percent": gain_percent(
                            candidate_row["mse"],
                            reference_row["mse"],
                        ),
                        "mae_gain_percent": gain_percent(
                            candidate_row["mae"],
                            reference_row["mae"],
                        ),
                        "factor_count": 1,
                    }
                )
    return rows


def composite_effect_rows(
    pairwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["effect"], row["dataset"], row["horizon"]): row
        for row in pairwise_rows
    }
    rows: list[dict[str, Any]] = []
    for effect, factors in COMPOSITE_EFFECTS.items():
        for dataset in DATASETS:
            for horizon in HORIZONS:
                selected = [
                    by_key[(factor, dataset, horizon)]
                    for factor in factors
                ]
                rows.append(
                    {
                        "effect": effect,
                        "candidate": "composite",
                        "reference": "matched_factorial_controls",
                        "dataset": dataset,
                        "horizon": horizon,
                        "mse_gain_percent": mean(
                            float(row["mse_gain_percent"])
                            for row in selected
                        ),
                        "mae_gain_percent": mean(
                            float(row["mae_gain_percent"])
                            for row in selected
                        ),
                        "factor_count": len(factors),
                    }
                )
    return rows


def summarize_effects(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for effect in (*COMPOSITE_EFFECTS, *PAIRWISE_EFFECTS):
        selected = [row for row in rows if row["effect"] == effect]
        dataset_gains = {
            dataset: mean(
                float(row["mse_gain_percent"])
                for row in selected
                if row["dataset"] == dataset
            )
            for dataset in DATASETS
        }
        summaries.append(
            {
                "effect": effect,
                "macro_mse_gain_percent": mean(
                    float(row["mse_gain_percent"])
                    for row in selected
                ),
                "macro_mae_gain_percent": mean(
                    float(row["mae_gain_percent"])
                    for row in selected
                ),
                "cell_wins": sum(
                    float(row["mse_gain_percent"]) > 0.0
                    for row in selected
                ),
                "cells": len(selected),
                "dataset_wins": sum(
                    value > 0.0 for value in dataset_gains.values()
                ),
                "datasets": len(DATASETS),
                "worst_dataset": min(
                    dataset_gains,
                    key=dataset_gains.__getitem__,
                ),
                "worst_dataset_mse_gain_percent": min(
                    dataset_gains.values()
                ),
            }
        )
    return summaries


def break_down_effects(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    breakdown: list[dict[str, Any]] = []
    for effect in (*COMPOSITE_EFFECTS, *PAIRWISE_EFFECTS):
        effect_rows = [row for row in rows if row["effect"] == effect]
        for axis, groups in (
            ("dataset", DATASETS),
            ("horizon", HORIZONS),
        ):
            for group in groups:
                selected = [
                    row for row in effect_rows if row[axis] == group
                ]
                breakdown.append(
                    {
                        "effect": effect,
                        "aggregation_axis": axis,
                        "group": group,
                        "macro_mse_gain_percent": mean(
                            float(row["mse_gain_percent"])
                            for row in selected
                        ),
                        "macro_mae_gain_percent": mean(
                            float(row["mae_gain_percent"])
                            for row in selected
                        ),
                        "cell_wins": sum(
                            float(row["mse_gain_percent"]) > 0.0
                            for row in selected
                        ),
                        "cells": len(selected),
                    }
                )
    return breakdown


def gate_result(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = {row["effect"]: row for row in summaries}

    def passes(effect: str, macro_min: float, wins_min: int) -> bool:
        row = lookup[effect]
        return bool(
            float(row["macro_mse_gain_percent"]) >= macro_min
            and int(row["dataset_wins"]) >= wins_min
        )

    gates = {
        "architecture_main_effect": passes(
            "architecture_main_effect",
            0.3,
            3,
        ),
        "mcca_main_effect": passes("mcca_main_effect", 0.2, 3),
        "joint_over_a6": passes("joint_over_a6", 0.3, 3),
        "ordered_over_constant": (
            float(lookup["ordered_over_constant"]["macro_mse_gain_percent"])
            > 0.0
        ),
        "ordered_over_permuted": (
            float(lookup["ordered_over_permuted"]["macro_mse_gain_percent"])
            > 0.0
        ),
        "capability_over_uniform_ot": (
            float(
                lookup["capability_over_uniform_ot"][
                    "macro_mse_gain_percent"
                ]
            )
            > 0.0
        ),
    }
    return {
        "candidate": "SC1-SIFF-v1/SC2-MCCA-v1",
        "evaluation_role": "retrospective_development_screen",
        "evaluation_split": "validation",
        "test_used": False,
        "seed": 2021,
        "paper_facing_horizons": list(HORIZONS),
        "checkpoint_rule_inherited": "best_validation_h720_mse",
        "checkpoint_reselected": False,
        "threshold_source": "original_frozen_step7b_gate",
        "cell_wins_role": "descriptive_only",
        "gates": gates,
        "method_pass": all(gates.values()),
        "decision": (
            "retrospective_standard_horizon_screen_pass"
            if all(gates.values())
            else "retrospective_standard_horizon_screen_fail"
        ),
    }


def main() -> None:
    args = parse_args()
    metric_rows, lookup = metric_table(args.artifact_root, args.seed)
    pairwise_rows = pairwise_effect_rows(lookup)
    composite_rows = composite_effect_rows(pairwise_rows)
    effect_rows = [*pairwise_rows, *composite_rows]
    summaries = summarize_effects(effect_rows)
    breakdown = break_down_effects(effect_rows)
    gate = gate_result(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "standard_horizon_metrics.csv", metric_rows)
    write_csv(args.output_dir / "standard_horizon_effects.csv", effect_rows)
    write_csv(args.output_dir / "standard_horizon_summary.csv", summaries)
    write_csv(
        args.output_dir / "standard_horizon_breakdown.csv",
        breakdown,
    )
    (args.output_dir / "standard_horizon_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
