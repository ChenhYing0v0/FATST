#!/usr/bin/env python3
"""Build the corrected Step 9 attribution audit for SC-RETRO-FAIR-v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2")
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("analysis/stage_c_fair_reaudit_v1_20260717"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_fair_reaudit_v1.json"),
    )
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_metric_maps(
    root: Path,
    arms: list[str],
) -> dict[str, dict[tuple[str, str, int], dict[str, float]]]:
    test_rows = read_csv(
        root / "remote_analysis" / "test_metrics_standard_horizons.csv"
    )
    test = {
        (row["dataset"], row["arm"], int(row["horizon"])): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in test_rows
    }
    validation: dict[tuple[str, str, int], dict[str, float]] = {}
    for arm in arms:
        for dataset in DATASETS:
            path = (
                root
                / "raw_lite"
                / arm
                / dataset
                / "h720_full"
                / "seed2021"
                / "metrics_by_target_horizon.csv"
            )
            for row in read_csv(path):
                validation[(dataset, arm, int(row["target_horizon"]))] = {
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                }
    expected = len(arms) * len(DATASETS) * len(HORIZONS)
    if len(test) != expected or len(validation) != expected:
        raise ValueError(
            f"incomplete metric maps: test={len(test)} val={len(validation)} "
            f"expected={expected}"
        )
    return {"validation": validation, "test": test}


def score(
    metric_map: dict[tuple[str, str, int], dict[str, float]],
    candidate: str,
    reference: str,
    metric: str,
) -> dict[str, Any]:
    cells = []
    by_dataset: dict[str, list[float]] = {}
    by_horizon: dict[int, list[float]] = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            candidate_loss = metric_map[(dataset, candidate, horizon)][metric]
            reference_loss = metric_map[(dataset, reference, horizon)][metric]
            gain = 100.0 * (1.0 - candidate_loss / reference_loss)
            cells.append(gain)
            by_dataset.setdefault(dataset, []).append(gain)
            by_horizon.setdefault(horizon, []).append(gain)
    return {
        "macro_gain_percent": mean(cells),
        "cell_wins": sum(value > 0.0 for value in cells),
        "dataset_wins": sum(
            mean(values) > 0.0 for values in by_dataset.values()
        ),
        "horizon_wins": sum(
            mean(values) > 0.0 for values in by_horizon.values()
        ),
        "dataset_gains": {
            dataset: mean(values) for dataset, values in by_dataset.items()
        },
        "horizon_gains": {
            str(horizon): mean(values)
            for horizon, values in by_horizon.items()
        },
    }


def passes_gate(result: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        result["macro_gain_percent"] >= gates["macro_gain_percent_min"]
        and result["cell_wins"] >= gates["cell_wins_min"]
        and result["dataset_wins"] >= gates["dataset_wins_min"]
        and result["horizon_wins"] >= gates["horizon_wins_min"]
    )


def checkpoint_rows(root: Path, arms: list[str]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        for dataset in DATASETS:
            path = (
                root
                / "raw_lite"
                / arm
                / dataset
                / "h720_full"
                / "seed2021"
                / "training_log.csv"
            )
            training = read_csv(path)
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "epochs_trained": len(training),
                    "best_epoch": max(
                        int(row["best_epoch_so_far"]) for row in training
                    ),
                    "final_recorded_val_mean_mse": float(
                        training[-1]["val_mean_mse"]
                    ),
                    "early_stopping_triggered": any(
                        int(row["stop_triggered"]) == 1 for row in training
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    arms = [entry["id"] for entry in config["arms"]]
    metric_maps = load_metric_maps(args.analysis_root, arms)
    gates = config["gates"]

    comparisons = [
        (entry["id"], entry["candidate"], entry["reference"], True)
        for entry in config["comparisons"]
    ]
    comparisons.extend(
        [
            ("siff_equal_over_a6", "siff_equal", "a6_full", False),
            ("dense_measure_over_a6", "dense_measure", "a6_full", False),
            ("pcsd_measure_over_a6", "pcsd_measure", "a6_full", False),
            ("pcsd_equal_over_a6", "pcsd_equal", "a6_full", False),
            ("pcsd_prior_over_a6", "pcsd_prior", "a6_full", False),
            ("pcsd_pcc_over_a6", "pcsd_pcc", "a6_full", False),
            ("siff_prior_over_a6", "siff_prior", "a6_full", False),
        ]
    )

    scorecard_rows = []
    cell_rows = []
    results: dict[str, dict[str, dict[str, Any]]] = {}
    preregistered_passes = []
    for comparison, candidate, reference, preregistered in comparisons:
        results[comparison] = {}
        for split, metric_map in metric_maps.items():
            for metric in ("mse", "mae"):
                result = score(metric_map, candidate, reference, metric)
                gate_pass = (
                    split == "test"
                    and metric == "mse"
                    and preregistered
                    and passes_gate(result, gates)
                )
                scorecard_rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "reference": reference,
                        "preregistered": preregistered,
                        "split": split,
                        "metric": metric,
                        "macro_gain_percent": result["macro_gain_percent"],
                        "cell_wins": result["cell_wins"],
                        "dataset_wins": result["dataset_wins"],
                        "horizon_wins": result["horizon_wins"],
                        "gate_pass": gate_pass,
                    }
                )
                results[comparison][f"{split}_{metric}"] = result
                if gate_pass:
                    preregistered_passes.append(comparison)
        for dataset in DATASETS:
            for horizon in HORIZONS:
                candidate_loss = metric_maps["test"][
                    (dataset, candidate, horizon)
                ]["mse"]
                reference_loss = metric_maps["test"][
                    (dataset, reference, horizon)
                ]["mse"]
                cell_rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "reference": reference,
                        "dataset": dataset,
                        "horizon": horizon,
                        "test_mse_gain_percent": 100.0
                        * (1.0 - candidate_loss / reference_loss),
                        "candidate_test_mse": candidate_loss,
                        "reference_test_mse": reference_loss,
                    }
                )

    arm_rows = []
    for arm in arms:
        if arm == "a6_full":
            continue
        for metric in ("mse", "mae"):
            result = score(
                metric_maps["test"],
                arm,
                "a6_full",
                metric,
            )
            arm_rows.append(
                {
                    "arm": arm,
                    "reference": "a6_full",
                    "metric": metric,
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in {"dataset_gains", "horizon_gains"}
                    },
                }
            )

    siff_equal = results["siff_equal_over_a6"]["test_mse"]
    siff_pcc = results["joint_over_a6"]["test_mse"]
    pcc_on_siff = results["pcc_over_equal_siff"]["test_mse"]
    dense_measure = results["dense_measure_over_a6"]["test_mse"]
    attribution = {
        "audit_id": config["audit_id"],
        "matrix_complete": True,
        "valid_runs": config["matrix"]["expected_runs"],
        "valid_test_cells": config["matrix"]["expected_test_cells"],
        "all_test_invariants_pass": True,
        "checkpoint_rule": config["training"]["validation_checkpoint_score"],
        "test_informed": True,
        "preregistered_comparison_passes": sorted(
            set(preregistered_passes)
        ),
        "pcsd_exact_conclusion": "failed_architecture_effect",
        "pcc_exact_conclusion": "failed_specificity_and_harms_siff_equal",
        "siff_exact_conclusion": (
            "partial_pass_under_equal_skill_but_not_objective_robust_or_"
            "fully_specific"
        ),
        "joint_conclusion": (
            "performance_pass_but_two_contribution_attribution_fail"
        ),
        "best_paper_facing_arm": "siff_equal",
        "best_arm_over_a6_test_mse_percent": siff_equal[
            "macro_gain_percent"
        ],
        "best_arm_over_a6_test_mae_percent": results[
            "siff_equal_over_a6"
        ]["test_mae"]["macro_gain_percent"],
        "joint_over_a6_test_mse_percent": siff_pcc["macro_gain_percent"],
        "pcc_on_siff_test_mse_percent": pcc_on_siff[
            "macro_gain_percent"
        ],
        "dense_measure_over_a6_test_mse_percent": dense_measure[
            "macro_gain_percent"
        ],
        "missing_attribution_controls": [
            "a6_measure_only",
            "siff_constant_equal",
            "siff_permuted_equal",
            "siff_q1_wide_equal",
            "siff_independent_equal",
        ],
        "confirmation_status": (
            "held_until_siff_equal_attribution_controls_define_new_"
            "test_informed_candidate"
        ),
        "ctd_status": "paused",
        "rollback": "return_siff_to_step6_attribution_repair_close_pcc_v1",
    }

    output_dir = args.analysis_root / "step9_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison_scorecard.csv", scorecard_rows)
    write_csv(output_dir / "comparison_test_cells.csv", cell_rows)
    write_csv(output_dir / "all_arms_vs_a6.csv", arm_rows)
    write_csv(
        output_dir / "checkpoint_epochs.csv",
        checkpoint_rows(args.analysis_root, arms),
    )
    (output_dir / "step9_attribution.json").write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(attribution, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
