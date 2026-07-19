#!/usr/bin/env python3
"""Build the D18 validation/test, checkpoint, and effect-decomposition audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DATASETS = ("Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2")
HORIZONS = (96, 192, 336)
PROBE_INTERVALS = {
    96: ((1, 48), (49, 96)),
    192: ((1, 96), (97, 192)),
    336: ((1, 96), (97, 192), (193, 336)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def metric_map(path: Path) -> dict[int, dict[str, float]]:
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(path)
    }


def gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def probe_interval_rows(
    dataset: str,
    horizon: int,
    specialist_dir: Path,
    measure_dir: Path,
) -> list[dict[str, Any]]:
    specialist = np.load(
        specialist_dir / "pcsd_test_audit_diagnostics.npz"
    )
    measure = np.load(measure_dir / "pcsd_test_audit_diagnostics.npz")
    specialist_targets = specialist["probe_targets"]
    measure_targets = measure["probe_targets"]
    target_gap = float(
        np.max(np.abs(specialist_targets - measure_targets))
    )
    if target_gap > 0.0:
        raise ValueError(
            f"probe target mismatch for {dataset} H{horizon}: {target_gap}"
        )
    specialist_prediction = specialist["probe_fused"]
    measure_prediction = measure["probe_fused"]
    rows = []
    for start, end in PROBE_INTERVALS[horizon]:
        target = specialist_targets[:, start - 1 : end]
        specialist_mse = float(
            np.mean(
                (
                    specialist_prediction[:, start - 1 : end]
                    - target
                )
                ** 2
            )
        )
        measure_mse = float(
            np.mean(
                (measure_prediction[:, start - 1 : end] - target) ** 2
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "specialist_horizon": horizon,
                "interval_start": start,
                "interval_end": end,
                "specialist_probe_mse": specialist_mse,
                "a6_measure_probe_mse": measure_mse,
                "gain_over_a6_measure_percent": gain(
                    specialist_mse,
                    measure_mse,
                ),
                "prediction_nrmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                specialist_prediction[
                                    :, start - 1 : end
                                ]
                                - measure_prediction[:, start - 1 : end]
                            )
                            ** 2
                        )
                    )
                    / (
                        np.sqrt(
                            np.mean(
                                measure_prediction[:, start - 1 : end]
                                ** 2
                            )
                        )
                        + 1e-12
                    )
                ),
            }
        )
    return rows


def aggregate(
    rows: list[dict[str, Any]],
    field: str,
    group: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row[group])].append(float(row[field]))
    return {key: mean(items) for key, items in values.items()}


def main() -> None:
    args = parse_args()
    remote_summary = json.loads(
        (args.raw_root / "analysis/summary.json").read_text(encoding="utf-8")
    )
    remote_cells = {
        (row["dataset"], int(row["horizon"])): row
        for row in read_csv(args.raw_root / "analysis/own_horizon_cells.csv")
    }

    cell_rows = []
    checkpoint_rows = []
    invariant_rows = []
    probe_rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            arm = f"a6_spec{horizon}"
            specialist_dir = (
                args.raw_root / arm / dataset / "h720_full" / "seed2021"
            )
            measure_dir = (
                args.control_root
                / "a6_measure"
                / dataset
                / "h720_full"
                / "seed2021"
            )
            remote_measure_dir = (
                args.raw_root
                / "a6_measure"
                / dataset
                / "h720_full"
                / "seed2021"
            )
            probe_rows.extend(
                probe_interval_rows(
                    dataset,
                    horizon,
                    specialist_dir,
                    remote_measure_dir,
                )
            )
            specialist_validation = metric_map(
                specialist_dir / "metrics_by_target_horizon.csv"
            )[horizon]
            measure_validation = metric_map(
                measure_dir / "metrics_by_target_horizon.csv"
            )[horizon]
            test = remote_cells[(dataset, horizon)]
            validation_gain = gain(
                specialist_validation["mse"],
                measure_validation["mse"],
            )
            test_gain = float(test["gain_over_a6_measure_percent"])
            cell_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "validation_specialist_mse": specialist_validation["mse"],
                    "validation_a6_measure_mse": measure_validation["mse"],
                    "validation_gain_over_a6_measure_percent": validation_gain,
                    "test_specialist_mse": float(test["specialist_mse"]),
                    "test_a6_measure_mse": float(test["a6_measure_mse"]),
                    "test_a6_full_mse": float(test["a6_full_mse"]),
                    "test_gain_over_a6_measure_percent": test_gain,
                    "test_gain_over_a6_full_percent": float(
                        test["gain_over_a6_full_percent"]
                    ),
                    "validation_test_sign_agreement": (
                        validation_gain > 0.0
                    )
                    == (test_gain > 0.0),
                }
            )
            training = read_csv(specialist_dir / "training_log.csv")
            best_validation = min(
                float(row["val_mean_mse"]) for row in training
            )
            best_epoch = next(
                int(row["epoch"])
                for row in training
                if math.isclose(
                    float(row["val_mean_mse"]),
                    best_validation,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "epochs_trained": len(training),
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_validation,
                    "early_stopping_triggered": any(
                        int(row["stop_triggered"]) == 1 for row in training
                    ),
                    "best_epoch_at_budget_boundary": best_epoch
                    == len(training),
                }
            )

    for invariant_path in sorted(
        args.raw_root.glob(
            "a6_*/**/h720_full/seed2021/test_audit_invariants.json"
        )
    ):
        invariant = json.loads(invariant_path.read_text(encoding="utf-8"))
        invariant_rows.append(
            {
                "dataset": invariant["dataset"],
                "readout_mode": invariant["readout_mode"],
                "checkpoint_sha256": invariant["checkpoint_sha256"],
                "checkpoint_retrained": invariant["checkpoint_retrained"],
                "all_finite": invariant["all_finite"],
                "protocol_pass": invariant["protocol_pass"],
                "readout_contract_pass": invariant[
                    "readout_contract_pass"
                ],
                "prefix_gap": invariant["full_prefix_max_abs"],
                "test_access_authorized": invariant[
                    "test_access_authorized"
                ],
                "pass": invariant["pass"],
            }
        )

    test_measure_over_full = [
        gain(float(row["test_a6_measure_mse"]), float(row["test_a6_full_mse"]))
        for row in cell_rows
    ]
    test_specialist_over_full = [
        float(row["test_gain_over_a6_full_percent"]) for row in cell_rows
    ]
    test_specialist_over_measure = [
        float(row["test_gain_over_a6_measure_percent"]) for row in cell_rows
    ]
    validation_specialist_over_measure = [
        float(row["validation_gain_over_a6_measure_percent"])
        for row in cell_rows
    ]
    decomposition_rows = [
        {
            "comparison": "a6_measure_over_a6_full",
            "split": "test",
            "macro_gain_percent": mean(test_measure_over_full),
            "positive_cells": sum(
                value > 0.0 for value in test_measure_over_full
            ),
        },
        {
            "comparison": "specialist_over_a6_full",
            "split": "test",
            "macro_gain_percent": mean(test_specialist_over_full),
            "positive_cells": sum(
                value > 0.0 for value in test_specialist_over_full
            ),
        },
        {
            "comparison": "specialist_over_a6_measure",
            "split": "test",
            "macro_gain_percent": mean(test_specialist_over_measure),
            "positive_cells": sum(
                value > 0.0 for value in test_specialist_over_measure
            ),
        },
        {
            "comparison": "specialist_over_a6_measure",
            "split": "validation",
            "macro_gain_percent": mean(validation_specialist_over_measure),
            "positive_cells": sum(
                value > 0.0 for value in validation_specialist_over_measure
            ),
        },
    ]
    validation_by_dataset = aggregate(
        cell_rows,
        "validation_gain_over_a6_measure_percent",
        "dataset",
    )
    validation_by_horizon = aggregate(
        cell_rows,
        "validation_gain_over_a6_measure_percent",
        "horizon",
    )
    summary = {
        "candidate_version": "SC-D18-SPC-v1",
        "matrix_complete": len(invariant_rows) == 25,
        "valid_runs": sum(row["pass"] is True for row in invariant_rows),
        "unique_checkpoint_hashes": len(
            {row["checkpoint_sha256"] for row in invariant_rows}
        ),
        "max_prefix_gap": max(
            float(row["prefix_gap"]) for row in invariant_rows
        ),
        "test_macro_specialist_over_measure_percent": mean(
            test_specialist_over_measure
        ),
        "test_macro_measure_over_full_percent": mean(test_measure_over_full),
        "test_macro_specialist_over_full_percent": mean(
            test_specialist_over_full
        ),
        "validation_macro_specialist_over_measure_percent": mean(
            validation_specialist_over_measure
        ),
        "validation_positive_cells": sum(
            value > 0.0 for value in validation_specialist_over_measure
        ),
        "validation_positive_datasets": sum(
            value > 0.0 for value in validation_by_dataset.values()
        ),
        "validation_positive_horizons": sum(
            value > 0.0 for value in validation_by_horizon.values()
        ),
        "validation_minimum_horizon_gain_percent": min(
            validation_by_horizon.values()
        ),
        "validation_test_sign_agreement_cells": sum(
            row["validation_test_sign_agreement"] for row in cell_rows
        ),
        "best_epoch_at_budget_boundary_runs": sum(
            row["best_epoch_at_budget_boundary"] for row in checkpoint_rows
        ),
        "h96_probe_positive_intervals": sum(
            row["gain_over_a6_measure_percent"] > 0.0
            for row in probe_rows
            if row["specialist_horizon"] == 96
        ),
        "h96_probe_total_intervals": sum(
            row["specialist_horizon"] == 96 for row in probe_rows
        ),
        "h96_probe_interval_macro_gain_percent": mean(
            row["gain_over_a6_measure_percent"]
            for row in probe_rows
            if row["specialist_horizon"] == 96
        ),
        "remote_gate_result": remote_summary,
        "failure_attribution": {
            "primary": "hypothesis_false",
            "exact_scope": (
                "stable_cross_dataset_multi_horizon_accuracy_cost_of_exact_"
                "projectivity_relative_to_a6_measure"
            ),
            "not_supported": [
                "optimization_or_numeric_pathology",
                "capacity_or_initialization_mismatch",
                "test_only_reversal_as_primary_explanation",
            ],
            "retained_signal": (
                "H96 specialization headroom is positive but not sufficient "
                "for a multi-horizon soft-projectivity mainline"
            ),
            "decision": (
                "measure_training_explains_close_soft_architecture_route_"
                "return_step2"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_test_cells.csv", cell_rows)
    write_csv(args.output_dir / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(args.output_dir / "protocol_invariants.csv", invariant_rows)
    write_csv(args.output_dir / "effect_decomposition.csv", decomposition_rows)
    write_csv(args.output_dir / "probe_interval_gains.csv", probe_rows)
    (args.output_dir / "deep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
