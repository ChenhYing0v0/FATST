#!/usr/bin/env python3
"""Audit D20 summary contribution direction and scale on saved test probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d20_d1_contribution_diagnostic.json"),
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / max(reference, 1e-12))


def region_metrics(
    fused: np.ndarray,
    target: np.ndarray,
    contribution: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float | bool]:
    fused_region = fused[:, start:end].astype(np.float64)
    target_region = target[:, start:end].astype(np.float64)
    contribution_region = contribution[:, start:end].astype(np.float64)
    base_region = fused_region - contribution_region
    residual = target_region - base_region
    denominator = float(np.sum(np.square(contribution_region)))
    numerator = float(np.sum(contribution_region * residual))
    optimal_alpha = numerator / max(denominator, 1e-18)
    clipped_alpha = min(max(optimal_alpha, 0.0), 1.0)
    oracle = base_region + optimal_alpha * contribution_region
    clipped = base_region + clipped_alpha * contribution_region
    mse_base = float(np.mean(np.square(base_region - target_region)))
    mse_fused = float(np.mean(np.square(fused_region - target_region)))
    mse_oracle = float(np.mean(np.square(oracle - target_region)))
    mse_clipped = float(np.mean(np.square(clipped - target_region)))
    residual_norm = float(np.sqrt(np.sum(np.square(residual))))
    contribution_norm = float(
        np.sqrt(np.sum(np.square(contribution_region)))
    )
    cosine = numerator / max(residual_norm * contribution_norm, 1e-18)
    reconstruction_gap = float(
        np.max(np.abs(base_region + contribution_region - fused_region))
    )
    values = (
        mse_base,
        mse_fused,
        mse_oracle,
        mse_clipped,
        optimal_alpha,
        cosine,
        reconstruction_gap,
    )
    return {
        "mse_base": mse_base,
        "mse_actual": mse_fused,
        "actual_gain_vs_base_percent": gain(mse_fused, mse_base),
        "optimal_alpha": optimal_alpha,
        "clipped_alpha_0_1": clipped_alpha,
        "mse_oracle_alpha": mse_oracle,
        "oracle_gain_vs_base_percent": gain(mse_oracle, mse_base),
        "mse_clipped_alpha": mse_clipped,
        "clipped_gain_vs_base_percent": gain(mse_clipped, mse_base),
        "contribution_residual_cosine": cosine,
        "contribution_rms": float(np.sqrt(np.mean(np.square(contribution_region)))),
        "base_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "reconstruction_max_abs": reconstruction_gap,
        "all_finite": all(math.isfinite(value) for value in values),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_root = args.raw_root or Path(config["source_root"])
    output_dir = args.output_dir or Path(config["output_root"])
    rows = []
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            run_dir = raw_root / arm / dataset / "h720_full" / "seed2021"
            path = run_dir / "pcsd_test_audit_diagnostics.npz"
            diagnostics = np.load(path)
            missing = set(config["required_arrays"]) - set(diagnostics.files)
            if missing:
                raise ValueError(f"missing arrays in {path}: {sorted(missing)}")
            fused = diagnostics["probe_fused"]
            targets = diagnostics["probe_targets"]
            contribution = diagnostics[
                "probe_history_prediction_contribution"
            ]
            if fused.shape != (256, 720) or targets.shape != fused.shape:
                raise ValueError(f"unexpected probe shape: {path} {fused.shape}")
            if contribution.shape != fused.shape:
                raise ValueError(f"unexpected contribution shape: {path}")
            for region in config["regions"]:
                result = region_metrics(
                    fused,
                    targets,
                    contribution,
                    int(region["start"]),
                    int(region["end"]),
                )
                rows.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "region": region["name"],
                        "start": int(region["start"]),
                        "end": int(region["end"]),
                        **result,
                    }
                )

    summaries = []
    for arm in config["arms"]:
        selected = [
            row for row in rows if row["arm"] == arm and row["region"] != "H1_720"
        ]
        full = [
            row for row in rows if row["arm"] == arm and row["region"] == "H1_720"
        ]
        summaries.append(
            {
                "arm": arm,
                "region_cells": len(selected),
                "actual_macro_gain_vs_base_percent": mean(
                    float(row["actual_gain_vs_base_percent"])
                    for row in selected
                ),
                "actual_help_cells": sum(
                    float(row["actual_gain_vs_base_percent"]) > 0.0
                    for row in selected
                ),
                "oracle_macro_gain_vs_base_percent": mean(
                    float(row["oracle_gain_vs_base_percent"])
                    for row in selected
                ),
                "median_optimal_alpha": median(
                    float(row["optimal_alpha"]) for row in selected
                ),
                "alpha_between_zero_and_one_cells": sum(
                    0.0 < float(row["optimal_alpha"]) < 1.0
                    for row in selected
                ),
                "alpha_nonpositive_cells": sum(
                    float(row["optimal_alpha"]) <= 0.0 for row in selected
                ),
                "actual_harm_but_positive_shrinkage_cells": sum(
                    float(row["actual_gain_vs_base_percent"]) <= 0.0
                    and 0.0 < float(row["optimal_alpha"]) < 1.0
                    for row in selected
                ),
                "full_horizon_actual_gain_mean_percent": mean(
                    float(row["actual_gain_vs_base_percent"]) for row in full
                ),
                "full_horizon_optimal_alpha_median": median(
                    float(row["optimal_alpha"]) for row in full
                ),
                "max_reconstruction_abs": max(
                    float(row["reconstruction_max_abs"]) for row in selected
                ),
                "all_finite": all(bool(row["all_finite"]) for row in selected),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "contribution_region_audit.csv", rows)
    write_csv(output_dir / "contribution_arm_summary.csv", summaries)
    payload = {
        "diagnostic_id": config["diagnostic_id"],
        "candidate_version": config["candidate_version"],
        "datasets": len(config["datasets"]),
        "arms": len(config["arms"]),
        "region_rows": len(rows),
        "all_finite": all(row["all_finite"] for row in summaries),
        "max_reconstruction_abs": max(
            row["max_reconstruction_abs"] for row in summaries
        ),
        "summaries": summaries,
        "test_oracle_diagnostic_only": True,
        "direction_rejection_allowed": False,
        "method_promotion_allowed": False,
    }
    (output_dir / "d1_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
