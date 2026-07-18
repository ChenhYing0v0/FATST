#!/usr/bin/env python3
"""Analyze internal arm health for the StageC fair re-audit."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def analyze_run(path: Path, raw_root: Path) -> dict[str, Any] | None:
    relative = path.relative_to(raw_root)
    arm, dataset, _, seed_dir, _ = relative.parts
    with np.load(path) as payload:
        required_arrays = {
            "fused_row_bin_mse",
            "arm_row_bin_mse",
            "policy_row_bin_usage",
            "probe_arms",
            "probe_targets",
        }
        if not required_arrays.issubset(payload.files):
            return None
        fused_loss = payload["fused_row_bin_mse"].astype(np.float64)
        arm_loss = payload["arm_row_bin_mse"].astype(np.float64)
        policy_usage = payload["policy_row_bin_usage"].astype(np.float64)
        probe_arms = payload["probe_arms"].astype(np.float64)
        probe_targets = payload["probe_targets"].astype(np.float64)

        fused_mean = float(fused_loss.mean())
        fixed_arm_means = arm_loss.mean(axis=(0, 1))
        row_bin_oracle = arm_loss.min(axis=-1)
        target_rms = max(
            float(np.sqrt(np.mean(np.square(probe_targets)))),
            1e-12,
        )
        pairwise_nrmse = [
            float(
                np.sqrt(
                    np.mean(
                        np.square(probe_arms[:, first] - probe_arms[:, second])
                    )
                )
                / target_rms
            )
            for first, second in combinations(range(probe_arms.shape[1]), 2)
        ]

        usage_mean = policy_usage.mean(axis=(0, 1))
        normalized_entropy = (
            -np.sum(
                policy_usage
                * np.log(np.clip(policy_usage, 1e-12, None)),
                axis=-1,
            )
            / np.log(policy_usage.shape[-1])
        )

    return {
        "arm": arm,
        "dataset": dataset,
        "seed": int(seed_dir.removeprefix("seed")),
        "oracle_headroom_percent": relative_gain(
            float(row_bin_oracle.mean()),
            fused_mean,
        ),
        "best_fixed_over_fused_percent": relative_gain(
            float(fixed_arm_means.min()),
            fused_mean,
        ),
        "worst_fixed_over_fused_percent": relative_gain(
            float(fixed_arm_means.max()),
            fused_mean,
        ),
        "arm_loss_cv_percent": (
            100.0 * float(fixed_arm_means.std() / fixed_arm_means.mean())
        ),
        "min_pairwise_probe_nrmse": min(pairwise_nrmse),
        "mean_pairwise_probe_nrmse": float(np.mean(pairwise_nrmse)),
        "policy_entropy": float(normalized_entropy.mean()),
        "usage_min": float(usage_mean.min()),
        "usage_max": float(usage_mean.max()),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "oracle_headroom_percent",
        "best_fixed_over_fused_percent",
        "worst_fixed_over_fused_percent",
        "arm_loss_cv_percent",
        "min_pairwise_probe_nrmse",
        "mean_pairwise_probe_nrmse",
        "policy_entropy",
        "usage_min",
        "usage_max",
    ]
    summaries: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summary: dict[str, Any] = {
            "arm": arm,
            "runs": len(arm_rows),
            "oracle_positive_runs": sum(
                float(row["oracle_headroom_percent"]) > 0.0
                for row in arm_rows
            ),
            "best_fixed_beats_fused_runs": sum(
                float(row["best_fixed_over_fused_percent"]) > 0.0
                for row in arm_rows
            ),
        }
        summary.update(
            {
                field: float(np.mean([float(row[field]) for row in arm_rows]))
                for field in numeric_fields
            }
        )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    diagnostic_paths = sorted(
        args.raw_root.glob(
            "*/*/h720_full/seed*/pcsd_test_audit_diagnostics.npz"
        )
    )
    if not diagnostic_paths:
        raise FileNotFoundError(
            f"No PCSD diagnostic artifacts under {args.raw_root}"
        )

    analyzed = [analyze_run(path, args.raw_root) for path in diagnostic_paths]
    rows = [row for row in analyzed if row is not None]
    if not rows:
        raise ValueError("No diagnostic artifacts contained arm-level arrays")
    summaries = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mechanism_health_by_run.csv", rows)
    write_csv(args.output_dir / "mechanism_health_summary.csv", summaries)

    payload = {
        "raw_root": str(args.raw_root),
        "runs": len(rows),
        "arms": len(summaries),
        "definitions": {
            "oracle_headroom_percent": (
                "Gain of a row-bin oracle choosing the lowest-loss arm over "
                "the learned fused forecast."
            ),
            "best_fixed_over_fused_percent": (
                "Gain of the single globally best same-run arm over the "
                "learned fused forecast."
            ),
            "arm_loss_cv_percent": (
                "Coefficient of variation across the five same-run arm MSEs."
            ),
            "pairwise_probe_nrmse": (
                "Pairwise arm prediction RMSE divided by target RMS on the "
                "saved probe batch."
            ),
            "policy_entropy": (
                "Mean row-bin policy entropy normalized by log(number of arms)."
            ),
        },
        "boundary": (
            "These same-run diagnostics measure arm diversity, skill spread, "
            "fusion headroom, and policy use. They do not replace comparisons "
            "against separately trained fixed-scope arms."
        ),
        "summary": summaries,
    }
    (args.output_dir / "mechanism_health.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
