#!/usr/bin/env python3
"""Screen full-rank scope conditioning on frozen ISCF validation probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from diagnose_stage_c_iscf_bsc_d0 import project_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_frsc_d1.json"),
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


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


def prefix_metric(
    forecast: np.ndarray,
    target: np.ndarray,
    horizon: int,
    metric: str,
) -> float:
    error = forecast[:, :horizon] - target[:, :horizon]
    if metric == "mse":
        return float(np.mean(error**2))
    if metric == "mae":
        return float(np.mean(np.abs(error)))
    raise ValueError(metric)


def projected_arms(
    arms: np.ndarray,
    *,
    scales: list[int],
    mode_rank: int,
    projection: str,
    partition: str,
) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(arms.astype(np.float64, copy=False))
    outputs = []
    for scale_index, scale in enumerate(scales):
        outputs.append(
            project_rows(
                tensor[:, scale_index],
                scale=scale,
                scale_index=scale_index,
                mode_rank=mode_rank,
                projection=projection,
                partition=partition,
            )
        )
    return torch.stack(outputs, dim=1).numpy()


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["arm"], row["alpha"], row["metric"]) for row in cells})
    summaries = []
    for arm, alpha, metric in keys:
        selected = [
            row
            for row in cells
            if row["arm"] == arm
            and row["alpha"] == alpha
            and row["metric"] == metric
        ]
        by_dataset: dict[str, list[float]] = {}
        by_horizon: dict[int, list[float]] = {}
        for row in selected:
            by_dataset.setdefault(row["dataset"], []).append(row["gain_percent"])
            by_horizon.setdefault(row["horizon"], []).append(row["gain_percent"])
        summaries.append(
            {
                "arm": arm,
                "alpha": alpha,
                "outside_eigenvalue": 1.0 - float(alpha),
                "metric": metric,
                "macro_gain_percent": mean(row["gain_percent"] for row in selected),
                "cell_wins": sum(row["gain_percent"] > 0 for row in selected),
                "dataset_wins": sum(mean(values) > 0 for values in by_dataset.values()),
                "horizon_wins": sum(mean(values) > 0 for values in by_horizon.values()),
            }
        )
    return summaries


def decide(
    summaries: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    canonical_mse = [
        row
        for row in summaries
        if row["arm"] == "frsc_scope_canonical" and row["metric"] == "mse"
    ]
    selected_mse = max(canonical_mse, key=lambda row: row["macro_gain_percent"])
    alpha = selected_mse["alpha"]
    selected_mae = next(
        row
        for row in summaries
        if row["arm"] == "frsc_scope_canonical"
        and row["metric"] == "mae"
        and row["alpha"] == alpha
    )
    control_mse = [
        row
        for row in summaries
        if row["metric"] == "mse"
        and row["alpha"] == alpha
        and row["arm"] != "frsc_scope_canonical"
    ]
    control_margin = selected_mse["macro_gain_percent"] - max(
        row["macro_gain_percent"] for row in control_mse
    )
    gate = config["diagnostic_lead"]
    lead = bool(
        selected_mse["macro_gain_percent"] >= gate["macro_mse_gain_percent_min"]
        and selected_mae["macro_gain_percent"] >= gate["macro_mae_gain_percent_min"]
        and selected_mse["dataset_wins"] >= gate["dataset_wins_min"]
        and selected_mse["horizon_wins"] >= gate["horizon_wins_min"]
        and selected_mse["cell_wins"] >= gate["cell_wins_min"]
        and control_margin >= gate["canonical_control_margin_percent_min"]
        and all(row["all_finite"] for row in audits)
        and max(row["barycenter_reconstruction_max_abs"] for row in audits)
        <= gate["barycenter_reconstruction_max_abs_max"]
    )
    return {
        "decision": (
            "diagnostic_positive_complete_FRSC_step4_6_design"
            if lead
            else "exact_frozen_FRSC_diagnostic_not_positive_return_step4"
        ),
        "diagnostic_lead": lead,
        "selected_alpha": alpha,
        "selected_outside_eigenvalue": 1.0 - float(alpha),
        "selected_macro_mse_gain_percent": selected_mse["macro_gain_percent"],
        "selected_macro_mae_gain_percent": selected_mae["macro_gain_percent"],
        "selected_dataset_wins": selected_mse["dataset_wins"],
        "selected_horizon_wins": selected_mse["horizon_wins"],
        "selected_cell_wins": selected_mse["cell_wins"],
        "canonical_control_margin_percent": control_margin,
        "method_effectiveness": False,
        "direction_level_rejection_allowed": False,
        "test_accessed": False,
    }


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.root is None or args.output_dir is None:
        raise ValueError("--root and --output-dir are required")
    cells: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        path = (
            args.root
            / config["source_arm"]
            / dataset
            / "h720_full"
            / "seed2021"
            / "pcsd_validation_diagnostics.npz"
        )
        with np.load(path) as arrays:
            row_limit = config["probe_rows"]
            arms = arrays["probe_arms"][:row_limit].astype(np.float64)
            weights = arrays["probe_direct_policy"][:row_limit].astype(np.float64)
            parent = arrays["probe_fused"][:row_limit].astype(np.float64)
            target = arrays["probe_targets"][:row_limit].astype(np.float64)
        expected = {
            "arms": (row_limit, len(config["scales"]), 720),
            "weights": (row_limit, 720, len(config["scales"])),
            "parent": (row_limit, 720),
            "target": (row_limit, 720),
        }
        actual = {
            "arms": arms.shape,
            "weights": weights.shape,
            "parent": parent.shape,
            "target": target.shape,
        }
        if actual != expected:
            raise ValueError(f"unexpected probe shapes for {dataset}: {actual}")
        recomposed = np.sum(weights.transpose(0, 2, 1) * arms, axis=1)
        target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-12)
        audit: dict[str, Any] = {
            "dataset": dataset,
            "rows": row_limit,
            "barycenter_reconstruction_max_abs": float(
                np.max(np.abs(recomposed - parent))
            ),
            "all_finite": bool(
                np.isfinite(arms).all()
                and np.isfinite(weights).all()
                and np.isfinite(parent).all()
                and np.isfinite(target).all()
            ),
        }
        for control in config["projection_controls"]:
            hard_arms = projected_arms(
                arms,
                scales=config["scales"],
                mode_rank=config["mode_ranks"][dataset],
                projection=control["projection"],
                partition=control["partition"],
            )
            for alpha in config["scope_strengths"]:
                conditioned = arms + float(alpha) * (hard_arms - arms)
                forecast = np.sum(
                    weights.transpose(0, 2, 1) * conditioned,
                    axis=1,
                )
                audit[f"{control['id']}_a{alpha}_change_normalized_rms"] = float(
                    np.sqrt(np.mean((forecast - parent) ** 2)) / target_rms
                )
                audit["all_finite"] = bool(
                    audit["all_finite"]
                    and np.isfinite(conditioned).all()
                    and np.isfinite(forecast).all()
                )
                for horizon in config["horizons"]:
                    for metric in ("mse", "mae"):
                        value = prefix_metric(forecast, target, horizon, metric)
                        parent_value = prefix_metric(parent, target, horizon, metric)
                        cells.append(
                            {
                                "dataset": dataset,
                                "arm": control["id"],
                                "alpha": alpha,
                                "horizon": horizon,
                                "metric": metric,
                                "value": value,
                                "parent_value": parent_value,
                                "gain_percent": 100.0 * (1.0 - value / parent_value),
                            }
                        )
        audits.append(audit)
    summaries = summarize(cells)
    decision = decide(summaries, audits, config)
    decision.update(
        {
            "diagnostic_id": config["diagnostic_id"],
            "datasets": len(config["datasets"]),
            "probe_rows_per_dataset": config["probe_rows"],
            "cells": len(cells),
            "evidence_limits": config["evidence_limits"],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "diagnostic_cells.csv", cells)
    write_csv(args.output_dir / "diagnostic_summary.csv", summaries)
    write_csv(args.output_dir / "run_audit.csv", audits)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_frsc_d1={decision['decision']} "
        f"alpha={decision['selected_alpha']} test_accessed=false"
    )


def synthetic_smoke(config: dict[str, Any]) -> None:
    generator = np.random.default_rng(20260722)
    arms = generator.normal(size=(8, 5, 720))
    logits = generator.normal(size=(8, 720, 5))
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    parent = np.sum(weights.transpose(0, 2, 1) * arms, axis=1)
    hard = projected_arms(
        arms,
        scales=config["scales"],
        mode_rank=109,
        projection="scope",
        partition="canonical",
    )
    alpha = config["scope_strengths"][0]
    forecast = np.sum(
        weights.transpose(0, 2, 1) * (arms + alpha * (hard - arms)),
        axis=1,
    )
    if forecast.shape != parent.shape or not np.isfinite(forecast).all():
        raise RuntimeError("FRSC synthetic shape/finite contract failed")
    if np.max(np.abs(forecast - parent)) <= 0 or not 0 < 1.0 - alpha < 1.0:
        raise RuntimeError("FRSC full-rank conditioning contract failed")
    print("iscf_frsc_d1_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    run(args, config)


if __name__ == "__main__":
    main()
