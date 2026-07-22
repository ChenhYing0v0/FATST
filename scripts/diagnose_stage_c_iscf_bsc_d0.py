#!/usr/bin/env python3
"""Run the frozen-validation barycentric scope-composition diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_bsc_d0.json"),
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


def dct_basis(length: int, rank: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    frequencies = torch.arange(rank, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * (positions + 0.5) * frequencies / length)
    basis[:, 0] *= math.sqrt(1.0 / length)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / length)
    return basis.to(dtype=dtype)


def group_indices(
    series_length: int,
    scale: int,
    scale_index: int,
    partition: str,
    seed: int = 15101,
) -> torch.Tensor:
    indices = torch.arange(series_length, dtype=torch.long)
    if partition == "random" and scale not in {1, series_length}:
        generator = torch.Generator(device="cpu").manual_seed(
            seed + 1009 * scale_index + scale
        )
        indices = indices[torch.randperm(series_length, generator=generator)]
    return indices.reshape(series_length // scale, scale)


def project_rows(
    values: torch.Tensor,
    *,
    scale: int,
    scale_index: int,
    mode_rank: int,
    projection: str,
    partition: str,
) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != 720:
        raise ValueError("values must have shape [N,720]")
    if projection == "global":
        rank = min(720, mode_rank)
        basis = dct_basis(720, rank, values.dtype)
        return (values @ basis) @ basis.T
    if projection != "scope":
        raise ValueError(f"unsupported projection: {projection}")
    rank = min(scale, max(1, round(mode_rank * scale / 720)))
    basis = dct_basis(scale, rank, values.dtype)
    indices = group_indices(720, scale, scale_index, partition)
    grouped = values[:, indices]
    reconstructed = torch.einsum(
        "ngs,sr->ngr", grouped, basis
    ) @ basis.T
    output = torch.zeros_like(values)
    output[:, indices.flatten()] = reconstructed.flatten(start_dim=1)
    return output


def transform(
    arms: np.ndarray,
    weights: np.ndarray,
    barycenter: np.ndarray,
    *,
    scales: list[int],
    mode_rank: int,
    projection: str,
    partition: str,
) -> tuple[np.ndarray, np.ndarray]:
    arms_tensor = torch.from_numpy(arms.astype(np.float64, copy=False))
    weights_tensor = torch.from_numpy(weights.astype(np.float64, copy=False))
    center = torch.from_numpy(barycenter.astype(np.float64, copy=False))
    affine_arms = []
    for scale_index, scale in enumerate(scales):
        deviation = arms_tensor[:, scale_index] - center
        projected = project_rows(
            deviation,
            scale=scale,
            scale_index=scale_index,
            mode_rank=mode_rank,
            projection=projection,
            partition=partition,
        )
        affine_arms.append(center + projected)
    stacked = torch.stack(affine_arms, dim=1)
    forecast = (weights_tensor.permute(0, 2, 1) * stacked).sum(dim=1)
    return forecast.numpy(), stacked.numpy()


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


def summarize(
    cells: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for control in config["projection_controls"]:
        for metric in ("mse", "mae"):
            selected = [
                row
                for row in cells
                if row["arm"] == control["id"] and row["metric"] == metric
            ]
            by_dataset: dict[str, list[float]] = {}
            by_horizon: dict[int, list[float]] = {}
            for row in selected:
                by_dataset.setdefault(row["dataset"], []).append(
                    row["gain_percent"]
                )
                by_horizon.setdefault(row["horizon"], []).append(
                    row["gain_percent"]
                )
            rows.append(
                {
                    "arm": control["id"],
                    "metric": metric,
                    "macro_gain_percent": mean(
                        row["gain_percent"] for row in selected
                    ),
                    "cell_wins": sum(
                        row["gain_percent"] > 0 for row in selected
                    ),
                    "dataset_wins": sum(
                        mean(values) > 0 for values in by_dataset.values()
                    ),
                    "horizon_wins": sum(
                        mean(values) > 0 for values in by_horizon.values()
                    ),
                }
            )
    return rows


def decide(
    summaries: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    scope = {
        row["metric"]: row
        for row in summaries
        if row["arm"] == "bsc_scope_canonical"
    }
    gate = config["diagnostic_lead"]
    lead = bool(
        scope["mse"]["macro_gain_percent"]
        >= gate["macro_mse_gain_percent_min"]
        and scope["mse"]["dataset_wins"] >= gate["dataset_wins_min"]
        and scope["mse"]["horizon_wins"] >= gate["horizon_wins_min"]
        and all(row["all_finite"] for row in audits)
        and max(row["barycenter_reconstruction_max_abs"] for row in audits)
        <= gate["barycenter_reconstruction_max_abs_max"]
    )
    return {
        "decision": (
            "diagnostic_positive_complete_BSC_step4_6_design"
            if lead
            else "exact_frozen_BSC_diagnostic_not_positive_return_step4"
        ),
        "diagnostic_lead": lead,
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
            arms = arrays["probe_arms"][:row_limit]
            weights = arrays["probe_direct_policy"][:row_limit]
            barycenter = arrays["probe_fused"][:row_limit]
            target = arrays["probe_targets"][:row_limit]
        expected_shapes = {
            "arms": (target.shape[0], len(config["scales"]), 720),
            "weights": (target.shape[0], 720, len(config["scales"])),
            "barycenter": (target.shape[0], 720),
            "target": (target.shape[0], 720),
        }
        actual_shapes = {
            "arms": arms.shape,
            "weights": weights.shape,
            "barycenter": barycenter.shape,
            "target": target.shape,
        }
        if target.shape[0] != row_limit or actual_shapes != expected_shapes:
            raise ValueError(
                f"unexpected probe shapes for {dataset}: {actual_shapes}"
            )
        recomposed = np.sum(weights.transpose(0, 2, 1) * arms, axis=1)
        reconstruction_gap = float(np.max(np.abs(recomposed - barycenter)))
        target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-12)
        audit = {
            "dataset": dataset,
            "rows": int(target.shape[0]),
            "barycenter_reconstruction_max_abs": reconstruction_gap,
            "all_finite": bool(
                np.isfinite(arms).all()
                and np.isfinite(weights).all()
                and np.isfinite(barycenter).all()
                and np.isfinite(target).all()
            ),
        }
        for control in config["projection_controls"]:
            forecast, affine_arms = transform(
                arms,
                weights,
                barycenter,
                scales=config["scales"],
                mode_rank=config["mode_ranks"][dataset],
                projection=control["projection"],
                partition=control["partition"],
            )
            audit["all_finite"] = bool(
                audit["all_finite"]
                and np.isfinite(forecast).all()
                and np.isfinite(affine_arms).all()
            )
            change = float(
                np.sqrt(np.mean((forecast - barycenter) ** 2)) / target_rms
            )
            pairwise = [
                float(
                    np.sqrt(
                        np.mean(
                            (
                                affine_arms[:, left]
                                - affine_arms[:, right]
                            )
                            ** 2
                        )
                    )
                    / target_rms
                )
                for left, right in combinations(range(len(config["scales"])), 2)
            ]
            audit[f"{control['id']}_change_normalized_rms"] = change
            audit[f"{control['id']}_pairwise_normalized_rms"] = mean(pairwise)
            for horizon in config["horizons"]:
                for metric in ("mse", "mae"):
                    parent_value = prefix_metric(
                        barycenter, target, horizon, metric
                    )
                    value = prefix_metric(forecast, target, horizon, metric)
                    cells.append(
                        {
                            "dataset": dataset,
                            "arm": control["id"],
                            "horizon": horizon,
                            "metric": metric,
                            "value": value,
                            "parent_value": parent_value,
                            "gain_percent": 100.0 * (1.0 - value / parent_value),
                        }
                    )
        audits.append(audit)
    summaries = summarize(cells, config)
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
        f"iscf_bsc_d0={decision['decision']} "
        f"datasets={len(audits)} test_accessed=false"
    )


def synthetic_smoke(config: dict[str, Any]) -> None:
    generator = np.random.default_rng(20260722)
    arms = generator.normal(size=(8, 5, 720))
    logits = generator.normal(size=(8, 720, 5))
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    barycenter = np.sum(weights.transpose(0, 2, 1) * arms, axis=1)
    scope, affine = transform(
        arms,
        weights,
        barycenter,
        scales=config["scales"],
        mode_rank=109,
        projection="scope",
        partition="canonical",
    )
    random, _ = transform(
        arms,
        weights,
        barycenter,
        scales=config["scales"],
        mode_rank=109,
        projection="scope",
        partition="random",
    )
    if scope.shape != (8, 720) or affine.shape != (8, 5, 720):
        raise RuntimeError("unexpected BSC synthetic shapes")
    if not np.isfinite(scope).all() or np.max(np.abs(scope - random)) <= 0:
        raise RuntimeError("BSC synthetic projection contract failed")
    print("iscf_bsc_d0_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
        return
    run(args, config)


if __name__ == "__main__":
    main()
