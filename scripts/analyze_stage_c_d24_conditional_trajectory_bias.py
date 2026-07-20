#!/usr/bin/env python3
"""Audit past-identifiable coarse trajectory bias on frozen validation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_stage_c_pcsd_cf_checkpoint import load_model, sequential_loader


DEFAULT_CONFIG = Path("configs/stage_c_d24_conditional_trajectory_bias.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ridge_predict(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    evaluation_features: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    if fit_features.ndim != 2 or evaluation_features.ndim != 2:
        raise ValueError("ridge features must be matrices")
    if fit_targets.ndim != 2:
        raise ValueError("ridge targets must be a matrix")
    if fit_features.shape[0] != fit_targets.shape[0]:
        raise ValueError("ridge feature/target row mismatch")
    if fit_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("fit/evaluation feature width mismatch")

    if fit_features.shape[1]:
        feature_mean = fit_features.mean(axis=0)
        feature_scale = fit_features.std(axis=0)
        feature_scale[feature_scale < 1e-8] = 1.0
        fit_scaled = (fit_features - feature_mean) / feature_scale
        evaluation_scaled = (
            evaluation_features - feature_mean
        ) / feature_scale
    else:
        fit_scaled = fit_features
        evaluation_scaled = evaluation_features

    fit_design = np.concatenate(
        [np.ones((fit_scaled.shape[0], 1)), fit_scaled],
        axis=1,
    )
    evaluation_design = np.concatenate(
        [np.ones((evaluation_scaled.shape[0], 1)), evaluation_scaled],
        axis=1,
    )
    gram = fit_design.T @ fit_design
    regularizer = np.eye(gram.shape[0]) * ridge_lambda
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer,
        fit_design.T @ fit_targets,
    )
    return evaluation_design @ coefficients


def corrected_mse(
    residual_sum: np.ndarray,
    residual_square_sum: np.ndarray,
    correction: np.ndarray,
    future_bin_size: int,
    horizon: int,
) -> float:
    bin_count = horizon // future_bin_size
    new_square_sum = (
        residual_square_sum[:, :bin_count]
        - 2.0 * correction[:, :bin_count] * residual_sum[:, :bin_count]
        + future_bin_size * np.square(correction[:, :bin_count])
    )
    denominator = residual_sum.shape[0] * horizon
    return float(new_square_sum.sum() / denominator)


def chronological_masks(
    origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_origins = np.unique(origins)
    if unique_origins.size < 6:
        raise ValueError("too few forecast origins for chronological thirds")
    first_boundary = unique_origins.size // 3
    second_boundary = 2 * unique_origins.size // 3
    fit_origins = unique_origins[:first_boundary]
    purge_origins = unique_origins[first_boundary:second_boundary]
    evaluation_origins = unique_origins[second_boundary:]
    return (
        np.isin(origins, fit_origins),
        np.isin(origins, purge_origins),
        np.isin(origins, evaluation_origins),
    )


def history_features(
    history_rows: torch.Tensor,
    history_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = history_rows.detach().to(torch.float64).cpu()
    if history.shape[1] % history_bins:
        raise ValueError("history length must divide history_bins")
    mean = history.mean(dim=1, keepdim=True)
    std = history.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    normalized = (history - mean) / std
    bin_width = history.shape[1] // history_bins
    ordered = normalized.reshape(-1, history_bins, bin_width).mean(dim=2)
    marginal = torch.cat(
        [
            mean,
            std,
            history[:, -1:] - history[:, :1],
            torch.diff(history, dim=1).square().mean(dim=1, keepdim=True).sqrt(),
        ],
        dim=1,
    )
    sorted_history = torch.sort(ordered, dim=1).values
    return (
        marginal.numpy(),
        ordered.numpy(),
        sorted_history.numpy(),
    )


def collect_validation_statistics(
    run_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, np.ndarray | int]:
    model, _effective, official_args = load_model(run_dir, device)
    loader = sequential_loader(official_args, "val")
    history_bins = int(config["history_bins"])
    future_bin_size = int(config["future_bin_size"])
    prediction_length = int(config["prediction_length"])
    future_bins = prediction_length // future_bin_size

    marginal_parts: list[np.ndarray] = []
    ordered_parts: list[np.ndarray] = []
    sorted_parts: list[np.ndarray] = []
    residual_sum_parts: list[np.ndarray] = []
    residual_square_sum_parts: list[np.ndarray] = []
    origin_parts: list[np.ndarray] = []
    channel_parts: list[np.ndarray] = []
    origin_offset = 0
    channel_count: int | None = None

    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            if batch_x.shape[1] != int(config["history_length"]):
                raise ValueError(
                    f"unexpected history length: {batch_x.shape[1]}"
                )
            target = batch_y[:, -prediction_length:, :].float().to(device)
            forecast = model(
                batch_x,
                target,
                is_training=False,
                target_prefix=prediction_length,
            )[0]
            if forecast.shape != target.shape:
                raise ValueError(
                    f"forecast/target shape mismatch: "
                    f"{forecast.shape} vs {target.shape}"
                )
            batch_size, _, current_channels = batch_x.shape
            if channel_count is None:
                channel_count = current_channels
            elif channel_count != current_channels:
                raise ValueError("channel count changed within validation")

            history_rows = batch_x.permute(0, 2, 1).reshape(
                -1,
                batch_x.shape[1],
            )
            marginal, ordered, sorted_history = history_features(
                history_rows,
                history_bins,
            )
            residual = (target - forecast).permute(0, 2, 1).reshape(
                -1,
                future_bins,
                future_bin_size,
            )
            marginal_parts.append(marginal)
            ordered_parts.append(ordered)
            sorted_parts.append(sorted_history)
            residual_sum_parts.append(
                residual.sum(dim=2).to(torch.float64).cpu().numpy()
            )
            residual_square_sum_parts.append(
                residual.square().sum(dim=2).to(torch.float64).cpu().numpy()
            )
            origins = np.arange(
                origin_offset,
                origin_offset + batch_size,
                dtype=np.int64,
            )
            origin_parts.append(np.repeat(origins, current_channels))
            channel_parts.append(
                np.tile(np.arange(current_channels), batch_size)
            )
            origin_offset += batch_size

    if channel_count is None:
        raise RuntimeError("validation loader produced no rows")
    payload: dict[str, np.ndarray | int] = {
        "marginal": np.concatenate(marginal_parts),
        "ordered": np.concatenate(ordered_parts),
        "sorted_history": np.concatenate(sorted_parts),
        "residual_sum": np.concatenate(residual_sum_parts),
        "residual_square_sum": np.concatenate(residual_square_sum_parts),
        "origins": np.concatenate(origin_parts),
        "channels": np.concatenate(channel_parts),
        "channel_count": channel_count,
        "origin_count": origin_offset,
    }
    arrays = [value for value in payload.values() if isinstance(value, np.ndarray)]
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("non-finite validation statistic")
    return payload


def feature_families(
    payload: dict[str, np.ndarray | int],
    recent_bins: int,
) -> dict[str, np.ndarray]:
    marginal = np.asarray(payload["marginal"])
    ordered = np.asarray(payload["ordered"])
    sorted_history = np.asarray(payload["sorted_history"])
    channels = np.asarray(payload["channels"])
    channel_count = int(payload["channel_count"])
    channel_features = np.eye(channel_count, dtype=np.float64)[channels]
    common = np.concatenate([channel_features, marginal], axis=1)
    return {
        "global": np.empty((marginal.shape[0], 0), dtype=np.float64),
        "channel": channel_features,
        "marginal": common,
        "recent": np.concatenate(
            [common, ordered[:, -recent_bins:]],
            axis=1,
        ),
        "sorted_history": np.concatenate(
            [common, sorted_history],
            axis=1,
        ),
        "ordered_history": np.concatenate([common, ordered], axis=1),
        "target_shuffled": np.concatenate([common, ordered], axis=1),
    }


def evaluate_run(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    if args.run_dir is None or args.output_dir is None:
        raise ValueError("run-dir and output-dir are required")
    if args.arm not in config["arms"]:
        raise ValueError(f"unexpected arm: {args.arm}")
    if args.dataset not in config["datasets"]:
        raise ValueError(f"unexpected dataset: {args.dataset}")
    if config["evaluation_split"] != "val":
        raise ValueError("D24 only permits validation inference")
    if config["authorization"]["official_test_access_authorized"]:
        raise ValueError("D24 config must keep official test disabled")
    if config["authorization"]["remote_training_authorized"]:
        raise ValueError("D24 config must keep remote training disabled")

    checkpoint = args.run_dir / "checkpoint.pt"
    checkpoint_before = file_sha256(checkpoint)
    payload = collect_validation_statistics(
        args.run_dir,
        torch.device(args.device),
        config,
    )
    checkpoint_after = file_sha256(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("frozen checkpoint mutated during D24 inference")

    origins = np.asarray(payload["origins"])
    fit_mask, purge_mask, evaluation_mask = chronological_masks(origins)
    if np.any(fit_mask & purge_mask) or np.any(fit_mask & evaluation_mask):
        raise RuntimeError("chronological partitions overlap")
    features = feature_families(
        payload,
        int(config["recent_control_bins"]),
    )
    residual_sum = np.asarray(payload["residual_sum"])
    residual_square_sum = np.asarray(payload["residual_square_sum"])
    fit_target = residual_sum[fit_mask] / int(config["future_bin_size"])
    evaluation_sum = residual_sum[evaluation_mask]
    evaluation_square_sum = residual_square_sum[evaluation_mask]

    rows: list[dict[str, Any]] = []
    fit_origin_count = np.unique(origins[fit_mask]).size
    evaluation_origin_count = np.unique(origins[evaluation_mask]).size
    for ridge_lambda in config["ridge_lambdas"]:
        for family, values in features.items():
            if family == "target_shuffled":
                rng = np.random.default_rng(
                    int(config["target_shuffle_seed"])
                )
                predictions = []
                fit_rows_per_origin = (
                    fit_mask.sum() // fit_origin_count
                )
                target_by_origin = fit_target.reshape(
                    fit_origin_count,
                    fit_rows_per_origin,
                    -1,
                )
                for _ in range(int(config["target_shuffle_repeats"])):
                    permutation = rng.permutation(fit_origin_count)
                    shuffled_target = target_by_origin[permutation].reshape(
                        fit_target.shape
                    )
                    predictions.append(
                        ridge_predict(
                            values[fit_mask],
                            shuffled_target,
                            values[evaluation_mask],
                            float(ridge_lambda),
                        )
                    )
                correction = np.mean(predictions, axis=0)
            else:
                correction = ridge_predict(
                    values[fit_mask],
                    fit_target,
                    values[evaluation_mask],
                    float(ridge_lambda),
                )
            for horizon in config["horizons"]:
                uncorrected = corrected_mse(
                    evaluation_sum,
                    evaluation_square_sum,
                    np.zeros_like(correction),
                    int(config["future_bin_size"]),
                    int(horizon),
                )
                mse = corrected_mse(
                    evaluation_sum,
                    evaluation_square_sum,
                    correction,
                    int(config["future_bin_size"]),
                    int(horizon),
                )
                rows.append(
                    {
                        "arm": args.arm,
                        "dataset": args.dataset,
                        "split": "validation_last_third",
                        "ridge_lambda": ridge_lambda,
                        "feature_family": family,
                        "feature_dimension": values.shape[1],
                        "horizon": horizon,
                        "mse": mse,
                        "uncorrected_mse": uncorrected,
                        "gain_vs_uncorrected_percent": (
                            100.0 * (1.0 - mse / uncorrected)
                        ),
                        "fit_origins": fit_origin_count,
                        "purge_origins": np.unique(origins[purge_mask]).size,
                        "evaluation_origins": evaluation_origin_count,
                        "channels": int(payload["channel_count"]),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics.csv", rows)
    metadata = {
        "candidate_version": config["candidate_version"],
        "arm": args.arm,
        "dataset": args.dataset,
        "evaluation_split": "val",
        "checkpoint_sha256": checkpoint_before,
        "checkpoint_mutated": False,
        "origin_count": int(payload["origin_count"]),
        "row_count": int(origins.size),
        "fit_origin_count": fit_origin_count,
        "purge_origin_count": int(np.unique(origins[purge_mask]).size),
        "evaluation_origin_count": evaluation_origin_count,
        "all_finite": True,
        "config_sha256": file_sha256(args.config),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"d24_run_done arm={args.arm} dataset={args.dataset} "
        f"origins={payload['origin_count']} rows={origins.size}"
    )


def comparison_summary(
    cell_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in cell_rows:
        grouped[
            (
                str(row["arm"]),
                float(row["ridge_lambda"]),
                str(row["comparison"]),
            )
        ].append(row)

    output = []
    for (arm, ridge_lambda, comparison), rows in sorted(grouped.items()):
        dataset_gains: dict[str, list[float]] = defaultdict(list)
        horizon_gains: dict[int, list[float]] = defaultdict(list)
        gains = []
        for row in rows:
            gain = float(row["mse_gain_percent"])
            gains.append(gain)
            dataset_gains[str(row["dataset"])].append(gain)
            horizon_gains[int(row["horizon"])].append(gain)
        output.append(
            {
                "arm": arm,
                "ridge_lambda": ridge_lambda,
                "comparison": comparison,
                "macro_mse_gain_percent": float(np.mean(gains)),
                "positive_cells": sum(value > 0 for value in gains),
                "positive_datasets": sum(
                    np.mean(value) > 0 for value in dataset_gains.values()
                ),
                "positive_horizons": sum(
                    np.mean(value) > 0 for value in horizon_gains.values()
                ),
            }
        )
    return output


def gate_pass(
    summary: dict[tuple[str, float, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    primary_lambda = float(config["primary_ridge_lambda"])
    gates = config["gates"]
    comparisons = {
        "ordered_vs_marginal": float(
            gates["ordered_vs_marginal_macro_mse_gain_percent_min"]
        ),
        "ordered_vs_sorted": float(
            gates["ordered_vs_sorted_macro_mse_gain_percent_min"]
        ),
        "ordered_vs_target_shuffled": float(
            gates[
                "ordered_vs_target_shuffled_macro_mse_gain_percent_min"
            ]
        ),
    }
    checks: dict[str, bool] = {}
    for arm in config["arms"]:
        for comparison, macro_min in comparisons.items():
            row = summary[(arm, primary_lambda, comparison)]
            checks[f"{arm}:{comparison}:primary"] = bool(
                float(row["macro_mse_gain_percent"]) >= macro_min
                and int(row["positive_cells"])
                >= int(gates["positive_cells_min"])
                and int(row["positive_datasets"])
                >= int(gates["positive_datasets_min"])
                and int(row["positive_horizons"])
                >= int(gates["positive_horizons_min"])
            )
            for ridge_lambda in config["ridge_lambdas"]:
                sensitivity = summary[
                    (arm, float(ridge_lambda), comparison)
                ]
                checks[
                    f"{arm}:{comparison}:lambda={ridge_lambda}"
                ] = bool(
                    float(sensitivity["macro_mse_gain_percent"]) > 0
                )
    return all(checks.values()), checks


def aggregate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.aggregate_root is None or args.output_dir is None:
        raise ValueError("aggregate-root and output-dir are required")
    metric_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    missing = []
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            directory = args.aggregate_root / arm / dataset
            metrics_path = directory / "metrics.csv"
            metadata_path = directory / "metadata.json"
            if not metrics_path.is_file() or not metadata_path.is_file():
                missing.append(f"{arm}/{dataset}")
                continue
            metric_rows.extend(read_csv(metrics_path))
            metadata_rows.append(
                json.loads(metadata_path.read_text(encoding="utf-8"))
            )
    if missing:
        raise FileNotFoundError(f"missing D24 runs: {missing}")
    expected_rows = (
        len(config["arms"])
        * len(config["datasets"])
        * len(config["ridge_lambdas"])
        * len(config["feature_families"])
        * len(config["horizons"])
    )
    if len(metric_rows) != expected_rows:
        raise ValueError(
            f"unexpected D24 metric rows: {len(metric_rows)} "
            f"!= {expected_rows}"
        )
    protocol_valid = all(
        row["candidate_version"] == config["candidate_version"]
        and row["evaluation_split"] == "val"
        and row["checkpoint_mutated"] is False
        and row["all_finite"] is True
        for row in metadata_rows
    )

    lookup: dict[tuple[str, str, float, str, int], float] = {}
    for row in metric_rows:
        key = (
            row["arm"],
            row["dataset"],
            float(row["ridge_lambda"]),
            row["feature_family"],
            int(row["horizon"]),
        )
        lookup[key] = float(row["mse"])
    references = {
        "ordered_vs_global": "global",
        "ordered_vs_channel": "channel",
        "ordered_vs_marginal": "marginal",
        "ordered_vs_recent": "recent",
        "ordered_vs_sorted": "sorted_history",
        "ordered_vs_target_shuffled": "target_shuffled",
    }
    cells = []
    for arm in config["arms"]:
        for dataset in config["datasets"]:
            for ridge_lambda in config["ridge_lambdas"]:
                for horizon in config["horizons"]:
                    ordered = lookup[
                        (
                            arm,
                            dataset,
                            float(ridge_lambda),
                            "ordered_history",
                            int(horizon),
                        )
                    ]
                    for comparison, reference_family in references.items():
                        reference = lookup[
                            (
                                arm,
                                dataset,
                                float(ridge_lambda),
                                reference_family,
                                int(horizon),
                            )
                        ]
                        cells.append(
                            {
                                "arm": arm,
                                "dataset": dataset,
                                "ridge_lambda": ridge_lambda,
                                "horizon": horizon,
                                "comparison": comparison,
                                "candidate_mse": ordered,
                                "reference_mse": reference,
                                "mse_gain_percent": (
                                    100.0 * (1.0 - ordered / reference)
                                ),
                            }
                        )
    summaries = comparison_summary(cells)
    summary_lookup = {
        (
            row["arm"],
            float(row["ridge_lambda"]),
            row["comparison"],
        ): row
        for row in summaries
    }
    supported, checks = gate_pass(summary_lookup, config)
    decision = (
        "past_identifiable_coarse_deformation_supported_on_validation"
        if protocol_valid and supported
        else "coarse_linear_deformation_not_supported"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cell_comparisons.csv", cells)
    write_csv(args.output_dir / "comparison_summary.csv", summaries)
    write_csv(args.output_dir / "run_audit.csv", metadata_rows)
    decision_payload = {
        "candidate_version": config["candidate_version"],
        "decision": decision,
        "protocol_valid": protocol_valid,
        "problem_gate_supported": supported,
        "checks": checks,
        "matrix": {
            "runs": len(metadata_rows),
            "metric_rows": len(metric_rows),
            "comparison_cells": len(cells),
            "official_test_cells": 0,
        },
        "consequence": (
            "return Step4 source/narrative gate; method/training/test remain false"
            if supported
            else "close exact coarse linear diagnostic; remain Step2/3"
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# SC-D24-CTB Validation Diagnostic",
        "",
        f"- decision: `{decision}`",
        f"- protocol_valid: `{str(protocol_valid).lower()}`",
        f"- runs: `{len(metadata_rows)}/10`",
        "- official test access: `0`",
        "",
        "| Arm | Lambda | Comparison | Macro gain | Cells | Datasets | Horizons |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if float(row["ridge_lambda"]) != float(
            config["primary_ridge_lambda"]
        ):
            continue
        report_lines.append(
            f"| {row['arm']} | {row['ridge_lambda']} | "
            f"{row['comparison']} | "
            f"{float(row['macro_mse_gain_percent']):+.4f}% | "
            f"{row['positive_cells']}/20 | "
            f"{row['positive_datasets']}/5 | "
            f"{row['positive_horizons']}/4 |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(
        f"d24_aggregate_done decision={decision} "
        f"protocol_valid={protocol_valid}"
    )


def synthetic_smoke(config: dict[str, Any]) -> None:
    rng = np.random.default_rng(20260720)
    fit_features = rng.normal(size=(200, 8))
    evaluation_features = rng.normal(size=(100, 8))
    coefficients = rng.normal(size=(8, 15))
    fit_targets = fit_features @ coefficients
    expected = evaluation_features @ coefficients
    predicted = ridge_predict(
        fit_features,
        fit_targets,
        evaluation_features,
        0.1,
    )
    if np.mean((expected - predicted) ** 2) > 1e-4:
        raise AssertionError("ridge synthetic recovery failed")

    residual_sum = np.full((100, 15), 48.0)
    residual_square_sum = np.full((100, 15), 48.0)
    zero_mse = corrected_mse(
        residual_sum,
        residual_square_sum,
        np.zeros((100, 15)),
        48,
        720,
    )
    exact_mse = corrected_mse(
        residual_sum,
        residual_square_sum,
        np.ones((100, 15)),
        48,
        720,
    )
    if not math.isclose(zero_mse, 1.0) or abs(exact_mse) > 1e-12:
        raise AssertionError("exact corrected-MSE identity failed")
    origins = np.repeat(np.arange(90), 2)
    fit_mask, purge_mask, evaluation_mask = chronological_masks(origins)
    if (
        np.unique(origins[fit_mask]).size != 30
        or np.unique(origins[purge_mask]).size != 30
        or np.unique(origins[evaluation_mask]).size != 30
    ):
        raise AssertionError("chronological split synthetic check failed")

    cells = []
    for arm in config["arms"]:
        for ridge_lambda in config["ridge_lambdas"]:
            for comparison in (
                "ordered_vs_marginal",
                "ordered_vs_sorted",
                "ordered_vs_target_shuffled",
            ):
                for dataset in config["datasets"]:
                    for horizon in config["horizons"]:
                        cells.append(
                            {
                                "arm": arm,
                                "dataset": dataset,
                                "ridge_lambda": ridge_lambda,
                                "horizon": horizon,
                                "comparison": comparison,
                                "mse_gain_percent": 1.0,
                            }
                        )
    summaries = comparison_summary(cells)
    summary_lookup = {
        (
            row["arm"],
            float(row["ridge_lambda"]),
            row["comparison"],
        ): row
        for row in summaries
    }
    supported, checks = gate_pass(summary_lookup, config)
    if not supported or not all(checks.values()):
        raise AssertionError("aggregate gate synthetic check failed")
    if config["authorization"]["official_test_access_authorized"]:
        raise AssertionError("synthetic config unexpectedly authorizes test")
    print("d24_ctb_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.synthetic_smoke:
        synthetic_smoke(config)
    elif args.aggregate_root is not None:
        aggregate(args, config)
    else:
        evaluate_run(args, config)


if __name__ == "__main__":
    main()
