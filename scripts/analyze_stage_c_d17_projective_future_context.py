#!/usr/bin/env python3
"""Run the D17 projective future-context offline diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MODELS = (
    "parent",
    "pointwise_wide",
    "causal_ordered",
    "causal_row_shuffled",
    "symmetric_ordered",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def diagnostics_path(
    root: Path,
    arm: str,
    dataset: str,
    seed: int,
    artifact_name: str,
) -> Path:
    return (
        root
        / arm
        / dataset
        / "h720_full"
        / f"seed{seed}"
        / artifact_name
    )


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
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coordinate_features(
    length: int,
    harmonics: int,
    domain_length: int,
) -> np.ndarray:
    coordinate = np.arange(length, dtype=np.float64)
    denominator = max(domain_length - 1, 1)
    normalized = coordinate / denominator
    features = [
        np.ones(length, dtype=np.float64),
        normalized,
        normalized**2,
    ]
    for frequency in range(1, harmonics + 1):
        phase = 2.0 * np.pi * frequency * normalized
        features.extend((np.sin(phase), np.cos(phase)))
    return np.stack(features, axis=-1)


def pointwise_features(
    forecast: np.ndarray,
    harmonics: int,
    domain_length: int,
) -> np.ndarray:
    coordinates = coordinate_features(
        forecast.shape[1],
        harmonics,
        domain_length,
    )
    coordinates = np.broadcast_to(
        coordinates[None, :, :],
        (forecast.shape[0], *coordinates.shape),
    )
    current = forecast[:, :, None]
    return np.concatenate((coordinates, current, current * coordinates), axis=-1)


def lag_context(
    forecast: np.ndarray,
    lags: list[int],
    *,
    direction: int,
) -> np.ndarray:
    rows, length = forecast.shape
    coordinate = np.arange(length)
    current = forecast[:, :, None]
    features = []
    for lag in lags:
        source = coordinate + direction * lag
        available = (source >= 0) & (source < length)
        clipped = np.clip(source, 0, length - 1)
        value = forecast[:, clipped, None]
        mask = np.broadcast_to(available[None, :, None], (rows, length, 1))
        masked_value = np.where(mask, value, current)
        features.extend(
            (
                masked_value,
                current - masked_value,
                mask.astype(np.float64),
            )
        )
    return np.concatenate(features, axis=-1)


def derangement(row_count: int, rng: np.random.Generator) -> np.ndarray:
    if row_count < 2:
        raise ValueError("row-shuffled control requires at least two rows")
    identity = np.arange(row_count)
    for _ in range(1000):
        candidate = rng.permutation(row_count)
        if np.all(candidate != identity):
            return candidate
    return np.roll(identity, 1)


def build_features(
    forecast: np.ndarray,
    lags: list[int],
    harmonics: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    pointwise = pointwise_features(
        forecast,
        harmonics,
        forecast.shape[1],
    )
    causal = lag_context(forecast, lags, direction=-1)
    shuffled_forecast = forecast[derangement(forecast.shape[0], rng)]
    shuffled = lag_context(shuffled_forecast, lags, direction=-1)
    future = lag_context(forecast, lags, direction=1)
    return {
        "pointwise_wide": pointwise,
        "causal_ordered": np.concatenate((pointwise, causal), axis=-1),
        "causal_row_shuffled": np.concatenate(
            (pointwise, shuffled),
            axis=-1,
        ),
        "symmetric_ordered": np.concatenate(
            (pointwise, causal, future),
            axis=-1,
        ),
    }


def fit_residual_correction(
    train_features: np.ndarray,
    train_residual: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(
        train_features.reshape(-1, train_features.shape[-1])
    )
    test_x = scaler.transform(
        test_features.reshape(-1, test_features.shape[-1])
    )
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(train_x, train_residual.reshape(-1))
    return model.predict(test_x).reshape(test_features.shape[:2])


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def relative_gain(candidate: float, reference: float) -> float:
    return 100.0 * (1.0 - candidate / reference)


def prefix_invariance_gap(
    forecast: np.ndarray,
    lags: list[int],
    harmonics: int,
    horizons: list[int],
) -> float:
    domain_length = forecast.shape[1]
    full_pointwise = pointwise_features(
        forecast,
        harmonics,
        domain_length,
    )
    full_causal = lag_context(forecast, lags, direction=-1)
    full = np.concatenate((full_pointwise, full_causal), axis=-1)
    maximum = 0.0
    for horizon in horizons:
        cropped = forecast[:, :horizon]
        cropped_features = np.concatenate(
            (
                pointwise_features(cropped, harmonics, domain_length),
                lag_context(cropped, lags, direction=-1),
            ),
            axis=-1,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(full[:, :horizon] - cropped_features))),
        )
    return maximum


def run_dataset(
    carrier: str,
    dataset: str,
    validation_forecast: np.ndarray,
    validation_target: np.ndarray,
    test_forecast: np.ndarray,
    test_target: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    seed = int(config["seed"])
    offset = seed + sum(map(ord, carrier + dataset))
    validation_rng = np.random.default_rng(offset)
    test_rng = np.random.default_rng(offset + 104729)
    lags = [int(value) for value in config["lags"]]
    harmonics = int(config["coordinate_harmonics"])
    alpha = float(config["ridge_alpha"])
    horizons = [int(value) for value in config["horizons"]]
    validation_features = build_features(
        validation_forecast,
        lags,
        harmonics,
        validation_rng,
    )
    test_features = build_features(
        test_forecast,
        lags,
        harmonics,
        test_rng,
    )
    validation_residual = validation_target - validation_forecast
    predictions = {"parent": test_forecast}
    for name in validation_features:
        correction = fit_residual_correction(
            validation_features[name],
            validation_residual,
            test_features[name],
            alpha,
        )
        predictions[name] = test_forecast + correction
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        for horizon in horizons:
            rows.append(
                {
                    "carrier": carrier,
                    "dataset": dataset,
                    "fit_split": config["fit_split"],
                    "evaluation_split": config["evaluation_split"],
                    "model": model_name,
                    "horizon": horizon,
                    "train_rows": int(validation_forecast.shape[0]),
                    "test_rows": int(test_forecast.shape[0]),
                    "feature_count": (
                        0
                        if model_name == "parent"
                        else int(validation_features[model_name].shape[-1])
                    ),
                    "mse": mse(
                        predictions[model_name][:, :horizon],
                        test_target[:, :horizon],
                    ),
                }
            )
    gap = max(
        prefix_invariance_gap(
            validation_forecast,
            lags,
            harmonics,
            horizons,
        ),
        prefix_invariance_gap(
            test_forecast,
            lags,
            harmonics,
            horizons,
        ),
    )
    return rows, gap


def aggregate_metrics(
    fold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[float]] = {}
    for row in fold_rows:
        key = (
            row["carrier"],
            row["dataset"],
            row["model"],
            row["horizon"],
        )
        groups.setdefault(key, []).append(float(row["mse"]))
    return [
        {
            "carrier": carrier,
            "dataset": dataset,
            "model": model,
            "horizon": horizon,
            "mse": float(np.mean(values)),
        }
        for (carrier, dataset, model, horizon), values in sorted(groups.items())
    ]


def comparison_rows(
    aggregate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cell = {
        (
            row["carrier"],
            row["dataset"],
            row["horizon"],
            row["model"],
        ): float(row["mse"])
        for row in aggregate_rows
    }
    rows = []
    cells = sorted(
        {
            (row["carrier"], row["dataset"], row["horizon"])
            for row in aggregate_rows
        }
    )
    for carrier, dataset, horizon in cells:
        parent = by_cell[(carrier, dataset, horizon, "parent")]
        pointwise = by_cell[(carrier, dataset, horizon, "pointwise_wide")]
        causal = by_cell[(carrier, dataset, horizon, "causal_ordered")]
        shuffled = by_cell[
            (carrier, dataset, horizon, "causal_row_shuffled")
        ]
        symmetric = by_cell[(carrier, dataset, horizon, "symmetric_ordered")]
        rows.append(
            {
                "carrier": carrier,
                "dataset": dataset,
                "horizon": horizon,
                "parent_mse": parent,
                "pointwise_mse": pointwise,
                "causal_mse": causal,
                "shuffled_mse": shuffled,
                "symmetric_mse": symmetric,
                "pointwise_gain_over_parent_percent": relative_gain(
                    pointwise,
                    parent,
                ),
                "causal_gain_over_parent_percent": relative_gain(
                    causal,
                    parent,
                ),
                "causal_gain_over_pointwise_percent": relative_gain(
                    causal,
                    pointwise,
                ),
                "causal_gain_over_shuffled_percent": relative_gain(
                    causal,
                    shuffled,
                ),
                "symmetric_gain_over_pointwise_percent": relative_gain(
                    symmetric,
                    pointwise,
                ),
            }
        )
    return rows


def macro_gain(
    rows: list[dict[str, Any]],
    candidate_field: str,
    reference_field: str,
) -> float:
    candidate = float(np.mean([row[candidate_field] for row in rows]))
    reference = float(np.mean([row[reference_field] for row in rows]))
    return relative_gain(candidate, reference)


def summarize(
    rows: list[dict[str, Any]],
    prefix_gaps: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    causal_vs_pointwise = macro_gain(rows, "causal_mse", "pointwise_mse")
    causal_vs_shuffled = macro_gain(rows, "causal_mse", "shuffled_mse")
    carrier_gains = {}
    for carrier in config["parent_arms"]:
        selected = [row for row in rows if row["carrier"] == carrier]
        carrier_gains[carrier] = macro_gain(
            selected,
            "causal_mse",
            "pointwise_mse",
        )
    dataset_gains = {}
    for dataset in config["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        dataset_gains[dataset] = macro_gain(
            selected,
            "causal_mse",
            "pointwise_mse",
        )
    carrier_dataset_gains = {}
    for carrier in config["parent_arms"]:
        for dataset in config["datasets"]:
            selected = [
                row
                for row in rows
                if row["carrier"] == carrier and row["dataset"] == dataset
            ]
            carrier_dataset_gains[f"{carrier}/{dataset}"] = macro_gain(
                selected,
                "causal_mse",
                "pointwise_mse",
            )
    horizon_gains = {}
    for horizon in config["horizons"]:
        selected = [row for row in rows if row["horizon"] == horizon]
        horizon_gains[str(horizon)] = macro_gain(
            selected,
            "causal_mse",
            "pointwise_mse",
        )
    maximum_gap = max(prefix_gaps.values())
    gates = config["gates"]
    gate_results = {
        "causal_gain_over_pointwise": (
            causal_vs_pointwise
            >= gates["causal_gain_over_pointwise_percent_min"]
        ),
        "causal_gain_over_shuffled": (
            causal_vs_shuffled
            >= gates["causal_gain_over_shuffled_percent_min"]
        ),
        "positive_datasets": (
            sum(value > 0.0 for value in dataset_gains.values())
            >= gates["positive_datasets_min"]
        ),
        "positive_carriers": (
            sum(value > 0.0 for value in carrier_gains.values())
            >= gates["positive_carriers_min"]
        ),
        "positive_carrier_datasets": (
            sum(value > 0.0 for value in carrier_dataset_gains.values())
            >= gates["positive_carrier_datasets_min"]
        ),
        "positive_horizons": (
            sum(value > 0.0 for value in horizon_gains.values())
            >= gates["positive_horizons_min"]
        ),
        "prefix_invariance": (
            maximum_gap <= gates["prefix_invariance_max_abs_gap_max"]
        ),
    }
    return {
        "causal_gain_over_pointwise_percent": causal_vs_pointwise,
        "causal_gain_over_shuffled_percent": causal_vs_shuffled,
        "pointwise_gain_over_parent_percent": macro_gain(
            rows,
            "pointwise_mse",
            "parent_mse",
        ),
        "causal_gain_over_parent_percent": macro_gain(
            rows,
            "causal_mse",
            "parent_mse",
        ),
        "symmetric_gain_over_pointwise_percent": macro_gain(
            rows,
            "symmetric_mse",
            "pointwise_mse",
        ),
        "carrier_causal_gain_over_pointwise_percent": carrier_gains,
        "dataset_causal_gain_over_pointwise_percent": dataset_gains,
        "carrier_dataset_causal_gain_over_pointwise_percent": (
            carrier_dataset_gains
        ),
        "horizon_causal_gain_over_pointwise_percent": horizon_gains,
        "positive_datasets": sum(
            value > 0.0 for value in dataset_gains.values()
        ),
        "positive_carriers": sum(
            value > 0.0 for value in carrier_gains.values()
        ),
        "positive_carrier_datasets": sum(
            value > 0.0 for value in carrier_dataset_gains.values()
        ),
        "positive_horizons": sum(
            value > 0.0 for value in horizon_gains.values()
        ),
        "prefix_invariance_max_abs_gap": maximum_gap,
        "prefix_invariance_by_dataset": prefix_gaps,
        "gate_results": gate_results,
        "all_gates_pass": all(gate_results.values()),
    }


def main() -> None:
    args = parse_args()
    config, config_hash = load_config(args.config)
    fold_rows: list[dict[str, Any]] = []
    prefix_gaps: dict[str, float] = {}
    for carrier in config["parent_arms"]:
        for dataset in config["datasets"]:
            validation_path = diagnostics_path(
                args.validation_root,
                carrier,
                dataset,
                int(config["seed"]),
                "pcsd_validation_diagnostics.npz",
            )
            test_path = diagnostics_path(
                args.test_root,
                carrier,
                dataset,
                int(config["seed"]),
                "pcsd_test_audit_diagnostics.npz",
            )
            if not validation_path.is_file():
                raise FileNotFoundError(validation_path)
            if not test_path.is_file():
                raise FileNotFoundError(test_path)
            with np.load(validation_path, allow_pickle=False) as payload:
                validation_forecast = payload["probe_fused"].astype(np.float64)
                validation_target = payload["probe_targets"].astype(np.float64)
            with np.load(test_path, allow_pickle=False) as payload:
                test_forecast = payload["probe_fused"].astype(np.float64)
                test_target = payload["probe_targets"].astype(np.float64)
            if (
                validation_forecast.shape != validation_target.shape
                or test_forecast.shape != test_target.shape
                or validation_forecast.shape[1] != 720
                or test_forecast.shape[1] != 720
            ):
                raise ValueError(
                    f"unexpected probe shape for {carrier}/{dataset}: "
                    f"validation={validation_forecast.shape}/"
                    f"{validation_target.shape}, "
                    f"test={test_forecast.shape}/{test_target.shape}"
                )
            dataset_rows, gap = run_dataset(
                carrier,
                dataset,
                validation_forecast,
                validation_target,
                test_forecast,
                test_target,
                config,
            )
            fold_rows.extend(dataset_rows)
            prefix_gaps[f"{carrier}/{dataset}"] = gap

    aggregate_rows = aggregate_metrics(fold_rows)
    comparisons = comparison_rows(aggregate_rows)
    summary = summarize(comparisons, prefix_gaps, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "transfer_metrics.csv", fold_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "comparison_cells.csv", comparisons)
    payload = {
        "candidate_id": config["candidate_id"],
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "validation_root": str(args.validation_root),
        "test_root": str(args.test_root),
        "definitions": {
            "pointwise_wide": (
                "Fixed-alpha ridge residual correction using only the current "
                "parent forecast and coordinate polynomial/Fourier features."
            ),
            "causal_ordered": (
                "The pointwise control plus ordered parent-forecast lags. Every "
                "coordinate uses only draft coordinates at or before itself."
            ),
            "causal_row_shuffled": (
                "A same-dimensional control whose lag context comes from a "
                "different probe row under a deterministic derangement."
            ),
            "symmetric_ordered": (
                "A non-projective upper control using both lagged and leading "
                "parent-forecast coordinates."
            ),
            "relative_gain_percent": (
                "100 * (1 - candidate MSE / reference MSE); positive is better."
            ),
        },
        "boundary": (
            "This is a test-informed diagnostic. A fixed residual corrector is "
            "fitted only on saved validation probes and evaluated on separate "
            "saved test probes. It may support a conditional problem "
            "hypothesis, but it is not an end-to-end forecasting method, formal "
            "ablation, or paper-facing effectiveness result."
        ),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
