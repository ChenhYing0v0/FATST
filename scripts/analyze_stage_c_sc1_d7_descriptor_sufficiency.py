#!/usr/bin/env python3
"""Analyze SC1-D7 RGNB descriptor-sufficiency artifacts and hard gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--d7-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    for name in ("input_root", "d7_config", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)


def reduction(candidate: float, control: float) -> float:
    if min(candidate, control) <= 0.0:
        raise ValueError("metrics must be positive")
    return (control - candidate) / control


def load_artifacts(
    input_root: Path, datasets: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    metrics: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    for dataset in datasets:
        metrics.extend(read_csv(input_root / dataset / "d7_probe_metrics.csv"))
        metadata.extend(
            json.loads(
                (input_root / dataset / "d7_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    return metrics, metadata


def metric_index(
    metrics: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in metrics:
        key = (str(row["dataset"]), int(row["checkpoint_seed"]), str(row["arm"]))
        if key in index:
            raise ValueError(f"duplicate D7 metric row: {key}")
        index[key] = row
    return index


def build_comparisons(
    metrics: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = metric_index(metrics)
    horizon_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for checkpoint_seed in config["checkpoint_seeds"]:
            free = index[(dataset, int(checkpoint_seed), "free_m0")]
            for width_code in ("c256", "m694"):
                geo = index[(dataset, int(checkpoint_seed), f"geo_{width_code}")]
                perm = index[(dataset, int(checkpoint_seed), f"perm_{width_code}")]
                random = index[(dataset, int(checkpoint_seed), f"random_{width_code}")]
                mse_gains = []
                mae_gains = []
                free_gaps = []
                for horizon in config["horizons"]:
                    geo_mse = float(geo[f"val_mse_h{horizon}"])
                    geo_mae = float(geo[f"val_mae_h{horizon}"])
                    control_mse = median(
                        [
                            float(perm[f"val_mse_h{horizon}"]),
                            float(random[f"val_mse_h{horizon}"]),
                        ]
                    )
                    control_mae = median(
                        [
                            float(perm[f"val_mae_h{horizon}"]),
                            float(random[f"val_mae_h{horizon}"]),
                        ]
                    )
                    free_mse = float(free[f"val_mse_h{horizon}"])
                    mse_gain = reduction(geo_mse, control_mse)
                    mae_gain = reduction(geo_mae, control_mae)
                    free_gap = reduction(geo_mse, free_mse)
                    mse_gains.append(mse_gain)
                    mae_gains.append(mae_gain)
                    free_gaps.append(free_gap)
                    horizon_rows.append(
                        {
                            "dataset": dataset,
                            "checkpoint_seed": checkpoint_seed,
                            "width": width_code,
                            "horizon": horizon,
                            "geo_mse": geo_mse,
                            "control_median_mse": control_mse,
                            "descriptor_gain_mse": mse_gain,
                            "geo_mae": geo_mae,
                            "control_median_mae": control_mae,
                            "descriptor_gain_mae": mae_gain,
                            "free_m0_mse": free_mse,
                            "free_gap_mse": free_gap,
                        }
                    )
                fit_geo = float(geo["fit_mse_eval_h720"])
                fit_control = median(
                    [
                        float(perm["fit_mse_eval_h720"]),
                        float(random["fit_mse_eval_h720"]),
                    ]
                )
                holdout_geo = float(geo["holdout_mse_eval_h720"])
                holdout_control = median(
                    [
                        float(perm["holdout_mse_eval_h720"]),
                        float(random["holdout_mse_eval_h720"]),
                    ]
                )
                fit_gain = reduction(fit_geo, fit_control)
                holdout_gain = reduction(holdout_geo, holdout_control)
                checkpoint_rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint_seed": checkpoint_seed,
                        "width": width_code,
                        "descriptor_gain_mse": mean(mse_gains),
                        "descriptor_gain_mae": mean(mae_gains),
                        "free_gap_mse": mean(free_gaps),
                        "fit_descriptor_gain_mse": fit_gain,
                        "holdout_descriptor_gain_mse": holdout_gain,
                        "fit_holdout_gain_gap": fit_gain - holdout_gain,
                    }
                )
    return horizon_rows, checkpoint_rows


def summarize_datasets(
    checkpoints: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        grouped[(str(row["dataset"]), str(row["width"]))].append(row)
    outputs = []
    for (dataset, width), rows in sorted(grouped.items()):
        gains = [float(row["descriptor_gain_mse"]) for row in rows]
        outputs.append(
            {
                "dataset": dataset,
                "width": width,
                "descriptor_gain_mse": mean(gains),
                "descriptor_gain_mae": mean(
                    [float(row["descriptor_gain_mae"]) for row in rows]
                ),
                "free_gap_mse": mean([float(row["free_gap_mse"]) for row in rows]),
                "positive_checkpoints": sum(value > 0.0 for value in gains),
                "checkpoint_count": len(rows),
            }
        )
    if len(outputs) != len(config["datasets"]) * 2:
        raise ValueError("incomplete D7 dataset summary")
    return outputs


def parameter_count_expected(state_width: int, arm: str, config: dict[str, Any]) -> int:
    branch = state_width * 256 + 256
    if arm == "free_m0":
        return branch + 256 * 720 + 720
    width = (
        int(config["compact_trunk_width"])
        if arm.endswith("c256")
        else int(config["matched_trunk_width"])
    )
    trunk = 8 * width + width + width * 256 + 256
    return branch + trunk + 720


def invariants_pass(
    metrics: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    expected_keys = {
        (dataset, int(seed), arm)
        for dataset in config["datasets"]
        for seed in config["checkpoint_seeds"]
        for arm in config["arms"]
    }
    observed_keys = {
        (str(row["dataset"]), int(row["checkpoint_seed"]), str(row["arm"]))
        for row in metrics
    }
    metadata_index = {
        (str(row["dataset"]), int(row["checkpoint_seed"])): row
        for row in metadata
    }
    gate = config["gate"]
    finite_metrics = all(
        math.isfinite(float(row[field])) and float(row[field]) > 0.0
        for row in metrics
        for field in (
            "fit_mse_eval_h720",
            "holdout_mse_eval_h720",
            *[
                f"val_{metric}_h{horizon}"
                for horizon in config["horizons"]
                for metric in ("mse", "mae")
            ],
        )
    )
    parameter_match = all(
        int(row["parameters"])
        == parameter_count_expected(
            int(metadata_index[(str(row["dataset"]), int(row["checkpoint_seed"]))]["state_width"]),
            str(row["arm"]),
            config,
        )
        for row in metrics
    )
    metadata_valid = (
        len(metadata) == len(config["datasets"]) * len(config["checkpoint_seeds"])
        and len(metadata_index) == len(metadata)
        and len({row["contract_hash"] for row in metadata}) == 1
        and len({row["d7_config_hash"] for row in metadata}) == 1
        and len(
            {
                json.dumps(row["descriptor_hashes"], sort_keys=True)
                for row in metadata
            }
        )
        == 1
        and all(
            not row["uses_test_split"]
            and not row["forecast_model_updated"]
            and not row["official_validation_used_for_early_stopping"]
            and int(row["validation_batch_offset"])
            == int(config["probe_contract"]["val_offset_batches"])
            and float(row["basis_orthogonality_max_abs"])
            <= float(gate["orthogonality_tolerance"])
            and float(row["random_descriptor_moment_max_abs"])
            <= float(gate["descriptor_moment_tolerance"])
            and float(row["coefficient_subset_max_abs"])
            <= float(gate["projectivity_tolerance"])
            and float(row["prefix_reconstruction_max_abs"])
            <= float(gate["projectivity_tolerance"])
            for row in metadata
        )
    )
    checks = {
        "fit_count": len(metrics),
        "expected_fit_count": int(config["fit_count"]),
        "metadata_count": len(metadata),
        "combination_complete": observed_keys == expected_keys,
        "finite_positive_metrics": finite_metrics,
        "parameter_algebra_match": parameter_match,
        "metadata_freeze_and_tensor_checks": metadata_valid,
    }
    passed = (
        len(metrics) == int(config["fit_count"])
        and observed_keys == expected_keys
        and finite_metrics
        and parameter_match
        and metadata_valid
    )
    return passed, checks


def build_summary(
    metrics: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon_rows, checkpoint_rows = build_comparisons(metrics, config)
    dataset_rows = summarize_datasets(checkpoint_rows, config)
    invariant_pass, invariant_checks = invariants_pass(metrics, metadata, config)
    gate = config["gate"]
    width_summary: dict[str, dict[str, Any]] = {}
    for width in ("c256", "m694"):
        horizons = [row for row in horizon_rows if row["width"] == width]
        checkpoints = [row for row in checkpoint_rows if row["width"] == width]
        datasets = [row for row in dataset_rows if row["width"] == width]
        width_summary[width] = {
            "descriptor_gain_mse": mean(
                [float(row["descriptor_gain_mse"]) for row in horizons]
            ),
            "descriptor_gain_mae": mean(
                [float(row["descriptor_gain_mae"]) for row in horizons]
            ),
            "free_gap_mse": mean([float(row["free_gap_mse"]) for row in horizons]),
            "positive_datasets": sum(
                int(row["positive_checkpoints"])
                >= int(gate["minimum_positive_checkpoints_per_dataset"])
                and float(row["descriptor_gain_mse"]) > 0.0
                for row in datasets
            ),
            "fit_holdout_gain_gap": mean(
                [float(row["fit_holdout_gain_gap"]) for row in checkpoints]
            ),
        }
    compact = width_summary["c256"]
    matched = width_summary["m694"]
    descriptor_gate = (
        compact["descriptor_gain_mse"] >= float(gate["minimum_descriptor_gain_mse"])
        and matched["descriptor_gain_mse"]
        >= float(gate["minimum_descriptor_gain_mse"])
        and compact["positive_datasets"] >= int(gate["minimum_positive_datasets"])
        and matched["positive_datasets"] >= int(gate["minimum_positive_datasets"])
        and compact["descriptor_gain_mae"] >= float(gate["mae_gain_floor"])
        and matched["descriptor_gain_mae"] >= float(gate["mae_gain_floor"])
        and matched["free_gap_mse"] >= float(gate["matched_free_m0_floor"])
        and compact["fit_holdout_gain_gap"]
        <= float(gate["maximum_fit_holdout_gain_gap"])
        and matched["fit_holdout_gain_gap"]
        <= float(gate["maximum_fit_holdout_gain_gap"])
    )
    if not invariant_pass:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif descriptor_gate:
        decision = "descriptor_sufficiency_supported_return_step6_freeze"
    else:
        decision = "descriptor_sufficiency_not_supported_close_paf"
    summary = {
        "candidate": "SC1-D7",
        "role": "diagnostic_only",
        "invariant_gate": {"pass": invariant_pass, **invariant_checks},
        "descriptor_gate": {
            "pass": descriptor_gate,
            "compact": compact,
            "matched": matched,
            "thresholds": gate,
        },
        "decision": decision,
        "authorization_if_passed": config["authorization_if_passed"],
        "method_training_authorized": False,
    }
    return summary, horizon_rows, checkpoint_rows, dataset_rows


def render_report(summary: dict[str, Any]) -> str:
    compact = summary["descriptor_gate"]["compact"]
    matched = summary["descriptor_gate"]["matched"]
    return "\n".join(
        [
            "# SC1-D7 RGNB Descriptor Sufficiency Report",
            "",
            f"- `decision`: `{summary['decision']}`",
            f"- invariant gate: `{summary['invariant_gate']['pass']}`",
            f"- descriptor gate: `{summary['descriptor_gate']['pass']}`",
            "- test used: `false`",
            "- forecast model updated: `false`",
            "",
            "| Width | MSE gain | MAE gain | Positive datasets | Free-M0 gap | Fit-holdout gain gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| compact-256 | {compact['descriptor_gain_mse']:.4%} | "
            f"{compact['descriptor_gain_mae']:.4%} | {compact['positive_datasets']}/5 | "
            f"{compact['free_gap_mse']:.4%} | {compact['fit_holdout_gain_gap']:.4%} |",
            f"| matched-694 | {matched['descriptor_gain_mse']:.4%} | "
            f"{matched['descriptor_gain_mae']:.4%} | {matched['positive_datasets']}/5 | "
            f"{matched['free_gap_mse']:.4%} | {matched['fit_holdout_gain_gap']:.4%} |",
            "",
        ]
    )


def synthetic_smoke() -> None:
    config = {
        "datasets": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"],
        "checkpoint_seeds": [2021, 2022, 2023],
        "horizons": [48, 96, 144, 192, 288, 336, 512, 720],
        "arms": [
            "free_m0",
            "geo_c256",
            "perm_c256",
            "random_c256",
            "geo_m694",
            "perm_m694",
            "random_m694",
        ],
        "compact_trunk_width": 256,
        "matched_trunk_width": 694,
        "fit_count": 105,
        "probe_contract": {"val_offset_batches": 16},
        "gate": {
            "minimum_descriptor_gain_mse": 0.005,
            "minimum_positive_datasets": 4,
            "minimum_positive_checkpoints_per_dataset": 2,
            "mae_gain_floor": 0.0,
            "matched_free_m0_floor": -0.005,
            "maximum_fit_holdout_gain_gap": 0.01,
            "orthogonality_tolerance": 2e-5,
            "descriptor_moment_tolerance": 1e-5,
            "projectivity_tolerance": 1e-5,
        },
        "authorization_if_passed": "return_step6_freeze_task_specific_method_contract",
    }
    metrics: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for seed in config["checkpoint_seeds"]:
            metadata.append(
                {
                    "dataset": dataset,
                    "checkpoint_seed": seed,
                    "state_width": 768,
                    "contract_hash": "contract",
                    "d7_config_hash": "config",
                    "descriptor_hashes": {"geo": "g", "perm": "p", "random": "r"},
                    "uses_test_split": False,
                    "forecast_model_updated": False,
                    "official_validation_used_for_early_stopping": False,
                    "validation_batch_offset": 16,
                    "basis_orthogonality_max_abs": 1e-6,
                    "random_descriptor_moment_max_abs": 1e-7,
                    "coefficient_subset_max_abs": 1e-7,
                    "prefix_reconstruction_max_abs": 1e-7,
                }
            )
            for arm in config["arms"]:
                value = 1.0
                if arm == "free_m0":
                    value = 0.98
                elif arm.startswith("geo_"):
                    value = 0.982 if arm.endswith("m694") else 0.98
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "checkpoint_seed": seed,
                    "arm": arm,
                    "parameters": parameter_count_expected(768, arm, config),
                    "fit_mse_eval_h720": value * 0.8,
                    "holdout_mse_eval_h720": value * 0.9,
                }
                for horizon in config["horizons"]:
                    row[f"val_mse_h{horizon}"] = value
                    row[f"val_mae_h{horizon}"] = math.sqrt(value)
                metrics.append(row)
    summary, _horizons, _checkpoints, _datasets = build_summary(
        metrics, metadata, config
    )
    if summary["decision"] != "descriptor_sufficiency_supported_return_step6_freeze":
        raise RuntimeError(f"D7 analyzer smoke failed: {summary}")
    print("stage_c_sc1_d7_analyzer_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    config = json.loads(args.d7_config.read_text(encoding="utf-8"))
    metrics, metadata = load_artifacts(args.input_root, config["datasets"])
    summary, horizon_rows, checkpoint_rows, dataset_rows = build_summary(
        metrics, metadata, config
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "d7_horizon_comparisons.csv", horizon_rows)
    write_csv(args.output_dir / "d7_checkpoint_summary.csv", checkpoint_rows)
    write_csv(args.output_dir / "d7_dataset_summary.csv", dataset_rows)
    (args.output_dir / "d7_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "d7_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d7_analysis=complete decision={summary['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
