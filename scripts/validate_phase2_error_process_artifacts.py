from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_HORIZONS = [96, 192, 336, 720]
REQUIRED_ROOT_FILES = [
    "metrics_by_target_horizon.csv",
    "prefix_consistency.csv",
    "effective_config.json",
    "training_log.csv",
]
REQUIRED_HORIZON_FILES = [
    "metrics.json",
    "metrics_by_horizon.csv",
    "metrics_by_segment.csv",
    "error_process_stats.csv",
]
ERROR_PROCESS_COLUMNS = [
    "scope",
    "residual_base_mae_ratio",
    "residual_energy",
    "residual_second_diff_smoothness",
    "error_process_state_norm",
    "segment_state_cosine",
    "base_prediction_mse",
    "final_prediction_mse",
    "residual_gain_mse_pct",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric(value: str, path: Path, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}: column {column} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: column {column} is not finite: {value!r}")
    return parsed


def check_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")
    elif path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")


def validate_error_process_stats(path: Path, errors: list[str]) -> dict[str, float]:
    rows = read_csv(path)
    if not rows:
        errors.append(f"empty csv rows: {path}")
        return {}
    columns = set(rows[0].keys())
    missing = [column for column in ERROR_PROCESS_COLUMNS if column not in columns]
    if missing:
        errors.append(f"{path}: missing columns {missing}")
        return {}
    ratios = []
    energies = []
    smoothness = []
    gains = []
    decompositions = []
    for row in rows:
        ratios.append(numeric(row["residual_base_mae_ratio"], path, "residual_base_mae_ratio"))
        energies.append(numeric(row["residual_energy"], path, "residual_energy"))
        smoothness.append(
            numeric(row["residual_second_diff_smoothness"], path, "residual_second_diff_smoothness")
        )
        numeric(row["error_process_state_norm"], path, "error_process_state_norm")
        numeric(row["segment_state_cosine"], path, "segment_state_cosine")
        numeric(row["base_prediction_mse"], path, "base_prediction_mse")
        numeric(row["final_prediction_mse"], path, "final_prediction_mse")
        gains.append(numeric(row["residual_gain_mse_pct"], path, "residual_gain_mse_pct"))
        if "prediction_decomposition_max_abs" in row:
            decomposition = numeric(
                row["prediction_decomposition_max_abs"],
                path,
                "prediction_decomposition_max_abs",
            )
            decompositions.append(decomposition)
            if decomposition > 1e-5:
                errors.append(
                    f"{path}: prediction decomposition max abs {decomposition:.6g} > 1e-5"
                )
    stats = {
        "mean_residual_base_mae_ratio": sum(ratios) / len(ratios),
        "mean_residual_energy": sum(energies) / len(energies),
        "mean_residual_second_diff_smoothness": sum(smoothness) / len(smoothness),
        "mean_residual_gain_mse_pct": sum(gains) / len(gains),
    }
    if decompositions:
        stats["max_prediction_decomposition_abs"] = max(decompositions)
    return stats


def validate_prefix(path: Path, threshold: float, errors: list[str]) -> float:
    rows = read_csv(path)
    if not rows:
        errors.append(f"empty csv rows: {path}")
        return float("nan")
    max_mismatch = max(numeric(row["prefix_mismatch_mse"], path, "prefix_mismatch_mse") for row in rows)
    if max_mismatch > threshold:
        errors.append(f"{path}: max prefix mismatch {max_mismatch:.6g} > threshold {threshold:.6g}")
    return max_mismatch


def validate_run(run_dir: Path, horizons: list[int], prefix_threshold: float) -> dict[str, object]:
    errors: list[str] = []
    for filename in REQUIRED_ROOT_FILES:
        check_exists(run_dir / filename, errors)
    for horizon in horizons:
        horizon_dir = run_dir / f"h{horizon}"
        for filename in REQUIRED_HORIZON_FILES:
            check_exists(horizon_dir / filename, errors)

    stats: dict[str, object] = {}
    if (run_dir / "prefix_consistency.csv").exists():
        stats["max_prefix_mismatch_mse"] = validate_prefix(
            run_dir / "prefix_consistency.csv",
            prefix_threshold,
            errors,
        )

    horizon_stats = {}
    for horizon in horizons:
        stats_path = run_dir / f"h{horizon}" / "error_process_stats.csv"
        if stats_path.exists() and stats_path.stat().st_size > 0:
            horizon_stats[str(horizon)] = validate_error_process_stats(stats_path, errors)
    stats["horizon_error_process_stats"] = horizon_stats
    return {
        "run_dir": str(run_dir),
        "pass": not errors,
        "errors": errors,
        "stats": stats,
    }


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase2-B error-process decoder artifacts.")
    parser.add_argument(
        "--run-dir",
        default=(
            "artifacts/runs/smoke_phase2_error_process_decoder/"
            "PatchEncoderErrorProcessDecoder/ETTh2/mixed_h96_h192_h336_h720/seed2021"
        ),
    )
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--prefix-threshold", type=float, default=1e-10)
    parser.add_argument("--summary-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_run(
        Path(args.run_dir),
        parse_horizons(args.horizons),
        args.prefix_threshold,
    )
    text = json.dumps(summary, indent=2)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n")
    print(text)
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
