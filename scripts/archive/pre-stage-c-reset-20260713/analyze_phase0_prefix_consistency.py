from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 0 fixed-head prefix consistency.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--model", default="PatchEncoderFixedHead")
    parser.add_argument("--datasets", nargs="+", default=["ETTh2", "ETTm1", "Weather"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--max-horizon", type=int, default=720)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def load_prediction(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "predictions_test.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return data["pred"], data["true"]


def mse(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff * diff))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def run_dir(root: Path, model: str, dataset: str, horizon: int, seed: int) -> Path:
    return root / model / dataset / f"h{horizon}" / f"seed{seed}"


def analyze_dataset(
    root: Path,
    model: str,
    dataset: str,
    horizons: list[int],
    max_horizon: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    max_pred, max_true = load_prediction(run_dir(root, model, dataset, max_horizon, seed))
    rows: list[dict[str, float | int | str]] = []

    for horizon in horizons:
        fixed_pred, fixed_true = load_prediction(run_dir(root, model, dataset, horizon, seed))
        if horizon > max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {max_horizon}")

        n_aligned = min(fixed_pred.shape[0], max_pred.shape[0])
        fixed_pred_aligned = fixed_pred[:n_aligned, :horizon, :]
        fixed_true_aligned = fixed_true[:n_aligned, :horizon, :]
        max_pred_prefix = max_pred[:n_aligned, :horizon, :]
        max_true_prefix = max_true[:n_aligned, :horizon, :]

        fixed_mse = mse(fixed_pred_aligned, fixed_true_aligned)
        max_prefix_mse = mse(max_pred_prefix, max_true_prefix)
        fixed_mae = mae(fixed_pred_aligned, fixed_true_aligned)
        max_prefix_mae = mae(max_pred_prefix, max_true_prefix)
        pred_mse = mse(fixed_pred_aligned, max_pred_prefix)
        pred_mae = mae(fixed_pred_aligned, max_pred_prefix)
        truth_alignment_mse = mse(fixed_true_aligned, max_true_prefix)

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "prefix_horizon": horizon,
                "max_horizon": max_horizon,
                "seed": seed,
                "n_aligned": n_aligned,
                "fixed_mse": fixed_mse,
                "fixed_mae": fixed_mae,
                "max_prefix_mse": max_prefix_mse,
                "max_prefix_mae": max_prefix_mae,
                "max_minus_fixed_mse": max_prefix_mse - fixed_mse,
                "max_minus_fixed_mae": max_prefix_mae - fixed_mae,
                "relative_mse_change": (max_prefix_mse - fixed_mse) / fixed_mse,
                "relative_mae_change": (max_prefix_mae - fixed_mae) / fixed_mae,
                "fixed_vs_max_pred_mse": pred_mse,
                "fixed_vs_max_pred_mae": pred_mae,
                "truth_alignment_mse": truth_alignment_mse,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    non_max_rows = [row for row in rows if row["prefix_horizon"] != row["max_horizon"]]
    max_relative = max(float(row["relative_mse_change"]) for row in non_max_rows)
    max_prediction_mse = max(float(row["fixed_vs_max_pred_mse"]) for row in non_max_rows)
    worst_relative = max(non_max_rows, key=lambda row: float(row["relative_mse_change"]))
    worst_prediction = max(non_max_rows, key=lambda row: float(row["fixed_vs_max_pred_mse"]))
    return {
        "n_rows": len(rows),
        "max_relative_mse_change": max_relative,
        "max_fixed_vs_max_pred_mse": max_prediction_mse,
        "worst_relative_mse_change": worst_relative,
        "worst_fixed_vs_max_pred_mse": worst_prediction,
        "truth_alignment_mse_max": max(float(row["truth_alignment_mse"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for dataset in args.datasets:
        rows.extend(
            analyze_dataset(
                root=root,
                model=args.model,
                dataset=dataset,
                horizons=args.horizons,
                max_horizon=args.max_horizon,
                seed=args.seed,
            )
        )

    raw_path = report_dir / "phase0_prefix_consistency_raw.csv"
    summary_path = report_dir / "phase0_prefix_consistency_summary.json"
    write_csv(raw_path, rows)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"raw_csv": str(raw_path), "summary_json": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
