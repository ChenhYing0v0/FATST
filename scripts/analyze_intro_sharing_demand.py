#!/usr/bin/env python3
"""Analyze and visualize the neutral sharing-extent pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCALES = (1, 8, 32, 128, 720)
REGION_LENGTH = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="NeutralSharingExtent")
    parser.add_argument("--dataset", default="Weather")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--crossing-margin", type=float, default=0.005)
    parser.add_argument("--headroom-threshold", type=float, default=0.005)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(args: argparse.Namespace) -> dict[int, dict[str, np.ndarray]]:
    artifacts: dict[int, dict[str, np.ndarray]] = {}
    for scale in SCALES:
        run_dir = (
            args.input_root
            / args.run_name
            / args.dataset
            / f"s{scale}"
            / f"seed{args.seed}"
        )
        path = run_dir / "predictions_val.npz"
        config_path = run_dir / "effective_config.json"
        if not path.is_file() or not config_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with np.load(path) as archive:
            artifacts[scale] = {
                "pred": archive["pred"].copy(),
                "true": archive["true"].copy(),
                "history": archive["history"].copy(),
                "origin_index": archive["origin_index"].copy(),
                "parameter_count": np.asarray(config["parameter_count"]),
                "best_epoch": np.asarray(config["best_epoch"]),
            }
    return artifacts


def validate_artifacts(
    artifacts: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    reference = artifacts[SCALES[0]]
    parameter_counts = {
        int(artifacts[scale]["parameter_count"])
        for scale in SCALES
    }
    if len(parameter_counts) != 1:
        raise RuntimeError(f"parameter mismatch: {sorted(parameter_counts)}")
    maximum_target_gap = 0.0
    maximum_history_gap = 0.0
    for scale in SCALES:
        item = artifacts[scale]
        if item["pred"].shape != reference["pred"].shape:
            raise RuntimeError(f"prediction shape mismatch at scale {scale}")
        maximum_target_gap = max(
            maximum_target_gap,
            float(np.max(np.abs(item["true"] - reference["true"]))),
        )
        maximum_history_gap = max(
            maximum_history_gap,
            float(np.max(np.abs(item["history"] - reference["history"]))),
        )
    if maximum_target_gap > 1e-6 or maximum_history_gap > 1e-6:
        raise RuntimeError(
            "artifact alignment failed: "
            f"target_gap={maximum_target_gap}, history_gap={maximum_history_gap}"
        )
    return {
        "parameter_count": parameter_counts.pop(),
        "maximum_target_alignment_gap": maximum_target_gap,
        "maximum_history_alignment_gap": maximum_history_gap,
        "origin_count": len(reference["pred"]),
    }


def moving_average(values: np.ndarray, window: int = 31) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("moving-average window must be odd")
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def compute_statistics(
    artifacts: dict[int, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    step_risks = []
    region_risks = []
    global_risks = []
    step_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        error = artifacts[scale]["pred"] - artifacts[scale]["true"]
        step_risk = np.mean(error * error, axis=(0, 2)).astype(np.float64)
        regions = np.asarray(
            [
                np.mean(step_risk[start : start + REGION_LENGTH])
                for start in range(0, len(step_risk), REGION_LENGTH)
            ],
            dtype=np.float64,
        )
        global_risk = float(np.mean(step_risk))
        step_risks.append(step_risk)
        region_risks.append(regions)
        global_risks.append(global_risk)
        smoothed = moving_average(step_risk)
        for index, (raw, smooth) in enumerate(zip(step_risk, smoothed), start=1):
            step_rows.append(
                {
                    "scale": scale,
                    "future_step": index,
                    "mse": float(raw),
                    "mse_smooth31": float(smooth),
                }
            )
        for region_index, risk in enumerate(regions, start=1):
            region_rows.append(
                {
                    "scale": scale,
                    "region": region_index,
                    "start_step": (region_index - 1) * REGION_LENGTH + 1,
                    "end_step": region_index * REGION_LENGTH,
                    "mse": float(risk),
                }
            )
    step_matrix = np.stack(step_risks)
    region_matrix = np.stack(region_risks)
    global_array = np.asarray(global_risks)
    fixed_index = int(np.argmin(global_array))
    fixed_scale = SCALES[fixed_index]
    relative_risk = (
        region_matrix - region_matrix[fixed_index : fixed_index + 1]
    ) / np.maximum(region_matrix[fixed_index : fixed_index + 1], 1e-12)
    best_indices = np.argmin(region_matrix, axis=0)
    oracle_risk = float(np.mean(np.min(region_matrix, axis=0)))
    fixed_risk = float(global_array[fixed_index])
    oracle_headroom = (fixed_risk - oracle_risk) / max(fixed_risk, 1e-12)

    crossing_pairs = []
    for left_index, left_scale in enumerate(SCALES):
        for right_index in range(left_index + 1, len(SCALES)):
            right_scale = SCALES[right_index]
            difference = (
                region_matrix[left_index] - region_matrix[right_index]
            ) / np.maximum(region_matrix[right_index], 1e-12)
            if (
                float(np.max(difference)) >= args.crossing_margin
                and float(np.min(difference)) <= -args.crossing_margin
            ):
                crossing_pairs.append(f"s{left_scale}_vs_s{right_scale}")
    summary = {
        "fixed_scale": fixed_scale,
        "fixed_validation_mse": fixed_risk,
        "region_oracle_validation_mse": oracle_risk,
        "region_oracle_headroom": oracle_headroom,
        "distinct_best_scales": sorted(
            {SCALES[int(index)] for index in best_indices}
        ),
        "best_scale_by_region": [
            SCALES[int(index)] for index in best_indices
        ],
        "crossing_pairs": crossing_pairs,
        "crossing_pair_count": len(crossing_pairs),
        "visualization_signal": bool(
            len(set(best_indices.tolist())) >= 2
            and bool(crossing_pairs)
            and oracle_headroom >= args.headroom_threshold
        ),
    }
    return (
        step_matrix,
        region_matrix,
        relative_risk,
        step_rows,
        region_rows,
        summary,
    )


def plot_figure(
    args: argparse.Namespace,
    step_matrix: np.ndarray,
    region_matrix: np.ndarray,
    relative_risk: np.ndarray,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    figure = plt.figure(figsize=(16.0, 4.7), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.2, 1.55, 1.0])
    heatmap_axis = figure.add_subplot(grid[0, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    bar_axis = figure.add_subplot(grid[0, 2])

    maximum = max(float(np.max(np.abs(relative_risk))), 1e-6)
    image = heatmap_axis.imshow(
        relative_risk * 100.0,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-maximum * 100.0,
        vmax=maximum * 100.0,
    )
    heatmap_axis.set_yticks(range(len(SCALES)), labels=SCALES)
    heatmap_axis.set_xticks(
        range(region_matrix.shape[1]),
        labels=range(1, region_matrix.shape[1] + 1),
    )
    heatmap_axis.set_xlabel("60-step future region")
    heatmap_axis.set_ylabel("Sharing extent")
    heatmap_axis.set_title("(a) Sharing-risk landscape")
    best_indices = np.argmin(region_matrix, axis=0)
    heatmap_axis.plot(
        np.arange(region_matrix.shape[1]),
        best_indices,
        color="#111111",
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        label="Descriptive argmin",
    )
    heatmap_axis.legend(loc="upper right", fontsize=8, frameon=True)
    colorbar = figure.colorbar(image, ax=heatmap_axis, shrink=0.82)
    colorbar.set_label("MSE change vs best fixed scale (%)")

    fixed_index = SCALES.index(int(summary["fixed_scale"]))
    fixed_step = step_matrix[fixed_index]
    future_steps = np.arange(1, step_matrix.shape[1] + 1)
    curve_colors = {1: "#D55E00", 32: "#0072B2", 720: "#009E73"}
    for scale in (1, 32, 720):
        scale_index = SCALES.index(scale)
        relative_curve = (
            step_matrix[scale_index] - fixed_step
        ) / np.maximum(fixed_step, 1e-12)
        curve_axis.plot(
            future_steps,
            moving_average(relative_curve) * 100.0,
            color=curve_colors[scale],
            linewidth=1.4,
            label=f"s={scale}",
        )
    curve_axis.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    for boundary in range(REGION_LENGTH, 720, REGION_LENGTH):
        curve_axis.axvline(boundary, color="#BBBBBB", linewidth=0.35)
    curve_axis.set_xlabel("Future time step")
    curve_axis.set_ylabel("MSE change vs best fixed scale (%)")
    curve_axis.set_title("(b) Step-wise sharing-risk differences")
    curve_axis.legend(frameon=False)
    curve_axis.grid(alpha=0.15)
    if not summary["visualization_signal"]:
        curve_axis.text(
            0.02,
            0.97,
            "No material crossover under the frozen margin",
            transform=curve_axis.transAxes,
            va="top",
            color="#555555",
            fontsize=9,
        )

    fixed_risk = float(summary["fixed_validation_mse"])
    gains = (fixed_risk - np.mean(region_matrix, axis=1)) / fixed_risk
    labels = [f"s={scale}" for scale in SCALES] + ["Region\noracle"]
    values = [*list(gains * 100.0), summary["region_oracle_headroom"] * 100.0]
    colors = ["#9ECAE1"] * len(SCALES) + ["#F28E2B"]
    bar_axis.bar(np.arange(len(labels)), values, color=colors)
    bar_axis.axhline(0.0, color="#555555", linewidth=0.9)
    bar_axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=30)
    bar_axis.set_ylabel("Gain vs best fixed scale (%)")
    bar_axis.set_title("(c) Validation descriptive headroom")
    bar_axis.text(
        0.02,
        0.98,
        "Exploratory; not out-of-sample",
        transform=bar_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    bar_axis.grid(axis="y", alpha=0.15)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "sharing_demand_visualization.svg"
    png_path = args.output_dir / "sharing_demand_visualization.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def main() -> None:
    args = parse_args()
    artifacts = load_predictions(args)
    invariants = validate_artifacts(artifacts)
    (
        step_matrix,
        region_matrix,
        relative_risk,
        step_rows,
        region_rows,
        summary,
    ) = compute_statistics(artifacts, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "step_risk.csv", step_rows)
    write_csv(args.output_dir / "region_risk.csv", region_rows)
    svg_path, png_path = plot_figure(
        args,
        step_matrix,
        region_matrix,
        relative_risk,
        summary,
    )
    output = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split": "validation",
        "test_accessed": False,
        **invariants,
        **summary,
        "figure_svg": str(svg_path),
        "figure_png": str(png_path),
        "claim_role": "exploratory_visualization_only",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
