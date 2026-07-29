#!/usr/bin/env python3
"""Analyze and visualize horizon-specific prefix disagreement."""

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


HORIZONS = (96, 192, 336, 720)
COLORS = {
    96: "#D55E00",
    192: "#E69F00",
    336: "#0072B2",
    720: "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="IntroDLinearPrefixViz")
    parser.add_argument("--dataset", default="Weather")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--sample-quantile", type=float, default=0.85)
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
    for horizon in HORIZONS:
        path = (
            args.input_root
            / args.run_name
            / args.dataset
            / f"h{horizon}"
            / f"seed{args.seed}"
            / "predictions_val.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            artifacts[horizon] = {
                key: archive[key].copy()
                for key in (
                    "pred",
                    "true",
                    "history",
                    "origin_index",
                    "train_mean",
                    "train_std",
                )
            }
    return artifacts


def validate_alignment(
    artifacts: dict[int, dict[str, np.ndarray]],
) -> tuple[int, float, float]:
    origin_count = min(len(artifacts[horizon]["pred"]) for horizon in HORIZONS)
    if origin_count <= 0:
        raise RuntimeError("no aligned validation origins")
    reference_history = artifacts[720]["history"][:origin_count]
    maximum_history_gap = 0.0
    maximum_target_gap = 0.0
    for horizon in HORIZONS:
        item = artifacts[horizon]
        if item["pred"].shape[1:] != item["true"].shape[1:]:
            raise RuntimeError(f"prediction/target shape mismatch for H{horizon}")
        if item["pred"].shape[1] != horizon:
            raise RuntimeError(f"unexpected prediction length for H{horizon}")
        maximum_history_gap = max(
            maximum_history_gap,
            float(
                np.max(
                    np.abs(item["history"][:origin_count] - reference_history)
                )
            ),
        )
        maximum_target_gap = max(
            maximum_target_gap,
            float(
                np.max(
                    np.abs(
                        item["true"][:origin_count]
                        - artifacts[720]["true"][:origin_count, :horizon]
                    )
                )
            ),
        )
    if maximum_history_gap > 1e-6 or maximum_target_gap > 1e-6:
        raise RuntimeError(
            "same-origin alignment failed: "
            f"history_gap={maximum_history_gap}, target_gap={maximum_target_gap}"
        )
    return origin_count, maximum_history_gap, maximum_target_gap


def pair_statistics(
    artifacts: dict[int, dict[str, np.ndarray]],
    origin_count: int,
    dataset: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.full((len(HORIZONS), len(HORIZONS)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    origin_scores: list[np.ndarray] = []
    channel_scores: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for left_index, short_horizon in enumerate(HORIZONS):
        short_prediction = artifacts[short_horizon]["pred"][:origin_count]
        short_target = artifacts[short_horizon]["true"][:origin_count]
        for right_index in range(left_index + 1, len(HORIZONS)):
            long_horizon = HORIZONS[right_index]
            long_prediction = artifacts[long_horizon]["pred"][
                :origin_count, :short_horizon
            ]
            difference = short_prediction - long_prediction
            absolute = np.abs(difference)
            squared = difference * difference
            overlap_error = 0.5 * (
                np.mean((short_prediction - short_target) ** 2)
                + np.mean((long_prediction - short_target) ** 2)
            )
            nchpd = float(np.mean(absolute))
            chpd2 = float(np.mean(squared))
            rda = float(np.sqrt(chpd2 / max(overlap_error, 1e-12)))
            matrix[left_index, right_index] = nchpd
            origin_scores.append(np.mean(absolute, axis=(1, 2)))
            channel_scores.append(np.mean(absolute, axis=(0, 1)))
            rows.append(
                {
                    "dataset": dataset,
                    "short_horizon": short_horizon,
                    "long_horizon": long_horizon,
                    "aligned_origins": origin_count,
                    "nchpd_l1": nchpd,
                    "chpd_l2": chpd2,
                    "relative_disagreement_amplitude": rda,
                }
            )
    return (
        rows,
        matrix,
        np.stack(origin_scores, axis=1),
        np.stack(channel_scores, axis=1),
    )


def nearest_quantile_index(values: np.ndarray, quantile: float) -> int:
    target = float(np.quantile(values, quantile))
    return int(np.argmin(np.abs(values - target)))


def inverse_scale(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return values * std + mean


def plot_overlay(
    args: argparse.Namespace,
    artifacts: dict[int, dict[str, np.ndarray]],
    origin_index: int,
    channel_index: int,
) -> tuple[Path, Path]:
    reference = artifacts[720]
    mean = reference["train_mean"].reshape(-1)
    std = reference["train_std"].reshape(-1)
    history = inverse_scale(
        reference["history"][origin_index, :, channel_index],
        mean[channel_index],
        std[channel_index],
    )
    target = inverse_scale(
        reference["true"][origin_index, :, channel_index],
        mean[channel_index],
        std[channel_index],
    )

    figure, (main_axis, inset_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    history_x = np.arange(-len(history), 0)
    future_x = np.arange(1, 721)
    main_axis.plot(history_x, history, color="#4D4D4D", linewidth=1.4, label="History")
    main_axis.plot(
        future_x,
        target,
        color="#000000",
        linewidth=1.5,
        alpha=0.85,
        label="Ground truth",
    )
    for horizon in HORIZONS:
        prediction = inverse_scale(
            artifacts[horizon]["pred"][origin_index, :, channel_index],
            mean[channel_index],
            std[channel_index],
        )
        main_axis.plot(
            np.arange(1, horizon + 1),
            prediction,
            color=COLORS[horizon],
            linewidth=1.25,
            label=f"H={horizon}",
        )
    main_axis.axvline(0, color="#777777", linestyle="--", linewidth=1.0)
    main_axis.set_title(
        "Same history, independently optimized horizon-specific forecasts"
    )
    main_axis.set_xlabel("Future time step")
    main_axis.set_ylabel(f"Channel {channel_index} value")
    main_axis.legend(ncol=3, fontsize=9, frameon=False)
    main_axis.grid(alpha=0.18)

    inset_axis.plot(
        np.arange(1, 97),
        target[:96],
        color="#000000",
        linewidth=1.6,
        label="Ground truth",
    )
    for horizon in HORIZONS:
        prediction = inverse_scale(
            artifacts[horizon]["pred"][origin_index, :96, channel_index],
            mean[channel_index],
            std[channel_index],
        )
        inset_axis.plot(
            np.arange(1, 97),
            prediction,
            color=COLORS[horizon],
            linewidth=1.35,
            label=f"H={horizon}",
        )
    inset_axis.set_title("Zoomed overlapping prefix: future steps 1–96")
    inset_axis.set_xlabel("Future time step")
    inset_axis.set_ylabel("Value")
    inset_axis.grid(alpha=0.18)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "prefix_disagreement_overlay.svg"
    png_path = args.output_dir / "prefix_disagreement_overlay.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def plot_heatmap(
    args: argparse.Namespace,
    matrix: np.ndarray,
) -> tuple[Path, Path]:
    masked = np.ma.masked_invalid(matrix)
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(masked, cmap="YlOrRd", vmin=0.0)
    axis.set_xticks(range(len(HORIZONS)), labels=HORIZONS)
    axis.set_yticks(range(len(HORIZONS)), labels=HORIZONS)
    axis.set_xlabel("Longer horizon")
    axis.set_ylabel("Shorter horizon")
    axis.set_title("Validation prefix disagreement (NCHPD)")
    for row in range(len(HORIZONS)):
        for column in range(len(HORIZONS)):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean absolute disagreement / train scale")
    svg_path = args.output_dir / "prefix_disagreement_heatmap.svg"
    png_path = args.output_dir / "prefix_disagreement_heatmap.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.sample_quantile < 1.0:
        raise ValueError("sample_quantile must be in [0.5, 1.0)")
    artifacts = load_predictions(args)
    origin_count, history_gap, target_gap = validate_alignment(artifacts)
    rows, matrix, origin_pair_scores, channel_pair_scores = pair_statistics(
        artifacts,
        origin_count,
        args.dataset,
    )
    origin_score = origin_pair_scores.mean(axis=1)
    selected_origin = nearest_quantile_index(origin_score, args.sample_quantile)
    channel_score = channel_pair_scores.mean(axis=1)
    selected_channel = nearest_quantile_index(
        channel_score,
        args.sample_quantile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pair_metrics.csv", rows)
    overlay_svg, overlay_png = plot_overlay(
        args,
        artifacts,
        selected_origin,
        selected_channel,
    )
    heatmap_svg, heatmap_png = plot_heatmap(args, matrix)
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split": "validation",
        "test_accessed": False,
        "aligned_origin_count": origin_count,
        "maximum_history_alignment_gap": history_gap,
        "maximum_target_alignment_gap": target_gap,
        "sample_quantile": args.sample_quantile,
        "selected_origin_index": selected_origin,
        "selected_origin_score": float(origin_score[selected_origin]),
        "selected_origin_percentile": float(
            np.mean(origin_score <= origin_score[selected_origin])
        ),
        "selected_channel_index": selected_channel,
        "selected_channel_score": float(channel_score[selected_channel]),
        "selected_channel_percentile": float(
            np.mean(channel_score <= channel_score[selected_channel])
        ),
        "macro_nchpd_l1": float(np.mean([row["nchpd_l1"] for row in rows])),
        "macro_rda": float(
            np.mean([row["relative_disagreement_amplitude"] for row in rows])
        ),
        "figures": {
            "overlay_svg": str(overlay_svg),
            "overlay_png": str(overlay_png),
            "heatmap_svg": str(heatmap_svg),
            "heatmap_png": str(heatmap_png),
        },
        "claim_role": "exploratory_visualization_only",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
