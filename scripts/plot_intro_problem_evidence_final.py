#!/usr/bin/env python3
"""Render publication-oriented Introduction problem-evidence figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HORIZONS = (96, 192, 336, 720)
SCALES = (1, 8, 32, 128, 720)
HORIZON_COLORS = {
    96: "#D55E00",
    192: "#E69F00",
    336: "#0072B2",
    720: "#009E73",
}
HORIZON_STYLES = {
    96: "-",
    192: "--",
    336: "-.",
    720: ":",
}
SCALE_COLORS = {
    1: "#D55E00",
    8: "#E69F00",
    32: "#0072B2",
    128: "#CC79A7",
    720: "#009E73",
}
DOUBLE_COLUMN_WIDTH = 183.0 / 25.4


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-source", type=Path, required=True)
    parser.add_argument("--prefix-summary", type=Path, required=True)
    parser.add_argument("--prefix-pairs", type=Path, required=True)
    parser.add_argument("--sharing-region-risk", type=Path, required=True)
    parser.add_argument("--sharing-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prefix_source(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_pair_matrix(path: Path) -> np.ndarray:
    matrix = np.full((len(HORIZONS), len(HORIZONS)), np.nan)
    np.fill_diagonal(matrix, 0.0)
    horizon_to_index = {
        horizon: index for index, horizon in enumerate(HORIZONS)
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left = horizon_to_index[int(row["short_horizon"])]
            right = horizon_to_index[int(row["long_horizon"])]
            matrix[left, right] = float(row["nchpd_l1"])
    return matrix


def read_region_risk(path: Path) -> np.ndarray:
    matrix = np.full((len(SCALES), 12), np.nan)
    scale_to_index = {scale: index for index, scale in enumerate(SCALES)}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            matrix[
                scale_to_index[int(row["scale"])],
                int(row["region"]) - 1,
            ] = float(row["mse"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("sharing region-risk matrix is incomplete")
    return matrix


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def export_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "tiff": output_dir / f"{stem}.tiff",
    }
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    figure.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return {key: str(value) for key, value in paths.items()}


def plot_prefix_figure(
    source_path: Path,
    pair_path: Path,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    rows = read_prefix_source(source_path)
    history_rows = [row for row in rows if row["phase"] == "history"][-48:]
    future_rows = [
        row
        for row in rows
        if row["phase"] == "future"
        and 1 <= int(row["relative_step"]) <= 96
    ]
    history_x = np.asarray(
        [int(row["relative_step"]) for row in history_rows]
    )
    history_y = np.asarray([float(row["history"]) for row in history_rows])
    future_x = np.asarray(
        [int(row["relative_step"]) for row in future_rows]
    )
    target = np.asarray(
        [float(row["ground_truth"]) for row in future_rows]
    )
    predictions = {
        horizon: np.asarray(
            [
                float(row[f"prediction_h{horizon}"])
                for row in future_rows
            ]
        )
        for horizon in HORIZONS
    }
    pair_matrix = read_pair_matrix(pair_path)

    figure = plt.figure(
        figsize=(DOUBLE_COLUMN_WIDTH, 4.05),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1.0, 0.86),
    )
    trajectory_axis = figure.add_subplot(grid[0, 0])
    difference_axis = figure.add_subplot(grid[1, 0])
    heatmap_axis = figure.add_subplot(grid[:, 1])

    trajectory_axis.plot(
        history_x,
        history_y,
        color="#6B6B6B",
        linewidth=1.0,
        label="History",
    )
    trajectory_axis.plot(
        future_x,
        target,
        color="#222222",
        linewidth=1.45,
        label="Ground truth",
    )
    for horizon in HORIZONS:
        trajectory_axis.plot(
            future_x,
            predictions[horizon],
            color=HORIZON_COLORS[horizon],
            linestyle=HORIZON_STYLES[horizon],
            label=f"H={horizon}",
        )
    trajectory_axis.axvline(
        0,
        color="#777777",
        linestyle="--",
        linewidth=0.75,
    )
    trajectory_axis.set_xlim(history_x[0], 96)
    trajectory_axis.set_xlabel("Relative future step")
    trajectory_axis.set_ylabel("Observed / predicted value")
    trajectory_axis.set_title(
        "Same history, different requested horizons",
        loc="left",
    )
    trajectory_axis.grid(alpha=0.16, linewidth=0.45)
    trajectory_axis.legend(
        ncol=3,
        frameon=False,
        loc="upper right",
        handlelength=2.5,
    )
    panel_label(trajectory_axis, "a")

    reference = predictions[720]
    for horizon in HORIZONS[:-1]:
        difference = predictions[horizon] - reference
        mean_absolute = float(np.mean(np.abs(difference)))
        difference_axis.plot(
            future_x,
            difference,
            color=HORIZON_COLORS[horizon],
            linestyle=HORIZON_STYLES[horizon],
            label=f"H={horizon}  (mean |Δ|={mean_absolute:.2f})",
        )
    difference_axis.axhline(
        0.0,
        color="#555555",
        linestyle="--",
        linewidth=0.75,
    )
    difference_axis.set_xlim(1, 96)
    difference_axis.set_xlabel("Overlapping future step")
    difference_axis.set_ylabel("Prediction − H=720")
    difference_axis.set_title(
        "Disagreement on the shared 96-step prefix",
        loc="left",
    )
    difference_axis.grid(alpha=0.16, linewidth=0.45)
    difference_axis.legend(
        frameon=False,
        loc="lower right",
        handlelength=2.5,
    )
    panel_label(difference_axis, "b")

    masked = np.ma.masked_invalid(pair_matrix)
    image = heatmap_axis.imshow(
        masked,
        cmap="YlOrRd",
        vmin=0.0,
        aspect="equal",
    )
    heatmap_axis.set_xticks(range(len(HORIZONS)), labels=HORIZONS)
    heatmap_axis.set_yticks(range(len(HORIZONS)), labels=HORIZONS)
    heatmap_axis.set_xlabel("Longer requested horizon")
    heatmap_axis.set_ylabel("Shorter requested horizon")
    heatmap_axis.set_title(
        "Validation-set prefix disagreement",
        loc="left",
    )
    for row in range(len(HORIZONS)):
        for column in range(len(HORIZONS)):
            value = pair_matrix[row, column]
            if np.isfinite(value):
                heatmap_axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                )
    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        shrink=0.62,
        pad=0.04,
    )
    colorbar.set_label("NCHPD")
    panel_label(heatmap_axis, "c")

    figure.text(
        0.01,
        -0.015,
        "Illustrative validation example selected by maximum aggregate "
        f"disagreement over {summary['searched_origin_channel_cells']:,} "
        "origin–channel candidates; heatmap uses all validation origins.",
        fontsize=5.6,
        color="#555555",
        ha="left",
    )
    paths = export_figure(
        figure,
        output_dir,
        "figure_intro_prefix_disagreement",
    )
    plt.close(figure)
    return paths


def plot_sharing_figure(
    region_path: Path,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    risk = read_region_risk(region_path)
    selected = summary["selected"]
    fixed_scale = int(selected["best_fixed_scale"])
    fixed_index = SCALES.index(fixed_scale)
    fixed_risk = risk[fixed_index]
    relative_to_fixed = (
        risk - fixed_risk[None, :]
    ) / np.maximum(fixed_risk[None, :], 1e-12)
    winners = np.argmin(risk, axis=0)
    winner_risk = np.min(risk, axis=0)
    realized_gain = (
        fixed_risk - winner_risk
    ) / np.maximum(fixed_risk, 1e-12)

    figure = plt.figure(
        figsize=(DOUBLE_COLUMN_WIDTH, 3.25),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.45, 1.0))
    heatmap_axis = figure.add_subplot(grid[0, 0])
    gain_axis = figure.add_subplot(grid[0, 1])

    magnitude = max(float(np.max(np.abs(relative_to_fixed))), 0.01)
    image = heatmap_axis.imshow(
        relative_to_fixed * 100.0,
        cmap="RdBu_r",
        vmin=-100.0 * magnitude,
        vmax=100.0 * magnitude,
        aspect="auto",
    )
    heatmap_axis.set_xticks(range(12), labels=range(1, 13))
    heatmap_axis.set_yticks(range(len(SCALES)), labels=SCALES)
    heatmap_axis.set_xlabel("60-step future region")
    heatmap_axis.set_ylabel("Cross-step sharing extent")
    heatmap_axis.set_title(
        f"Region-wise risk relative to fixed s={fixed_scale}",
        loc="left",
    )
    heatmap_axis.scatter(
        np.arange(12),
        winners,
        marker="s",
        s=25,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.9,
        label="Region winner",
    )
    heatmap_axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.88,
        borderpad=0.3,
    )
    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        shrink=0.78,
        pad=0.03,
    )
    colorbar.set_label("ΔMSE (%)")
    panel_label(heatmap_axis, "a")

    region_x = np.arange(1, 13)
    winner_scales = [SCALES[int(index)] for index in winners]
    gain_axis.bar(
        region_x,
        realized_gain * 100.0,
        color=[SCALE_COLORS[scale] for scale in winner_scales],
        width=0.78,
    )
    gain_axis.axhline(0.0, color="#555555", linewidth=0.7)
    gain_axis.set_xticks(region_x)
    gain_axis.set_xlabel("60-step future region")
    gain_axis.set_ylabel(f"Gain vs fixed s={fixed_scale} (%)")
    gain_axis.set_title(
        "Different regions favor different sharing extents",
        loc="left",
    )
    gain_axis.grid(axis="y", alpha=0.16, linewidth=0.45)
    gain_axis.scatter(
        region_x,
        np.full_like(region_x, 0.55, dtype=np.float64),
        marker="s",
        s=13,
        c=[SCALE_COLORS[scale] for scale in winner_scales],
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    legend_handles = [
        Patch(facecolor=SCALE_COLORS[scale], label=f"s={scale}")
        for scale in SCALES
    ]
    gain_axis.set_ylim(
        0.0,
        max(1.0, float(np.max(realized_gain * 100.0)) * 1.18),
    )
    gain_axis.legend(
        handles=legend_handles,
        title="Region winner",
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=0.45,
        handlelength=0.9,
        handletextpad=0.3,
        fontsize=5.4,
        title_fontsize=5.6,
    )
    gain_axis.text(
        0.98,
        0.82,
        "All 5 extents win 2–3 regions\n"
        f"Region-oracle headroom: "
        f"{100.0 * selected['sample_oracle_headroom']:.1f}%",
        transform=gain_axis.transAxes,
        va="bottom",
        ha="right",
        fontsize=6.2,
        color="#333333",
        bbox={
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "boxstyle": "round,pad=0.25",
            "alpha": 0.9,
        },
    )
    panel_label(gain_axis, "b")

    figure.text(
        0.01,
        -0.015,
        "Maximum-heterogeneity validation example; each region aggregates "
        "60 future steps and all channels. Squares, markers, and bar colors "
        "identify the descriptive region winner.",
        fontsize=5.6,
        color="#555555",
        ha="left",
    )
    paths = export_figure(
        figure,
        output_dir,
        "figure_intro_sharing_heterogeneity",
    )
    plt.close(figure)
    return paths


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    prefix_summary = read_json(args.prefix_summary)
    sharing_summary = read_json(args.sharing_summary)
    if prefix_summary["test_accessed"] or sharing_summary["test_accessed"]:
        raise RuntimeError("final Introduction figures must not access test")

    prefix_paths = plot_prefix_figure(
        args.prefix_source,
        args.prefix_pairs,
        prefix_summary,
        args.output_dir,
    )
    sharing_paths = plot_sharing_figure(
        args.sharing_region_risk,
        sharing_summary,
        args.output_dir,
    )
    manifest = {
        "split": "validation",
        "test_accessed": False,
        "prefix": {
            "dataset": prefix_summary["dataset"],
            "seed": prefix_summary["seed"],
            "origin_index": prefix_summary["selected_origin_index"],
            "channel_index": prefix_summary["selected_channel_index"],
            "selection_mode": prefix_summary["selection_mode"],
            "selected_joint_score": prefix_summary["selected_joint_score"],
            "searched_origin_channel_cells": (
                prefix_summary["searched_origin_channel_cells"]
            ),
            "source_data": {
                "forecast": str(args.prefix_source),
                "pair_metrics": str(args.prefix_pairs),
                "summary": str(args.prefix_summary),
            },
            "outputs": prefix_paths,
        },
        "sharing": {
            "dataset": sharing_summary["dataset"],
            "seed": sharing_summary["seed"],
            "selected": sharing_summary["selected"],
            "source_data": {
                "region_risk": str(args.sharing_region_risk),
                "summary": str(args.sharing_summary),
            },
            "outputs": sharing_paths,
        },
        "claim_boundary": [
            "illustrative validation example",
            "not prevalence evidence",
            "not method effectiveness evidence",
            "not an untouched test claim",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
