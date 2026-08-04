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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Patch


HORIZONS = (96, 192, 336, 720)
SCALES = (1, 8, 32, 128, 720)
HORIZON_COLORS = {
    96: "#B85C38",
    192: "#C9952D",
    336: "#4C78A8",
    720: "#2A7F62",
}
HORIZON_MARKERS = {
    96: "o",
    192: "s",
    336: "^",
    720: "D",
}
HORIZON_MARKER_OFFSETS = {
    96: 2,
    192: 6,
    336: 10,
    720: 14,
}
SCALE_COLORS = {
    1: "#425A8B",
    8: "#607CAD",
    32: "#86A6C8",
    128: "#A784B8",
    720: "#C36B85",
}
DOUBLE_COLUMN_WIDTH = 183.0 / 25.4
PREFIX_CMAP = LinearSegmentedColormap.from_list(
    "prefix_disagreement",
    ("#F4F5FA", "#C7D2E8", "#7995C2", "#2D4573"),
)
SHARING_CMAP = LinearSegmentedColormap.from_list(
    "sharing_excess_risk",
    ("#F7F7F7", "#F3D8CF", "#D98B78", "#8E3B46"),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
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
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.4,
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
    figure.savefig(paths["svg"])
    figure.savefig(paths["pdf"])
    figure.savefig(paths["png"], dpi=300)
    figure.savefig(
        paths["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return {key: str(value) for key, value in paths.items()}


def plot_prefix_figure(
    source_path: Path,
    pair_path: Path,
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
        figsize=(DOUBLE_COLUMN_WIDTH, 3.08),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.72, 1.0),
        wspace=0.10,
    )
    trajectory_axis = figure.add_subplot(grid[0, 0])
    heatmap_axis = figure.add_subplot(grid[0, 1])

    trajectory_axis.plot(
        history_x,
        history_y,
        color="#999999",
        linewidth=1.05,
        zorder=1,
        label="History",
    )
    trajectory_axis.axvspan(
        history_x[0],
        0,
        color="#F1F1F1",
        zorder=-2,
    )
    trajectory_axis.plot(
        future_x,
        target,
        color="#444444",
        linewidth=1.15,
        alpha=0.9,
        zorder=2,
        label="Ground truth",
    )
    mean_absolute_differences: dict[int, float] = {}
    reference = predictions[720]
    for horizon in HORIZONS:
        mean_absolute_differences[horizon] = float(
            np.mean(np.abs(predictions[horizon] - reference))
        )
        line_width = 0.95 if horizon == 720 else 0.82
        line_zorder = {
            96: 6,
            192: 5,
            336: 4,
            720: 3,
        }[horizon]
        line, = trajectory_axis.plot(
            future_x,
            predictions[horizon],
            color=HORIZON_COLORS[horizon],
            linestyle="-",
            linewidth=line_width,
            marker=HORIZON_MARKERS[horizon],
            markevery=(HORIZON_MARKER_OFFSETS[horizon], 18),
            markersize=2.25 if horizon == 720 else 2.0,
            markerfacecolor=HORIZON_COLORS[horizon],
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=line_zorder,
            label=f"$H={horizon}$",
        )
        line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=line_width + 0.38,
                    foreground="white",
                    alpha=0.72,
                ),
                path_effects.Normal(),
            ]
        )
    trajectory_axis.axvline(
        0,
        color="#777777",
        linestyle="--",
        linewidth=0.75,
    )
    trajectory_axis.set_xlim(history_x[0], 96)
    trajectory_axis.set_xlabel("Time step relative to forecast origin")
    trajectory_axis.set_ylabel("Value")
    trajectory_axis.set_title(
        "Same future steps, different horizon-specific forecasts",
        loc="left",
        pad=6,
    )
    trajectory_axis.legend(
        ncol=3,
        loc="upper right",
        handlelength=2.3,
        columnspacing=0.9,
        borderaxespad=0.25,
        numpoints=1,
    )
    delta_header = TextArea(
        r"Mean $|\Delta|$ from $H=720$",
        textprops={
            "fontsize": 5.8,
            "color": "#4A4A4A",
        },
    )
    delta_items = [
        TextArea(
            f"$H={horizon}$: {mean_absolute_differences[horizon]:.2f}",
            textprops={
                "fontsize": 6.0,
                "color": HORIZON_COLORS[horizon],
                "fontweight": "bold",
            },
        )
        for horizon in HORIZONS[:-1]
    ]
    delta_row = HPacker(
        children=delta_items,
        align="center",
        pad=0,
        sep=7,
    )
    delta_box = VPacker(
        children=[delta_header, delta_row],
        align="left",
        pad=0,
        sep=2,
    )
    delta_annotation = AnchoredOffsetbox(
        loc="lower right",
        child=delta_box,
        frameon=True,
        bbox_to_anchor=(0.99, 0.02),
        bbox_transform=trajectory_axis.transAxes,
        borderpad=0.0,
        pad=0.2,
    )
    delta_annotation.patch.set_facecolor("white")
    delta_annotation.patch.set_edgecolor("#D6D6D6")
    delta_annotation.patch.set_linewidth(0.55)
    delta_annotation.patch.set_alpha(0.94)
    trajectory_axis.add_artist(delta_annotation)
    panel_label(trajectory_axis, "a")

    pair_display = pair_matrix[:-1, 1:]
    masked = np.ma.masked_invalid(pair_display)
    PREFIX_CMAP.set_bad("#FFFFFF")
    image = heatmap_axis.imshow(
        masked,
        cmap=PREFIX_CMAP,
        vmin=0.0,
        vmax=float(np.nanmax(pair_display)),
        aspect="equal",
    )
    heatmap_axis.set_xticks(
        range(len(HORIZONS) - 1),
        labels=HORIZONS[1:],
    )
    heatmap_axis.set_yticks(
        range(len(HORIZONS) - 1),
        labels=HORIZONS[:-1],
    )
    heatmap_axis.set_xlabel("Longer requested horizon")
    heatmap_axis.set_ylabel("Shorter requested horizon")
    heatmap_axis.set_title(
        "Cross-horizon disagreement",
        loc="left",
        pad=6,
    )
    heatmap_axis.set_xticks(
        np.arange(-0.5, len(HORIZONS) - 1, 1),
        minor=True,
    )
    heatmap_axis.set_yticks(
        np.arange(-0.5, len(HORIZONS) - 1, 1),
        minor=True,
    )
    heatmap_axis.grid(
        which="minor",
        color="white",
        linewidth=1.0,
    )
    heatmap_axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(len(HORIZONS) - 1):
        for column in range(len(HORIZONS) - 1):
            value = pair_display[row, column]
            if np.isfinite(value):
                text_color = (
                    "white"
                    if value > 0.63 * float(np.nanmax(pair_display))
                    else "#202020"
                )
                heatmap_axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=text_color,
                )
    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        shrink=0.67,
        pad=0.04,
    )
    colorbar.set_label("Mean NCHPD")
    panel_label(heatmap_axis, "b")
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
    winners = np.argmin(risk, axis=0)
    winner_risk = np.min(risk, axis=0)
    excess_over_region_best = (
        risk - winner_risk[None, :]
    ) / np.maximum(winner_risk[None, :], 1e-12)
    realized_gain = (
        fixed_risk - winner_risk
    ) / np.maximum(fixed_risk, 1e-12)

    figure = plt.figure(
        figsize=(DOUBLE_COLUMN_WIDTH, 3.08),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.52, 1.0),
        wspace=0.12,
    )
    heatmap_axis = figure.add_subplot(grid[0, 0])
    gain_axis = figure.add_subplot(grid[0, 1])

    image = heatmap_axis.imshow(
        excess_over_region_best * 100.0,
        cmap=SHARING_CMAP,
        vmin=0.0,
        vmax=float(np.max(excess_over_region_best * 100.0)),
        aspect="auto",
    )
    heatmap_axis.set_xticks(range(12), labels=range(1, 13))
    heatmap_axis.set_yticks(range(len(SCALES)), labels=SCALES)
    heatmap_axis.set_xlabel("60-step future region")
    heatmap_axis.set_ylabel(r"Cross-step sharing extent $s$")
    heatmap_axis.set_title(
        "Excess MSE above each region's best extent",
        loc="left",
        pad=6,
    )
    heatmap_axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    heatmap_axis.set_yticks(
        np.arange(-0.5, len(SCALES), 1),
        minor=True,
    )
    heatmap_axis.grid(
        which="minor",
        color="white",
        linewidth=0.8,
    )
    heatmap_axis.tick_params(which="minor", bottom=False, left=False)
    heatmap_axis.scatter(
        np.arange(12),
        winners,
        marker="s",
        s=25,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.9,
        label="Region best",
    )
    heatmap_axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        borderpad=0.2,
        handletextpad=0.35,
    )
    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        shrink=0.82,
        pad=0.03,
        fraction=0.045,
    )
    colorbar.set_label("Excess MSE (%)")
    panel_label(heatmap_axis, "a")

    region_x = np.arange(1, 13)
    winner_scales = [SCALES[int(index)] for index in winners]
    bars = gain_axis.bar(
        region_x,
        realized_gain * 100.0,
        color=[SCALE_COLORS[scale] for scale in winner_scales],
        width=0.78,
        edgecolor="white",
        linewidth=0.45,
    )
    gain_axis.axhline(0.0, color="#555555", linewidth=0.75)
    mean_gain = 100.0 * float(selected["sample_oracle_headroom"])
    gain_axis.axhline(
        mean_gain,
        color="#666666",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )
    gain_axis.set_xticks(region_x)
    gain_axis.set_xlabel("60-step future region")
    gain_axis.set_ylabel(f"Gain vs fixed s={fixed_scale} (%)")
    gain_axis.set_title(
        "Region-wise gain over the best fixed extent",
        loc="left",
        pad=6,
    )
    gain_axis.grid(
        axis="y",
        color="#D9D9D9",
        alpha=0.35,
        linewidth=0.45,
        zorder=-2,
    )
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
        max(1.0, float(np.max(realized_gain * 100.0)) * 1.2),
    )
    gain_axis.legend(
        handles=legend_handles,
        title="Winning extent",
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=0.45,
        handlelength=0.9,
        handletextpad=0.3,
        fontsize=5.4,
        title_fontsize=5.6,
    )
    for bar, value in zip(bars, realized_gain * 100.0):
        if value >= 1.0:
            gain_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.55,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#3F3F3F",
            )
    gain_axis.text(
        0.98,
        0.80,
        "All five extents win 2–3 regions",
        transform=gain_axis.transAxes,
        va="top",
        ha="right",
        fontsize=6.1,
        color="#4A4A4A",
    )
    gain_axis.text(
        12.35,
        mean_gain + 0.45,
        f"Mean = {mean_gain:.1f}%",
        va="bottom",
        ha="right",
        fontsize=5.6,
        color="#5A5A5A",
    )
    panel_label(gain_axis, "b")
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
        "figure_contract": {
            "archetype": "asymmetric quantitative composite",
            "final_width_mm": 183.0,
            "backend": "Python/matplotlib",
        },
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
            "layout": (
                "two-panel trajectory hero plus validation NCHPD heatmap"
            ),
            "trajectory_encoding": (
                "thin solid horizon colors with sparse staggered markers "
                "and subtle white separation strokes; H=720 is the "
                "lower-layer reference"
            ),
            "delta_summary": (
                "mean absolute prediction difference from H=720 over "
                "the shared 96 future steps"
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
            "heatmap_encoding": (
                "percent MSE excess over the best sharing extent "
                "within each future region"
            ),
            "bar_reference": (
                "percent MSE gain over the sample-best fixed extent s=720"
            ),
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
