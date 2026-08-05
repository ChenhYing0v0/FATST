#!/usr/bin/env python3
"""Render the manuscript-facing ISCF-BSCA architecture overview."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


DOUBLE_COLUMN_WIDTH = 183.0 / 25.4
FIGURE_HEIGHT = 112.0 / 25.4

COLORS = {
    "ink": "#26323B",
    "muted": "#66737E",
    "guide": "#B8C0C7",
    "panel": "#FBFCFD",
    "history": "#6675A8",
    "history_light": "#E8EAF4",
    "coordinate": "#D9A441",
    "coordinate_light": "#FAEFD5",
    "field": "#367F91",
    "field_light": "#DCEEF1",
    "allocation": "#765A9B",
    "allocation_light": "#EBE3F2",
    "forecast": "#246B5B",
    "forecast_light": "#DDEEE9",
    "training": "#C76858",
    "training_light": "#FAE7E3",
    "single": "#7D8790",
    "single_light": "#E9ECEF",
}

SCOPE_COLORS = (
    "#DCEAF5",
    "#B8D6EA",
    "#88B6D3",
    "#5E8FB8",
    "#355F91",
)
SCOPE_EDGES = (
    "#7599B6",
    "#648BAA",
    "#4E789A",
    "#3F688A",
    "#2C527B",
)
SCOPE_LABELS = (r"$s=1$", r"$s=48$", r"$s=144$", r"$s=360$", r"$s=720$")


def configure_style() -> None:
    """Configure a compact publication-oriented matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 6.5,
            "axes.titlesize": 7.6,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def rounded_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 5.8,
    textcolor: str | None = None,
    linewidth: float = 0.8,
    radius: float = 0.012,
    zorder: int = 3,
) -> FancyBboxPatch:
    """Draw one rounded module box in axis coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=axis.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor or COLORS["ink"],
        transform=axis.transAxes,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    linewidth: float = 0.8,
    linestyle: str = "-",
    mutation_scale: float = 7.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
) -> FancyArrowPatch:
    """Draw a directed edge in axis coordinates."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        transform=axis.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def panel_frame(axis: Axes, label: str, title: str) -> None:
    """Apply a common frame, title and panel label."""
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    frame = FancyBboxPatch(
        (0.006, 0.008),
        0.988,
        0.982,
        boxstyle="round,pad=0.004,rounding_size=0.016",
        facecolor=COLORS["panel"],
        edgecolor="#D6DBE0",
        linewidth=0.75,
        transform=axis.transAxes,
        clip_on=False,
        zorder=-10,
    )
    axis.add_patch(frame)
    axis.text(
        0.025,
        0.955,
        label,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["ink"],
        transform=axis.transAxes,
    )
    axis.text(
        0.078,
        0.953,
        title,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["ink"],
        transform=axis.transAxes,
    )


def draw_waveform(
    axis: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    *,
    color: str,
    phase: float = 0.0,
    linewidth: float = 1.0,
    zorder: int = 4,
) -> None:
    """Draw a deterministic schematic trajectory."""
    x = np.linspace(0.0, 1.0, 160)
    y = 0.5 + 0.24 * np.sin(2.0 * np.pi * (1.5 * x + phase))
    y += 0.08 * np.sin(2.0 * np.pi * (4.2 * x + 0.35 * phase))
    axis.plot(
        x0 + width * x,
        y0 + height * y,
        color=color,
        linewidth=linewidth,
        transform=axis.transAxes,
        clip_on=False,
        zorder=zorder,
    )


def draw_scope_blocks(
    axis: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    *,
    group_size: int,
    facecolor: str,
    edgecolor: str,
    columns: int = 12,
) -> None:
    """Draw contiguous future-step groups for one sharing scope."""
    cell_width = width / columns
    for start in range(0, columns, group_size):
        count = min(group_size, columns - start)
        rect = Rectangle(
            (x0 + start * cell_width, y0),
            count * cell_width - 0.002,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.65,
            transform=axis.transAxes,
            clip_on=False,
            zorder=3,
        )
        axis.add_patch(rect)
        if count > 1:
            for offset in range(1, count):
                axis.plot(
                    [x0 + (start + offset) * cell_width] * 2,
                    [y0, y0 + height],
                    color="white",
                    linewidth=0.35,
                    alpha=0.72,
                    transform=axis.transAxes,
                    zorder=4,
                )


def draw_matrix(
    axis: Axes,
    x0: float,
    y0: float,
    width: float,
    height: float,
    *,
    colors: tuple[str, ...],
    alpha_values: np.ndarray,
    edgecolor: str = "white",
) -> None:
    """Draw a schematic scope-by-target tensor slice."""
    rows, columns = alpha_values.shape
    cell_width = width / columns
    cell_height = height / rows
    for row in range(rows):
        for column in range(columns):
            base = mpl.colors.to_rgb(colors[row])
            alpha = float(alpha_values[row, column])
            mixed = tuple(1.0 - alpha * (1.0 - channel) for channel in base)
            rect = Rectangle(
                (x0 + column * cell_width, y0 + (rows - row - 1) * cell_height),
                cell_width,
                cell_height,
                facecolor=mixed,
                edgecolor=edgecolor,
                linewidth=0.30,
                transform=axis.transAxes,
                clip_on=False,
                zorder=3,
            )
            axis.add_patch(rect)
    border = Rectangle(
        (x0, y0),
        width,
        height,
        facecolor="none",
        edgecolor="#78838C",
        linewidth=0.65,
        transform=axis.transAxes,
        clip_on=False,
        zorder=5,
    )
    axis.add_patch(border)


def plot_single_scope(axis: Axes) -> None:
    """Draw the fixed-sharing comparison path."""
    panel_frame(axis, "a", "Single-scope forecasting")
    rounded_box(
        axis,
        0.055,
        0.60,
        0.20,
        0.12,
        "History state\n" + r"$\mathbf{R}$",
        facecolor=COLORS["history_light"],
        edgecolor=COLORS["history"],
    )
    rounded_box(
        axis,
        0.36,
        0.60,
        0.23,
        0.12,
        "One history\nprojection",
        facecolor=COLORS["single_light"],
        edgecolor=COLORS["single"],
    )
    arrow(axis, (0.255, 0.66), (0.36, 0.66))
    arrow(axis, (0.59, 0.66), (0.70, 0.66))
    draw_scope_blocks(
        axis,
        0.70,
        0.61,
        0.25,
        0.10,
        group_size=4,
        facecolor=COLORS["single_light"],
        edgecolor=COLORS["single"],
    )
    axis.text(
        0.79,
        0.755,
        "one fixed sharing extent",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color=COLORS["single"],
        transform=axis.transAxes,
    )
    rounded_box(
        axis,
        0.22,
        0.27,
        0.56,
        0.12,
        "The same latent-state sharing pattern is applied\nthroughout the future domain",
        facecolor="white",
        edgecolor="#CDD3D8",
        fontsize=5.5,
    )
    axis.plot(
        [0.825, 0.825],
        [0.61, 0.43],
        color=COLORS["guide"],
        linestyle=(0, (2, 2)),
        linewidth=0.7,
        transform=axis.transAxes,
        zorder=1,
    )
    axis.text(
        0.50,
        0.12,
        r"fixed $s$ across $\tau=1,\ldots,T$",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )


def plot_scope_field(axis: Axes) -> None:
    """Draw the ISCF scope-indexed forecast field construction."""
    panel_frame(axis, "b", "Independent Scope-Conditioned Forecasting")
    rounded_box(
        axis,
        0.025,
        0.43,
        0.14,
        0.15,
        "Shared\nhistory state\n" + r"$\mathbf{R}$",
        facecolor=COLORS["history_light"],
        edgecolor=COLORS["history"],
        fontsize=5.5,
    )
    row_centers = np.linspace(0.77, 0.23, 5)
    group_sizes = (1, 2, 3, 6, 12)
    for row, (center, label) in enumerate(zip(row_centers, SCOPE_LABELS)):
        rounded_box(
            axis,
            0.215,
            center - 0.045,
            0.13,
            0.09,
            "independent\nprojection\n" + label,
            facecolor="white",
            edgecolor=SCOPE_EDGES[row],
            fontsize=5.0,
            linewidth=0.75,
        )
        arrow(
            axis,
            (0.165, 0.505),
            (0.215, center),
            color=SCOPE_EDGES[row],
            connectionstyle=f"arc3,rad={(2 - row) * 0.05}",
        )
        draw_scope_blocks(
            axis,
            0.385,
            center - 0.035,
            0.31,
            0.07,
            group_size=group_sizes[row],
            facecolor=SCOPE_COLORS[row],
            edgecolor=SCOPE_EDGES[row],
        )
        arrow(
            axis,
            (0.345, center),
            (0.385, center),
            color=SCOPE_EDGES[row],
            mutation_scale=6.0,
        )
        arrow(
            axis,
            (0.695, center),
            (0.745, center),
            color=SCOPE_EDGES[row],
            mutation_scale=6.0,
        )

    rounded_box(
        axis,
        0.745,
        0.17,
        0.105,
        0.66,
        "Shared\nstep-specific\nsynthesis\n" + r"$(\mathbf{a}_\tau,\mathbf{n}_\tau)$",
        facecolor=COLORS["coordinate_light"],
        edgecolor=COLORS["coordinate"],
        fontsize=5.0,
    )
    field_values = np.array(
        [
            [0.35, 0.55, 0.72, 0.44, 0.28, 0.62, 0.78, 0.48, 0.32, 0.58, 0.70, 0.42],
            [0.42, 0.60, 0.66, 0.38, 0.32, 0.68, 0.74, 0.45, 0.36, 0.64, 0.67, 0.40],
            [0.50, 0.58, 0.62, 0.46, 0.40, 0.63, 0.68, 0.54, 0.44, 0.60, 0.64, 0.48],
            [0.60, 0.54, 0.50, 0.56, 0.62, 0.52, 0.48, 0.58, 0.66, 0.50, 0.46, 0.61],
            [0.68, 0.64, 0.58, 0.54, 0.50, 0.47, 0.45, 0.48, 0.52, 0.58, 0.63, 0.69],
        ]
    )
    draw_matrix(
        axis,
        0.895,
        0.22,
        0.08,
        0.56,
        colors=SCOPE_EDGES,
        alpha_values=field_values,
    )
    for center in row_centers:
        arrow(
            axis,
            (0.85, center),
            (0.895, center),
            color=COLORS["field"],
            mutation_scale=6.0,
        )
    axis.text(
        0.90,
        0.84,
        "scope-indexed\nforecast field",
        ha="center",
        va="bottom",
        fontsize=5.3,
        fontweight="bold",
        color=COLORS["field"],
        transform=axis.transAxes,
    )
    axis.text(
        0.90,
        0.13,
        r"$\mathcal{F}(\mathbf{X})\in\mathbb{R}^{B\times C\times T\times S}$",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["field"],
        transform=axis.transAxes,
    )
    axis.text(
        0.54,
        0.10,
        "block width = latent-state sharing extent; each cell remains step specific",
        ha="center",
        va="center",
        fontsize=5.3,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )


def plot_allocation(axis: Axes) -> None:
    """Draw target-conditioned scope contraction and nested prefixes."""
    panel_frame(axis, "c", "Target-conditioned scope allocation")
    field_values = np.array(
        [
            [0.35, 0.55, 0.72, 0.44, 0.28, 0.62, 0.78, 0.48, 0.32, 0.58, 0.70, 0.42],
            [0.42, 0.60, 0.66, 0.38, 0.32, 0.68, 0.74, 0.45, 0.36, 0.64, 0.67, 0.40],
            [0.50, 0.58, 0.62, 0.46, 0.40, 0.63, 0.68, 0.54, 0.44, 0.60, 0.64, 0.48],
            [0.60, 0.54, 0.50, 0.56, 0.62, 0.52, 0.48, 0.58, 0.66, 0.50, 0.46, 0.61],
            [0.68, 0.64, 0.58, 0.54, 0.50, 0.47, 0.45, 0.48, 0.52, 0.58, 0.63, 0.69],
        ]
    )
    allocation_values = np.array(
        [
            [0.82, 0.72, 0.56, 0.38, 0.28, 0.24, 0.26, 0.32, 0.46, 0.62, 0.73, 0.78],
            [0.48, 0.58, 0.72, 0.76, 0.64, 0.45, 0.31, 0.26, 0.30, 0.44, 0.55, 0.60],
            [0.28, 0.34, 0.46, 0.62, 0.78, 0.82, 0.70, 0.52, 0.36, 0.30, 0.34, 0.42],
            [0.25, 0.28, 0.32, 0.38, 0.44, 0.58, 0.76, 0.84, 0.70, 0.48, 0.34, 0.28],
            [0.34, 0.28, 0.24, 0.26, 0.32, 0.40, 0.52, 0.66, 0.82, 0.86, 0.72, 0.54],
        ]
    )
    draw_matrix(
        axis,
        0.035,
        0.31,
        0.22,
        0.40,
        colors=SCOPE_EDGES,
        alpha_values=field_values,
    )
    axis.text(
        0.145,
        0.76,
        "scope field",
        ha="center",
        va="bottom",
        fontsize=5.5,
        fontweight="bold",
        color=COLORS["field"],
        transform=axis.transAxes,
    )
    axis.text(
        0.145,
        0.24,
        r"$\mathcal{F}:[B,C,T,S]$",
        ha="center",
        va="center",
        fontsize=5.3,
        color=COLORS["field"],
        transform=axis.transAxes,
    )
    axis.text(
        0.282,
        0.51,
        r"$\odot$",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["ink"],
        transform=axis.transAxes,
    )
    draw_matrix(
        axis,
        0.31,
        0.31,
        0.22,
        0.40,
        colors=(COLORS["allocation"],) * 5,
        alpha_values=allocation_values,
    )
    rounded_box(
        axis,
        0.315,
        0.80,
        0.21,
        0.09,
        r"$[\mathbf{R};\boldsymbol{\phi}_\tau]\;\rightarrow$ allocation MLP",
        facecolor=COLORS["allocation_light"],
        edgecolor=COLORS["allocation"],
        fontsize=5.1,
    )
    arrow(
        axis,
        (0.42, 0.80),
        (0.42, 0.72),
        color=COLORS["allocation"],
    )
    axis.text(
        0.42,
        0.24,
        r"$\boldsymbol{\Pi}:[B,C,T,S]$",
        ha="center",
        va="center",
        fontsize=5.3,
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )
    arrow(axis, (0.535, 0.51), (0.60, 0.51), color=COLORS["forecast"])
    rounded_box(
        axis,
        0.60,
        0.42,
        0.15,
        0.18,
        "weighted\ncontraction\n" + r"$\sum_s\pi_s\mathcal{F}_s$",
        facecolor=COLORS["forecast_light"],
        edgecolor=COLORS["forecast"],
        fontsize=5.4,
    )
    arrow(axis, (0.75, 0.51), (0.79, 0.51), color=COLORS["forecast"])
    draw_waveform(
        axis,
        0.80,
        0.39,
        0.17,
        0.25,
        color=COLORS["forecast"],
        phase=0.12,
        linewidth=1.35,
    )
    for x, label in ((0.845, r"$H_1$"), (0.905, r"$H_2$"), (0.968, r"$H_3$")):
        axis.plot(
            [x, x],
            [0.35, 0.67],
            color=COLORS["guide"],
            linestyle=(0, (2, 2)),
            linewidth=0.65,
            transform=axis.transAxes,
            zorder=2,
        )
        axis.text(
            x,
            0.70,
            label,
            ha="center",
            va="bottom",
            fontsize=5.2,
            color=COLORS["muted"],
            transform=axis.transAxes,
        )
    axis.text(
        0.79,
        0.25,
        "one trajectory\nwith nested request prefixes",
        ha="center",
        va="center",
        fontsize=5.25,
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.10,
        "conditioning uses history and target identity, not future observations",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )


def plot_bsca(axis: Axes) -> None:
    """Draw the training-only Balanced Scope Co-Adaptation objective."""
    panel_frame(axis, "d", "Balanced Scope Co-Adaptation")
    rounded_box(
        axis,
        0.055,
        0.72,
        0.18,
        0.10,
        "scope field\n" + r"$\mathcal{F}$",
        facecolor=COLORS["field_light"],
        edgecolor=COLORS["field"],
        fontsize=5.3,
    )
    rounded_box(
        axis,
        0.055,
        0.55,
        0.18,
        0.10,
        "allocation\n" + r"$\boldsymbol{\Pi}$",
        facecolor=COLORS["allocation_light"],
        edgecolor=COLORS["allocation"],
        fontsize=5.3,
    )
    rounded_box(
        axis,
        0.34,
        0.64,
        0.20,
        0.12,
        "weighted\ncontraction",
        facecolor=COLORS["forecast_light"],
        edgecolor=COLORS["forecast"],
        fontsize=5.4,
    )
    arrow(axis, (0.235, 0.77), (0.34, 0.72), color=COLORS["forecast"])
    arrow(axis, (0.235, 0.60), (0.34, 0.68), color=COLORS["forecast"])
    rounded_box(
        axis,
        0.65,
        0.65,
        0.16,
        0.10,
        "forecast\n" + r"$\widehat{\mathbf{Y}}$",
        facecolor="white",
        edgecolor=COLORS["forecast"],
        fontsize=5.4,
    )
    arrow(axis, (0.54, 0.70), (0.65, 0.70), color=COLORS["forecast"])
    axis.text(
        0.91,
        0.70,
        "inference",
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )
    axis.plot(
        [0.035, 0.965],
        [0.49, 0.49],
        color="#D7DCE0",
        linewidth=0.8,
        transform=axis.transAxes,
    )
    training_region = FancyBboxPatch(
        (0.035, 0.08),
        0.93,
        0.34,
        boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor=COLORS["training_light"],
        edgecolor=COLORS["training"],
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        transform=axis.transAxes,
        clip_on=False,
        zorder=0,
    )
    axis.add_patch(training_region)
    axis.text(
        0.055,
        0.405,
        "training only",
        ha="left",
        va="top",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["training"],
        transform=axis.transAxes,
    )
    rounded_box(
        axis,
        0.08,
        0.19,
        0.20,
        0.10,
        "fused loss\n" + r"$\mathcal{L}_{\mathrm{fuse}}$",
        facecolor="white",
        edgecolor=COLORS["training"],
        fontsize=5.2,
    )
    rounded_box(
        axis,
        0.35,
        0.19,
        0.22,
        0.10,
        "direct slice skill\n" + r"$\mathcal{L}_{\mathrm{skill}}$",
        facecolor="white",
        edgecolor=COLORS["training"],
        fontsize=5.2,
    )
    rounded_box(
        axis,
        0.64,
        0.19,
        0.22,
        0.10,
        "uniform anchor\n" + r"$\mathcal{L}_{\mathrm{anchor}}$",
        facecolor="white",
        edgecolor=COLORS["training"],
        fontsize=5.2,
    )
    for x in (0.18, 0.46, 0.75):
        arrow(
            axis,
            (x, 0.19),
            (0.50, 0.115),
            color=COLORS["training"],
            linewidth=0.75,
            linestyle="--",
            mutation_scale=6.0,
        )
    axis.text(
        0.50,
        0.095,
        r"$\mathcal{L}_{\mathrm{BSCA}}$",
        ha="center",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=COLORS["training"],
        transform=axis.transAxes,
        zorder=5,
    )
    axis.text(
        0.91,
        0.58,
        "BSCA adds no\ninference path",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )


def build_figure() -> Figure:
    """Construct the four-panel method overview."""
    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT))
    outer = figure.add_gridspec(
        2,
        height_ratios=(1.04, 0.96),
        left=0.012,
        right=0.988,
        bottom=0.025,
        top=0.985,
        hspace=0.045,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(0.82, 1.68), wspace=0.025)
    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.08, 1.42), wspace=0.025)
    plot_single_scope(figure.add_subplot(top[0, 0]))
    plot_scope_field(figure.add_subplot(top[0, 1]))
    plot_allocation(figure.add_subplot(bottom[0, 0]))
    plot_bsca(figure.add_subplot(bottom[0, 1]))
    return figure


def save_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Save the editable and raster publication formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_iscf_bsca_method_overview"
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    figure.savefig(paths["svg"])
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(paths["pdf"])
    figure.savefig(paths["png"], dpi=300)
    figure.savefig(
        paths["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return {key: str(path) for key, path in paths.items()}


def sync_manuscript_assets(
    outputs: dict[str, str],
    manuscript_dir: Path,
) -> dict[str, str]:
    """Copy the generated bundle into the stable manuscript asset directory."""
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    synced: dict[str, str] = {}
    for kind, source in outputs.items():
        destination = manuscript_dir / Path(source).name
        shutil.copy2(source, destination)
        synced[kind] = str(destination)
    return synced


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render the ISCF-BSCA Method overview figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_bsca_method_figure_20260805"),
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=Path("paper-figures"),
    )
    return parser.parse_args()


def main() -> None:
    """Render the figure and write its machine-readable manifest."""
    args = parse_args()
    configure_style()
    figure = build_figure()
    outputs = save_bundle(figure, args.output_dir)
    plt.close(figure)
    manuscript_outputs = sync_manuscript_assets(outputs, args.manuscript_dir)
    manifest = {
        "figure_id": "figure_iscf_bsca_method_overview",
        "figure_number": 4,
        "status": "initial_draft_for_author_review",
        "backend": "Python/matplotlib",
        "final_width_mm": 183.0,
        "final_height_mm": 112.0,
        "data_role": "architecture schematic",
        "empirical_data_used": False,
        "test_accessed": False,
        "panels": {
            "a": "single-scope forecasting with one fixed sharing extent",
            "b": "ISCF scope-indexed forecast field construction",
            "c": "target-conditioned scope allocation and nested prefixes",
            "d": "BSCA training-only objective and unchanged inference path",
        },
        "claim_boundary": [
            "explains the frozen architecture only",
            "does not report learned allocation behavior",
            "does not establish component effectiveness",
            "does not establish transferability or main-table superiority",
        ],
        "outputs": outputs,
        "manuscript_outputs": manuscript_outputs,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
