#!/usr/bin/env python3
"""Render the graphics-first ISCF architecture concept figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


FIGURE_WIDTH = 183.0 / 25.4
FIGURE_HEIGHT = 116.0 / 25.4

COLORS = {
    "ink": "#24313A",
    "muted": "#66737D",
    "guide": "#C8CFD5",
    "history": "#596AA5",
    "history_light": "#E8EBF5",
    "coordinate": "#D59C32",
    "coordinate_light": "#FAEED3",
    "allocation": "#76549A",
    "allocation_light": "#ECE3F2",
    "forecast": "#237264",
    "forecast_light": "#DDEFEA",
}

SCOPE_COLORS = (
    "#C9E2EC",
    "#9CC8D9",
    "#71ACC3",
    "#4E88AA",
    "#325E8E",
)
SCOPE_EDGES = (
    "#73A8BA",
    "#5E98AF",
    "#46829E",
    "#356C8F",
    "#274F7D",
)
SCOPE_LABELS = ("s=1", "s=48", "s=144", "s=360", "s=720")


def configure_style() -> None:
    """Configure compact publication-oriented defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 6.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    linewidth: float = 0.9,
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 7.0,
    linestyle: str = "-",
    zorder: int = 5,
) -> FancyArrowPatch:
    """Draw one directed edge in axis coordinates."""
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


def module_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 6.0,
) -> None:
    """Draw one of the two intentionally retained named modules."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        transform=axis.transAxes,
        clip_on=False,
        zorder=5,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["ink"],
        transform=axis.transAxes,
        zorder=6,
    )


def blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    """Blend a color with white at a bounded strength."""
    rgb = np.asarray(mpl.colors.to_rgb(color))
    value = float(np.clip(strength, 0.0, 1.0))
    return tuple(1.0 - value * (1.0 - rgb))


def draw_history(axis: Axes) -> None:
    """Draw a multivariate history as direct curve marks."""
    x0, y0, width, height = 0.018, 0.685, 0.112, 0.175
    axis.text(
        x0,
        y0 + height + 0.035,
        "multivariate history",
        ha="left",
        va="bottom",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["history"],
        transform=axis.transAxes,
    )
    time = np.linspace(0.0, 1.0, 220)
    phases = (0.00, 0.23, 0.47)
    offsets = (0.73, 0.50, 0.27)
    alphas = (1.0, 0.72, 0.48)
    for phase, offset, alpha in zip(phases, offsets, alphas):
        signal = np.sin(2.0 * np.pi * (1.45 * time + phase))
        signal += 0.38 * np.sin(2.0 * np.pi * (3.8 * time + 0.5 * phase))
        signal /= 1.38
        axis.plot(
            x0 + width * time,
            y0 + height * (offset + 0.13 * signal),
            color=COLORS["history"],
            linewidth=1.25,
            alpha=alpha,
            transform=axis.transAxes,
            clip_on=False,
            zorder=4,
        )
    axis.plot(
        [x0, x0 + width],
        [y0 - 0.012, y0 - 0.012],
        color=COLORS["guide"],
        linewidth=0.7,
        transform=axis.transAxes,
        zorder=1,
    )
    axis.text(
        x0,
        y0 - 0.035,
        r"$1$",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width,
        y0 - 0.035,
        r"$L$",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )


def draw_history_state(axis: Axes) -> None:
    """Draw the encoded history state as a feature vector."""
    x0, y0, width, height = 0.238, 0.675, 0.025, 0.18
    values = (0.25, 0.48, 0.72, 0.42, 0.88, 0.57, 0.32, 0.68, 0.51)
    cell_height = height / len(values)
    for index, value in enumerate(values):
        axis.add_patch(
            Rectangle(
                (x0, y0 + index * cell_height),
                width,
                cell_height - 0.001,
                facecolor=blend_with_white(COLORS["history"], value),
                edgecolor="white",
                linewidth=0.35,
                transform=axis.transAxes,
                zorder=4,
            )
        )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["history"],
            linewidth=0.9,
            transform=axis.transAxes,
            zorder=5,
        )
    )
    axis.text(
        x0 + width / 2.0,
        y0 + height + 0.045,
        "history state",
        ha="center",
        va="bottom",
        fontsize=6.1,
        fontweight="bold",
        color=COLORS["history"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        y0 + height + 0.014,
        r"$\mathbf{r}_{b,c}\in\mathbb{R}^R$",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color=COLORS["history"],
        transform=axis.transAxes,
    )


def draw_coordinate_field(axis: Axes) -> None:
    """Draw four DCT-style future coordinates as parallel curves."""
    x0, y0, width, height = 0.018, 0.175, 0.235, 0.235
    axis.text(
        x0,
        y0 + height + 0.035,
        "future-coordinate field",
        ha="left",
        va="bottom",
        fontsize=6.3,
        fontweight="bold",
        color=COLORS["coordinate"],
        transform=axis.transAxes,
    )
    time = np.linspace(0.0, 1.0, 260)
    colors = ("#E6BE68", "#D9A43D", "#C88B28", "#AA6D1D")
    for index, color in enumerate(colors):
        center = y0 + height * (0.84 - 0.22 * index)
        axis.plot(
            [x0, x0 + width],
            [center, center],
            color="#E5E1D8",
            linewidth=0.45,
            transform=axis.transAxes,
            zorder=1,
        )
        if index == 0:
            values = np.zeros_like(time)
        else:
            values = np.cos(np.pi * index * (time + 0.002))
        axis.plot(
            x0 + width * time,
            center + height * 0.075 * values,
            color=color,
            linewidth=1.05,
            transform=axis.transAxes,
            clip_on=False,
            zorder=4,
        )
        axis.text(
            x0 - 0.006,
            center,
            rf"$\phi_{{\tau,{index}}}$",
            ha="right",
            va="center",
            fontsize=5.0,
            color=color,
            transform=axis.transAxes,
        )
    axis.plot(
        [x0, x0 + width],
        [y0 - 0.012, y0 - 0.012],
        color=COLORS["guide"],
        linewidth=0.7,
        transform=axis.transAxes,
        zorder=1,
    )
    axis.text(
        x0,
        y0 - 0.035,
        r"$\tau=1$",
        ha="center",
        va="top",
        fontsize=5.1,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width,
        y0 - 0.035,
        r"$T$",
        ha="center",
        va="top",
        fontsize=5.1,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        y0 - 0.072,
        r"$\boldsymbol{\Phi}\in\mathbb{R}^{T\times D_q}$",
        ha="center",
        va="top",
        fontsize=5.4,
        color=COLORS["coordinate"],
        transform=axis.transAxes,
    )


def draw_scope_matrices(axis: Axes, row_centers: np.ndarray) -> None:
    """Draw the independent scope-conditioned mode matrices."""
    x0, width, height = 0.302, 0.050, 0.044
    axis.text(
        x0 + width / 2.0,
        0.925,
        "scope matrices",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=SCOPE_EDGES[-1],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        0.905,
        r"$\mathbf{M}^{(s)}\in\mathbb{R}^{D_q\times K}$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    for row, center in enumerate(row_centers):
        rows, columns = 4, 6
        cell_width = width / columns
        cell_height = height / rows
        for r_index in range(rows):
            for c_index in range(columns):
                strength = 0.28 + 0.62 * (
                    0.5 + 0.5 * np.sin((row + 1) * (r_index + 1) + c_index)
                )
                axis.add_patch(
                    Rectangle(
                        (x0 + c_index * cell_width, center - height / 2 + r_index * cell_height),
                        cell_width,
                        cell_height,
                        facecolor=blend_with_white(SCOPE_COLORS[row], strength),
                        edgecolor="white",
                        linewidth=0.25,
                        transform=axis.transAxes,
                        zorder=4,
                    )
                )
        axis.add_patch(
            Rectangle(
                (x0, center - height / 2),
                width,
                height,
                facecolor="none",
                edgecolor=SCOPE_EDGES[row],
                linewidth=0.75,
                transform=axis.transAxes,
                zorder=5,
            )
        )
        axis.text(
            x0 - 0.007,
            center,
            SCOPE_LABELS[row],
            ha="right",
            va="center",
            fontsize=5.0,
            color=SCOPE_EDGES[row],
            transform=axis.transAxes,
        )


def draw_region_descriptors(axis: Axes, row_centers: np.ndarray) -> None:
    """Draw scope-dependent pooling of future coordinates."""
    x0, width, height = 0.392, 0.105, 0.032
    group_sizes = (1, 2, 3, 6, 12)
    axis.text(
        x0 + width / 2.0,
        0.925,
        "region descriptors",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=COLORS["coordinate"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        0.905,
        r"$\overline{\boldsymbol{\phi}}^{(s)}_g$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    columns = 12
    cell_width = width / columns
    for row, (center, group_size) in enumerate(zip(row_centers, group_sizes)):
        for start in range(0, columns, group_size):
            count = min(group_size, columns - start)
            group_index = start // group_size
            strength = 0.34 + 0.50 * (
                0.5 + 0.5 * np.cos(0.9 * group_index + row)
            )
            axis.add_patch(
                Rectangle(
                    (x0 + start * cell_width, center - height / 2),
                    count * cell_width - 0.001,
                    height,
                    facecolor=blend_with_white(COLORS["coordinate"], strength),
                    edgecolor="#C99535",
                    linewidth=0.45,
                    transform=axis.transAxes,
                    zorder=4,
                )
            )
            for offset in range(1, count):
                axis.plot(
                    [x0 + (start + offset) * cell_width] * 2,
                    [center - height / 2, center + height / 2],
                    color="white",
                    linewidth=0.3,
                    transform=axis.transAxes,
                    zorder=5,
                )


def draw_region_representations(axis: Axes, row_centers: np.ndarray) -> None:
    """Draw groupwise latent vectors reused across future regions."""
    x0, width, height = 0.548, 0.108, 0.040
    group_sizes = (1, 2, 3, 6, 12)
    axis.text(
        x0 + width / 2.0,
        0.925,
        "region representations",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=SCOPE_EDGES[-1],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        0.905,
        r"$\mathbf{z}^{(s)}_g\in\mathbb{R}^K$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    columns = 12
    cell_width = width / columns
    for row, (center, group_size) in enumerate(zip(row_centers, group_sizes)):
        for start in range(0, columns, group_size):
            count = min(group_size, columns - start)
            group_width = count * cell_width - 0.001
            base_strength = 0.34 + 0.10 * ((start // group_size + row) % 4)
            for stripe in range(4):
                stripe_width = group_width / 4.0
                axis.add_patch(
                    Rectangle(
                        (
                            x0 + start * cell_width + stripe * stripe_width,
                            center - height / 2,
                        ),
                        stripe_width,
                        height,
                        facecolor=blend_with_white(
                            SCOPE_COLORS[row],
                            base_strength + 0.11 * stripe,
                        ),
                        edgecolor="white",
                        linewidth=0.25,
                        transform=axis.transAxes,
                        zorder=4,
                    )
                )
            axis.add_patch(
                Rectangle(
                    (x0 + start * cell_width, center - height / 2),
                    group_width,
                    height,
                    facecolor="none",
                    edgecolor=SCOPE_EDGES[row],
                    linewidth=0.5,
                    transform=axis.transAxes,
                    zorder=5,
                )
            )


def draw_synthesis_tensor(axis: Axes, row_centers: np.ndarray) -> None:
    """Draw the shared step-specific synthesis parameters as a tensor."""
    x0, y0, width, height = 0.690, 0.555, 0.034, 0.292
    rows, columns = 14, 5
    for row in range(rows):
        for column in range(columns):
            strength = 0.25 + 0.68 * (
                0.5 + 0.5 * np.sin(0.72 * row + 1.15 * column)
            )
            color = COLORS["coordinate"] if column < 3 else COLORS["allocation"]
            axis.add_patch(
                Rectangle(
                    (x0 + column * width / columns, y0 + row * height / rows),
                    width / columns,
                    height / rows,
                    facecolor=blend_with_white(color, strength),
                    edgecolor="white",
                    linewidth=0.22,
                    transform=axis.transAxes,
                    zorder=4,
                )
            )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor="#A67D36",
            linewidth=0.85,
            transform=axis.transAxes,
            zorder=5,
        )
    )
    axis.text(
        x0 + width / 2.0,
        0.925,
        "step-specific\nsynthesis",
        ha="center",
        va="bottom",
        fontsize=5.3,
        fontweight="bold",
        color="#9A732F",
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        y0 - 0.022,
        r"$\mathbf{a}_\tau,\mathbf{n}_\tau,\beta_\tau$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    for row, center in enumerate(row_centers):
        arrow(
            axis,
            (0.656, center),
            (x0, center),
            color=SCOPE_EDGES[row],
            linewidth=0.75,
            mutation_scale=5.7,
        )


def scope_wave(time: np.ndarray, row: int) -> np.ndarray:
    """Return one deterministic schematic scope prediction."""
    base = 0.55 * np.sin(2.0 * np.pi * (1.25 * time + 0.05 * row))
    local = (0.30 - 0.035 * row) * np.sin(
        2.0 * np.pi * ((3.2 - 0.25 * row) * time + 0.13 * row)
    )
    trend = 0.15 * (row - 2) * (time - 0.5)
    return base + local + trend


def draw_scope_forecasts(axis: Axes, row_centers: np.ndarray) -> None:
    """Draw five complete scope-conditioned forecast trajectories."""
    x0, width, height = 0.758, 0.105, 0.048
    time = np.linspace(0.0, 1.0, 220)
    axis.text(
        x0 + width / 2.0,
        0.925,
        "scope-wise\nglobal forecasts",
        ha="center",
        va="bottom",
        fontsize=5.6,
        fontweight="bold",
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        0.905,
        r"$\mathcal{F}_{b,c,\tau,s}$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    for row, center in enumerate(row_centers):
        values = scope_wave(time, row)
        axis.plot(
            [x0, x0 + width],
            [center, center],
            color="#E3E7EA",
            linewidth=0.45,
            transform=axis.transAxes,
            zorder=1,
        )
        axis.plot(
            x0 + width * time,
            center + height * values,
            color=SCOPE_EDGES[row],
            linewidth=1.05,
            transform=axis.transAxes,
            clip_on=False,
            zorder=5,
        )
        arrow(
            axis,
            (0.724, center),
            (x0, center),
            color=SCOPE_EDGES[row],
            linewidth=0.75,
            mutation_scale=5.7,
        )
    axis.plot(
        [x0, x0 + width],
        [0.535, 0.535],
        color=COLORS["guide"],
        linewidth=0.55,
        transform=axis.transAxes,
        zorder=1,
    )


def draw_condition_field(axis: Axes) -> None:
    """Draw history-plus-coordinate condition vectors over future targets."""
    x0, y0, width, height = 0.390, 0.205, 0.113, 0.145
    rows, columns = 7, 12
    for row in range(rows):
        for column in range(columns):
            if row < 3:
                color = COLORS["history"]
                strength = 0.48 + 0.08 * row
            else:
                color = COLORS["coordinate"]
                strength = 0.25 + 0.65 * (
                    0.5 + 0.5 * np.cos((row - 2) * column * np.pi / columns)
                )
            axis.add_patch(
                Rectangle(
                    (x0 + column * width / columns, y0 + row * height / rows),
                    width / columns,
                    height / rows,
                    facecolor=blend_with_white(color, strength),
                    edgecolor="white",
                    linewidth=0.25,
                    transform=axis.transAxes,
                    zorder=4,
                )
            )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["allocation"],
            linewidth=0.8,
            transform=axis.transAxes,
            zorder=5,
        )
    )
    axis.text(
        x0 + width / 2.0,
        y0 + height + 0.028,
        "target-wise condition vectors",
        ha="center",
        va="bottom",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        y0 - 0.021,
        r"$[\mathbf{u}_{b,c};\boldsymbol{\phi}_\tau]$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 - 0.008,
        y0 + 0.025,
        "history",
        ha="right",
        va="center",
        fontsize=5.0,
        color=COLORS["history"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 - 0.008,
        y0 + 0.108,
        "target",
        ha="right",
        va="center",
        fontsize=5.0,
        color=COLORS["coordinate"],
        transform=axis.transAxes,
    )


def allocation_probabilities() -> np.ndarray:
    """Return a deterministic schematic target-wise probability field."""
    target = np.linspace(0.0, 1.0, 12)
    logits = np.stack(
        (
            1.2 * np.cos(2.0 * np.pi * target),
            0.9 * np.sin(2.0 * np.pi * target + 0.5),
            0.8 * np.cos(3.0 * np.pi * target + 0.8),
            0.7 * np.sin(2.5 * np.pi * target + 1.5),
            0.6 * np.cos(1.5 * np.pi * target + 2.2),
        ),
        axis=0,
    )
    exponential = np.exp(logits - logits.max(axis=0, keepdims=True))
    return exponential / exponential.sum(axis=0, keepdims=True)


def draw_probability_field(axis: Axes) -> None:
    """Draw the target-wise scope probability tensor and one vector slice."""
    x0, y0, width, height = 0.665, 0.185, 0.145, 0.185
    probabilities = allocation_probabilities()
    rows, columns = probabilities.shape
    for row in range(rows):
        for column in range(columns):
            strength = 0.15 + 0.85 * probabilities[row, column] / probabilities.max()
            axis.add_patch(
                Rectangle(
                    (
                        x0 + column * width / columns,
                        y0 + (rows - row - 1) * height / rows,
                    ),
                    width / columns,
                    height / rows,
                    facecolor=blend_with_white(SCOPE_COLORS[row], strength),
                    edgecolor="white",
                    linewidth=0.3,
                    transform=axis.transAxes,
                    zorder=4,
                )
            )
        axis.text(
            x0 - 0.007,
            y0 + (rows - row - 0.5) * height / rows,
            SCOPE_LABELS[row],
            ha="right",
            va="center",
            fontsize=5.0,
            color=SCOPE_EDGES[row],
            transform=axis.transAxes,
        )
    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["allocation"],
            linewidth=0.8,
            transform=axis.transAxes,
            zorder=5,
        )
    )
    selected_column = 8
    axis.add_patch(
        Rectangle(
            (x0 + selected_column * width / columns, y0),
            width / columns,
            height,
            facecolor="none",
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            transform=axis.transAxes,
            zorder=6,
        )
    )
    axis.text(
        x0 + width / 2.0,
        y0 + height + 0.030,
        "scope-probability field",
        ha="center",
        va="bottom",
        fontsize=5.9,
        fontweight="bold",
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        y0 - 0.023,
        r"$\boldsymbol{\Pi}\in\mathbb{R}^{T\times S}$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
    )
    vector_x = 0.833
    selected = probabilities[:, selected_column]
    for row, value in enumerate(selected):
        y = y0 + (rows - row - 0.5) * height / rows
        axis.scatter(
            [vector_x],
            [y],
            s=12.0 + 85.0 * float(value),
            color=SCOPE_EDGES[row],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.45,
            transform=axis.transAxes,
            zorder=7,
        )
    axis.text(
        vector_x,
        y0 + height + 0.030,
        r"$\boldsymbol{\pi}_\tau$",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )


def draw_fusion_and_output(axis: Axes) -> None:
    """Draw pointwise weighted contraction and the final trajectory."""
    fusion_x, fusion_y = 0.892, 0.535
    axis.scatter(
        [fusion_x],
        [fusion_y],
        s=255,
        facecolor="white",
        edgecolor=COLORS["forecast"],
        linewidth=1.1,
        transform=axis.transAxes,
        zorder=7,
    )
    axis.text(
        fusion_x,
        fusion_y + 0.008,
        r"$\sum_s$",
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=COLORS["forecast"],
        transform=axis.transAxes,
        zorder=8,
    )
    axis.text(
        fusion_x,
        fusion_y - 0.050,
        "target-wise\nweighted contraction",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["muted"],
        transform=axis.transAxes,
        zorder=8,
    )
    arrow(
        axis,
        (0.863, 0.705),
        (fusion_x - 0.017, fusion_y + 0.018),
        color=COLORS["forecast"],
        connectionstyle="arc3,rad=0.18",
        linewidth=1.05,
    )
    arrow(
        axis,
        (0.845, 0.278),
        (fusion_x - 0.016, fusion_y - 0.018),
        color=COLORS["allocation"],
        connectionstyle="arc3,rad=-0.18",
        linewidth=1.05,
    )
    arrow(
        axis,
        (fusion_x + 0.018, fusion_y),
        (0.923, fusion_y),
        color=COLORS["forecast"],
        linewidth=1.15,
    )
    x0, width, height = 0.925, 0.060, 0.17
    time = np.linspace(0.0, 1.0, 240)
    values = 0.57 * np.sin(2.0 * np.pi * (1.35 * time + 0.04))
    values += 0.24 * np.sin(2.0 * np.pi * (3.1 * time + 0.18))
    axis.plot(
        x0 + width * time,
        fusion_y + height * 0.42 * values,
        color=COLORS["forecast"],
        linewidth=1.8,
        transform=axis.transAxes,
        clip_on=False,
        zorder=7,
    )
    axis.fill_between(
        x0 + width * time,
        fusion_y,
        fusion_y + height * 0.42 * values,
        color=COLORS["forecast_light"],
        alpha=0.55,
        transform=axis.transAxes,
        zorder=2,
    )
    axis.text(
        x0 + width / 2.0,
        fusion_y + 0.142,
        "one predicted\ntrajectory",
        ha="center",
        va="bottom",
        fontsize=5.8,
        fontweight="bold",
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )
    axis.text(
        x0 + width / 2.0,
        fusion_y - 0.125,
        r"$\widehat{\mathbf{Y}}_{1:T}$",
        ha="center",
        va="top",
        fontsize=5.7,
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )


def build_figure() -> Figure:
    """Construct the single-canvas ISCF concept figure."""
    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis = figure.add_axes((0.01, 0.02, 0.98, 0.96))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    axis.text(
        0.29,
        0.978,
        "Scope-conditioned forecasting field",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=SCOPE_EDGES[-1],
        transform=axis.transAxes,
    )
    axis.text(
        0.29,
        0.455,
        "Target-conditioned scope allocation",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )
    axis.plot(
        [0.286, 0.865],
        [0.485, 0.485],
        color="#E2E6E9",
        linewidth=0.75,
        transform=axis.transAxes,
        zorder=1,
    )

    draw_history(axis)
    module_box(
        axis,
        0.150,
        0.725,
        0.058,
        0.083,
        "Encoder",
        facecolor=COLORS["history_light"],
        edgecolor=COLORS["history"],
    )
    draw_history_state(axis)
    arrow(axis, (0.130, 0.767), (0.150, 0.767), color=COLORS["history"])
    arrow(axis, (0.208, 0.767), (0.238, 0.767), color=COLORS["history"])
    draw_coordinate_field(axis)

    row_centers = np.asarray((0.825, 0.762, 0.699, 0.636, 0.573))
    draw_scope_matrices(axis, row_centers)
    draw_region_descriptors(axis, row_centers)
    draw_region_representations(axis, row_centers)
    draw_synthesis_tensor(axis, row_centers)
    draw_scope_forecasts(axis, row_centers)

    arrow(
        axis,
        (0.263, 0.767),
        (0.302, 0.767),
        color=COLORS["history"],
        linewidth=1.0,
    )
    arrow(
        axis,
        (0.252, 0.365),
        (0.430, 0.552),
        color=COLORS["coordinate"],
        connectionstyle="arc3,rad=-0.13",
        linewidth=0.95,
    )
    for row, center in enumerate(row_centers):
        arrow(
            axis,
            (0.352, center),
            (0.365, center),
            color=SCOPE_EDGES[row],
            linewidth=0.65,
            mutation_scale=5.5,
        )
        axis.text(
            0.373,
            center,
            r"$\times$",
            ha="center",
            va="center",
            fontsize=7.0,
            color=COLORS["ink"],
            transform=axis.transAxes,
        )
        arrow(
            axis,
            (0.381, center),
            (0.392, center),
            color=SCOPE_EDGES[row],
            linewidth=0.65,
            mutation_scale=5.5,
        )
        arrow(
            axis,
            (0.497, center),
            (0.548, center),
            color=SCOPE_EDGES[row],
            linewidth=0.65,
            mutation_scale=5.5,
        )

    draw_condition_field(axis)
    module_box(
        axis,
        0.548,
        0.235,
        0.072,
        0.085,
        "Allocation\nMLP",
        facecolor=COLORS["allocation_light"],
        edgecolor=COLORS["allocation"],
        fontsize=5.8,
    )
    draw_probability_field(axis)
    arrow(
        axis,
        (0.252, 0.675),
        (0.405, 0.350),
        color=COLORS["history"],
        connectionstyle="arc3,rad=0.15",
        linewidth=0.95,
    )
    arrow(
        axis,
        (0.253, 0.275),
        (0.390, 0.275),
        color=COLORS["coordinate"],
        linewidth=0.95,
    )
    arrow(axis, (0.503, 0.277), (0.548, 0.277), color=COLORS["allocation"])
    arrow(axis, (0.620, 0.277), (0.665, 0.277), color=COLORS["allocation"])

    draw_fusion_and_output(axis)
    axis.text(
        0.874,
        0.674,
        r"$\mathcal{F}_{\tau,:}$",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["forecast"],
        transform=axis.transAxes,
    )
    axis.text(
        0.870,
        0.385,
        r"$\boldsymbol{\pi}_{\tau,:}$",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["allocation"],
        transform=axis.transAxes,
    )
    axis.text(
        0.985,
        0.065,
        "schematic tensors and probabilities; no empirical values",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#919AA1",
        transform=axis.transAxes,
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Parse output options."""
    parser = argparse.ArgumentParser(
        description="Render the graphics-first ISCF architecture concept."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_bsca_method_figure_redesign_20260805"),
    )
    return parser.parse_args()


def save_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Save the concept-review bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_iscf_architecture_concept_v2"
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
    return {kind: str(path) for kind, path in paths.items()}


def main() -> None:
    """Render the figure and write a machine-readable manifest."""
    args = parse_args()
    configure_style()
    figure = build_figure()
    outputs = save_bundle(figure, args.output_dir)
    plt.close(figure)
    manifest = {
        "figure_id": "figure_iscf_architecture_concept_v2",
        "status": "concept_draft_for_information_hierarchy_review",
        "backend": "Python/matplotlib",
        "final_width_mm": 183.0,
        "final_height_mm": 116.0,
        "data_role": "architecture schematic",
        "empirical_data_used": False,
        "manuscript_replacement": False,
        "named_module_boxes": ["Encoder", "Allocation MLP"],
        "visual_objects": [
            "history curves",
            "history-state vector",
            "four-channel future-coordinate curves",
            "scope matrices",
            "segmented region descriptors",
            "region-representation vectors",
            "step-specific synthesis tensor",
            "scope-wise forecast curves",
            "target-wise condition tensor",
            "scope-probability field and vector",
            "weighted contraction",
            "final trajectory",
        ],
        "claim_boundary": [
            "explains ISCF architecture only",
            "omits single-scope comparison and BSCA",
            "uses schematic probabilities rather than learned values",
            "does not establish effectiveness or specialization",
        ],
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
