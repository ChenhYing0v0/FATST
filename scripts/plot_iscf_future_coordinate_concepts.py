#!/usr/bin/env python3
"""Render three local design candidates for the ISCF future-coordinate glyph.

The figure is a deterministic architecture-design study. All curves are analytic
DCT basis functions; no empirical observation, learned activation, or model
output is displayed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


FIGURE_ID = "figure_iscf_future_coordinate_component_concepts_v1"
WIDTH_MM = 183.0
HEIGHT_MM = 78.0

COLORS = {
    "ink": "#25282D",
    "neutral": "#72777F",
    "hairline": "#D8DDE3",
    "panel": "#F8FAFC",
    "white": "#FFFFFF",
    "indigo": "#454D7A",
    "periwinkle": "#7585B7",
    "blue_soft": "#A8B8D9",
    "teal": "#319A9A",
    "teal_soft": "#A8DAD6",
    "violet": "#7864D8",
    "violet_soft": "#E6E1FA",
    "ochre": "#C88B2B",
}

BASIS_COLORS = (
    COLORS["indigo"],
    COLORS["periwinkle"],
    COLORS["blue_soft"],
    COLORS["teal"],
)


def configure_style() -> None:
    """Configure a restrained conference-paper style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def dct_basis(total_steps: int = 240, dimensions: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return the displayed centered and scaled DCT coordinate channels."""
    tau = np.arange(1, total_steps + 1, dtype=float)
    field = np.ones((dimensions, total_steps), dtype=float)
    for dim in range(1, dimensions):
        raw = np.cos(np.pi * (tau - 0.5) * dim / total_steps)
        field[dim] = np.sqrt(2.0) * (raw - raw.mean())
    scale = np.maximum(np.abs(field).max(axis=1, keepdims=True), 1e-8)
    return tau, field / scale


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral"],
    linewidth: float = 0.8,
    mutation_scale: float = 6.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 6,
) -> None:
    """Draw a compact directed connector."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def panel_header(
    ax: Axes,
    x0: float,
    x1: float,
    label: str,
    title: str,
    subtitle: str,
    *,
    recommended: bool = False,
) -> None:
    """Draw a consistent candidate heading."""
    ax.text(
        x0,
        66.0,
        label,
        ha="left",
        va="center",
        fontsize=7.5,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        x0 + 4.0,
        66.0,
        title,
        ha="left",
        va="center",
        fontsize=7.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        (x0 + x1) / 2,
        62.4,
        subtitle,
        ha="center",
        va="center",
        fontsize=5.3,
        color=COLORS["neutral"],
    )
    if recommended:
        badge_x = x1 - 13.0
        ax.add_patch(
            FancyBboxPatch(
                (badge_x, 64.6),
                12.4,
                2.8,
                boxstyle="round,pad=0.25,rounding_size=1.2",
                facecolor=COLORS["violet_soft"],
                edgecolor="none",
                zorder=1,
            )
        )
        ax.text(
            badge_x + 6.2,
            66.0,
            "recommended",
            ha="center",
            va="center",
            fontsize=5.0,
            color=COLORS["violet"],
            fontweight="bold",
            zorder=2,
        )


def signed_vector(
    ax: Axes,
    x: float,
    y_center: float,
    values: np.ndarray,
    *,
    label: str,
    scale: float = 3.0,
    compact: bool = False,
) -> None:
    """Represent a coordinate vector as signed nodes around a vertical spine."""
    spacing = 3.6 if compact else 5.0
    y_values = y_center + spacing * (np.arange(values.size)[::-1] - (values.size - 1) / 2)
    ax.plot(
        [x, x],
        [y_values.min() - 1.5, y_values.max() + 1.5],
        color=COLORS["hairline"],
        linewidth=1.1,
        zorder=1,
    )
    for dim, (value, y_value) in enumerate(zip(values, y_values)):
        endpoint = x + scale * float(value)
        ax.plot(
            [x, endpoint],
            [y_value, y_value],
            color=BASIS_COLORS[dim],
            linewidth=1.3,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [endpoint],
            [y_value],
            s=17 if compact else 23,
            facecolor=COLORS["white"],
            edgecolor=BASIS_COLORS[dim],
            linewidth=0.85,
            zorder=4,
        )
    ax.text(
        x,
        y_values.min() - 3.5,
        label,
        ha="center",
        va="top",
        fontsize=6.0,
        color=COLORS["violet"],
        fontweight="bold",
    )


def draw_harmonic_ribbon(ax: Axes, x0: float, x1: float) -> None:
    """Draw Candidate A: aligned harmonic lanes sampled at one future step."""
    panel_header(
        ax,
        x0,
        x1,
        "a",
        "Sampled harmonic ribbon",
        "field and target vector remain visible together",
        recommended=True,
    )
    tau, field = dct_basis()
    plot_x0, plot_x1 = x0 + 8.8, x1 - 13.5
    mapped_x = plot_x0 + (tau - 1.0) / (tau[-1] - 1.0) * (plot_x1 - plot_x0)
    lane_centers = np.array([52.0, 44.5, 37.0, 29.5])
    target_index = 158
    target_x = mapped_x[target_index]

    for dim, center in enumerate(lane_centers):
        amplitude = 2.1 if dim else 0.0
        curve_y = center + amplitude * field[dim]
        ax.plot(
            [plot_x0, plot_x1],
            [center, center],
            color=COLORS["hairline"],
            linewidth=0.45,
            zorder=1,
        )
        if dim == 0:
            curve_y = np.full_like(mapped_x, center + 1.2)
        ax.fill_between(
            mapped_x,
            center,
            curve_y,
            color=BASIS_COLORS[dim],
            alpha=0.075,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            mapped_x,
            curve_y,
            color=BASIS_COLORS[dim],
            linewidth=1.25,
            solid_capstyle="round",
            zorder=3,
        )
        channel_label = ("$d=0$", "$d=1$", "$d=2$", "$d=D_q-1$")[dim]
        ax.text(
            plot_x0 - 0.9,
            center,
            channel_label,
            ha="right",
            va="center",
            fontsize=5.0,
            color=COLORS["neutral"],
        )
        sampled_y = curve_y[target_index]
        ax.scatter(
            [target_x],
            [sampled_y],
            s=23,
            facecolor=COLORS["white"],
            edgecolor=COLORS["violet"],
            linewidth=1.05,
            zorder=7,
        )

    ax.plot(
        [target_x, target_x],
        [26.2, 55.2],
        color=COLORS["violet"],
        linewidth=1.1,
        zorder=6,
    )
    ax.text(target_x, 24.7, r"$\tau$", ha="center", va="top", fontsize=6.2, color=COLORS["violet"])
    ax.text(plot_x0, 24.7, "$1$", ha="center", va="top", fontsize=5.2, color=COLORS["neutral"])
    ax.text(plot_x1, 24.7, "$T$", ha="center", va="top", fontsize=5.2, color=COLORS["neutral"])

    vector_values = field[:, target_index]
    vector_x = x1 - 7.0
    arrow(
        ax,
        (target_x + 0.6, 54.8),
        (vector_x - 2.8, 54.8),
        color=COLORS["violet"],
        linewidth=0.8,
        connectionstyle="arc3,rad=-0.13",
    )
    signed_vector(ax, vector_x, 42.0, vector_values, label=r"$\boldsymbol{\phi}_{\tau}$", scale=2.4)

    bracket_x0 = plot_x0 + 0.22 * (plot_x1 - plot_x0)
    bracket_x1 = plot_x0 + 0.48 * (plot_x1 - plot_x0)
    bracket_y = 20.0
    ax.plot(
        [bracket_x0, bracket_x0, bracket_x1, bracket_x1],
        [bracket_y + 1.1, bracket_y, bracket_y, bracket_y + 1.1],
        color=COLORS["ochre"],
        linewidth=0.75,
    )
    ax.text(
        (bracket_x0 + bracket_x1) / 2,
        bracket_y - 1.0,
        r"mean over $\mathcal{G}_g^{(s)}$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["ochre"],
    )
    arrow(
        ax,
        ((bracket_x0 + bracket_x1) / 2, bracket_y - 3.1),
        ((bracket_x0 + bracket_x1) / 2, 14.2),
        color=COLORS["ochre"],
        linewidth=0.7,
        mutation_scale=5.0,
    )
    ax.text(
        (bracket_x0 + bracket_x1) / 2,
        12.7,
        r"$\overline{\boldsymbol{\phi}}^{(s)}_g$",
        ha="center",
        va="top",
        fontsize=5.8,
        color=COLORS["ochre"],
        fontweight="bold",
    )
    ax.text(
        (x0 + x1) / 2,
        7.0,
        "best semantic fidelity · low visual ambiguity",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )


def draw_spectral_fan(ax: Axes, x0: float, x1: float) -> None:
    """Draw Candidate B: perspective-separated basis curves."""
    panel_header(
        ax,
        x0,
        x1,
        "b",
        "Spectral fan",
        "more dimensional, but perspective is interpretive",
    )
    tau, field = dct_basis()
    base_x0, base_x1 = x0 + 6.0, x1 - 10.0
    target_index = 158
    target_fraction = target_index / (tau.size - 1)
    sampled_points: list[tuple[float, float]] = []

    for dim in range(3, -1, -1):
        depth = dim
        lane_x0 = base_x0 + 1.7 * depth
        lane_x1 = base_x1 + 1.7 * depth
        mapped_x = lane_x0 + (tau - 1.0) / (tau[-1] - 1.0) * (lane_x1 - lane_x0)
        center = 29.0 + 6.0 * depth
        amplitude = 2.2 if dim else 0.0
        curve_y = center + amplitude * field[dim]
        if dim == 0:
            curve_y = np.full_like(mapped_x, center + 1.2)
        ax.plot(
            mapped_x,
            curve_y,
            color=BASIS_COLORS[dim],
            linewidth=1.45,
            alpha=0.92,
            solid_capstyle="round",
            zorder=3 + dim,
        )
        ax.plot(
            [lane_x0, lane_x1],
            [center, center],
            color=COLORS["hairline"],
            linewidth=0.4,
            zorder=1,
        )
        sample_x = lane_x0 + target_fraction * (lane_x1 - lane_x0)
        sample_y = curve_y[target_index]
        sampled_points.append((sample_x, sample_y))
        ax.scatter(
            [sample_x],
            [sample_y],
            s=21,
            facecolor=COLORS["white"],
            edgecolor=COLORS["violet"],
            linewidth=0.95,
            zorder=9,
        )

    blade = np.array(
        [
            [sampled_points[-1][0] - 0.7, 25.2],
            [sampled_points[0][0] - 0.7, 49.5],
            [sampled_points[0][0] + 0.9, 54.0],
            [sampled_points[-1][0] + 0.9, 29.7],
        ]
    )
    ax.add_patch(
        Polygon(
            blade,
            closed=True,
            facecolor=COLORS["violet_soft"],
            edgecolor=COLORS["violet"],
            linewidth=0.65,
            alpha=0.48,
            zorder=2,
        )
    )
    ax.text(
        sampled_points[-1][0],
        23.7,
        r"$\tau$",
        ha="center",
        va="top",
        fontsize=6.1,
        color=COLORS["violet"],
    )
    vector_x = x1 - 5.3
    vector_values = field[:, target_index]
    arrow(
        ax,
        (sampled_points[0][0] + 1.5, 53.0),
        (vector_x - 2.8, 53.0),
        color=COLORS["violet"],
        linewidth=0.75,
        connectionstyle="arc3,rad=-0.13",
    )
    signed_vector(ax, vector_x, 41.0, vector_values, label=r"$\boldsymbol{\phi}_{\tau}$", scale=2.2)
    ax.text(
        (x0 + x1) / 2,
        12.0,
        "strong visual depth",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        (x0 + x1) / 2,
        8.7,
        "risk: apparent geometric dimension",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )


def mini_basis(ax: Axes, x0: float, x1: float, center: float, dim: int) -> None:
    """Draw one small harmonic stroke for the compact function bank."""
    tau, field = dct_basis(total_steps=100, dimensions=4)
    x = x0 + (tau - 1.0) / (tau[-1] - 1.0) * (x1 - x0)
    y = center + (0.0 if dim == 0 else 1.15 * field[dim])
    if dim == 0:
        y = np.full_like(x, center + 0.65)
    ax.plot(x, y, color=BASIS_COLORS[dim], linewidth=1.0, solid_capstyle="round")


def draw_coordinate_capsule(ax: Axes, x0: float, x1: float) -> None:
    """Draw Candidate C: a compact coordinate-to-vector mapping."""
    panel_header(
        ax,
        x0,
        x1,
        "c",
        "Coordinate capsule",
        "compact operator glyph for a narrow architecture row",
    )
    timeline_x0, timeline_x1 = x0 + 5.0, x0 + 20.5
    timeline_y = 43.0
    target_x = timeline_x0 + 0.66 * (timeline_x1 - timeline_x0)
    ax.plot(
        [timeline_x0, timeline_x1],
        [timeline_y, timeline_y],
        color=COLORS["hairline"],
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        np.linspace(timeline_x0, timeline_x1, 9),
        np.full(9, timeline_y),
        s=7,
        facecolor=COLORS["hairline"],
        edgecolor="none",
        zorder=2,
    )
    ax.scatter(
        [target_x],
        [timeline_y],
        s=55,
        facecolor=COLORS["white"],
        edgecolor=COLORS["violet"],
        linewidth=1.2,
        zorder=4,
    )
    ax.text(target_x, timeline_y + 4.0, r"$\tau$", ha="center", fontsize=6.6, color=COLORS["violet"])
    ax.text(timeline_x0, timeline_y - 3.0, "$1$", ha="center", fontsize=5.0, color=COLORS["neutral"])
    ax.text(timeline_x1, timeline_y - 3.0, "$T$", ha="center", fontsize=5.0, color=COLORS["neutral"])

    bank_x0, bank_x1 = x0 + 27.0, x0 + 39.0
    bank_centers = (51.0, 46.0, 41.0, 36.0)
    arrow(ax, (timeline_x1 + 1.2, timeline_y), (bank_x0 - 1.2, timeline_y), color=COLORS["violet"])
    for dim, center in enumerate(bank_centers):
        mini_basis(ax, bank_x0, bank_x1, center, dim)
    ax.plot(
        [bank_x0 - 1.1, bank_x0 - 1.1, bank_x0 - 0.2],
        [33.0, 54.0, 54.0],
        color=COLORS["neutral"],
        linewidth=0.6,
    )
    ax.plot(
        [bank_x1 + 1.1, bank_x1 + 1.1, bank_x1 + 0.2],
        [33.0, 54.0, 54.0],
        color=COLORS["neutral"],
        linewidth=0.6,
    )
    ax.text(
        (bank_x0 + bank_x1) / 2,
        29.8,
        "fixed DCT map",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["neutral"],
    )
    vector_x = x1 - 5.3
    _, field = dct_basis()
    vector_values = field[:, 158]
    arrow(ax, (bank_x1 + 1.7, timeline_y), (vector_x - 3.2, timeline_y), color=COLORS["violet"])
    signed_vector(ax, vector_x, 43.0, vector_values, label=r"$\boldsymbol{\phi}_{\tau}$", scale=2.0, compact=True)

    ax.text(
        (x0 + x1) / 2,
        17.0,
        r"$\tau\;\mapsto\;\{\phi_{\tau,d}\}_{d=0}^{D_q-1}$",
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["ink"],
    )
    ax.text(
        (x0 + x1) / 2,
        11.8,
        "highest compactness",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        (x0 + x1) / 2,
        8.7,
        "trade-off: shared field is less explicit",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )


def build_figure() -> Figure:
    """Assemble the component comparison sheet."""
    configure_style()
    figure = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, WIDTH_MM)
    axis.set_ylim(0.0, HEIGHT_MM)
    axis.set_axis_off()

    axis.text(
        4.0,
        74.1,
        "Future-step coordinate · local component study",
        ha="left",
        va="center",
        fontsize=9.1,
        color=COLORS["ink"],
        fontweight="bold",
    )
    axis.text(
        179.0,
        74.1,
        "fixed, parameter-free DCT field",
        ha="right",
        va="center",
        fontsize=5.5,
        color=COLORS["neutral"],
    )
    axis.plot([4.0, 179.0], [70.8, 70.8], color=COLORS["hairline"], linewidth=0.7)
    axis.plot([62.0, 62.0], [5.2, 68.8], color=COLORS["hairline"], linewidth=0.55)
    axis.plot([123.0, 123.0], [5.2, 68.8], color=COLORS["hairline"], linewidth=0.55)

    draw_harmonic_ribbon(axis, 4.0, 60.0)
    draw_spectral_fan(axis, 65.0, 121.0)
    draw_coordinate_capsule(axis, 126.0, 179.0)
    return figure


def build_recommended_component() -> Figure:
    """Build the recommended glyph at a realistic architecture-panel size."""
    configure_style()
    width_mm, height_mm = 82.0, 44.0
    figure = plt.figure(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, width_mm)
    axis.set_ylim(0.0, height_mm)
    axis.set_axis_off()

    axis.text(
        3.0,
        41.0,
        "fixed future-step coordinates",
        ha="left",
        va="center",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    axis.text(
        79.0,
        41.0,
        r"$\boldsymbol{\Phi}\in\mathbb{R}^{T\times D_q}$",
        ha="right",
        va="center",
        fontsize=5.6,
        color=COLORS["neutral"],
    )
    axis.plot([3.0, 79.0], [38.4, 38.4], color=COLORS["hairline"], linewidth=0.65)

    tau, field = dct_basis()
    plot_x0, plot_x1 = 12.0, 57.0
    mapped_x = plot_x0 + (tau - 1.0) / (tau[-1] - 1.0) * (plot_x1 - plot_x0)
    centers = np.array([33.0, 27.0, 21.0, 15.0])
    target_index = 158
    target_x = mapped_x[target_index]
    channel_labels = ("$d=0$", "$d=1$", "$d=2$", "$d=D_q-1$")

    for dim, center in enumerate(centers):
        curve_y = center + (0.0 if dim == 0 else 1.75 * field[dim])
        if dim == 0:
            curve_y = np.full_like(mapped_x, center + 1.0)
        axis.plot(
            [plot_x0, plot_x1],
            [center, center],
            color=COLORS["hairline"],
            linewidth=0.4,
            zorder=1,
        )
        axis.fill_between(
            mapped_x,
            center,
            curve_y,
            color=BASIS_COLORS[dim],
            alpha=0.07,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            mapped_x,
            curve_y,
            color=BASIS_COLORS[dim],
            linewidth=1.15,
            solid_capstyle="round",
            zorder=3,
        )
        axis.text(
            plot_x0 - 1.1,
            center,
            channel_labels[dim],
            ha="right",
            va="center",
            fontsize=5.0,
            color=COLORS["neutral"],
        )
        axis.scatter(
            [target_x],
            [curve_y[target_index]],
            s=20,
            facecolor=COLORS["white"],
            edgecolor=COLORS["violet"],
            linewidth=0.95,
            zorder=7,
        )

    axis.plot(
        [target_x, target_x],
        [12.2, 35.7],
        color=COLORS["violet"],
        linewidth=1.05,
        zorder=6,
    )
    axis.text(target_x, 10.8, r"$\tau$", ha="center", va="top", fontsize=6.0, color=COLORS["violet"])
    axis.text(plot_x0, 10.8, "$1$", ha="center", va="top", fontsize=5.0, color=COLORS["neutral"])
    axis.text(plot_x1, 10.8, "$T$", ha="center", va="top", fontsize=5.0, color=COLORS["neutral"])

    vector_x = 70.5
    arrow(
        axis,
        (target_x + 0.5, 35.1),
        (vector_x - 3.0, 35.1),
        color=COLORS["violet"],
        linewidth=0.75,
        connectionstyle="arc3,rad=-0.12",
    )
    signed_vector(
        axis,
        vector_x,
        24.5,
        field[:, target_index],
        label=r"$\boldsymbol{\phi}_{\tau}$",
        scale=2.4,
        compact=True,
    )
    axis.text(
        vector_x,
        10.7,
        "target identity",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["neutral"],
    )

    bracket_x0 = plot_x0 + 0.18 * (plot_x1 - plot_x0)
    bracket_x1 = plot_x0 + 0.40 * (plot_x1 - plot_x0)
    axis.plot(
        [bracket_x0, bracket_x0, bracket_x1, bracket_x1],
        [8.3, 7.4, 7.4, 8.3],
        color=COLORS["ochre"],
        linewidth=0.7,
    )
    axis.text(
        (bracket_x0 + bracket_x1) / 2,
        6.5,
        r"pool $\mathcal{G}_g^{(s)}$",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["ochre"],
    )
    arrow(
        axis,
        ((bracket_x0 + bracket_x1) / 2, 4.6),
        ((bracket_x0 + bracket_x1) / 2, 2.5),
        color=COLORS["ochre"],
        linewidth=0.65,
        mutation_scale=4.8,
    )
    axis.text(
        (bracket_x0 + bracket_x1) / 2 + 2.1,
        2.4,
        r"$\overline{\boldsymbol{\phi}}_g^{(s)}$",
        ha="left",
        va="center",
        fontsize=5.2,
        color=COLORS["ochre"],
        fontweight="bold",
    )
    return figure


def normalize_svg(path: Path) -> None:
    """Strip renderer-introduced line-end whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Export editable and review formats plus a machine-readable manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / FIGURE_ID
    outputs = {
        "svg": f"{prefix}.svg",
        "pdf": f"{prefix}.pdf",
        "png": f"{prefix}.png",
        "tiff": f"{prefix}.tiff",
    }
    figure.savefig(outputs["svg"], format="svg")
    normalize_svg(Path(outputs["svg"]))
    figure.savefig(outputs["pdf"], format="pdf")
    figure.savefig(outputs["png"], format="png", dpi=300)
    figure.savefig(
        outputs["tiff"],
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    manifest = {
        "figure_id": FIGURE_ID,
        "status": "local_design_study_for_author_review",
        "backend": "Python/matplotlib",
        "final_width_mm": WIDTH_MM,
        "final_height_mm": HEIGHT_MM,
        "empirical_data_used": False,
        "analytic_basis": "centered and scaled DCT cosine channels",
        "candidates": [
            "sampled_harmonic_ribbon_recommended",
            "spectral_fan",
            "coordinate_capsule",
        ],
        "main_figure_replacement": False,
        "outputs": outputs,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plt.close(figure)
    return outputs


def save_recommended_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Export the standalone recommended coordinate glyph."""
    prefix = output_dir / "figure_iscf_future_coordinate_component_recommended_v1"
    outputs = {
        "svg": f"{prefix}.svg",
        "pdf": f"{prefix}.pdf",
        "png": f"{prefix}.png",
        "tiff": f"{prefix}.tiff",
    }
    figure.savefig(outputs["svg"], format="svg")
    normalize_svg(Path(outputs["svg"]))
    figure.savefig(outputs["pdf"], format="pdf")
    figure.savefig(outputs["png"], format="png", dpi=300)
    figure.savefig(
        outputs["tiff"],
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_bsca_future_coordinate_component_study_20260805"),
    )
    return parser.parse_args()


def main() -> None:
    """Render the comparison sheet."""
    args = parse_args()
    figure = build_figure()
    comparison_outputs = save_bundle(figure, args.output_dir)
    recommended_figure = build_recommended_component()
    recommended_outputs = save_recommended_bundle(recommended_figure, args.output_dir)
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_outputs"] = recommended_outputs
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "comparison": comparison_outputs,
                "recommended": recommended_outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
