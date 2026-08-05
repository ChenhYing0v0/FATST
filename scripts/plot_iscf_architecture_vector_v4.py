#!/usr/bin/env python3
"""Draw the deterministic ISCF architecture v4 vector concept.

The figure is an architecture schematic. All curves, matrices, coordinates and
probabilities are deterministic visual glyphs and do not represent observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 112.0
FIGURE_ID = "figure_iscf_architecture_vector_v4"

COLORS = {
    "ink": "#272727",
    "neutral": "#767676",
    "neutral_soft": "#D8D8D8",
    "indigo": "#484878",
    "periwinkle": "#7884B4",
    "periwinkle_soft": "#B4C0E4",
    "aqua": "#77D7D1",
    "teal": "#33B5A5",
    "teal_dark": "#176D73",
    "lilac": "#B9A7E8",
    "violet": "#7C6CCF",
    "ochre": "#C88A23",
    "peach": "#F0E0D0",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    """Apply the saved Python publication backend style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def mix(color_a: str, color_b: str, weight: float) -> tuple[float, float, float]:
    """Linearly mix two colors."""
    a = np.asarray(to_rgb(color_a))
    b = np.asarray(to_rgb(color_b))
    return tuple((1.0 - weight) * a + weight * b)


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral"],
    linewidth: float = 0.9,
    mutation_scale: float = 6.5,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 4,
) -> None:
    """Draw one consistent directed connector."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_polyline_arrow(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    linewidth: float = 0.9,
) -> None:
    """Draw an orthogonal polyline whose last segment has an arrowhead."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=2,
        )
    draw_arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        linewidth=linewidth,
    )


def draw_module_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    edgecolor: str,
    facecolor: str,
    fontsize: float = 7.0,
) -> None:
    """Draw one of the two permitted named module boxes."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.18,rounding_size=0.8",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight="medium",
        zorder=5,
    )


def multiscale_wave(x: np.ndarray, phase: float, kind: int) -> np.ndarray:
    """Return a deterministic multiscale schematic waveform."""
    base = 0.55 * np.sin(1.55 * x + phase)
    medium = 0.22 * np.sin((3.2 + 0.25 * kind) * x - 0.5 * phase)
    fine = 0.08 * np.sin((8.6 - 0.35 * kind) * x + 0.7 * phase)
    trend = 0.12 * np.cos(0.62 * x + 0.4 * kind)
    return base + medium + fine + trend


def draw_waveform(
    ax: Axes,
    x0: float,
    x1: float,
    center_y: float,
    *,
    phase: float,
    kind: int,
    color: str,
    scale: float,
    final: bool = False,
) -> None:
    """Draw a line-only multiscale waveform glyph without uncertainty bands."""
    t = np.linspace(0, 2 * np.pi, 220)
    x = np.linspace(x0, x1, t.size)
    y = center_y + scale * multiscale_wave(t, phase, kind)
    echo = y + 0.13 * np.sin(6.1 * t + phase) + 0.14
    detail = y - 0.11 * np.sin(9.3 * t - phase) - 0.13
    ax.plot(x, echo, color=mix(color, COLORS["white"], 0.58), linewidth=0.55, zorder=3)
    ax.plot(x, detail, color=mix(color, COLORS["white"], 0.72), linewidth=0.48, zorder=3)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.65 if final else 1.15,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [x0, x1 + 0.65],
        [center_y - 1.9 * scale, center_y - 1.9 * scale],
        color=COLORS["neutral_soft"],
        linewidth=0.55,
        zorder=1,
    )
    draw_arrow(
        ax,
        (x1 + 0.25, center_y - 1.9 * scale),
        (x1 + 0.65, center_y - 1.9 * scale),
        color=COLORS["neutral"],
        linewidth=0.55,
        mutation_scale=4.5,
        zorder=2,
    )


def draw_history(ax: Axes) -> None:
    """Draw three aligned history-variable waveforms."""
    ax.text(
        9.7,
        70.0,
        "multivariate history",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=COLORS["indigo"],
        fontweight="bold",
    )
    for index, center in enumerate((62.2, 57.2, 52.2)):
        draw_waveform(
            ax,
            1.5,
            14.4,
            center,
            phase=0.45 * index,
            kind=index,
            color=[COLORS["indigo"], COLORS["periwinkle"], COLORS["periwinkle_soft"]][index],
            scale=1.45,
        )
    ax.text(1.5, 48.7, "$1$", fontsize=5.4, color=COLORS["neutral"], ha="left")
    ax.text(15.1, 48.7, "$L$", fontsize=5.4, color=COLORS["neutral"], ha="right")


def draw_history_state(ax: Axes) -> None:
    """Draw the variable-wise history state as grouped feature bands."""
    x0, y0, width, height = 26.2, 49.6, 2.2, 15.0
    values = np.array([0.25, 0.42, 0.72, 0.50, 0.85, 0.38, 0.67, 0.31, 0.78, 0.56])
    cell_h = height / values.size
    for idx, value in enumerate(values):
        color = mix(COLORS["white"], COLORS["indigo"], 0.20 + 0.72 * value)
        ax.add_patch(
            Rectangle(
                (x0, y0 + idx * cell_h),
                width,
                cell_h - 0.08,
                facecolor=color,
                edgecolor=COLORS["white"],
                linewidth=0.28,
                zorder=3,
            )
        )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=COLORS["indigo"],
            linewidth=0.9,
            zorder=4,
        )
    )
    ax.text(
        x0 + width / 2,
        66.1,
        "history state",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=COLORS["indigo"],
        fontweight="bold",
    )
    ax.text(
        x0 + width / 2,
        48.3,
        "$\\mathbf{r}_{b,c}\\in\\mathbb{R}^{R}$",
        ha="center",
        va="top",
        fontsize=5.5,
        color=COLORS["neutral"],
    )


def coordinate_field() -> np.ndarray:
    """Construct the exact displayed centered DCT coordinate family."""
    total_steps = 64
    dimensions = 10
    tau = np.arange(1, total_steps + 1, dtype=float)
    field = np.ones((dimensions, total_steps), dtype=float)
    for dim in range(1, dimensions):
        raw = np.cos(np.pi * (tau - 0.5) * dim / total_steps)
        field[dim] = np.sqrt(2.0) * (raw - raw.mean())
    row_scale = np.maximum(np.max(np.abs(field), axis=1, keepdims=True), 1e-8)
    return field / row_scale


def draw_coordinate_atlas(ax: Axes) -> None:
    """Draw the DCT coordinate atlas and a pulled-out target coordinate."""
    x0, x1 = 2.0, 22.7
    y0, y1 = 9.0, 26.0
    field = coordinate_field()
    cmap = LinearSegmentedColormap.from_list(
        "coordinate_family",
        [mix(COLORS["ochre"], COLORS["white"], 0.20), COLORS["white"], COLORS["ochre"]],
    )
    ax.imshow(
        field,
        extent=(x0, x1, y0, y1),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=1,
    )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["ochre"],
            linewidth=0.85,
            zorder=4,
        )
    )
    target_index = 43
    target_x = x0 + (target_index + 0.5) * (x1 - x0) / field.shape[1]
    target_w = (x1 - x0) / field.shape[1]
    ax.add_patch(
        Rectangle(
            (target_x - target_w / 2, y0),
            target_w,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["violet"],
            linewidth=1.15,
            zorder=5,
        )
    )

    vector_x0, vector_x1 = 25.0, 27.2
    vector = field[:, target_index]
    cell_h = (y1 - y0) / vector.size
    for idx, value in enumerate(vector):
        color = cmap((value + 1.0) / 2.0)
        ax.add_patch(
            Rectangle(
                (vector_x0, y0 + idx * cell_h),
                vector_x1 - vector_x0,
                cell_h - 0.08,
                facecolor=color,
                edgecolor=COLORS["white"],
                linewidth=0.25,
                zorder=3,
            )
        )
    ax.add_patch(
        Rectangle(
            (vector_x0, y0),
            vector_x1 - vector_x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["ochre"],
            linewidth=0.85,
            zorder=4,
        )
    )
    draw_arrow(
        ax,
        (target_x + 0.25, y1 + 0.35),
        (vector_x0 - 0.25, y1 + 0.35),
        color=COLORS["ochre"],
        linewidth=0.8,
        mutation_scale=5.0,
    )
    ax.text(
        14.5,
        30.0,
        "future-step coordinate field",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=COLORS["ochre"],
        fontweight="bold",
    )
    ax.text(
        14.5,
        28.6,
        "$\\boldsymbol{\\Phi}=[\\boldsymbol{\\phi}_1,\\ldots,\\boldsymbol{\\phi}_T]^\\top\\in\\mathbb{R}^{T\\times D_q}$",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color=COLORS["neutral"],
    )
    ax.text(
        x0 + 0.35,
        y0 + 0.75,
        "$d=0$",
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["white"], "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
        zorder=6,
    )
    ax.text(
        x0 + 0.35,
        y1 - 0.75,
        "$d=D_q-1$",
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["white"], "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
        zorder=6,
    )
    ax.text(x0, y0 - 1.15, "$\\tau=1$", fontsize=5.2, color=COLORS["neutral"], ha="left")
    ax.text(target_x, y0 - 1.15, "$\\tau$", fontsize=5.2, color=COLORS["violet"], ha="center")
    ax.text(x1, y0 - 1.15, "$T$", fontsize=5.2, color=COLORS["neutral"], ha="right")
    ax.text(
        (vector_x0 + vector_x1) / 2,
        y0 - 1.15,
        "$\\boldsymbol{\\phi}_{\\tau}$",
        fontsize=5.5,
        color=COLORS["ochre"],
        ha="center",
    )
    ax.text(
        12.3,
        6.1,
        "coordinate channels × future steps",
        fontsize=5.2,
        color=COLORS["neutral"],
        ha="center",
    )


def stripe_matrix(row: int, rows: int = 9, cols: int = 7) -> np.ndarray:
    """Create a deterministic low-rank matrix glyph."""
    r = np.linspace(-1.0, 1.0, rows)[:, None]
    c = np.linspace(-1.0, 1.0, cols)[None, :]
    value = 0.55 * np.cos((row + 1.2) * np.pi * r) + 0.45 * np.sin((row + 1.8) * np.pi * c)
    value += 0.24 * np.outer(np.sin(np.pi * np.linspace(0, 1, rows)), np.cos(np.pi * np.linspace(0, 1, cols)))
    value -= value.min()
    return value / max(value.max(), 1e-8)


def draw_scope_matrix(ax: Axes, x0: float, y: float, row: int) -> None:
    """Draw one scope-conditioned D_q by K matrix."""
    width, height = 6.2, 5.8
    cmap = LinearSegmentedColormap.from_list(
        "scope_matrix",
        [COLORS["white"], COLORS["periwinkle_soft"], COLORS["indigo"]],
    )
    ax.imshow(
        stripe_matrix(row),
        extent=(x0, x0 + width, y - height / 2, y + height / 2),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (x0, y - height / 2),
            width,
            height,
            fill=False,
            edgecolor=COLORS["indigo"],
            linewidth=0.75,
            zorder=4,
        )
    )
    for fraction in (0.30, 0.68):
        ax.plot(
            [x0 + fraction * width, x0 + fraction * width],
            [y - height / 2, y + height / 2],
            color=mix(COLORS["indigo"], COLORS["white"], 0.40),
            linewidth=0.55,
            zorder=4,
        )


def draw_region_descriptor(
    ax: Axes,
    x0: float,
    y: float,
    row: int,
) -> None:
    """Draw scope-dependent contiguous coordinate summaries."""
    width, height = 11.0, 3.0
    segment_counts = (12, 5, 2)
    count = segment_counts[row]
    segment_width = width / count
    for segment in range(count):
        intensity = 0.20 + 0.55 * (0.5 + 0.5 * np.cos((segment + 1) * (row + 1) * 0.67))
        face = mix(COLORS["white"], COLORS["peach"], intensity)
        ax.add_patch(
            Rectangle(
                (x0 + segment * segment_width, y - height / 2),
                segment_width - 0.08,
                height,
                facecolor=face,
                edgecolor=COLORS["ochre"],
                linewidth=0.48,
                zorder=2,
            )
        )
        if segment_width > 1.2:
            for offset in (-0.45, 0.0, 0.45):
                length = 0.34 + 0.15 * np.sin(segment + offset + row)
                ax.plot(
                    [x0 + segment * segment_width + 0.22, x0 + segment * segment_width + segment_width * (0.62 + length / 3)],
                    [y + offset, y + offset],
                    color=mix(COLORS["ochre"], COLORS["white"], 0.32),
                    linewidth=0.45,
                    zorder=4,
                )


def draw_region_representation(
    ax: Axes,
    x0: float,
    y: float,
    row: int,
) -> None:
    """Draw the region-indexed latent states as structured vector tiles."""
    width, height = 11.0, 3.8
    segment_counts = (12, 5, 2)
    count = segment_counts[row]
    segment_width = width / count
    for segment in range(count):
        value = 0.35 + 0.45 * (0.5 + 0.5 * np.sin(0.73 * segment + 0.8 * row))
        face = mix(COLORS["white"], COLORS["aqua"], value)
        ax.add_patch(
            Rectangle(
                (x0 + segment * segment_width, y - height / 2),
                segment_width - 0.08,
                height,
                facecolor=face,
                edgecolor=COLORS["teal_dark"],
                linewidth=0.48,
                zorder=2,
            )
        )
        if segment_width > 1.15:
            for mode in range(3):
                y_line = y - 0.78 + mode * 0.78
                x_start = x0 + segment * segment_width + 0.24
                x_end = x0 + segment * segment_width + segment_width - 0.28
                ax.plot(
                    [x_start, x_end],
                    [y_line, y_line + 0.11 * np.sin(segment + mode)],
                    color=mix(COLORS["teal_dark"], COLORS["white"], 0.32 + 0.15 * mode),
                    linewidth=0.46,
                    zorder=4,
                )


def draw_synthesis_bank(ax: Axes, x0: float, y: float) -> None:
    """Draw a shared-looking step-specific synthesis basis bank."""
    width, height = 10.2, 2.9
    count = 34
    for index in range(count):
        fraction = index / (count - 1)
        if fraction < 0.55:
            color = mix(COLORS["indigo"], COLORS["periwinkle_soft"], fraction / 0.55)
        else:
            color = mix(COLORS["periwinkle_soft"], COLORS["aqua"], (fraction - 0.55) / 0.45)
        bar_height = height * (0.64 + 0.28 * (0.5 + 0.5 * np.sin(0.75 * index)))
        x = x0 + index * width / count
        ax.add_patch(
            Rectangle(
                (x, y - bar_height / 2),
                width / count * 0.62,
                bar_height,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )


def draw_scope_forecast(ax: Axes, y: float, row: int) -> None:
    """Draw one scope slice of the common forecast field."""
    draw_waveform(
        ax,
        91.5,
        104.0,
        y,
        phase=0.75 + 0.65 * row,
        kind=row + 1,
        color=[COLORS["teal"], COLORS["teal_dark"], COLORS["indigo"]][row],
        scale=1.45,
    )


def allocation_field() -> np.ndarray:
    """Create a normalized three-scope schematic allocation field."""
    tau = np.linspace(0.0, 1.0, 42)
    logits = np.stack(
        [
            0.9 * np.cos(2.0 * np.pi * tau) + 0.25 * np.sin(6.0 * np.pi * tau),
            0.55 * np.sin(2.0 * np.pi * tau + 0.8) + 0.45 * np.cos(4.0 * np.pi * tau),
            -0.45 * np.cos(2.0 * np.pi * tau - 0.4) + 0.30 * np.sin(5.0 * np.pi * tau),
        ]
    )
    logits -= logits.max(axis=0, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=0, keepdims=True)


def draw_condition_tensor(ax: Axes) -> None:
    """Draw concatenated history and coordinate features."""
    x0, x1 = 43.8, 54.8
    y0, y1 = 13.2, 24.8
    bands = 24
    for index in range(bands):
        if index < 13:
            base = COLORS["periwinkle"]
            value = 0.30 + 0.55 * (0.5 + 0.5 * np.sin(index * 0.83))
        else:
            base = COLORS["ochre"]
            value = 0.28 + 0.58 * (0.5 + 0.5 * np.cos(index * 0.91))
        face = mix(COLORS["white"], base, value)
        width = (x1 - x0) / bands
        ax.add_patch(
            Rectangle(
                (x0 + index * width, y0),
                width - 0.03,
                y1 - y0,
                facecolor=face,
                edgecolor=COLORS["white"],
                linewidth=0.2,
                zorder=2,
            )
        )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["neutral"],
            linewidth=0.75,
            zorder=4,
        )
    )
    ax.plot([49.8, 49.8], [y0, y1], color=COLORS["white"], linewidth=1.15, zorder=4)
    ax.text(
        (x0 + x1) / 2,
        26.6,
        "condition vector",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        (x0 + x1) / 2,
        11.7,
        "$[\\mathbf{u}_{b,c};\\boldsymbol{\\phi}_{\\tau}]$",
        ha="center",
        fontsize=5.6,
        color=COLORS["neutral"],
    )


def draw_probability_field(ax: Axes) -> None:
    """Draw the target-wise three-scope probability field."""
    probs = allocation_field()
    x0, x1 = 72.0, 90.0
    y0, y1 = 13.2, 24.8
    cmap = LinearSegmentedColormap.from_list(
        "allocation_family",
        [COLORS["white"], COLORS["lilac"], COLORS["violet"]],
    )
    ax.imshow(
        probs,
        extent=(x0, x1, y0, y1),
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=0.75,
        interpolation="bilinear",
        zorder=1,
    )
    for boundary in (y0 + (y1 - y0) / 3, y0 + 2 * (y1 - y0) / 3):
        ax.plot([x0, x1], [boundary, boundary], color=COLORS["white"], linewidth=0.7, zorder=4)
    target_index = 29
    target_x = x0 + (target_index + 0.5) * (x1 - x0) / probs.shape[1]
    target_width = (x1 - x0) / probs.shape[1]
    ax.add_patch(
        Rectangle(
            (target_x - target_width / 2, y0),
            target_width,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["violet"],
            linewidth=0.75,
            zorder=4,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        26.6,
        "scope-probability field",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COLORS["violet"],
        fontweight="bold",
    )
    labels = ("$s_0$", "$s_1$", "$s_2$")
    row_height = (y1 - y0) / 3
    for idx, label in enumerate(labels):
        row_y = y1 - (idx + 0.5) * row_height
        ax.text(x0 - 0.9, row_y, label, ha="right", va="center", fontsize=5.8, color=COLORS["neutral"])
    ax.text(
        (x0 + x1) / 2,
        11.7,
        "$\\boldsymbol{\\Pi}\\in\\mathbb{R}^{T\\times S}$",
        ha="center",
        fontsize=5.5,
        color=COLORS["neutral"],
    )

    selected = probs[:, target_index]
    vector_x = 94.2
    vector_ys = (22.7, 19.0, 15.3)
    for value, y in zip(selected, vector_ys):
        radius = 0.50 + 1.05 * value
        ax.add_patch(
            Circle(
                (vector_x, y),
                radius,
                facecolor=mix(COLORS["white"], COLORS["violet"], 0.40 + 0.55 * value),
                edgecolor=COLORS["violet"],
                linewidth=0.55,
                zorder=4,
            )
        )
    ax.text(vector_x, 26.6, "$\\boldsymbol{\\pi}_{\\tau}$", ha="center", fontsize=6.0, color=COLORS["violet"])
    draw_arrow(ax, (90.6, 19.0), (92.7, 19.0), color=COLORS["neutral"], linewidth=0.8)


def draw_fusion_and_output(ax: Axes) -> None:
    """Draw the only merge point and final trajectory."""
    collector_x = 107.5
    row_centers = (59.5, 51.5, 43.5)
    for y in row_centers:
        ax.plot([104.8, collector_x], [y, y], color=COLORS["neutral"], linewidth=0.75, zorder=2)
    ax.plot([collector_x, collector_x], [43.5, 59.5], color=COLORS["neutral"], linewidth=0.75, zorder=2)
    draw_polyline_arrow(
        ax,
        [(collector_x, 43.5), (collector_x, 31.0), (110.5, 31.0), (110.5, 22.6)],
        color=COLORS["neutral"],
        linewidth=0.8,
    )
    draw_arrow(ax, (96.0, 19.0), (106.5, 19.0), color=COLORS["violet"], linewidth=0.85)

    center = (110.5, 19.0)
    ax.add_patch(
        Circle(
            center,
            3.5,
            facecolor=mix(COLORS["white"], COLORS["aqua"], 0.18),
            edgecolor=COLORS["teal_dark"],
            linewidth=1.15,
            zorder=4,
        )
    )
    ax.text(
        center[0],
        center[1] + 0.15,
        "$\\sum_s$",
        ha="center",
        va="center",
        fontsize=10.0,
        color=COLORS["teal_dark"],
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        center[0],
        13.4,
        "target-wise\nweighted contraction",
        ha="center",
        va="top",
        fontsize=5.8,
        color=COLORS["neutral"],
        linespacing=1.05,
    )
    draw_arrow(ax, (114.2, 19.0), (117.0, 19.0), color=COLORS["teal_dark"], linewidth=1.0)
    draw_waveform(
        ax,
        117.5,
        128.7,
        19.4,
        phase=1.15,
        kind=4,
        color=COLORS["teal_dark"],
        scale=2.35,
        final=True,
    )
    ax.text(
        123.0,
        27.0,
        "one predicted\ntrajectory",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COLORS["teal_dark"],
        fontweight="bold",
    )
    ax.text(
        123.0,
        11.7,
        "$\\widehat{\\mathbf{Y}}_{1:T}$",
        ha="center",
        fontsize=6.3,
        color=COLORS["teal_dark"],
        fontweight="bold",
    )


def draw_main_paths(ax: Axes) -> None:
    """Assemble the complete two-lane ISCF information flow."""
    draw_history(ax)
    draw_module_box(
        ax,
        (17.7, 54.0),
        6.4,
        7.2,
        "Encoder",
        edgecolor=COLORS["indigo"],
        facecolor=mix(COLORS["white"], COLORS["periwinkle_soft"], 0.22),
        fontsize=7.3,
    )
    draw_arrow(ax, (14.8, 57.6), (17.4, 57.6), color=COLORS["indigo"], linewidth=0.95)
    draw_arrow(ax, (24.3, 57.6), (25.8, 57.6), color=COLORS["indigo"], linewidth=0.95)
    draw_history_state(ax)
    draw_coordinate_atlas(ax)

    row_centers = (59.5, 51.5, 43.5)
    scope_labels = ("$s_0$", "$s_1$", "$s_2$")
    ax.text(
        71.5,
        71.4,
        "Scope-conditioned forecasting",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color=COLORS["ink"],
        fontweight="bold",
    )
    headings = (
        (37.3, "scope\nmatrix"),
        (51.3, "region\ndescriptor"),
        (66.5, "region\nrepresentation"),
        (82.0, "step-specific\nsynthesis"),
        (97.8, "scope-wise\nforecasts"),
    )
    for x, label in headings:
        ax.text(
            x,
            66.3,
            label,
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=COLORS["ink"],
            fontweight="bold",
            linespacing=0.92,
        )
    ax.text(
        30.9,
        63.4,
        "scope\nindex",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["neutral"],
        linespacing=1.0,
    )
    ax.plot([29.7, 29.7], [42.5, 60.5], color=COLORS["periwinkle"], linewidth=0.75, zorder=2)
    ax.plot([29.7, 31.0], [60.5, 60.5], color=COLORS["periwinkle"], linewidth=0.75, zorder=2)
    ax.plot([29.7, 31.0], [42.5, 42.5], color=COLORS["periwinkle"], linewidth=0.75, zorder=2)

    for row, (y, scope_label) in enumerate(zip(row_centers, scope_labels)):
        ax.text(32.2, y, scope_label, ha="right", va="center", fontsize=6.4, color=COLORS["indigo"], fontweight="bold")
        draw_scope_matrix(ax, 34.2, y, row)
        ax.text(43.0, y, "$\\times$", ha="center", va="center", fontsize=9.0, color=COLORS["ink"])
        draw_region_descriptor(ax, 46.0, y, row)
        ax.text(58.5, y, "$=$", ha="center", va="center", fontsize=8.0, color=COLORS["neutral"])
        draw_region_representation(ax, 61.0, y, row)
        draw_synthesis_bank(ax, 77.0, y)
        draw_scope_forecast(ax, y, row)
        draw_arrow(ax, (40.7, y), (41.9, y), color=COLORS["neutral"], linewidth=0.75)
        draw_arrow(ax, (44.0, y), (45.2, y), color=COLORS["neutral"], linewidth=0.75)
        draw_arrow(ax, (72.2, y), (75.8, y), color=COLORS["teal_dark"], linewidth=0.75)
        draw_arrow(ax, (87.4, y), (90.3, y), color=COLORS["teal_dark"], linewidth=0.75)

    draw_arrow(ax, (28.7, 57.6), (33.3, 57.6), color=COLORS["indigo"], linewidth=0.9)
    ax.plot([32.9, 32.9], [43.5, 59.5], color=COLORS["indigo"], linewidth=0.75, zorder=2)
    for y in row_centers:
        draw_arrow(ax, (32.9, y), (33.7, y), color=COLORS["indigo"], linewidth=0.75, mutation_scale=5.0)

    coordinate_branch_x = 44.7
    ax.plot([27.6, coordinate_branch_x], [21.8, 21.8], color=COLORS["ochre"], linewidth=0.85, zorder=2)
    ax.plot([coordinate_branch_x, coordinate_branch_x], [21.8, 59.5], color=COLORS["ochre"], linewidth=0.85, zorder=2)
    for y in row_centers:
        draw_arrow(ax, (coordinate_branch_x, y), (45.3, y), color=COLORS["ochre"], linewidth=0.75, mutation_scale=5.0)

    ax.plot([31.5, 107.5], [34.6, 34.6], color=mix(COLORS["neutral_soft"], COLORS["white"], 0.25), linewidth=0.7, zorder=1)
    ax.text(
        72.0,
        31.4,
        "Target-conditioned scope allocation",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=COLORS["violet"],
        fontweight="bold",
    )
    draw_condition_tensor(ax)
    draw_module_box(
        ax,
        (58.0, 15.1),
        8.2,
        7.7,
        "Allocation\nMLP",
        edgecolor=COLORS["violet"],
        facecolor=mix(COLORS["white"], COLORS["lilac"], 0.23),
        fontsize=6.8,
    )
    draw_probability_field(ax)
    draw_arrow(ax, (55.2, 19.0), (57.4, 19.0), color=COLORS["neutral"], linewidth=0.8)
    draw_arrow(ax, (66.5, 19.0), (71.2, 19.0), color=COLORS["violet"], linewidth=0.85)

    draw_polyline_arrow(
        ax,
        [(27.4, 49.0), (27.4, 28.4), (42.0, 28.4), (42.0, 22.0), (43.2, 22.0)],
        color=COLORS["indigo"],
        linewidth=0.85,
    )
    draw_polyline_arrow(
        ax,
        [(27.4, 13.8), (39.7, 13.8), (39.7, 16.0), (43.2, 16.0)],
        color=COLORS["ochre"],
        linewidth=0.85,
    )
    ax.text(39.6, 23.1, "history", fontsize=5.0, color=COLORS["indigo"], ha="right")
    ax.text(39.6, 15.0, "coordinate", fontsize=5.0, color=COLORS["ochre"], ha="right")

    draw_fusion_and_output(ax)
    ax.text(
        129.0,
        3.0,
        "schematic tensors and trajectories; no empirical values",
        ha="right",
        va="center",
        fontsize=5.0,
        color=mix(COLORS["neutral"], COLORS["white"], 0.25),
    )


def build_figure() -> Figure:
    """Build the exact-size publication canvas."""
    configure_style()
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, 130.0)
    axis.set_ylim(0.0, 74.0)
    axis.set_axis_off()
    draw_main_paths(axis)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_bsca_method_figure_vector_redesign_20260805"),
    )
    return parser.parse_args()


def save_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Save vector and raster review formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / FIGURE_ID
    outputs = {
        "svg": f"{prefix}.svg",
        "pdf": f"{prefix}.pdf",
        "png": f"{prefix}.png",
        "tiff": f"{prefix}.tiff",
    }
    figure.savefig(outputs["svg"], format="svg")
    svg_path = Path(outputs["svg"])
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
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
        "status": "vector_draft_for_author_review",
        "backend": "Python/matplotlib",
        "final_width_mm": FIGURE_WIDTH_MM,
        "final_height_mm": FIGURE_HEIGHT_MM,
        "data_role": "architecture schematic",
        "empirical_data_used": False,
        "manuscript_replacement": False,
        "representative_scope_rows": ["s_0", "s_1", "s_2"],
        "coordinate_visualization": "DCT basis atlas plus highlighted target coordinate vector",
        "named_module_boxes": ["Encoder", "Allocation MLP"],
        "claim_boundary": [
            "explains ISCF inference architecture only",
            "omits single-scope comparison and BSCA",
            "scope rows are representative rather than exact implementation count",
            "uses schematic tensors, probabilities and trajectories",
            "does not establish effectiveness or specialization",
        ],
        "outputs": outputs,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plt.close(figure)
    return outputs


def main() -> None:
    """Render the figure and print its manifest."""
    args = parse_args()
    figure = build_figure()
    outputs = save_bundle(figure, args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
