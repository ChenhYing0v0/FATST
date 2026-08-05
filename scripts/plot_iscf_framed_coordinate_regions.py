#!/usr/bin/env python3
"""Render a framed history and region-averaged future-coordinate design study.

All coordinate curves and cell values are analytic consequences of the fixed
DCT coordinate definition. History curves are deterministic schematic signals.
No empirical sample, learned activation, or performance result is displayed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle


COMBINED_ID = "figure_iscf_unified_framed_fields_v1"
COORDINATE_ID = "figure_iscf_framed_coordinate_region_v1"
COMBINED_WIDTH_MM = 183.0
COMBINED_HEIGHT_MM = 83.0
COORDINATE_WIDTH_MM = 120.0
COORDINATE_HEIGHT_MM = 62.0

COLORS = {
    "ink": "#25282D",
    "neutral": "#737A82",
    "hairline": "#D5DBE1",
    "white": "#FFFFFF",
    "history_fill": "#F2F5F7",
    "history_curve": "#4C5360",
    "frame": "#31495A",
    "negative": "#8196C5",
    "zero": "#F7F7F3",
    "positive": "#D89B78",
    "selection": "#B97831",
    "selection_soft": "#F4E7D8",
    "violet": "#7563D4",
}


def configure_style() -> None:
    """Configure a compact publication-oriented visual style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def coordinate_field(
    total_steps: int = 720,
    dimensions: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the exact displayed centered and scaled DCT coordinates."""
    tau = np.arange(1, total_steps + 1, dtype=float)
    field = np.ones((dimensions, total_steps), dtype=float)
    for dim in range(1, dimensions):
        raw = np.cos(np.pi * (tau - 0.5) * dim / total_steps)
        field[dim] = np.sqrt(2.0) * (raw - raw.mean())
    return tau, field


def region_means(field: np.ndarray, regions: int = 5) -> np.ndarray:
    """Average every coordinate channel over equal contiguous regions."""
    indices = np.array_split(np.arange(field.shape[1]), regions)
    return np.stack([field[:, index].mean(axis=1) for index in indices], axis=1)


def value_cmap() -> LinearSegmentedColormap:
    """Return a restrained signed-value color map."""
    return LinearSegmentedColormap.from_list(
        "coordinate_region_mean",
        [COLORS["negative"], COLORS["zero"], COLORS["positive"]],
    )


def schematic_history(total_steps: int = 240) -> np.ndarray:
    """Create three deterministic, data-free waveform glyphs."""
    t = np.linspace(0.0, 1.0, total_steps)
    first = (
        0.36 * np.sin(2.0 * np.pi * 1.35 * t + 0.30)
        + 0.16 * np.sin(2.0 * np.pi * 5.7 * t + 0.80)
        + 0.08 * np.sin(2.0 * np.pi * 13.0 * t)
        + 0.40 * np.exp(-((t - 0.86) / 0.055) ** 2)
    )
    second = (
        0.60 * (t - 0.5)
        + 0.18 * np.sin(2.0 * np.pi * 1.15 * t - 0.65)
        + 0.10 * np.sin(2.0 * np.pi * 7.0 * t + 0.40)
        + 0.26 * np.tanh(13.0 * (t - 0.38))
    )
    third = (
        0.30 * np.sin(2.0 * np.pi * 1.8 * t - 0.50)
        + 0.17 * np.sin(2.0 * np.pi * 4.2 * t + 0.75)
        + 0.10 * np.sin(2.0 * np.pi * 9.5 * t)
        + 0.36 * np.exp(-((t - 0.90) / 0.045) ** 2)
    )
    waves = np.stack([first, second, third])
    waves -= waves.mean(axis=1, keepdims=True)
    waves /= np.maximum(np.abs(waves).max(axis=1, keepdims=True), 1e-8)
    return waves


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral"],
    linewidth: float = 0.8,
    mutation_scale: float = 6.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 8,
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


def draw_history_frame(
    ax: Axes,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    title_y: float,
) -> None:
    """Draw a framed, channel-wise history glyph."""
    waves = schematic_history()
    rows = waves.shape[0]
    row_height = (y1 - y0) / rows
    curve_x = np.linspace(x0 + 2.0, x1 - 2.0, waves.shape[1])

    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=COLORS["history_fill"],
            edgecolor=COLORS["frame"],
            linewidth=1.2,
            zorder=1,
        )
    )
    for row in range(1, rows):
        y = y0 + row * row_height
        ax.plot([x0, x1], [y, y], color=COLORS["frame"], linewidth=0.8, zorder=4)
    for row, wave in enumerate(waves):
        center = y1 - (row + 0.5) * row_height
        curve_y = center + 0.28 * row_height * wave
        ax.plot(
            curve_x,
            curve_y,
            color=COLORS["history_curve"],
            linewidth=1.35,
            solid_capstyle="round",
            zorder=5,
        )

    ax.text(
        (x0 + x1) / 2,
        title_y,
        r"observed history  $\mathbf{X}$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        x0 - 4.0,
        (y0 + y1) / 2,
        "channels",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        color=COLORS["neutral"],
    )
    ax.text(
        (x0 + x1) / 2,
        y0 - 3.2,
        "past time",
        ha="center",
        va="center",
        fontsize=5.7,
        color=COLORS["neutral"],
    )
    ax.text(
        x0,
        y0 - 3.2,
        "$1$",
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )
    ax.text(
        x1,
        y0 - 3.2,
        "$L$",
        ha="right",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )


def draw_descriptor(
    ax: Axes,
    *,
    x: float,
    y_centers: np.ndarray,
    values: np.ndarray,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    square_size: float,
    label_y: float,
) -> None:
    """Draw the extracted four-dimensional region descriptor."""
    for value, center in zip(values, y_centers):
        ax.add_patch(
            Rectangle(
                (x - square_size / 2, center - square_size / 2),
                square_size,
                square_size,
                facecolor=cmap(norm(float(value))),
                edgecolor=COLORS["frame"],
                linewidth=0.8,
                zorder=5,
            )
        )
    ax.text(
        x,
        label_y,
        r"$\overline{\boldsymbol{\phi}}_{g^\star}^{(s)}$",
        ha="center",
        va="top",
        fontsize=6.6,
        color=COLORS["selection"],
        fontweight="bold",
    )
    ax.text(
        x,
        label_y - 3.3,
        "region descriptor",
        ha="center",
        va="top",
        fontsize=5.0,
        color=COLORS["neutral"],
    )


def draw_signed_legend(
    ax: Axes,
    *,
    x0: float,
    y: float,
    cmap: LinearSegmentedColormap,
) -> None:
    """Draw a compact signed-value legend."""
    values = (-1.0, 0.0, 1.0)
    labels = ("negative", "zero", "positive")
    for index, (value, label) in enumerate(zip(values, labels)):
        x = x0 + 9.0 * index
        ax.add_patch(
            Rectangle(
                (x, y),
                3.6,
                3.0,
                facecolor=cmap((value + 1.0) / 2.0),
                edgecolor=COLORS["hairline"],
                linewidth=0.45,
            )
        )
        ax.text(
            x + 1.8,
            y - 1.1,
            label,
            ha="center",
            va="top",
            fontsize=5.0,
            color=COLORS["neutral"],
        )


def draw_coordinate_frame(
    ax: Axes,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    descriptor_x: float,
    title_y: float,
    show_formula: bool,
) -> None:
    """Draw the framed DCT field, region means and one extracted descriptor."""
    tau, field = coordinate_field()
    regions = 5
    means = region_means(field, regions=regions)
    max_abs = float(np.max(np.abs(means)))
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    cmap = value_cmap()
    rows = field.shape[0]
    row_height = (y1 - y0) / rows
    region_width = (x1 - x0) / regions
    selected_region = 2

    for dim in range(rows):
        for region in range(regions):
            cell_y = y1 - (dim + 1) * row_height
            cell_x = x0 + region * region_width
            ax.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    region_width,
                    row_height,
                    facecolor=cmap(norm(float(means[dim, region]))),
                    edgecolor="none",
                    alpha=0.66,
                    zorder=1,
                )
            )

    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["frame"],
            linewidth=1.25,
            zorder=5,
        )
    )
    for row in range(1, rows):
        y = y0 + row * row_height
        ax.plot([x0, x1], [y, y], color=COLORS["frame"], linewidth=0.75, zorder=5)
    for region in range(1, regions):
        x = x0 + region * region_width
        ax.plot([x, x], [y0, y1], color=COLORS["frame"], linewidth=0.75, zorder=5)

    mapped_x = x0 + (tau - 1.0) / (tau[-1] - 1.0) * (x1 - x0)
    curve_color = COLORS["history_curve"]
    for dim in range(rows):
        center = y1 - (dim + 0.5) * row_height
        scale = 0.27 * row_height / max(float(np.max(np.abs(field[dim]))), 1e-8)
        curve_y = center + scale * field[dim]
        ax.plot(
            mapped_x,
            curve_y,
            color=curve_color,
            linewidth=1.15,
            solid_capstyle="round",
            zorder=6,
        )
        ax.text(
            x0 - 1.6,
            center,
            f"$d={dim}$",
            ha="right",
            va="center",
            fontsize=5.2,
            color=COLORS["neutral"],
        )

    selected_x = x0 + selected_region * region_width
    ax.add_patch(
        Rectangle(
            (selected_x, y0),
            region_width,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["selection"],
            linewidth=1.55,
            zorder=7,
        )
    )
    ax.text(
        selected_x + region_width / 2,
        y1 + 2.0,
        r"$g^\star$",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["selection"],
        fontweight="bold",
    )

    row_centers = y1 - (np.arange(rows) + 0.5) * row_height
    descriptor_values = means[:, selected_region]
    draw_descriptor(
        ax,
        x=descriptor_x,
        y_centers=row_centers,
        values=descriptor_values,
        cmap=cmap,
        norm=norm,
        square_size=0.48 * row_height,
        label_y=y0 - 2.2,
    )
    arrow(
        ax,
        (selected_x + region_width / 2, y1 + 4.2),
        (descriptor_x, y1 + 4.2),
        color=COLORS["selection"],
        linewidth=0.85,
        connectionstyle="angle3,angleA=0,angleB=-90",
    )
    ax.plot(
        [descriptor_x, descriptor_x],
        [y1 + 4.2, row_centers[0] + 0.35 * row_height],
        color=COLORS["selection"],
        linewidth=0.75,
        zorder=7,
    )

    ax.text(
        (x0 + x1) / 2,
        title_y,
        r"fixed future-coordinate field  $\boldsymbol{\Phi}^{\mathsf{T}}$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        x0 - 10.0,
        (y0 + y1) / 2,
        "basis channel",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        color=COLORS["neutral"],
    )
    ax.text(
        (x0 + x1) / 2,
        y0 - 3.0,
        r"future step $\tau$",
        ha="center",
        va="center",
        fontsize=5.7,
        color=COLORS["neutral"],
    )
    ax.text(x0, y0 - 3.0, "$1$", ha="left", va="center", fontsize=5.0, color=COLORS["neutral"])
    ax.text(x1, y0 - 3.0, "$T$", ha="right", va="center", fontsize=5.0, color=COLORS["neutral"])

    if show_formula:
        ax.text(
            (x0 + x1) / 2,
            y0 - 7.0,
            r"cell fill: $\overline{\phi}_{g,d}^{(s)}"
            r"=|\mathcal{G}_g^{(s)}|^{-1}\sum_{\tau\in\mathcal{G}_g^{(s)}}\phi_{\tau,d}$",
            ha="center",
            va="center",
            fontsize=5.0,
            color=COLORS["neutral"],
        )
    legend_x = descriptor_x - 11.0
    draw_signed_legend(ax, x0=legend_x, y=y0 - 10.8, cmap=cmap)


def build_combined_figure() -> Figure:
    """Build the shared visual-language comparison."""
    configure_style()
    figure = plt.figure(
        figsize=(COMBINED_WIDTH_MM / 25.4, COMBINED_HEIGHT_MM / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, COMBINED_WIDTH_MM)
    axis.set_ylim(0.0, COMBINED_HEIGHT_MM)
    axis.set_axis_off()

    axis.text(
        5.0,
        79.0,
        "Unified framed signal fields",
        ha="left",
        va="center",
        fontsize=9.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    axis.text(
        178.0,
        79.0,
        "history channels ↔ coordinate channels",
        ha="right",
        va="center",
        fontsize=5.5,
        color=COLORS["neutral"],
    )
    axis.plot([5.0, 178.0], [75.8, 75.8], color=COLORS["hairline"], linewidth=0.7)
    axis.text(5.0, 70.8, "a", fontsize=7.5, color=COLORS["ink"], fontweight="bold")
    axis.text(79.0, 70.8, "b", fontsize=7.5, color=COLORS["ink"], fontweight="bold")

    draw_history_frame(
        axis,
        x0=13.0,
        x1=68.0,
        y0=22.0,
        y1=62.0,
        title_y=69.2,
    )
    draw_coordinate_frame(
        axis,
        x0=91.0,
        x1=151.0,
        y0=22.0,
        y1=62.0,
        descriptor_x=168.0,
        title_y=69.2,
        show_formula=True,
    )
    axis.text(
        40.5,
        8.0,
        "schematic input signals",
        ha="center",
        va="center",
        fontsize=5.1,
        color=COLORS["neutral"],
    )
    axis.text(
        123.0,
        8.0,
        "analytic DCT curves + exact region means",
        ha="center",
        va="center",
        fontsize=5.1,
        color=COLORS["neutral"],
    )
    return figure


def build_coordinate_figure() -> Figure:
    """Build the standalone framed coordinate component."""
    configure_style()
    figure = plt.figure(
        figsize=(COORDINATE_WIDTH_MM / 25.4, COORDINATE_HEIGHT_MM / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, COORDINATE_WIDTH_MM)
    axis.set_ylim(0.0, COORDINATE_HEIGHT_MM)
    axis.set_axis_off()
    draw_coordinate_frame(
        axis,
        x0=14.0,
        x1=82.0,
        y0=18.0,
        y1=48.0,
        descriptor_x=104.0,
        title_y=57.0,
        show_formula=True,
    )
    axis.text(
        116.0,
        57.0,
        "region-aware view",
        ha="right",
        va="center",
        fontsize=5.2,
        color=COLORS["neutral"],
    )
    return figure


def normalize_svg(path: Path) -> None:
    """Strip renderer-introduced line-end whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_figure_bundle(
    figure: Figure,
    output_dir: Path,
    figure_id: str,
) -> dict[str, str]:
    """Save editable and raster review formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / figure_id
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
        default=Path("analysis/iscf_bsca_framed_coordinate_regions_20260805"),
    )
    return parser.parse_args()


def main() -> None:
    """Render the combined study and standalone coordinate component."""
    args = parse_args()
    combined_outputs = save_figure_bundle(
        build_combined_figure(),
        args.output_dir,
        COMBINED_ID,
    )
    coordinate_outputs = save_figure_bundle(
        build_coordinate_figure(),
        args.output_dir,
        COORDINATE_ID,
    )
    manifest = {
        "status": "local_design_draft_for_author_review",
        "backend": "Python/matplotlib",
        "method_change": False,
        "main_figure_replacement": False,
        "empirical_data_used": False,
        "history_role": "deterministic schematic signals",
        "coordinate_role": "analytic DCT curves and exact region means",
        "region_count": 5,
        "selected_region": 2,
        "color_semantics": {
            "negative": COLORS["negative"],
            "zero": COLORS["zero"],
            "positive": COLORS["positive"],
        },
        "combined_outputs": combined_outputs,
        "coordinate_outputs": coordinate_outputs,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
