#!/usr/bin/env python3
"""Export reusable ISCF history and future-coordinate figure components.

History traces are deterministic schematic signals. Exact coordinate curves
follow the frozen DCT definition; frequency-separated curves are design-only
glyphs and must not be presented as the implemented basis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT_WIDTH_MM = 118.0
HISTORY_HEIGHT_MM = 52.0
CURVES_HEIGHT_MM = 62.0
COMPONENT_HEIGHT_MM = 68.0
CONTACT_WIDTH_MM = 183.0
CONTACT_HEIGHT_MM = 150.0

COLORS = {
    "ink": "#252A30",
    "neutral": "#6F7780",
    "frame": "#344957",
    "hairline": "#D4DADE",
    "history": "#4F535B",
    "selection": "#7B5AA6",
    "selection_soft": "#E8E0F1",
    "white": "#FFFFFF",
}

ROW_COLORS = ("#555A91", "#6387B5", "#3F9293", "#C28A52")
ROW_DARK = ("#353A6C", "#3E648E", "#246D70", "#8B5E32")


def configure_style() -> None:
    """Set one publication-oriented style for all exported components."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def history_curves(total_steps: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Create three high-occupancy deterministic history glyphs."""
    t = np.linspace(0.0, 1.0, total_steps)
    envelope = 0.88 + 0.12 * np.sin(2.0 * np.pi * t - 0.4)
    first = envelope * (
        0.64 * np.sin(2.0 * np.pi * 1.45 * t + 0.25)
        + 0.28 * np.sin(2.0 * np.pi * 5.8 * t + 0.70)
        + 0.16 * np.sin(2.0 * np.pi * 13.0 * t)
    ) + 0.62 * np.exp(-((t - 0.90) / 0.045) ** 2)
    second = (
        0.78 * np.tanh(7.5 * (t - 0.43))
        + 0.34 * np.sin(2.0 * np.pi * 1.05 * t - 0.55)
        + 0.20 * np.sin(2.0 * np.pi * 7.2 * t + 0.10)
        - 0.46 * np.exp(-((t - 0.12) / 0.065) ** 2)
    )
    third = (
        0.60 * np.sin(2.0 * np.pi * 2.05 * t - 0.45)
        + 0.31 * np.sin(2.0 * np.pi * 4.7 * t + 0.95)
        + 0.18 * np.sin(2.0 * np.pi * 10.2 * t)
        + 0.74 * np.exp(-((t - 0.88) / 0.040) ** 2)
    )
    curves = np.stack([first, second, third])
    curves -= curves.mean(axis=1, keepdims=True)
    curves /= np.maximum(np.abs(curves).max(axis=1, keepdims=True), 1e-8)
    return t, curves


def exact_coordinate_curves(
    total_steps: int = 720,
    dimensions: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the exact frozen coordinate field."""
    tau = np.arange(1, total_steps + 1, dtype=float)
    field = np.ones((dimensions, total_steps), dtype=float)
    for dim in range(1, dimensions):
        raw = np.cos(np.pi * (tau - 0.5) * dim / total_steps)
        field[dim] = np.sqrt(2.0) * (raw - raw.mean())
    return tau, field


def design_coordinate_curves(
    total_steps: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a frequency-separated coordinate glyph for visual design only."""
    tau = np.arange(1, total_steps + 1, dtype=float)
    t = (tau - 0.5) / total_steps
    cycles = (0.0, 0.75, 2.0, 5.0)
    phases = (0.0, 0.08, 0.14, 0.20)
    field = np.ones((4, total_steps), dtype=float)
    for dim in range(1, 4):
        field[dim] = np.cos(2.0 * np.pi * cycles[dim] * t + phases[dim])
    return tau, field


def normalize_rows(field: np.ndarray) -> np.ndarray:
    """Map every nonconstant coordinate row to the displayed [-1, 1] range."""
    normalized = field.copy()
    normalized[0] = 1.0
    for row in range(1, field.shape[0]):
        centered = field[row] - field[row].mean()
        normalized[row] = centered / max(float(np.max(np.abs(centered))), 1e-8)
    return normalized


def region_means(field: np.ndarray, regions: int) -> np.ndarray:
    """Average every coordinate row over equal contiguous future regions."""
    indices = np.array_split(np.arange(field.shape[1]), regions)
    return np.stack([field[:, index].mean(axis=1) for index in indices], axis=1)


def value_alpha(value: float) -> float:
    """Map a signed value monotonically to an opacity."""
    clipped = float(np.clip(value, -1.0, 1.0))
    return 0.10 + 0.78 * (clipped + 1.0) / 2.0


def add_frame(
    ax: Axes,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    rows: int,
    columns: int | None = None,
) -> None:
    """Draw a restrained outer frame and internal separators."""
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["frame"],
            linewidth=1.0,
            zorder=8,
        )
    )
    row_height = (y1 - y0) / rows
    for row in range(1, rows):
        y = y0 + row * row_height
        ax.plot([x0, x1], [y, y], color=COLORS["frame"], linewidth=0.55, zorder=8)
    if columns is not None:
        column_width = (x1 - x0) / columns
        for column in range(1, columns):
            x = x0 + column * column_width
            ax.plot([x, x], [y0, y1], color=COLORS["frame"], linewidth=0.42, zorder=8)


def build_history_figure(*, transparent: bool) -> Figure:
    """Build a frameless standalone three-channel history material."""
    configure_style()
    facecolor = "none" if transparent else COLORS["white"]
    figure = plt.figure(
        figsize=(OUTPUT_WIDTH_MM / 25.4, HISTORY_HEIGHT_MM / 25.4),
        facecolor=facecolor,
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, OUTPUT_WIDTH_MM)
    axis.set_ylim(0.0, HISTORY_HEIGHT_MM)
    axis.set_axis_off()
    axis.patch.set_alpha(0.0 if transparent else 1.0)

    x0, x1, y0, y1 = 7.0, 111.0, 5.0, 47.0
    _, curves = history_curves()
    row_height = (y1 - y0) / curves.shape[0]
    mapped_x = np.linspace(x0 + 1.5, x1 - 1.5, curves.shape[1])
    for row, curve in enumerate(curves):
        center = y1 - (row + 0.5) * row_height
        curve_y = center + 0.42 * row_height * curve
        axis.plot(
            mapped_x,
            curve_y,
            color=COLORS["history"],
            linewidth=3.70,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5,
        )
    return figure


def build_coordinate_curves_figure(
    field: np.ndarray,
    *,
    transparent: bool,
    monochrome: bool,
) -> Figure:
    """Build a frameless standalone four-channel coordinate-curve material."""
    configure_style()
    facecolor = "none" if transparent else COLORS["white"]
    figure = plt.figure(
        figsize=(OUTPUT_WIDTH_MM / 25.4, CURVES_HEIGHT_MM / 25.4),
        facecolor=facecolor,
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, OUTPUT_WIDTH_MM)
    axis.set_ylim(0.0, CURVES_HEIGHT_MM)
    axis.set_axis_off()
    axis.patch.set_alpha(0.0 if transparent else 1.0)

    x0, x1, y0, y1 = 10.0, 112.0, 5.0, 57.0
    display = normalize_rows(field)
    row_height = (y1 - y0) / display.shape[0]
    mapped_x = np.linspace(x0 + 1.2, x1 - 1.2, display.shape[1])
    for row, curve in enumerate(display):
        center = y1 - (row + 0.5) * row_height
        curve_geometry = np.zeros_like(curve) if row == 0 else curve
        curve_y = center + 0.38 * row_height * curve_geometry
        curve_color = COLORS["history"] if monochrome else ROW_DARK[row]
        axis.plot(
            mapped_x,
            curve_y,
            color=curve_color,
            linewidth=3.10,
            solid_capstyle="round",
            zorder=5,
        )
        axis.text(
            x0 - 1.8,
            center,
            f"$d_{row}$",
            ha="right",
            va="center",
            fontsize=6.0,
            color=curve_color,
        )
    return figure


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    """Draw a compact selection-to-descriptor connector."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=5.5,
            linewidth=0.8,
            color=COLORS["selection"],
            connectionstyle="angle3,angleA=0,angleB=-90",
            shrinkA=0,
            shrinkB=0,
            zorder=12,
        )
    )


def build_coordinate_component(*, transparent: bool) -> Figure:
    """Build the revised frequency-separated coordinate-region component."""
    configure_style()
    facecolor = "none" if transparent else COLORS["white"]
    figure = plt.figure(
        figsize=(OUTPUT_WIDTH_MM / 25.4, COMPONENT_HEIGHT_MM / 25.4),
        facecolor=facecolor,
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, OUTPUT_WIDTH_MM)
    axis.set_ylim(0.0, COMPONENT_HEIGHT_MM)
    axis.set_axis_off()
    axis.patch.set_alpha(0.0 if transparent else 1.0)

    _, field = design_coordinate_curves()
    display = normalize_rows(field)
    regions = 15
    means = np.clip(region_means(display, regions), -1.0, 1.0)
    x0, x1, y0, y1 = 10.0, 94.0, 16.0, 58.0
    row_height = (y1 - y0) / display.shape[0]
    region_width = (x1 - x0) / regions
    selected_region = 8

    for row in range(display.shape[0]):
        cell_y = y1 - (row + 1) * row_height
        for region in range(regions):
            cell_x = x0 + region * region_width
            axis.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    region_width,
                    row_height,
                    facecolor=ROW_COLORS[row],
                    edgecolor="none",
                    alpha=value_alpha(float(means[row, region])),
                    zorder=1,
                )
            )

    mapped_x = np.linspace(x0 + 0.4, x1 - 0.4, display.shape[1])
    for row, curve in enumerate(display):
        center = y1 - (row + 0.5) * row_height
        curve_geometry = np.zeros_like(curve) if row == 0 else curve
        curve_y = center + 0.34 * row_height * curve_geometry
        axis.plot(
            mapped_x,
            curve_y,
            color=ROW_DARK[row],
            linewidth=1.25,
            solid_capstyle="round",
            zorder=6,
        )
        axis.text(
            x0 - 1.7,
            center,
            f"$d_{row}$",
            ha="right",
            va="center",
            fontsize=5.7,
            color=ROW_DARK[row],
        )

    add_frame(
        axis,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        rows=4,
        columns=regions,
    )
    selected_x = x0 + selected_region * region_width
    axis.add_patch(
        Rectangle(
            (selected_x, y0),
            region_width,
            y1 - y0,
            fill=False,
            edgecolor=COLORS["selection"],
            linewidth=1.35,
            zorder=10,
        )
    )
    axis.text(
        selected_x + region_width / 2,
        y1 + 1.8,
        r"$\mathcal{G}_{g}^{(s)}$",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COLORS["selection"],
        fontweight="bold",
    )

    descriptor_x = x1 + 5.2
    row_centers = y1 - (np.arange(4) + 0.5) * row_height
    descriptor_values = means[:, selected_region]
    square_size = 0.54 * row_height
    for row, (center, value) in enumerate(zip(row_centers, descriptor_values)):
        axis.add_patch(
            Rectangle(
                (descriptor_x - square_size / 2, center - square_size / 2),
                square_size,
                square_size,
                facecolor=ROW_COLORS[row],
                edgecolor=ROW_DARK[row],
                linewidth=0.75,
                alpha=value_alpha(float(value)),
                zorder=9,
            )
        )
    arrow(
        axis,
        (selected_x + region_width / 2, y1 + 5.2),
        (descriptor_x, y1 + 5.2),
    )
    axis.plot(
        [descriptor_x, descriptor_x],
        [y1 + 5.2, row_centers[0] + square_size / 2],
        color=COLORS["selection"],
        linewidth=0.7,
        zorder=11,
    )
    axis.text(
        descriptor_x,
        y0 - 1.8,
        r"$\overline{\boldsymbol{\phi}}_{g}^{(s)}$",
        ha="center",
        va="top",
        fontsize=6.2,
        color=COLORS["selection"],
        fontweight="bold",
    )

    legend_y = 6.2
    legend_x = 53.5
    axis.text(
        legend_x - 4.3,
        legend_y + 2.1,
        "region mean",
        ha="right",
        va="center",
        fontsize=5.0,
        color=COLORS["neutral"],
    )
    for index, value in enumerate((-1.0, 0.0, 1.0)):
        x = legend_x + 8.0 * index
        axis.add_patch(
            Rectangle(
                (x, legend_y),
                4.0,
                4.0,
                facecolor=ROW_COLORS[1],
                edgecolor=COLORS["hairline"],
                linewidth=0.4,
                alpha=value_alpha(value),
            )
        )
        axis.text(
            x + 2.0,
            legend_y - 1.0,
            f"{value:+.0f}" if value else "0",
            ha="center",
            va="top",
            fontsize=5.0,
            color=COLORS["neutral"],
        )
    return figure


def build_contact_sheet(
    history: np.ndarray,
    exact: np.ndarray,
    design: np.ndarray,
) -> Figure:
    """Build a compact author-review sheet from the generated curve arrays."""
    configure_style()
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(CONTACT_WIDTH_MM / 25.4, CONTACT_HEIGHT_MM / 25.4),
        constrained_layout=True,
    )
    figure.patch.set_facecolor(COLORS["white"])
    titles = (
        "History material — enlarged within-row variation",
        "Exact frozen coordinates — method-faithful",
        "Frequency-separated coordinates — design-only schematic",
    )
    arrays = (history, normalize_rows(exact), normalize_rows(design))
    line_sets = (
        (COLORS["history"],) * 3,
        ROW_DARK,
        ROW_DARK,
    )
    for axis, title, curves, colors in zip(axes, titles, arrays, line_sets):
        axis.set_title(title, loc="left", fontsize=7.0, fontweight="bold", pad=4)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.1, curves.shape[0] + 0.1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["left", "bottom"]].set_visible(False)
        for row, (curve, color) in enumerate(zip(curves, colors)):
            center = curves.shape[0] - row - 0.5
            scale = 0.39 if curves.shape[0] == 4 else 0.43
            curve_geometry = (
                np.zeros_like(curve)
                if curves.shape[0] == 4 and row == 0
                else curve
            )
            axis.plot(
                np.linspace(0.0, 1.0, curve.size),
                center + scale * curve_geometry,
                color=color,
                linewidth=3.0,
                solid_capstyle="round",
            )
    return figure


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_white_bundle(figure: Figure, output_dir: Path, figure_id: str) -> dict[str, str]:
    """Save vector and high-resolution white-background formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / figure_id
    outputs = {
        "svg": f"{prefix}.svg",
        "pdf": f"{prefix}.pdf",
        "png": f"{prefix}.png",
        "tiff": f"{prefix}.tiff",
    }
    figure.savefig(outputs["svg"], format="svg", bbox_inches="tight", facecolor="white")
    normalize_svg(Path(outputs["svg"]))
    figure.savefig(outputs["pdf"], format="pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["png"], format="png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        outputs["tiff"],
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return outputs


def save_transparent_bundle(
    figure: Figure,
    output_dir: Path,
    figure_id: str,
) -> dict[str, str]:
    """Save transparent SVG and PNG component assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / figure_id
    outputs = {"svg": f"{prefix}.svg", "png": f"{prefix}.png"}
    figure.savefig(
        outputs["svg"],
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    normalize_svg(Path(outputs["svg"]))
    figure.savefig(
        outputs["png"],
        format="png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(figure)
    return outputs


def write_source_data(
    output_dir: Path,
    history_t: np.ndarray,
    history: np.ndarray,
    tau: np.ndarray,
    exact: np.ndarray,
    design: np.ndarray,
) -> dict[str, str]:
    """Write deterministic source arrays for traceability."""
    history_path = output_dir / "history_schematic_source.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["normalized_time", "channel_0", "channel_1", "channel_2"])
        for index, time_value in enumerate(history_t):
            writer.writerow([time_value, *(history[:, index])])

    coordinate_path = output_dir / "coordinate_source.csv"
    with coordinate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "future_step",
                *(f"exact_d{dim}" for dim in range(4)),
                *(f"design_d{dim}" for dim in range(4)),
            ]
        )
        for index, step in enumerate(tau):
            writer.writerow([int(step), *(exact[:, index]), *(design[:, index])])
    return {"history": str(history_path), "coordinates": str(coordinate_path)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_bsca_component_materials_v2_20260805"),
    )
    return parser.parse_args()


def main() -> None:
    """Render all standalone materials and write their manifest."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_t, history = history_curves()
    tau, exact = exact_coordinate_curves()
    _, design = design_coordinate_curves()

    outputs = {
        "history_white": save_white_bundle(
            build_history_figure(transparent=False),
            args.output_dir,
            "history_curves_white",
        ),
        "history_transparent": save_transparent_bundle(
            build_history_figure(transparent=True),
            args.output_dir,
            "history_curves_transparent",
        ),
        "coordinate_exact_white": save_white_bundle(
            build_coordinate_curves_figure(
                exact,
                transparent=False,
                monochrome=False,
            ),
            args.output_dir,
            "coordinate_curves_exact_white",
        ),
        "coordinate_exact_transparent": save_transparent_bundle(
            build_coordinate_curves_figure(
                exact,
                transparent=True,
                monochrome=False,
            ),
            args.output_dir,
            "coordinate_curves_exact_transparent",
        ),
        "coordinate_exact_monochrome_white": save_white_bundle(
            build_coordinate_curves_figure(
                exact,
                transparent=False,
                monochrome=True,
            ),
            args.output_dir,
            "coordinate_curves_exact_monochrome_white",
        ),
        "coordinate_exact_monochrome_transparent": save_transparent_bundle(
            build_coordinate_curves_figure(
                exact,
                transparent=True,
                monochrome=True,
            ),
            args.output_dir,
            "coordinate_curves_exact_monochrome_transparent",
        ),
        "coordinate_design_white": save_white_bundle(
            build_coordinate_curves_figure(
                design,
                transparent=False,
                monochrome=False,
            ),
            args.output_dir,
            "coordinate_curves_frequency_separated_design_white",
        ),
        "coordinate_design_transparent": save_transparent_bundle(
            build_coordinate_curves_figure(
                design,
                transparent=True,
                monochrome=False,
            ),
            args.output_dir,
            "coordinate_curves_frequency_separated_design_transparent",
        ),
        "coordinate_design_monochrome_white": save_white_bundle(
            build_coordinate_curves_figure(
                design,
                transparent=False,
                monochrome=True,
            ),
            args.output_dir,
            "coordinate_curves_frequency_separated_design_monochrome_white",
        ),
        "coordinate_design_monochrome_transparent": save_transparent_bundle(
            build_coordinate_curves_figure(
                design,
                transparent=True,
                monochrome=True,
            ),
            args.output_dir,
            "coordinate_curves_frequency_separated_design_monochrome_transparent",
        ),
        "coordinate_component_white": save_white_bundle(
            build_coordinate_component(transparent=False),
            args.output_dir,
            "coordinate_region_component_v2_design_white",
        ),
        "coordinate_component_transparent": save_transparent_bundle(
            build_coordinate_component(transparent=True),
            args.output_dir,
            "coordinate_region_component_v2_design_transparent",
        ),
        "contact_sheet": save_white_bundle(
            build_contact_sheet(history, exact, design),
            args.output_dir,
            "component_materials_v2_contact_sheet",
        ),
    }
    source_data = write_source_data(
        args.output_dir,
        history_t,
        history,
        tau,
        exact,
        design,
    )
    manifest = {
        "status": "local_design_draft_for_author_review",
        "backend": "Python/matplotlib",
        "method_change": False,
        "main_figure_replacement": False,
        "empirical_data_used": False,
        "history_role": "deterministic high-occupancy schematic signals",
        "standalone_curve_style": "frameless with twofold line-width increase",
        "exact_coordinate_role": "frozen D_q=4 coordinate definition",
        "design_coordinate_role": (
            "frequency-separated schematic with 0.75, 2, and 5 cycles; "
            "not the implemented basis"
        ),
        "coordinate_component_role": "author-review visual design using schematic curves",
        "region_count": 15,
        "region_rationale": "matches T=720 divided by the valid s=48 scope",
        "opacity_mapping": "alpha = 0.10 + 0.78 * (clip(v,-1,1)+1)/2",
        "row_colors": list(ROW_COLORS),
        "source_data": source_data,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
