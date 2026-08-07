#!/usr/bin/env python3
"""Render ISCF scope-allocation, fusion, and varied-horizon output designs.

All probabilities and trajectories are deterministic schematic values used for
method explanation. They do not represent learned routing or empirical results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


EXTENSION_WIDTH_MM = 183.0
EXTENSION_HEIGHT_MM = 84.0

COLORS = {
    "ink": "#25282D",
    "muted": "#6D747C",
    "guide": "#D7DDE1",
    "allocation": "#7773A8",
    "allocation_light": "#EFEDF7",
    "scope_0": "#8AB9CA",
    "scope_1": "#68A3B8",
    "scope_2": "#4685A0",
    "scope_0_light": "#E8F2F5",
    "scope_1_light": "#DCECEF",
    "scope_2_light": "#D2E4E9",
    "coordinate": "#C96C2B",
    "forecast": "#2D7068",
    "forecast_light": "#DCEDE9",
    "white": "#FFFFFF",
}

SCOPE_COLORS = (
    COLORS["scope_0"],
    COLORS["scope_1"],
    COLORS["scope_2"],
)
SCOPE_LIGHTS = (
    COLORS["scope_0_light"],
    COLORS["scope_1_light"],
    COLORS["scope_2_light"],
)


def configure_style() -> None:
    """Configure publication-oriented defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def blend_with_white(color: str, strength: float) -> tuple[float, float, float]:
    """Blend one scope color with white using a bounded strength."""
    rgb = np.asarray(mpl.colors.to_rgb(color))
    strength = float(np.clip(strength, 0.0, 1.0))
    return tuple((1.0 - strength) * np.ones(3) + strength * rgb)


def arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["ink"],
    linewidth: float = 0.9,
    dashed: bool = True,
    connectionstyle: str = "arc3,rad=0",
    mutation_scale: float = 7.0,
) -> None:
    """Draw one connector in figure-millimetre coordinates."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=9,
        )
    )


def load_curve_source(path: Path) -> dict[str, np.ndarray]:
    """Load the deterministic curve and weight source generated previously."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {
        "t",
        "scope_0",
        "scope_1",
        "scope_2",
        "weight_0",
        "weight_1",
        "weight_2",
        "fused",
    }
    if not data.dtype.names or not required.issubset(data.dtype.names):
        raise ValueError(f"curve source must contain {sorted(required)}")
    return {name: np.asarray(data[name], dtype=float) for name in required}


def draw_forecast_slice(
    axis: Axes,
    *,
    x0: float,
    y: float,
    width: float,
    height: float,
    t: np.ndarray,
    values: np.ndarray,
    color: str,
    light_color: str,
    label: str,
) -> None:
    """Draw one compact scope-conditioned forecast lane."""
    axis.add_patch(
        Rectangle(
            (x0, y - height / 2),
            width,
            height,
            facecolor=light_color,
            edgecolor=COLORS["ink"],
            linewidth=0.75,
            zorder=2,
        )
    )
    for fraction in np.linspace(0.2, 0.8, 4):
        axis.plot(
            [x0 + width * fraction] * 2,
            [y - height / 2, y + height / 2],
            color=COLORS["white"],
            linewidth=0.65,
            zorder=3,
        )
    axis.plot(
        x0 + width * t,
        y + 0.40 * height * values,
        color=color,
        linewidth=1.8,
        zorder=5,
    )
    axis.text(
        x0 - 2.2,
        y,
        label,
        ha="right",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=color,
    )


def probability_bins(
    weights: np.ndarray,
    columns: int = 18,
) -> np.ndarray:
    """Average the full schematic probability field into display columns."""
    indices = np.array_split(np.arange(weights.shape[1]), columns)
    return np.column_stack([weights[:, index].mean(axis=1) for index in indices])


def draw_probability_field(
    axis: Axes,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    weights: np.ndarray,
    selected_column: int = 11,
    title: bool = True,
    expose_vector: bool = True,
) -> tuple[float, float]:
    """Draw a target-wise scope-probability field and one vector slice."""
    displayed = probability_bins(weights)
    rows, columns = displayed.shape
    cell_width = width / columns
    cell_height = height / rows
    if not 0 <= selected_column < columns:
        raise ValueError("selected probability column is out of range")

    for row in range(rows):
        for column in range(columns):
            value = displayed[row, column]
            strength = 0.16 + 0.84 * value / displayed.max()
            axis.add_patch(
                Rectangle(
                    (
                        x0 + column * cell_width,
                        y0 + (rows - row - 1) * cell_height,
                    ),
                    cell_width,
                    cell_height,
                    facecolor=blend_with_white(SCOPE_COLORS[row], strength),
                    edgecolor=COLORS["white"],
                    linewidth=0.55,
                    zorder=3,
                )
            )
        axis.text(
            x0 - 2.0,
            y0 + (rows - row - 0.5) * cell_height,
            rf"$s_{row}$",
            ha="right",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=SCOPE_COLORS[row],
        )

    axis.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["allocation"],
            linewidth=0.95,
            zorder=5,
        )
    )
    selected_x = x0 + selected_column * cell_width
    axis.add_patch(
        Rectangle(
            (selected_x, y0),
            cell_width,
            height,
            facecolor="none",
            edgecolor=COLORS["coordinate"],
            linewidth=1.15,
            zorder=6,
        )
    )
    if title:
        axis.text(
            x0 + width / 2,
            y0 + height + 3.4,
            "Target-wise Scope Probability",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=COLORS["ink"],
        )
        axis.text(
            x0 + width / 2,
            y0 - 2.8,
            r"$\boldsymbol{\Pi}_{b,c}\in\mathbb{R}^{T\times3}$",
            ha="center",
            va="top",
            fontsize=5.6,
            color=COLORS["muted"],
        )

    vector_x = x0 + width + 6.5
    vector_y = y0 + height / 2
    if expose_vector:
        selected = displayed[:, selected_column]
        arrow(
            axis,
            (selected_x + cell_width, y0 + height + 1.0),
            (vector_x - 2.5, y0 + height + 1.0),
            color=COLORS["coordinate"],
            linewidth=0.85,
            dashed=False,
            connectionstyle="arc3,rad=-0.25",
        )
        bar_x = vector_x - 2.0
        bar_width = 9.0
        bar_height = 2.2
        for row, value in enumerate(selected):
            y = vector_y + (1 - row) * 4.0
            axis.add_patch(
                FancyBboxPatch(
                    (bar_x, y - bar_height / 2),
                    bar_width,
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0.6",
                    facecolor=COLORS["white"],
                    edgecolor=COLORS["guide"],
                    linewidth=0.55,
                    zorder=3,
                )
            )
            axis.add_patch(
                FancyBboxPatch(
                    (bar_x, y - bar_height / 2),
                    bar_width * float(value),
                    bar_height,
                    boxstyle="round,pad=0,rounding_size=0.6",
                    facecolor=SCOPE_COLORS[row],
                    edgecolor="none",
                    zorder=4,
                )
            )
        axis.text(
            vector_x + 2.5,
            y0 + height + 2.4,
            r"$\boldsymbol{\pi}_{b,c,\tau}$",
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=COLORS["coordinate"],
        )
        axis.text(
            vector_x + 2.5,
            y0 - 1.0,
            r"$\sum_s\pi_s=1$",
            ha="center",
            va="top",
            fontsize=5.0,
            color=COLORS["muted"],
        )
    return vector_x + 7.0, vector_y


def draw_fusion_operator(
    axis: Axes,
    *,
    center: tuple[float, float],
    radius: float = 5.0,
) -> None:
    """Draw the scope-axis weighted contraction as a compact operator."""
    x, y = center
    axis.add_patch(
        Circle(
            center,
            radius,
            facecolor=COLORS["white"],
            edgecolor=COLORS["forecast"],
            linewidth=1.25,
            zorder=7,
        )
    )
    axis.text(
        x,
        y + 0.3,
        r"$\sum_s$",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["forecast"],
        zorder=8,
    )
    axis.text(
        x,
        y - radius - 2.4,
        "Target-wise\nWeighted Fusion",
        ha="center",
        va="top",
        fontsize=5.5,
        fontweight="bold",
        color=COLORS["muted"],
        linespacing=0.95,
    )


def draw_varied_horizon_output(
    axis: Axes,
    *,
    x0: float,
    y0: float,
    width: float,
    trajectory_height: float,
    t: np.ndarray,
    fused: np.ndarray,
    title: bool = True,
) -> None:
    """Draw one full trajectory and four exact nested-prefix views."""
    trajectory_center = y0 + 30.0
    if title:
        axis.text(
            x0 + width / 2,
            trajectory_center + trajectory_height / 2 + 3.2,
            "One Prediction Trajectory",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=COLORS["ink"],
        )
    axis.add_patch(
        FancyBboxPatch(
            (x0, trajectory_center - trajectory_height / 2),
            width,
            trajectory_height,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=COLORS["forecast_light"],
            edgecolor=COLORS["forecast"],
            linewidth=0.85,
            alpha=0.75,
            zorder=2,
        )
    )
    curve_y = trajectory_center + 0.39 * trajectory_height * fused
    curve_x = x0 + width * t
    axis.plot(
        curve_x,
        curve_y,
        color=COLORS["forecast"],
        linewidth=2.2,
        zorder=5,
    )

    endpoints = (0.24, 0.46, 0.69, 1.0)
    labels = (r"$H_1$", r"$H_2$", r"$H_3$", r"$H_4=T$")
    rows_y = (y0 + 13.8, y0 + 9.9, y0 + 6.0, y0 + 2.1)
    axis.text(
        x0,
        y0 + 17.8,
        "Varied-horizon Forecasts",
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["forecast"],
    )
    axis.text(
        x0 + width,
        y0 + 17.8,
        "nested prefixes of the same trajectory",
        ha="right",
        va="bottom",
        fontsize=5.1,
        color=COLORS["muted"],
    )

    for fraction, label, row_y in zip(endpoints, labels, rows_y):
        active = t <= fraction + 1e-12
        prefix_t = t[active]
        prefix = fused[active]
        prefix_x = x0 + width * prefix_t
        axis.plot(
            prefix_x,
            row_y + 1.35 * prefix,
            color=COLORS["forecast"],
            linewidth=1.35,
            zorder=5,
        )
        endpoint_x = x0 + width * fraction
        endpoint_index = int(np.argmin(np.abs(t - fraction)))
        endpoint_y = trajectory_center + 0.39 * trajectory_height * fused[endpoint_index]
        axis.plot(
            [endpoint_x, endpoint_x],
            [row_y - 1.8, row_y + 1.8],
            color=COLORS["coordinate"],
            linewidth=1.0,
            zorder=6,
        )
        axis.scatter(
            [endpoint_x],
            [endpoint_y],
            s=13.0,
            facecolor=COLORS["coordinate"],
            edgecolor=COLORS["white"],
            linewidth=0.55,
            zorder=7,
        )
        is_last = fraction >= 0.99
        label_x = endpoint_x - 1.3 if is_last else endpoint_x + 1.3
        label_y = row_y + 1.8 if is_last else row_y
        label_alignment = "right" if fraction >= 0.99 else "left"
        axis.text(
            label_x,
            label_y,
            label,
            ha=label_alignment,
            va="bottom" if is_last else "center",
            fontsize=5.3,
            color=COLORS["coordinate"],
        )


def build_extension_figure(curves: dict[str, np.ndarray]) -> Figure:
    """Build the self-contained right-side continuation design."""
    configure_style()
    figure = plt.figure(
        figsize=(EXTENSION_WIDTH_MM / 25.4, EXTENSION_HEIGHT_MM / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, EXTENSION_WIDTH_MM)
    axis.set_ylim(0.0, EXTENSION_HEIGHT_MM)
    axis.set_axis_off()

    axis.text(
        36.5,
        81.0,
        "Scope-conditioned Forecasts",
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    row_centers = (70.0, 58.0, 46.0)
    for row, y in enumerate(row_centers):
        draw_forecast_slice(
            axis,
            x0=8.0,
            y=y,
            width=58.0,
            height=8.2,
            t=curves["t"],
            values=curves[f"scope_{row}"],
            color=SCOPE_COLORS[row],
            light_color=SCOPE_LIGHTS[row],
            label=rf"$s_{row}$",
        )

    weights = np.vstack(
        (curves["weight_0"], curves["weight_1"], curves["weight_2"])
    )
    probability_out = draw_probability_field(
        axis,
        x0=8.0,
        y0=8.0,
        width=58.0,
        height=17.0,
        weights=weights,
    )
    fusion_center = (91.0, 46.0)
    for row, y in enumerate(row_centers):
        arrow(
            axis,
            (67.5, y),
            (fusion_center[0] - 5.2, fusion_center[1] + (1 - row) * 1.3),
            color=SCOPE_COLORS[row],
            linewidth=0.9,
            connectionstyle=f"arc3,rad={0.07 * (1 - row):.3f}",
        )
    arrow(
        axis,
        probability_out,
        (fusion_center[0] - 4.8, fusion_center[1] - 2.0),
        color=COLORS["allocation"],
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.18",
    )
    draw_fusion_operator(axis, center=fusion_center)
    arrow(
        axis,
        (fusion_center[0] + 5.0, fusion_center[1]),
        (104.0, fusion_center[1]),
        color=COLORS["forecast"],
        linewidth=1.2,
        dashed=False,
    )
    draw_varied_horizon_output(
        axis,
        x0=105.0,
        y0=5.0,
        width=73.0,
        trajectory_height=16.0,
        t=curves["t"],
        fused=curves["fused"],
    )
    axis.text(
        177.5,
        1.0,
        "schematic probabilities and trajectories; no empirical values",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#959DA3",
    )
    return figure


def build_probability_asset(curves: dict[str, np.ndarray]) -> Figure:
    """Build one transparent-ready standalone probability component."""
    configure_style()
    figure = plt.figure(figsize=(104.0 / 25.4, 39.0 / 25.4), facecolor="none")
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, 104.0)
    axis.set_ylim(0.0, 39.0)
    axis.set_axis_off()
    weights = np.vstack(
        (curves["weight_0"], curves["weight_1"], curves["weight_2"])
    )
    draw_probability_field(
        axis,
        x0=10.0,
        y0=9.0,
        width=70.0,
        height=19.0,
        weights=weights,
    )
    return figure


def build_trajectory_asset(curves: dict[str, np.ndarray]) -> Figure:
    """Build one transparent-ready varied-horizon output component."""
    configure_style()
    figure = plt.figure(figsize=(112.0 / 25.4, 64.0 / 25.4), facecolor="none")
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, 112.0)
    axis.set_ylim(0.0, 64.0)
    axis.set_axis_off()
    draw_varied_horizon_output(
        axis,
        x0=6.0,
        y0=5.0,
        width=100.0,
        trajectory_height=18.0,
        t=curves["t"],
        fused=curves["fused"],
    )
    return figure


def build_integrated_mockup(
    base_image: Path,
    curves: dict[str, np.ndarray],
) -> Figure:
    """Place the continuation after the user-provided current figure draft."""
    configure_style()
    image = plt.imread(base_image)
    base_width = 194.0
    total_width = 286.0
    height = 84.0
    figure = plt.figure(
        figsize=(total_width / 25.4, height / 25.4),
        facecolor=COLORS["white"],
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, total_width)
    axis.set_ylim(0.0, height)
    axis.set_axis_off()
    axis.imshow(image, extent=(0.0, base_width, 0.0, height), aspect="auto", zorder=0)

    weights = np.vstack(
        (curves["weight_0"], curves["weight_1"], curves["weight_2"])
    )
    probability_out = draw_probability_field(
        axis,
        x0=108.0,
        y0=7.5,
        width=58.0,
        height=17.0,
        weights=weights,
    )
    fusion_center = (207.0, 49.0)
    forecast_rows = (74.1, 61.7, 49.2)
    for row, y in enumerate(forecast_rows):
        arrow(
            axis,
            (190.0, y),
            (fusion_center[0] - 5.0, fusion_center[1] + (1 - row) * 1.2),
            color=SCOPE_COLORS[row],
            linewidth=0.9,
            connectionstyle=f"arc3,rad={0.065 * (1 - row):.3f}",
        )
    arrow(
        axis,
        probability_out,
        (fusion_center[0] - 4.6, fusion_center[1] - 2.0),
        color=COLORS["allocation"],
        linewidth=1.0,
        connectionstyle="arc3,rad=-0.18",
    )
    draw_fusion_operator(axis, center=fusion_center)
    arrow(
        axis,
        (fusion_center[0] + 5.0, fusion_center[1]),
        (218.0, fusion_center[1]),
        color=COLORS["forecast"],
        linewidth=1.2,
        dashed=False,
    )
    draw_varied_horizon_output(
        axis,
        x0=220.0,
        y0=5.0,
        width=62.0,
        trajectory_height=15.0,
        t=curves["t"],
        fused=curves["fused"],
    )
    return figure


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_bundle(
    figure: Figure,
    output_dir: Path,
    stem: str,
    *,
    transparent: bool = False,
    tiff: bool = True,
) -> dict[str, str]:
    """Save vector and review formats from one Matplotlib figure."""
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(svg_path, transparent=transparent)
    normalize_svg(svg_path)
    figure.savefig(pdf_path, transparent=transparent)
    figure.savefig(png_path, dpi=600, transparent=transparent)
    outputs = {
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }
    if tiff:
        path = output_dir / f"{stem}.tiff"
        figure.savefig(
            path,
            dpi=600,
            transparent=transparent,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        outputs["tiff"] = str(path)
    plt.close(figure)
    return outputs


def write_source_csv(path: Path, curves: dict[str, np.ndarray]) -> None:
    """Persist every schematic value used in the design."""
    columns = (
        "t",
        "scope_0",
        "scope_1",
        "scope_2",
        "weight_0",
        "weight_1",
        "weight_2",
        "fused",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(curves[column] for column in columns)))


def parse_args() -> argparse.Namespace:
    """Parse input and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-source",
        type=Path,
        default=Path(
            "analysis/iscf_forecast_curve_assets_20260806/"
            "schematic_curve_values.csv"
        ),
    )
    parser.add_argument(
        "--base-image",
        type=Path,
        default=None,
        help="Optional current author draft for a raster-only layout mockup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_allocation_trajectory_design_20260807"),
    )
    return parser.parse_args()


def main() -> None:
    """Render the extension, standalone components, and optional mockup."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curve_source(args.curve_source)
    outputs = {
        "extension": save_bundle(
            build_extension_figure(curves),
            args.output_dir,
            "figure_iscf_allocation_trajectory_extension_v1",
        ),
        "scope_probability": save_bundle(
            build_probability_asset(curves),
            args.output_dir,
            "asset_scope_probability_field_v1",
            transparent=True,
        ),
        "varied_horizon_output": save_bundle(
            build_trajectory_asset(curves),
            args.output_dir,
            "asset_varied_horizon_trajectory_v1",
            transparent=True,
        ),
    }
    if args.base_image is not None:
        mockup = build_integrated_mockup(args.base_image, curves)
        preview = args.output_dir / "figure_iscf_full_layout_mockup_v1.png"
        mockup.savefig(preview, dpi=300, facecolor=COLORS["white"])
        plt.close(mockup)
        outputs["integrated_mockup"] = {"png": str(preview)}

    source_csv = args.output_dir / "schematic_probability_and_trajectory.csv"
    write_source_csv(source_csv, curves)
    manifest = {
        "figure_id": "iscf_allocation_trajectory_design_v1",
        "status": "local_design_draft_for_author_review",
        "backend": "Python/matplotlib",
        "reuse_level": "style-only inheritance from the current author draft",
        "empirical_data_used": False,
        "method_change": False,
        "visual_claim": (
            "target-wise scope probabilities contract scope-conditioned "
            "forecasts into one trajectory whose prefixes answer varied-"
            "horizon requests"
        ),
        "claim_boundary": (
            "all probabilities and trajectories are deterministic schematic "
            "glyphs rather than learned or empirical values"
        ),
        "palette": COLORS,
        "source_values": str(source_csv),
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
