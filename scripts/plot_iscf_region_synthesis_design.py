#!/usr/bin/env python3
"""Render a local design study for ISCF region-to-forecast synthesis.

The figure is a schematic continuation component for author review. It shows
the exact information hierarchy of shared step-specific synthesis, but no
empirical or learned values.
"""

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


WIDTH_MM = 183.0
HEIGHT_MM = 72.0

COLORS = {
    "ink": "#25282D",
    "muted": "#6D747C",
    "guide": "#D4D9DD",
    "step_guide": "#E5E9EC",
    "module_fill": "#F7EEE7",
    "module_edge": "#2D2D2D",
    "scope_0": "#8AB9CA",
    "scope_1": "#68A3B8",
    "scope_2": "#4685A0",
    "scope_0_light": "#E8F2F5",
    "scope_1_light": "#DCECEF",
    "scope_2_light": "#D2E4E9",
    "accent": "#C87832",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    """Configure compact publication-oriented defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
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
    dashed: bool = True,
) -> None:
    """Draw one connector in figure-millimetre coordinates."""
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=linewidth,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )


def draw_region_representation(
    axis: Axes,
    *,
    y: float,
    regions: int,
    color: str,
    light_color: str,
    scope_label: str,
) -> None:
    """Draw one region-indexed latent-state sequence."""
    x0, width, height = 10.0, 42.0, 6.3
    axis.text(
        5.5,
        y,
        scope_label,
        ha="right",
        va="center",
        fontsize=7.0,
        color=color,
        fontweight="bold",
    )
    values = 0.5 + 0.5 * np.sin(np.arange(regions) * 1.7 + regions * 0.31)
    region_width = width / regions
    for region, value in enumerate(values):
        blend = mpl.colors.to_rgb(light_color)
        strong = mpl.colors.to_rgb(color)
        face = tuple(
            (1.0 - 0.58 * value) * base + 0.58 * value * target
            for base, target in zip(blend, strong)
        )
        axis.add_patch(
            Rectangle(
                (x0 + region * region_width, y - height / 2),
                region_width,
                height,
                facecolor=face,
                edgecolor=COLORS["white"],
                linewidth=0.8,
                zorder=3,
            )
        )
    axis.add_patch(
        Rectangle(
            (x0, y - height / 2),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["ink"],
            linewidth=0.9,
            zorder=4,
        )
    )


def draw_shared_synthesis(axis: Axes, row_centers: tuple[float, ...]) -> None:
    """Draw one shared module with aligned row-wise input/output ports."""
    x0, y0, width, height = 68.0, 11.0, 16.0, 49.0
    axis.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.6,rounding_size=1.3",
            facecolor=COLORS["module_fill"],
            edgecolor=COLORS["module_edge"],
            linewidth=1.0,
            zorder=5,
        )
    )
    axis.text(
        x0 + width / 2,
        y0 + height / 2,
        "Step-specific\nsynthesis",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.0,
        color=COLORS["ink"],
        fontweight="bold",
        linespacing=0.95,
        zorder=7,
    )
    axis.text(
        x0 + width / 2,
        y0 - 2.6,
        r"shared across scopes",
        ha="center",
        va="top",
        fontsize=5.3,
        color=COLORS["muted"],
    )
    axis.text(
        x0 + width / 2,
        y0 - 6.0,
        r"$\mathbf{a}_\tau,\mathbf{n}_\tau,\beta_\tau$",
        ha="center",
        va="top",
        fontsize=5.7,
        color=COLORS["accent"],
    )
    for y in row_centers:
        axis.plot(
            [x0, x0 + 2.1],
            [y, y],
            color=COLORS["module_edge"],
            linewidth=1.0,
            zorder=8,
        )
        axis.plot(
            [x0 + width - 2.1, x0 + width],
            [y, y],
            color=COLORS["module_edge"],
            linewidth=1.0,
            zorder=8,
        )


def forecast_curve(
    x: np.ndarray,
    row_index: int,
) -> np.ndarray:
    """Return one deterministic schematic scope-forecast curve."""
    normalized = (x - x.min()) / max(float(np.ptp(x)), 1e-8)
    if row_index == 0:
        return (
            0.58 * np.sin(2.0 * np.pi * (1.45 * normalized + 0.08))
            + 0.23 * np.sin(2.0 * np.pi * 4.2 * normalized)
        )
    if row_index == 1:
        return (
            0.64 * np.sin(2.0 * np.pi * (1.15 * normalized + 0.22))
            + 0.18 * np.sin(2.0 * np.pi * 3.3 * normalized + 0.4)
        )
    return (
        0.70 * np.sin(2.0 * np.pi * (0.88 * normalized + 0.34))
        + 0.13 * np.sin(2.0 * np.pi * 2.8 * normalized + 0.7)
    )


def draw_scope_forecast(
    axis: Axes,
    *,
    y: float,
    row_index: int,
    regions: int,
    color: str,
    light_color: str,
) -> None:
    """Draw one region-aware scope-conditioned forecast ribbon."""
    x0, width, height = 96.0, 78.0, 11.0
    axis.add_patch(
        Rectangle(
            (x0, y - height / 2),
            width,
            height,
            facecolor=light_color,
            edgecolor="none",
            alpha=0.68,
            zorder=1,
        )
    )
    region_width = width / regions
    steps_per_region = 3
    for region in range(regions + 1):
        x = x0 + region * region_width
        axis.plot(
            [x, x],
            [y - height / 2, y + height / 2],
            color=color,
            linewidth=0.9,
            alpha=0.75,
            zorder=2,
        )
    for region in range(regions):
        for step in range(1, steps_per_region):
            x = x0 + region * region_width + step * region_width / steps_per_region
            axis.plot(
                [x, x],
                [y - height / 2, y + height / 2],
                color=COLORS["step_guide"],
                linewidth=0.45,
                zorder=2,
            )
    curve_x = np.linspace(x0, x0 + width, 360)
    curve_y = y + 0.37 * height * forecast_curve(curve_x, row_index)
    axis.plot(
        curve_x,
        curve_y,
        color=color,
        linewidth=1.8,
        solid_capstyle="round",
        zorder=5,
    )
    axis.add_patch(
        Rectangle(
            (x0, y - height / 2),
            width,
            height,
            facecolor="none",
            edgecolor=COLORS["ink"],
            linewidth=0.8,
            zorder=6,
        )
    )


def build_figure() -> Figure:
    """Build the recommended region-to-forecast continuation design."""
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
        31.0,
        67.0,
        "Scope-region States",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        76.0,
        67.0,
        "Shared Step-specific Synthesis",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        135.0,
        67.0,
        "Scope-conditioned Forecasts",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
    )

    row_centers = (52.0, 36.0, 20.0)
    region_counts = (8, 4, 2)
    scope_colors = (COLORS["scope_0"], COLORS["scope_1"], COLORS["scope_2"])
    scope_lights = (
        COLORS["scope_0_light"],
        COLORS["scope_1_light"],
        COLORS["scope_2_light"],
    )
    scope_labels = (r"$s_0$", r"$s_1$", r"$s_2$")

    for row_index, (y, regions, color, light_color, label) in enumerate(
        zip(
            row_centers,
            region_counts,
            scope_colors,
            scope_lights,
            scope_labels,
        )
    ):
        draw_region_representation(
            axis,
            y=y,
            regions=regions,
            color=color,
            light_color=light_color,
            scope_label=label,
        )
        arrow(axis, (53.5, y), (67.0, y), color=COLORS["ink"])
        arrow(axis, (85.0, y), (94.5, y), color=COLORS["ink"])
        draw_scope_forecast(
            axis,
            y=y,
            row_index=row_index,
            regions=regions,
            color=color,
            light_color=light_color,
        )

    draw_shared_synthesis(axis, row_centers)

    highlight_x0 = 96.0 + 78.0 / region_counts[1]
    highlight_width = 78.0 / region_counts[1]
    axis.add_patch(
        Rectangle(
            (highlight_x0, row_centers[1] - 5.5),
            highlight_width,
            11.0,
            facecolor="none",
            edgecolor=COLORS["accent"],
            linewidth=1.2,
            zorder=9,
        )
    )
    axis.text(
        highlight_x0 + highlight_width / 2,
        row_centers[1] - 7.5,
        r"one shared $\mathbf{z}_g^{(s)}$; distinct step predictions",
        ha="center",
        va="top",
        fontsize=5.2,
        color=COLORS["accent"],
    )
    axis.text(
        135.0,
        4.0,
        "bold separators: region boundaries    ·    fine separators: future steps",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["muted"],
    )
    return figure


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_bundle(figure: Figure, output_dir: Path) -> dict[str, str]:
    """Save vector and review formats from the same Python source."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_iscf_region_synthesis_design_v1"
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    figure.savefig(paths["svg"])
    normalize_svg(paths["svg"])
    figure.savefig(paths["pdf"])
    figure.savefig(paths["png"], dpi=300)
    figure.savefig(
        paths["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return {kind: str(path) for kind, path in paths.items()}


def parse_args() -> argparse.Namespace:
    """Parse output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_region_synthesis_design_20260806"),
    )
    return parser.parse_args()


def main() -> None:
    """Render the design study and write its manifest."""
    args = parse_args()
    outputs = save_bundle(build_figure(), args.output_dir)
    manifest = {
        "figure_id": "figure_iscf_region_synthesis_design_v1",
        "status": "local_design_draft_for_author_review",
        "backend": "Python/matplotlib",
        "source_role": "continuation design for the current author draft",
        "reuse_level": "style-only inheritance",
        "empirical_data_used": False,
        "method_change": False,
        "recommended_names": {
            "input": "Scope-region states",
            "operation": "Shared step-specific synthesis",
            "output": "Scope-conditioned forecasts",
            "stacked_output": "Scope-indexed forecast field",
        },
        "visual_claim": (
            "one region state is reused within each region while shared "
            "step-specific parameters produce distinct future-step values"
        ),
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
