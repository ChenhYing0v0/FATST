#!/usr/bin/env python3
"""Export registered layers from the ISCF varied-horizon output panel."""

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
from matplotlib.patches import FancyBboxPatch


WIDTH_MM = 112.0
HEIGHT_MM = 54.0
PANEL_X0 = 6.0
PANEL_Y0 = 6.0
PANEL_WIDTH = 100.0
PANEL_HEIGHT = 38.0
INNER_X0 = PANEL_X0 + 2.0
INNER_WIDTH = PANEL_WIDTH - 4.0

COLORS = {
    "forecast": "#2D7068",
    "forecast_light": "#DCEDE9",
    "white": "#FFFFFF",
}

PREFIX_FRACTIONS = (0.24, 0.46, 0.69)
PREFIX_ROWS_Y = tuple(PANEL_Y0 + 2.8 + row * 2.15 for row in range(3))
PREFIX_ALPHAS = (0.20, 0.27, 0.34)


def configure_style() -> None:
    """Configure transparent publication-vector output."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": True,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def load_fused_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact schematic trajectory used by the assembled panel."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    if not data.dtype.names or not {"t", "fused"}.issubset(data.dtype.names):
        raise ValueError("curve source must contain t and fused columns")
    return np.asarray(data["t"], dtype=float), np.asarray(data["fused"], dtype=float)


def blank_figure() -> tuple[Figure, Axes]:
    """Create one transparent registered overlay canvas."""
    figure = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor="none",
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, WIDTH_MM)
    axis.set_ylim(0.0, HEIGHT_MM)
    axis.set_axis_off()
    return figure, axis


def draw_background(axis: Axes) -> None:
    """Draw only the rounded prediction-panel background."""
    axis.add_patch(
        FancyBboxPatch(
            (PANEL_X0, PANEL_Y0),
            PANEL_WIDTH,
            PANEL_HEIGHT,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=COLORS["forecast_light"],
            edgecolor=COLORS["forecast"],
            linewidth=0.85,
            alpha=0.75,
        )
    )


def draw_curve(axis: Axes, t: np.ndarray, fused: np.ndarray) -> None:
    """Draw only the dark-teal trajectory curve."""
    trajectory_center = PANEL_Y0 + 0.64 * PANEL_HEIGHT
    curve_x = INNER_X0 + INNER_WIDTH * t
    curve_y = trajectory_center + 0.20 * PANEL_HEIGHT * fused
    axis.plot(
        curve_x,
        curve_y,
        color=COLORS["forecast"],
        linewidth=2.2,
        zorder=5,
    )


def draw_prefix_bar(axis: Axes, index: int) -> None:
    """Draw one start-aligned lower prefix bar without label or guide."""
    fraction = PREFIX_FRACTIONS[index]
    row_y = PREFIX_ROWS_Y[index]
    endpoint_x = INNER_X0 + INNER_WIDTH * fraction
    axis.add_patch(
        FancyBboxPatch(
            (INNER_X0, row_y - 0.52),
            endpoint_x - INNER_X0,
            1.04,
            boxstyle="round,pad=0,rounding_size=0.5",
            facecolor=COLORS["forecast"],
            edgecolor="none",
            alpha=PREFIX_ALPHAS[index],
        )
    )


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_asset(figure: Figure, output_dir: Path, stem: str) -> dict[str, str]:
    """Save one transparent registered layer in four formats."""
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    tiff_path = output_dir / f"{stem}.tiff"
    figure.savefig(svg_path, transparent=True)
    normalize_svg(svg_path)
    figure.savefig(pdf_path, transparent=True)
    figure.savefig(png_path, dpi=600, transparent=True)
    figure.savefig(
        tiff_path,
        dpi=600,
        transparent=True,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return {
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
        "tiff": str(tiff_path),
    }


def build_overlay_preview(t: np.ndarray, fused: np.ndarray) -> Figure:
    """Build a label-free preview of the five registered layers combined."""
    figure, axis = blank_figure()
    draw_background(axis)
    for index in range(3):
        draw_prefix_bar(axis, index)
    draw_curve(axis, t, fused)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-source",
        type=Path,
        default=Path(
            "analysis/iscf_allocation_trajectory_design_20260807/"
            "schematic_probability_and_trajectory.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_varied_horizon_component_assets_20260807"),
    )
    return parser.parse_args()


def main() -> None:
    """Render five registered component layers and one overlay preview."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    t, fused = load_fused_curve(args.curve_source)

    curve_figure, curve_axis = blank_figure()
    draw_curve(curve_axis, t, fused)
    background_figure, background_axis = blank_figure()
    draw_background(background_axis)

    outputs = {
        "trajectory_curve": save_asset(
            curve_figure,
            args.output_dir,
            "trajectory_curve_only",
        ),
        "rounded_background": save_asset(
            background_figure,
            args.output_dir,
            "trajectory_background_only",
        ),
    }
    for index in range(3):
        bar_figure, bar_axis = blank_figure()
        draw_prefix_bar(bar_axis, index)
        outputs[f"prefix_bar_h{index + 1}"] = save_asset(
            bar_figure,
            args.output_dir,
            f"prefix_bar_h{index + 1}_only",
        )

    preview = args.output_dir / "component_overlay_preview.png"
    preview_figure = build_overlay_preview(t, fused)
    preview_figure.savefig(preview, dpi=300, transparent=True)
    plt.close(preview_figure)

    manifest = {
        "asset_set": "iscf_varied_horizon_registered_layers_v1",
        "status": "author_composition_assets",
        "backend": "Python/matplotlib",
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "registered_coordinates": True,
        "empirical_data_used": False,
        "included_layers": [
            "trajectory curve",
            "rounded background",
            "H1 prefix bar",
            "H2 prefix bar",
            "H3 prefix bar",
        ],
        "omitted_by_request": [
            "title",
            "horizon labels",
            "endpoint markers",
            "endpoint guides",
            "H4 full-length bar",
        ],
        "curve_source": str(args.curve_source),
        "overlay_preview": str(preview),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "component_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
