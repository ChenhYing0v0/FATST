#!/usr/bin/env python3
"""Render standalone schematic forecast curves for the ISCF main figure.

The three scope-conditioned curves and their target-wise fused trajectory are
deterministic visual assets. They do not contain empirical or learned values.
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
from matplotlib.figure import Figure


WIDTH_MM = 78.0
HEIGHT_MM = 24.0
LINE_WIDTH = 3.2

COLORS = {
    "scope_0": "#8AB9CA",
    "scope_1": "#68A3B8",
    "scope_2": "#4685A0",
    "fused": "#2D7068",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    """Configure publication-oriented vector and raster defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def gaussian(
    x: np.ndarray,
    center: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    """Return a localized smooth feature for a schematic trajectory."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def build_curves() -> dict[str, np.ndarray]:
    """Construct three distinctive forecasts and one convexly fused curve."""
    t = np.linspace(0.0, 1.0, 720)
    shared = (
        0.36 * np.sin(2.0 * np.pi * (1.12 * t + 0.08))
        + 0.17 * np.sin(2.0 * np.pi * (2.65 * t - 0.12))
        + 0.18 * (t - 0.45)
        + gaussian(t, 0.23, 0.045, 0.44)
        - gaussian(t, 0.58, 0.065, 0.34)
        + gaussian(t, 0.84, 0.034, 0.39)
    )

    scope_0 = (
        shared
        + 0.16 * np.sin(2.0 * np.pi * (8.4 * t + 0.11))
        + 0.08 * np.sin(2.0 * np.pi * 15.2 * t)
        - gaussian(t, 0.40, 0.026, 0.24)
        + gaussian(t, 0.71, 0.020, 0.30)
    )
    scope_1 = (
        0.90 * shared
        + 0.22 * np.sin(2.0 * np.pi * (4.55 * t + 0.16))
        + gaussian(t, 0.34, 0.050, 0.28)
        - gaussian(t, 0.73, 0.047, 0.27)
    )
    scope_2 = (
        0.72 * shared
        + 0.33 * np.sin(2.0 * np.pi * (1.72 * t - 0.18))
        - 0.13 * np.cos(2.0 * np.pi * 3.05 * t)
        + gaussian(t, 0.49, 0.095, 0.32)
        - gaussian(t, 0.90, 0.055, 0.30)
    )

    logits = np.column_stack(
        (
            0.85 * np.sin(2.0 * np.pi * (1.25 * t + 0.08))
            + gaussian(t, 0.72, 0.12, 1.05),
            0.72 * np.cos(2.0 * np.pi * (0.92 * t - 0.16))
            + gaussian(t, 0.38, 0.14, 0.92),
            0.62 * np.sin(2.0 * np.pi * (0.68 * t + 0.42))
            + gaussian(t, 0.88, 0.10, 1.12),
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    fused = np.sum(
        weights * np.column_stack((scope_0, scope_1, scope_2)),
        axis=1,
    )

    all_curves = np.concatenate((scope_0, scope_1, scope_2, fused))
    scale = float(np.max(np.abs(all_curves)))
    scope_0, scope_1, scope_2, fused = (
        curve / scale for curve in (scope_0, scope_1, scope_2, fused)
    )
    return {
        "t": t,
        "scope_0": scope_0,
        "scope_1": scope_1,
        "scope_2": scope_2,
        "weight_0": weights[:, 0],
        "weight_1": weights[:, 1],
        "weight_2": weights[:, 2],
        "fused": fused,
    }


def build_figure(
    t: np.ndarray,
    values: np.ndarray,
    color: str,
    *,
    transparent: bool,
) -> Figure:
    """Build one frameless standalone curve at a common physical size."""
    facecolor = "none" if transparent else COLORS["white"]
    figure = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        facecolor=facecolor,
    )
    axis = figure.add_axes([0.025, 0.09, 0.95, 0.82])
    axis.plot(t, values, color=color, linewidth=LINE_WIDTH)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-1.08, 1.08)
    axis.set_axis_off()
    return figure


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def save_curve_bundle(
    output_dir: Path,
    name: str,
    t: np.ndarray,
    values: np.ndarray,
    color: str,
) -> dict[str, str]:
    """Save transparent and white-background variants of one curve."""
    outputs: dict[str, str] = {}
    transparent_figure = build_figure(t, values, color, transparent=True)
    transparent_png = output_dir / f"{name}_transparent.png"
    transparent_svg = output_dir / f"{name}_transparent.svg"
    transparent_figure.savefig(
        transparent_png,
        dpi=600,
        transparent=True,
    )
    transparent_figure.savefig(transparent_svg, transparent=True)
    normalize_svg(transparent_svg)
    plt.close(transparent_figure)
    outputs["transparent_png"] = str(transparent_png)
    outputs["transparent_svg"] = str(transparent_svg)

    white_figure = build_figure(t, values, color, transparent=False)
    white_png = output_dir / f"{name}_white.png"
    white_pdf = output_dir / f"{name}_white.pdf"
    white_tiff = output_dir / f"{name}_white.tiff"
    white_figure.savefig(white_png, dpi=600, facecolor=COLORS["white"])
    white_figure.savefig(white_pdf, facecolor=COLORS["white"])
    white_figure.savefig(
        white_tiff,
        dpi=600,
        facecolor=COLORS["white"],
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(white_figure)
    outputs["white_png"] = str(white_png)
    outputs["white_pdf"] = str(white_pdf)
    outputs["white_tiff"] = str(white_tiff)
    return outputs


def write_source_csv(path: Path, curves: dict[str, np.ndarray]) -> None:
    """Persist every schematic value and fusion weight for auditability."""
    columns = list(curves)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(curves[column] for column in columns)))


def parse_args() -> argparse.Namespace:
    """Parse output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/iscf_forecast_curve_assets_20260806"),
    )
    return parser.parse_args()


def main() -> None:
    """Render all standalone curves and write reproducibility metadata."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    curves = build_curves()
    outputs = {
        output_name: save_curve_bundle(
            args.output_dir,
            output_name,
            curves["t"],
            curves[curve_key],
            COLORS[color_key],
        )
        for output_name, curve_key, color_key in (
            ("scope_forecast_s0", "scope_0", "scope_0"),
            ("scope_forecast_s1", "scope_1", "scope_1"),
            ("scope_forecast_s2", "scope_2", "scope_2"),
            ("final_fused_forecast", "fused", "fused"),
        )
    }
    csv_path = args.output_dir / "schematic_curve_values.csv"
    write_source_csv(csv_path, curves)
    manifest = {
        "asset_set": "iscf_forecast_curve_assets_v1",
        "status": "local_design_assets_for_author_review",
        "backend": "Python/matplotlib",
        "empirical_data_used": False,
        "method_change": False,
        "palette": COLORS,
        "curve_width_pt": LINE_WIDTH,
        "curve_role": {
            "scope_forecast_s0": "fine-scale scope-conditioned forecast",
            "scope_forecast_s1": "mid-scale scope-conditioned forecast",
            "scope_forecast_s2": "broad-scale scope-conditioned forecast",
            "final_fused_forecast": (
                "target-wise convex mixture of the three schematic forecasts"
            ),
        },
        "claim_boundary": (
            "deterministic schematic curves only; weights and values are not "
            "learned predictions or empirical allocation evidence"
        ),
        "source_values": str(csv_path),
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
