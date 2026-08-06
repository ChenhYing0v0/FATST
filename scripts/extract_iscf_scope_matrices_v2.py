#!/usr/bin/env python3
"""Export the five scope-matrix glyphs from the ISCF v2 concept figure.

The cell patterns, scope colors, edge colors, and line hierarchy are exact
reuses of ``plot_iscf_architecture_concept_v2.py``. No empirical or learned
matrix values are represented.
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
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from plot_iscf_architecture_concept_v2 import (
    SCOPE_COLORS,
    SCOPE_EDGES,
    SCOPE_LABELS,
    blend_with_white,
)


FIGURE_WIDTH_MM = 46.0
FIGURE_HEIGHT_MM = 30.0
MATRIX_WIDTH_MM = 40.0
MATRIX_HEIGHT_MM = 22.3
ROWS = 4
COLUMNS = 6


def configure_style() -> None:
    """Configure vector-safe publication exports."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "none",
            "savefig.transparent": True,
        }
    )


def build_scope_matrix(scope_index: int) -> Figure:
    """Build one standalone scope-matrix glyph using the v2 visual contract."""
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        facecolor="none",
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_xlim(0.0, FIGURE_WIDTH_MM)
    axis.set_ylim(0.0, FIGURE_HEIGHT_MM)
    axis.set_axis_off()
    axis.patch.set_alpha(0.0)

    x0 = (FIGURE_WIDTH_MM - MATRIX_WIDTH_MM) / 2.0
    y0 = (FIGURE_HEIGHT_MM - MATRIX_HEIGHT_MM) / 2.0
    cell_width = MATRIX_WIDTH_MM / COLUMNS
    cell_height = MATRIX_HEIGHT_MM / ROWS

    for row_index in range(ROWS):
        for column_index in range(COLUMNS):
            strength = 0.28 + 0.62 * (
                0.5
                + 0.5
                * np.sin(
                    (scope_index + 1) * (row_index + 1) + column_index
                )
            )
            axis.add_patch(
                Rectangle(
                    (
                        x0 + column_index * cell_width,
                        y0 + row_index * cell_height,
                    ),
                    cell_width,
                    cell_height,
                    facecolor=blend_with_white(
                        SCOPE_COLORS[scope_index],
                        strength,
                    ),
                    edgecolor="white",
                    linewidth=0.75,
                    zorder=2,
                )
            )

    axis.add_patch(
        Rectangle(
            (x0, y0),
            MATRIX_WIDTH_MM,
            MATRIX_HEIGHT_MM,
            facecolor="none",
            edgecolor=SCOPE_EDGES[scope_index],
            linewidth=2.25,
            zorder=3,
        )
    )
    return figure


def normalize_svg(path: Path) -> None:
    """Remove renderer-introduced trailing whitespace."""
    svg_text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def export_scope_matrix(
    figure: Figure,
    output_dir: Path,
    scope_label: str,
) -> dict[str, str]:
    """Export one scope matrix in transparent and publication formats."""
    scope_token = scope_label.replace("=", "").replace(" ", "")
    stem = output_dir / f"scope_matrix_{scope_token}"
    paths = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "tiff": stem.with_suffix(".tiff"),
    }
    figure.savefig(paths["png"], dpi=600, transparent=True)
    figure.savefig(paths["svg"], transparent=True)
    normalize_svg(paths["svg"])
    figure.savefig(paths["pdf"], transparent=True)
    figure.savefig(
        paths["tiff"],
        dpi=600,
        facecolor="white",
        transparent=False,
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
        default=Path("analysis/iscf_scope_matrix_assets_v2_20260806"),
    )
    return parser.parse_args()


def main() -> None:
    """Export all five scope matrices and write a provenance manifest."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    outputs: dict[str, dict[str, str]] = {}
    for scope_index, scope_label in enumerate(SCOPE_LABELS):
        outputs[scope_label] = export_scope_matrix(
            build_scope_matrix(scope_index),
            args.output_dir,
            scope_label,
        )

    manifest = {
        "asset_id": "iscf_scope_matrix_assets_v2",
        "status": "extracted_component_assets",
        "backend": "Python/matplotlib",
        "source_figure": "figure_iscf_architecture_concept_v2",
        "reuse_level": "exact visual-logic reuse",
        "empirical_data_used": False,
        "learned_values_shown": False,
        "scope_labels": list(SCOPE_LABELS),
        "matrix_shape": [ROWS, COLUMNS],
        "primary_format": "transparent PNG at 600 dpi",
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

