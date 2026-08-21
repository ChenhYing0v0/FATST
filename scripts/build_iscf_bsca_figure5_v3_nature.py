#!/usr/bin/env python3
"""Build the Nature-oriented author-review redesign of ISCF-BSCA Figure 5."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"


DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather"]
REGIONS = [
    "1–48",
    "49–96",
    "97–144",
    "145–192",
    "193–288",
    "289–336",
    "337–512",
    "513–720",
]
SCOPES = [1, 48, 144, 360, 720]
SCOPE_COLORS = {
    1: "#405F73",
    48: "#648395",
    144: "#91ABB5",
    360: "#9188AD",
    720: "#B47D96",
}
ALLOCATION_COLOR = "#6F70A8"
BSCA_COLOR = "#C96C4A"
NEUTRAL_DARK = "#30343A"
NEUTRAL_MID = "#6B7076"
NEUTRAL_LIGHT = "#DDE1E5"
TRACK_COLOR = "#ECEEF1"
ROW_BAND = "#F7F8FA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_section5_6_figure5_v2_positive_evidence_"
            "20260821/source_data"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_section5_6_figure5_v3_nature_redesign_"
            "20260821"
        ),
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def marker_area(gap_percent: float) -> float:
    """Map a non-negative MSE gap to a readable marker area in points squared."""
    if not np.isfinite(gap_percent) or gap_percent < 0:
        raise ValueError(f"invalid gap: {gap_percent}")
    return 78.0 + 19.0 * gap_percent


def summarize_scope_preference(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    expected = len(DATASETS) * len(REGIONS) * len(SCOPES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} scope rows, found {len(rows)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["future_region"])].append(row)

    summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for region in REGIONS:
            cell = grouped[(dataset, region)]
            if len(cell) != len(SCOPES):
                raise ValueError(f"incomplete scope cell for {dataset}/{region}")
            winners = [row for row in cell if row["regional_best"] == "1"]
            if len(winners) != 1:
                raise ValueError(
                    f"expected one preferred scope for {dataset}/{region}, "
                    f"found {len(winners)}"
                )
            winner = winners[0]
            gap = max(float(row["excess_mse_percent"]) for row in cell)
            summary.append(
                {
                    "dataset": dataset,
                    "future_region": region,
                    "preferred_scope": int(winner["scope"]),
                    "preferred_scope_mean_mse": winner["mean_mse"],
                    "best_to_worst_excess_mse_percent": f"{gap:.8f}",
                    "validation_rows": winner["validation_rows"],
                    "split": "validation",
                    "evidence_role": "aggregate_internal_diagnostic",
                }
            )
    return summary


def text_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.57 else NEUTRAL_DARK


def add_panel_label(
    axis: plt.Axes,
    label: str,
    x: float = -0.07,
    y: float = 1.10,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )


def draw_preference_map(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    x_lookup = {region: index for index, region in enumerate(REGIONS)}
    y_lookup = {dataset: len(DATASETS) - 1 - index for index, dataset in enumerate(DATASETS)}

    for row_index in range(len(DATASETS)):
        y = len(DATASETS) - 1 - row_index
        if row_index % 2 == 0:
            axis.axhspan(y - 0.48, y + 0.48, color=ROW_BAND, zorder=0)
    for boundary in np.arange(-0.5, len(REGIONS), 1.0):
        axis.axvline(boundary, color="#EEF0F2", linewidth=0.55, zorder=0)

    for row in rows:
        x = x_lookup[str(row["future_region"])]
        y = y_lookup[str(row["dataset"])]
        scope = int(row["preferred_scope"])
        gap = float(row["best_to_worst_excess_mse_percent"])
        color = SCOPE_COLORS[scope]
        axis.scatter(
            x,
            y,
            s=marker_area(gap),
            color=color,
            edgecolor="white",
            linewidth=1.15,
            alpha=0.96,
            zorder=3,
        )
        axis.text(
            x,
            y,
            str(scope),
            ha="center",
            va="center",
            fontsize=5.25,
            fontweight="bold",
            color=text_color(color),
            zorder=4,
        )

    axis.set_xlim(-0.55, len(REGIONS) - 0.45)
    axis.set_ylim(-0.60, len(DATASETS) - 0.40)
    axis.set_xticks(np.arange(len(REGIONS)), labels=REGIONS)
    axis.set_yticks(
        np.arange(len(DATASETS))[::-1],
        labels=DATASETS,
    )
    axis.set_xlabel("Future region", labelpad=5)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_title(
        "Preferred sharing scope varies across future regions",
        loc="left",
        fontsize=8.6,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=21,
    )
    axis.text(
        0.0,
        1.035,
        "Number = lowest-MSE scope; marker area = best-to-worst MSE gap",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL_MID,
    )

    legend_gaps = [2.0, 6.0, 10.0]
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=np.sqrt(marker_area(gap)),
            markerfacecolor="#C8CDD3",
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=f"{gap:.0f}%",
        )
        for gap in legend_gaps
    ]
    legend = axis.legend(
        handles=handles,
        title="Error gap",
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.185),
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(5.8)
    legend.get_title().set_color(NEUTRAL_MID)
    add_panel_label(axis, "a", x=-0.075, y=1.17)


def draw_effect_panel(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    color: str,
    title: str,
    subtitle: str,
    panel_label: str,
) -> None:
    labels = [row["dataset"] for row in rows]
    values = np.asarray(
        [float(row["mse_reduction_percent"]) for row in rows],
        dtype=float,
    )
    if labels != DATASETS + ["Mean"]:
        raise ValueError(f"unexpected effect-panel labels: {labels}")
    if not np.all(values > 0):
        raise ValueError(f"effect panel contains non-positive values: {values}")

    y_positions = np.arange(len(labels))[::-1]
    axis.axhspan(-0.42, 0.42, color="#F5F6F8", zorder=0)
    for index, (y, value) in enumerate(zip(y_positions, values)):
        is_mean = labels[index] == "Mean"
        line_width = 3.2 if is_mean else 2.45
        alpha = 1.0 if is_mean else 0.62
        axis.plot(
            [0, 7.0],
            [y, y],
            color=TRACK_COLOR,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [0, value],
            [y, y],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
        axis.scatter(
            value,
            y,
            s=31 if is_mean else 25,
            color=color,
            edgecolor="white",
            linewidth=0.75,
            alpha=1.0 if is_mean else 0.90,
            zorder=3,
        )
        axis.text(
            value + 0.13,
            y,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=6.2,
            color=color,
            fontweight="bold" if is_mean else "normal",
        )

    axis.set_yticks(y_positions, labels=labels)
    for tick in axis.get_yticklabels():
        if tick.get_text() == "Mean":
            tick.set_fontweight("bold")
    axis.set_ylim(-0.55, len(labels) - 0.20)
    axis.set_xlim(0, 7.0)
    axis.set_xticks([0, 2, 4, 6])
    axis.set_xlabel("MSE reduction (%)", labelpad=5)
    axis.tick_params(axis="both", length=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#AEB2B7")
    axis.spines["bottom"].set_linewidth(0.7)
    axis.set_title(
        title,
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        color=NEUTRAL_DARK,
        pad=18,
    )
    axis.text(
        0.0,
        1.025,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL_MID,
    )
    add_panel_label(axis, panel_label, x=-0.16, y=1.13)


def build_figure(
    output_dir: Path,
    preference_rows: list[dict[str, Any]],
    allocation_rows: list[dict[str, str]],
    bsca_rows: list[dict[str, str]],
) -> dict[str, Path]:
    configure_style()
    figure = plt.figure(figsize=(7.087, 4.331))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.06, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.12,
        top=0.88,
        hspace=0.57,
        wspace=0.38,
    )
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    draw_preference_map(axis_a, preference_rows)
    draw_effect_panel(
        axis_b,
        allocation_rows,
        ALLOCATION_COLOR,
        "Target-adaptive allocation",
        "Full model versus equal scope fusion",
        "b",
    )
    draw_effect_panel(
        axis_c,
        bsca_rows,
        BSCA_COLOR,
        "Balanced scope co-adaptation",
        "Full model versus prefix-only training",
        "c",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_5_scope_preference_component_gains"
    outputs = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    figure.savefig(outputs["svg"])
    normalize_svg(outputs["svg"])
    figure.savefig(outputs["pdf"])
    figure.savefig(outputs["png"], dpi=300)
    figure.savefig(outputs["tiff"], dpi=600)
    plt.close(figure)

    with Image.open(outputs["tiff"]) as image:
        image.save(outputs["tiff"], compression="tiff_lzw", dpi=(600, 600))
    return outputs


def main() -> None:
    args = parse_args()
    full_competence_rows = read_csv(
        args.source_dir / "panel_a_scope_competence.csv"
    )
    allocation_rows = read_csv(args.source_dir / "panel_b_allocation_gain.csv")
    bsca_rows = read_csv(args.source_dir / "panel_c_bsca_gain.csv")
    preference_rows = summarize_scope_preference(full_competence_rows)

    source_dir = args.output_dir / "source_data"
    write_csv(
        source_dir / "panel_a_scope_competence_full.csv",
        full_competence_rows,
    )
    write_csv(
        source_dir / "panel_a_scope_preference.csv",
        preference_rows,
    )
    write_csv(source_dir / "panel_b_allocation_gain.csv", allocation_rows)
    write_csv(source_dir / "panel_c_bsca_gain.csv", bsca_rows)
    outputs = build_figure(
        args.output_dir,
        preference_rows,
        allocation_rows,
        bsca_rows,
    )
    print(
        "Figure 5 v3 generated: "
        f"full_scope_rows={len(full_competence_rows)}, "
        f"preference_cells={len(preference_rows)}, "
        f"allocation_rows={len(allocation_rows)}, "
        f"bsca_rows={len(bsca_rows)}, "
        f"outputs={','.join(path.name for path in outputs.values())}"
    )


if __name__ == "__main__":
    main()
