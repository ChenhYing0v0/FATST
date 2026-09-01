"""Reproduce the submission-facing UVHF generalization figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data/figure7_decoder_transfer.csv"
OUTPUT = HERE / "outputs"

INK = "#27343B"
BLUE = "#5E7FAE"
CORAL = "#C66B6D"
PANEL_BACKGROUND = "#EEF0F5"
OUTLINE = "#6D7278"
AXIS_FLOOR = 0.20


def configure_style() -> None:
    """Configure publication-safe typography and export settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def read_rows() -> list[dict[str, str]]:
    """Read the frozen four-horizon mean MSE values."""
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: mpl.figure.Figure) -> None:
    """Export editable vector and high-resolution raster artifacts."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "figure_7_decoder_transfer"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_transfer() -> None:
    """Plot Original Decoder before UVHF for each backbone and dataset."""
    rows = read_rows()
    datasets = ["Weather", "ETTm1", "ETTm2", "Avg."]
    panels = [
        ("dlinear_style", "DLinear-style"),
        ("patchtst_style", "PatchTST-style"),
    ]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.70),
        sharey=True,
        constrained_layout=True,
    )
    width = 0.33
    positions = np.arange(len(datasets))

    for panel_index, ((backbone, title), axis) in enumerate(zip(panels, axes)):
        panel_rows = [row for row in rows if row["backbone"] == backbone]
        originals = [float(row["original_decoder_mse"]) for row in panel_rows]
        uvhf = [float(row["iscf_bsca_mse"]) for row in panel_rows]
        reductions = [
            float(row["relative_mse_reduction_percent"]) for row in panel_rows
        ]

        original_bars = axis.bar(
            positions - width / 2,
            np.asarray(originals) - AXIS_FLOOR,
            width,
            bottom=AXIS_FLOOR,
            color=BLUE,
            edgecolor="white",
            linewidth=0.9,
            hatch="//",
            label="Original Decoder",
            zorder=2,
        )
        uvhf_bars = axis.bar(
            positions + width / 2,
            np.asarray(uvhf) - AXIS_FLOOR,
            width,
            bottom=AXIS_FLOOR,
            color=CORAL,
            edgecolor="white",
            linewidth=0.9,
            hatch="xx",
            label="UVHF (MSD + BCA)",
            zorder=2,
        )
        for bar in [*original_bars, *uvhf_bars]:
            axis.add_patch(
                mpl.patches.Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    fill=False,
                    edgecolor=OUTLINE,
                    linewidth=0.45,
                    zorder=2.2,
                )
            )

        for index, (original, uvhf_value, reduction) in enumerate(
            zip(originals, uvhf, reductions)
        ):
            axis.annotate(
                "",
                xy=(index + width / 2, uvhf_value + 0.003),
                xytext=(index - width / 2, original + 0.004),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": INK,
                    "linewidth": 0.75,
                    "mutation_scale": 7.5,
                    "connectionstyle": "arc3,rad=-0.52",
                },
                annotation_clip=False,
            )
            axis.text(
                index,
                max(original, uvhf_value) + 0.014,
                f"−{reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=INK,
                fontweight="semibold",
            )

        axis.set_title(title, pad=6)
        axis.set_xticks(positions, datasets)
        axis.set_ylim(AXIS_FLOOR, 0.385)
        axis.set_yticks([0.20, 0.24, 0.28, 0.32, 0.36])
        axis.set_facecolor(PANEL_BACKGROUND)
        axis.grid(axis="y", color="white", linewidth=0.85, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(INK)
        axis.tick_params(colors=INK)
        axis.plot(
            (-0.013, 0.004),
            (-0.012, 0.010),
            transform=axis.transAxes,
            color=INK,
            linewidth=0.8,
            clip_on=False,
        )
        axis.plot(
            (0.000, 0.017),
            (-0.012, 0.010),
            transform=axis.transAxes,
            color=INK,
            linewidth=0.8,
            clip_on=False,
        )
        axis.text(
            0.018,
            0.975,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=9.6,
            fontweight="bold",
            va="top",
        )

    axes[0].set_ylabel("Four-horizon mean MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D3D6DC",
        framealpha=0.96,
    )
    save_figure(figure)
    plt.close(figure)


if __name__ == "__main__":
    configure_style()
    plot_transfer()
