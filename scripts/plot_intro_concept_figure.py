#!/usr/bin/env python3
"""Render the constructed-data Introduction problem illustration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


DOUBLE_COLUMN_WIDTH = 183.0 / 25.4
FIGURE_HEIGHT = 2.90

COLORS = {
    "history": "#8E9298",
    "neutral": "#53575C",
    "guide": "#9CA1A8",
}
HORIZON_COLORS = {
    "short": "#A85F78",
    "medium": "#6A74A5",
    "long": "#3E7A89",
}
SHARING_COLORS = {
    "fine": "#B55D42",
    "medium": "#4D77A8",
    "broad": "#2E7D68",
}


def configure_style() -> None:
    """Configure a compact publication-oriented matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 6.6,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def constructed_forecasts() -> dict[str, np.ndarray]:
    """Return deterministic curves used only for conceptual illustration."""
    history_x = np.linspace(-30.0, 0.0, 91)
    history = (
        0.18 * np.sin((history_x + 30.0) / 3.2)
        + 0.07 * np.cos((history_x + 30.0) / 1.7)
        + 0.004 * (history_x + 30.0)
    )
    future_x = np.linspace(0.0, 100.0, 401)
    base = (
        0.12
        + 0.30 * np.sin(future_x / 8.2)
        + 0.09 * np.sin(future_x / 3.5)
    )
    history = history - history[-1] + base[0]
    short = (
        base
        + 0.24 * (1.0 - np.exp(-future_x / 10.0))
        + 0.025 * np.sin(future_x / 2.8)
    )
    medium = (
        base
        - 0.17 * (1.0 - np.exp(-future_x / 13.0))
        + 0.025 * (np.cos(future_x / 4.1) - 1.0)
    )
    long = base + 0.025 * np.sin(future_x / 5.6)
    return {
        "history_x": history_x,
        "history": history,
        "future_x": future_x,
        "short": short,
        "medium": medium,
        "long": long,
    }


def constructed_risks() -> dict[str, np.ndarray]:
    """Return crossing risk curves with region-dependent minima."""
    region_x = np.linspace(0.0, 1.0, 401)
    fine = 0.22 + 0.60 * region_x + 0.04 * np.sin(2.0 * np.pi * region_x)
    medium = 0.34 + 0.85 * np.square(region_x - 0.50)
    broad = 0.78 - 0.55 * region_x + 0.02 * np.cos(
        2.0 * np.pi * region_x
    )
    return {
        "region_x": region_x,
        "fine": fine,
        "medium": medium,
        "broad": broad,
    }


def plot_prefix_panel(axis: plt.Axes) -> None:
    """Illustrate disagreement at the same future step."""
    curves = constructed_forecasts()
    future_x = curves["future_x"]
    horizons = {
        "short": 42.0,
        "medium": 70.0,
        "long": 100.0,
    }
    labels = {
        "short": r"short $H_1$",
        "medium": r"medium $H_2$",
        "long": r"long $H_3$",
    }
    markers = {"short": "o", "medium": "s", "long": "D"}

    axis.axvspan(-30.0, 0.0, color="#F2F2F2", zorder=-3)
    axis.axvspan(0.0, horizons["short"], color="#F1F5FA", zorder=-3)
    axis.plot(
        curves["history_x"],
        curves["history"],
        color=COLORS["history"],
        linewidth=1.25,
        label="Observed history",
        zorder=2,
    )
    axis.text(
        -26.5,
        0.51,
        "observed\nhistory",
        color="#72767C",
        fontsize=5.6,
        ha="left",
        va="top",
    )

    for key in ("long", "medium", "short"):
        horizon = horizons[key]
        mask = future_x <= horizon
        axis.plot(
            future_x[mask],
            curves[key][mask],
            color=HORIZON_COLORS[key],
            linewidth=1.30,
            label=labels[key],
            zorder={"long": 3, "medium": 4, "short": 5}[key],
        )
        end_index = int(np.flatnonzero(mask)[-1])
        axis.scatter(
            [future_x[end_index]],
            [curves[key][end_index]],
            marker=markers[key],
            s=15,
            color=HORIZON_COLORS[key],
            edgecolor="white",
            linewidth=0.45,
            zorder=7,
        )

    target_step = 26.0
    target_index = int(np.argmin(np.abs(future_x - target_step)))
    target_values = [
        curves[key][target_index] for key in ("short", "medium", "long")
    ]
    axis.axvline(
        target_step,
        color=COLORS["guide"],
        linestyle=(0, (2, 2)),
        linewidth=0.75,
        zorder=1,
    )
    for key, value in zip(("short", "medium", "long"), target_values):
        axis.scatter(
            [target_step],
            [value],
            marker=markers[key],
            s=20,
            color=HORIZON_COLORS[key],
            edgecolor="white",
            linewidth=0.55,
            zorder=8,
        )
    value_min = min(target_values)
    value_max = max(target_values)
    axis.annotate(
        "",
        xy=(target_step + 4.2, value_max),
        xytext=(target_step + 4.2, value_min),
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["neutral"],
            "linewidth": 0.8,
            "mutation_scale": 8,
        },
    )
    axis.text(
        target_step + 6.5,
        0.5 * (value_min + value_max),
        "different values\nat the same step",
        fontsize=5.6,
        color=COLORS["neutral"],
        va="center",
        ha="left",
    )

    axis.axvline(
        0.0,
        color="#777B80",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    axis.text(
        20.5,
        -0.58,
        "shared future steps",
        fontsize=5.4,
        color="#657181",
        ha="center",
        va="bottom",
    )
    axis.set_xlim(-30.0, 103.0)
    axis.set_ylim(-0.62, 0.66)
    axis.set_xticks(
        [0.0, target_step, horizons["short"], horizons["medium"], 100.0],
        ["origin", r"$\tau^\star$", r"$H_1$", r"$H_2$", r"$H_3$"],
    )
    axis.set_yticks([])
    axis.set_ylabel("Forecast value")
    axis.set_title(
        "Horizon-specific predictions need not agree",
        loc="left",
        pad=7,
    )
    handles, legend_labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(legend_labels, handles))
    ordered_labels = [
        "Observed history",
        labels["short"],
        labels["medium"],
        labels["long"],
    ]
    axis.legend(
        [handle_by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper left",
        bbox_to_anchor=(0.36, 1.00),
        ncol=2,
        columnspacing=0.9,
        handlelength=2.0,
        borderaxespad=0.0,
    )
    axis.text(
        -0.13,
        1.08,
        "a",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )


def plot_sharing_panel(axis: plt.Axes) -> None:
    """Illustrate future-region changes in preferred sharing extent."""
    risks = constructed_risks()
    region_x = risks["region_x"]
    series = (
        ("fine", "Fine sharing"),
        ("medium", "Intermediate sharing"),
        ("broad", "Broad sharing"),
    )
    regions = (
        (0.00, 1.0 / 3.0, "Early", "fine"),
        (1.0 / 3.0, 2.0 / 3.0, "Middle", "medium"),
        (2.0 / 3.0, 1.00, "Late", "broad"),
    )
    preferred_labels = {
        "fine": "Fine preferred",
        "medium": "Intermediate preferred",
        "broad": "Broad preferred",
    }

    for start, end, _, winner in regions:
        axis.axvspan(
            start,
            end,
            color=SHARING_COLORS[winner],
            alpha=0.055,
            zorder=-3,
        )
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        axis.axvline(
            boundary,
            color="#B8BCC2",
            linestyle=(0, (2, 2)),
            linewidth=0.65,
            zorder=-1,
        )

    for key, label in series:
        axis.plot(
            region_x,
            risks[key],
            color=SHARING_COLORS[key],
            linewidth=1.45,
            label=label,
            zorder=3,
        )

    ribbon_bottom = 0.135
    ribbon_top = 0.185
    for start, end, _region_label, winner in regions:
        center = 0.5 * (start + end)
        axis.fill_between(
            [start + 0.008, end - 0.008],
            ribbon_bottom,
            ribbon_top,
            color=SHARING_COLORS[winner],
            linewidth=0.0,
            zorder=4,
        )
        axis.text(
            center,
            0.5 * (ribbon_bottom + ribbon_top),
            preferred_labels[winner],
            color="white",
            fontsize=5.0 if winner == "medium" else 5.2,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        winner_index = int(np.argmin(np.abs(region_x - center)))
        axis.scatter(
            [center],
            [risks[winner][winner_index]],
            s=20,
            color=SHARING_COLORS[winner],
            edgecolor="white",
            linewidth=0.55,
            zorder=6,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.08, 0.88)
    axis.set_xticks(
        [1.0 / 6.0, 0.5, 5.0 / 6.0],
        [region[2] for region in regions],
    )
    axis.tick_params(axis="x", length=0, pad=2.5)
    axis.set_yticks([])
    axis.set_xlabel("Future region", labelpad=7)
    axis.set_ylabel("Illustrative risk\n(lower is better)")
    axis.set_title(
        "Preferred sharing extent can change by region",
        loc="left",
        pad=7,
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.51, 1.00),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    axis.text(
        -0.13,
        1.08,
        "b",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )


def save_figure(figure: plt.Figure, output_dir: Path) -> dict[str, str]:
    """Save an exact-width publication bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_intro_conceptual_problem"
    paths = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    figure.savefig(paths["svg"])
    figure.savefig(paths["pdf"])
    figure.savefig(paths["png"], dpi=300)
    figure.savefig(
        paths["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render the constructed Introduction problem illustration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_intro_concept_figure_20260730"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Render the figure and write its machine-readable manifest."""
    args = parse_args()
    configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"width_ratios": (1.08, 1.0)},
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.19,
        top=0.84,
        wspace=0.20,
    )
    plot_prefix_panel(axes[0])
    plot_sharing_panel(axes[1])
    figure.text(
        0.985,
        0.025,
        "Conceptual illustration with constructed curves; not empirical data",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#777B80",
        fontstyle="italic",
    )
    outputs = save_figure(figure, args.output_dir)
    plt.close(figure)

    manifest = {
        "figure_id": "figure_intro_conceptual_problem",
        "status": "approved_for_manuscript_draft",
        "approval_date": "2026-07-30",
        "paper_asset_directory": "paper-figures",
        "backend": "Python/matplotlib",
        "final_width_mm": 183.0,
        "data_role": "constructed conceptual illustration",
        "empirical_data_used": False,
        "panels": {
            "a": (
                "constructed horizon-specific predictions disagree at the "
                "same future step"
            ),
            "b": (
                "constructed fine, intermediate, and broad sharing-risk "
                "curves have different region-wise minima"
            ),
        },
        "claim_boundary": [
            "illustrates definitions only",
            "does not establish empirical existence",
            "does not report model effectiveness",
            "real-data evidence is deferred to Section 3",
        ],
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
