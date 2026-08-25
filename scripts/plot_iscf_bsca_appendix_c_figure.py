#!/usr/bin/env python3
"""Render Appendix C Figure C1 from frozen validation prediction arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar"]
HORIZONS = [96, 192, 336, 720]
PREDICTION_COLOR = "#2B7777"
TARGET_COLOR = "#3D4850"
HORIZON_COLOR = "#C96C4A"
PREFIX_COLORS = ["#5B8F9B", "#729DA6", "#91AEB2", "#C2A383"]
GRID_COLOR = "#D9DEE2"
TICK_COLOR = "#5F686F"
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 188


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "analysis/iscf_bsca_appendix_c_prediction_export_20260825",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def normalize_svg(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
        encoding="utf-8",
    )


def load_sources(input_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_dir = input_dir / dataset
        npz_path = dataset_dir / "appendix_c_predictions.npz"
        metadata_path = dataset_dir / "metadata.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing Appendix C source for {dataset}")
        with np.load(npz_path, allow_pickle=False) as source:
            prediction = source["prediction"].astype(np.float64)
            target = source["ground_truth"].astype(np.float64)
            horizons = source["horizons"].astype(int).tolist()
            channel = int(source["channel"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if horizons != HORIZONS:
            raise ValueError(f"horizon contract mismatch for {dataset}: {horizons}")
        if prediction.shape != (2, 720) or target.shape != (2, 720):
            raise ValueError(
                f"unexpected prediction shape for {dataset}: "
                f"{prediction.shape}, {target.shape}"
            )
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f"non-finite prediction source for {dataset}")
        if metadata.get("ablation_checkpoint") is not False:
            raise ValueError(f"ablation checkpoint provenance for {dataset}")
        if metadata.get("test_labels_accessed") is not False:
            raise ValueError(f"test-label provenance for {dataset}")
        selected = metadata.get("selected", [])
        if len(selected) != 2:
            raise ValueError(f"selection metadata mismatch for {dataset}")
        sources[dataset] = {
            "prediction": prediction,
            "target": target,
            "channel": channel,
            "metadata": metadata,
        }
        for rank, selected_row in enumerate(selected, start=1):
            for step, (target_value, prediction_value) in enumerate(
                zip(target[rank - 1], prediction[rank - 1]), start=1
            ):
                csv_rows.append(
                    {
                        "dataset": dataset,
                        "sample_rank": rank,
                        "validation_window_index": selected_row[
                            "validation_window_index"
                        ],
                        "raw_forecast_origin": selected_row["raw_forecast_origin"],
                        "channel": channel,
                        "future_step": step,
                        "ground_truth": float(target_value),
                        "prediction": float(prediction_value),
                    }
                )
    return sources, csv_rows


def robust_limits(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([target.ravel(), prediction.ravel()])
    low, high = np.nanpercentile(values, [0.5, 99.5])
    full_low, full_high = float(values.min()), float(values.max())
    low = min(float(low), full_low)
    high = max(float(high), full_high)
    span = max(high - low, 1e-8)
    margin = 0.08 * span
    return low - margin, high + margin


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_figure(
    sources: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    configure_style()
    width_in = FIGURE_WIDTH_MM / 25.4
    height_in = FIGURE_HEIGHT_MM / 25.4
    figure = plt.figure(figsize=(width_in, height_in), facecolor="white")
    grid = figure.add_gridspec(
        nrows=len(DATASETS),
        ncols=2,
        left=0.105,
        right=0.985,
        top=0.850,
        bottom=0.075,
        hspace=0.34,
        wspace=0.12,
    )
    axes = np.empty((len(DATASETS), 2), dtype=object)
    x = np.arange(1, 721)
    for row_index, dataset in enumerate(DATASETS):
        source = sources[dataset]
        target = source["target"]
        prediction = source["prediction"]
        y_low, y_high = robust_limits(target, prediction)
        for column in range(2):
            ax = figure.add_subplot(grid[row_index, column])
            axes[row_index, column] = ax
            ax.plot(
                x,
                target[column],
                color=TARGET_COLOR,
                linewidth=0.9,
                solid_capstyle="round",
                label="Ground truth",
                zorder=3,
            )
            ax.plot(
                x,
                prediction[column],
                color=PREDICTION_COLOR,
                linewidth=1.05,
                solid_capstyle="round",
                label="HoriScope",
                zorder=4,
            )
            # Keep the prefix boundaries visible without repeating the strong
            # dashed markers used in the previous draft. The shared ruler above
            # the grid carries the explicit horizon labels.
            for horizon in HORIZONS[:-1]:
                ax.axvline(
                    horizon,
                    color="#B9C5C8",
                    linewidth=0.42,
                    linestyle="-",
                    alpha=0.62,
                    zorder=0,
                )
            ax.set_xlim(1, 720)
            ax.set_ylim(y_low, y_high)
            ax.set_yticks(np.linspace(y_low, y_high, 3))
            ax.tick_params(
                axis="both",
                colors=TICK_COLOR,
                width=0.55,
                length=2.2,
                pad=1.6,
            )
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.65)
            ax.set_axisbelow(True)
            ax.spines["left"].set_color("#A7B0B6")
            ax.spines["bottom"].set_color("#A7B0B6")
            if column == 1:
                ax.set_yticklabels([])
            if row_index < len(DATASETS) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticks(HORIZONS)
                ax.set_xticklabels([str(h) for h in HORIZONS])
            if column == 0:
                ax.text(
                    -0.20,
                    0.5,
                    dataset,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.7,
                    fontweight="bold",
                    color="#303940",
                )
            if row_index == 0:
                ax.text(
                    0.02,
                    1.08,
                    f"sample {column + 1}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.4,
                    fontweight="bold",
                    color="#303940",
                )
    axes[-1, 0].set_xlabel("Future step", color=TICK_COLOR, labelpad=3.5)
    axes[-1, 1].set_xlabel("Future step", color=TICK_COLOR, labelpad=3.5)
    figure.text(
        0.028,
        0.52,
        "Value",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.8,
        color=TICK_COLOR,
    )
    figure.text(
        0.105,
        0.965,
        "Representative validation trajectories",
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#283238",
    )
    figure.text(
        0.105,
        0.932,
        "Nested prefixes returned by one unified model",
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#6C777C",
        style="italic",
    )

    # Place one ruler directly above each sample column so that the prefix
    # lengths share the same horizontal geometry as the traces below.
    for column, show_labels in enumerate((True, False)):
        panel_position = axes[0, column].get_position()
        ruler = figure.add_axes(
            [panel_position.x0, 0.886, panel_position.width, 0.039]
        )
        ruler.set_xlim(1, 720)
        ruler.set_ylim(-0.35, 3.8)
        ruler.axis("off")
        for level, (horizon, color) in enumerate(zip(HORIZONS, PREFIX_COLORS)):
            y = 3.15 - level
            ruler.plot(
                [1, horizon],
                [y, y],
                color=color,
                linewidth=2.0 if horizon < 720 else 1.8,
                solid_capstyle="round",
                zorder=2,
            )
            ruler.plot(
                horizon,
                y,
                marker="o",
                markersize=2.7,
                color=HORIZON_COLOR if horizon < 720 else "#6C777C",
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
            if show_labels:
                ruler.text(
                    horizon,
                    y + 0.28,
                    f"H={horizon}",
                    ha="center" if horizon < 650 else "right",
                    va="bottom",
                    fontsize=5.0,
                    color="#4D5A60",
                )
        ruler.plot(
            [1, 720],
            [-0.04, -0.04],
            color="#D4DCDD",
            linewidth=0.55,
            solid_capstyle="round",
            zorder=1,
        )
    legend_handles = [
        Line2D([0], [0], color=TARGET_COLOR, linewidth=1.0, label="Ground truth"),
        Line2D([0], [0], color=PREDICTION_COLOR, linewidth=1.15, label="HoriScope"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
        ncol=2,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure_c1_varied_horizon_forecasts"
    figure.savefig(f"{base}.svg", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(
        f"{base}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    normalize_svg(base.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    sources, csv_rows = load_sources(args.input_dir)
    write_source_csv(args.output_dir / "figure_c1_source_data.csv", csv_rows)
    render_figure(sources, args.output_dir)
    print(f"wrote Figure C1 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
