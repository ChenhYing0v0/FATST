from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
SOURCE = Path(__file__).resolve().parent / "source_data"

EFFICIENCY = (
    ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "efficiency_accuracy_memory_storage_20260817/efficiency_system_macro_results.csv"
)
TRANSFER = (
    ROOT
    / "analysis/iscf_bsca_paper_experiment_consolidation_20260731"
    / "decoder_transfer_three_dataset_scope_20260816"
    / "framework_portability_dataset_means.csv"
)

TEAL = "#187B74"
BLUE = "#4B78A8"
PURPLE = "#8A83B8"
GRAY = "#A8B0B8"
INK = "#27343B"
GRID = "#E4E8EA"
ORANGE = "#D7652C"
EFFICIENCY_COLORS = {
    "ISCF-BSCA": "#C94F3D",
    "TimeAlign": "#3B78B4",
    "QDF": "#8C6BB1",
    "AMD": "#D69E2E",
    "DLinear": "#6BAF45",
    "iTransformer": "#7E8AA2",
    "PatchTST": "#2A9D8F",
    "TimeMixer": "#56B4C2",
}
ARCHITECTURE_EQUIVALENT = {
    "DLinear",
    "iTransformer",
    "PatchTST",
    "TimeMixer",
}


def configure_style() -> None:
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(
        OUT / f"{stem}.tiff",
        bbox_inches="tight",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_efficiency() -> None:
    raw_rows = read_rows(EFFICIENCY)
    if len(raw_rows) != 9 or {row["system"] for row in raw_rows} != {
        "ISCF-BSCA",
        "TimeAlign",
        "QDF",
        "AMD",
        "SimpleTM",
        "DLinear",
        "iTransformer",
        "PatchTST",
        "TimeMixer",
    }:
        raise RuntimeError("expected the frozen nine-system efficiency source")
    efficiency = {row["system"]: row for row in raw_rows}
    order = [
        "ISCF-BSCA",
        "TimeAlign",
        "QDF",
        "AMD",
        "DLinear",
        "iTransformer",
        "PatchTST",
        "TimeMixer",
    ]
    plot_rows: list[dict[str, object]] = []
    for system in order:
        row = efficiency[system]
        plot_rows.append(
            {
                "system": system,
                "resident_model_count": int(row["model_count"]),
                "main_i_macro_mse": float(row["main_i_mse"]),
                "checkpoint_storage_mib": float(row["checkpoint_storage_mib"]),
                "peak_inference_memory_mib": float(
                    row["peak_inference_memory_mib"]
                ),
                "resource_evidence_role": (
                    "official_architecture_equivalent"
                    if system in ARCHITECTURE_EQUIVALENT
                    else "actual_trained_checkpoint"
                ),
            }
        )
    storage_values = np.asarray(
        [float(row["checkpoint_storage_mib"]) for row in plot_rows],
        dtype=np.float64,
    )
    if np.any(storage_values <= 0):
        raise RuntimeError("log-scale checkpoint storage must be strictly positive")
    write_rows(SOURCE / "figure6_accuracy_system_cost.csv", plot_rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.65), constrained_layout=True)
    bubble_scale = 5.2
    for row in plot_rows:
        memory = float(row["peak_inference_memory_mib"])
        system = str(row["system"])
        ax.scatter(
            row["checkpoint_storage_mib"],
            row["main_i_macro_mse"],
            s=memory * bubble_scale,
            color=EFFICIENCY_COLORS[system],
            edgecolor="#FFFFFF" if system != "ISCF-BSCA" else "#7A241C",
            linewidth=1.1 if system != "ISCF-BSCA" else 1.5,
            alpha=0.74 if system != "ISCF-BSCA" else 0.90,
            zorder=3,
        )
        ax.scatter(
            row["checkpoint_storage_mib"],
            row["main_i_macro_mse"],
            s=5.0,
            color="white",
            linewidth=0,
            zorder=4,
        )

    label_offsets = {
        "ISCF-BSCA": (13, 1),
        "TimeAlign": (13, -2),
        "QDF": (-13, 15),
        "AMD": (-13, 15),
        "DLinear": (13, -7),
        "iTransformer": (13, 15),
        "PatchTST": (-14, -19),
        "TimeMixer": (15, -6),
    }
    label_alignment = {
        "ISCF-BSCA": "left",
        "TimeAlign": "left",
        "QDF": "right",
        "AMD": "right",
        "DLinear": "left",
        "iTransformer": "left",
        "PatchTST": "right",
        "TimeMixer": "left",
    }
    for row in plot_rows:
        system = str(row["system"])
        display_name = (
            r"HoriScope$^{\mathrm{ours}}$"
            if system == "ISCF-BSCA"
            else system
        )
        ax.annotate(
            f"{display_name}\n"
            f"{float(row['checkpoint_storage_mib']):.1f} MiB storage · "
            f"{float(row['peak_inference_memory_mib']):.1f} MiB peak",
            (row["checkpoint_storage_mib"], row["main_i_macro_mse"]),
            xytext=label_offsets[system],
            textcoords="offset points",
            color=EFFICIENCY_COLORS[system],
            fontsize=7.2,
            fontweight="semibold",
            ha=label_alignment[system],
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": EFFICIENCY_COLORS[system],
                "alpha": 0.48,
                "lw": 0.55,
            },
            zorder=5,
        )

    legend_values = (25, 100, 225)
    legend_ax = ax.inset_axes([0.745, 0.035, 0.24, 0.20], zorder=8)
    legend_ax.set_facecolor((1.0, 1.0, 1.0, 0.95))
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_color("#D5DADD")
        spine.set_linewidth(0.8)
    legend_ax.text(
        0.5,
        0.86,
        "Peak memory (MiB)",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="semibold",
        color=INK,
    )
    for x_position, value in zip((0.17, 0.50, 0.83), legend_values):
        legend_ax.scatter(
            x_position,
            0.36,
            s=value * bubble_scale,
            color="#BCC4C9",
            edgecolor="white",
            linewidth=0.9,
            alpha=0.78,
            clip_on=False,
            zorder=1,
        )
        legend_ax.text(
            x_position,
            0.36,
            str(value),
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="semibold",
            color=INK,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Four-horizon checkpoint storage (MiB, log scale)")
    ax.set_ylabel("MSE")
    ax.set_title("Accuracy-storage trade-off for four-horizon services", pad=7)
    ax.set_xlim(2.4, 330)
    ax.set_ylim(0.255, 0.306)
    ax.set_xticks([3, 10, 30, 100, 300], labels=["3", "10", "30", "100", "300"])
    ax.set_yticks(np.arange(0.26, 0.306, 0.01))
    ax.grid(True, which="major", color=GRID, linewidth=0.68, zorder=0)
    ax.grid(True, which="minor", axis="x", color=GRID, linewidth=0.38, alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.text(
        0.012,
        0.018,
        "Lower-left is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#6A747A",
    )
    save_figure(fig, "figure_6_accuracy_system_cost")
    plt.close(fig)


def plot_transfer() -> None:
    raw = read_rows(TRANSFER)
    datasets = ["Weather", "ETTm1", "ETTm2", "Avg."]
    panels = [
        ("dlinear_style", "DLinear-style"),
        ("patchtst_style", "PatchTST-style"),
    ]
    plot_rows: list[dict[str, object]] = []
    for backbone, _ in panels:
        original_id = f"{backbone.removesuffix('_style')}_original"
        ours_id = f"{backbone.removesuffix('_style')}_iscf_bsca"
        for dataset in datasets:
            original = next(
                float(row["mean_mse"])
                for row in raw
                if row["backbone"] == backbone
                and row["arm_id"] == original_id
                and row["dataset"] == dataset
            )
            ours = next(
                float(row["mean_mse"])
                for row in raw
                if row["backbone"] == backbone
                and row["arm_id"] == ours_id
                and row["dataset"] == dataset
            )
            plot_rows.append(
                {
                    "backbone": backbone,
                    "dataset": dataset,
                    "original_decoder_mse": original,
                    "iscf_bsca_mse": ours,
                    "relative_mse_reduction_percent": 100.0 * (original - ours) / original,
                }
            )
    write_rows(SOURCE / "figure7_decoder_transfer.csv", plot_rows)

    coral = "#C66B6D"
    blue = "#5E7FAE"
    panel_background = "#EEF0F5"
    outline = "#6D7278"
    axis_floor = 0.20
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.85, 2.70),
        sharey=True,
        constrained_layout=True,
    )
    width = 0.33
    x = np.arange(len(datasets))
    for panel_index, ((backbone, title), ax) in enumerate(zip(panels, axes)):
        rows = [row for row in plot_rows if row["backbone"] == backbone]
        originals = [float(row["original_decoder_mse"]) for row in rows]
        ours = [float(row["iscf_bsca_mse"]) for row in rows]
        reductions = [float(row["relative_mse_reduction_percent"]) for row in rows]
        ours_bars = ax.bar(
            x - width / 2,
            np.asarray(ours) - axis_floor,
            width,
            bottom=axis_floor,
            color=coral,
            edgecolor="white",
            linewidth=0.9,
            hatch="xx",
            label="HoriScope",
            zorder=2,
        )
        original_bars = ax.bar(
            x + width / 2,
            np.asarray(originals) - axis_floor,
            width,
            bottom=axis_floor,
            color=blue,
            edgecolor="white",
            linewidth=0.9,
            hatch="//",
            label="Original Decoder",
            zorder=2,
        )
        for bar in [*ours_bars, *original_bars]:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (bar.get_x(), bar.get_y()),
                    bar.get_width(),
                    bar.get_height(),
                    fill=False,
                    edgecolor=outline,
                    linewidth=0.45,
                    zorder=2.2,
                )
            )
        for index, (original, ours_value, reduction) in enumerate(
            zip(originals, ours, reductions)
        ):
            top = max(original, ours_value)
            ax.annotate(
                "",
                xy=(index - width / 2, ours_value + 0.003),
                xytext=(index + width / 2, original + 0.004),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": INK,
                    "linewidth": 0.75,
                    "mutation_scale": 7.5,
                    "connectionstyle": "arc3,rad=0.52",
                },
                annotation_clip=False,
            )
            ax.text(
                index,
                top + 0.014,
                f"−{reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=INK,
                fontweight="semibold",
            )
        ax.set_title(title, pad=6)
        ax.set_xticks(x, datasets)
        ax.set_ylim(axis_floor, 0.385)
        ax.set_yticks([0.20, 0.24, 0.28, 0.32, 0.36])
        ax.set_facecolor(panel_background)
        ax.grid(axis="y", color="white", linewidth=0.85, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(INK)
        ax.tick_params(colors=INK)
        ax.plot(
            (-0.013, 0.004),
            (-0.012, 0.010),
            transform=ax.transAxes,
            color=INK,
            linewidth=0.8,
            clip_on=False,
        )
        ax.plot(
            (0.000, 0.017),
            (-0.012, 0.010),
            transform=ax.transAxes,
            color=INK,
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            0.018,
            0.975,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=9.6,
            fontweight="bold",
            va="top",
        )
    axes[0].set_ylabel("Four-horizon mean MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D3D6DC",
        framealpha=0.96,
    )
    save_figure(fig, "figure_7_decoder_transfer")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Section 5 manuscript figures.")
    parser.add_argument("--figure", choices=("6", "7", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    if args.figure in {"6", "all"}:
        plot_efficiency()
    if args.figure in {"7", "all"}:
        plot_transfer()


if __name__ == "__main__":
    main()
