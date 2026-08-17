from __future__ import annotations

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
    / "efficiency_accuracy_params_epoch_20260817/efficiency_system_macro_results.csv"
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
    efficiency = {
        row["system"]: row
        for row in read_rows(EFFICIENCY)
        if row["system"] in {"ISCF-BSCA", "TimeAlign", "QDF"}
    }
    order = ["ISCF-BSCA", "TimeAlign", "QDF"]
    colors = {"ISCF-BSCA": TEAL, "TimeAlign": BLUE, "QDF": PURPLE}
    plot_rows: list[dict[str, object]] = []
    for system in order:
        row = efficiency[system]
        plot_rows.append(
            {
                "system": system,
                "trained_model_count": int(row["model_count"]),
                "deployed_parameters_million": float(
                    row["total_parameters_million"]
                ),
                "main_i_macro_mse": float(row["main_i_mse"]),
                "one_epoch_cycle_seconds": float(
                    row["one_epoch_cycle_seconds"]
                ),
            }
        )
    write_rows(SOURCE / "figure6_accuracy_system_cost.csv", plot_rows)

    fig, ax = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    max_seconds = max(float(row["one_epoch_cycle_seconds"]) for row in plot_rows)
    for row in plot_rows:
        seconds = float(row["one_epoch_cycle_seconds"])
        area = 130.0 + 620.0 * seconds / max_seconds
        ax.scatter(
            row["deployed_parameters_million"],
            row["main_i_macro_mse"],
            s=area,
            color=colors[str(row["system"])],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
            zorder=3,
        )

    label_offsets = {
        "ISCF-BSCA": (18, -3),
        "TimeAlign": (-13, -2),
        "QDF": (7, -2),
    }
    label_alignment = {"ISCF-BSCA": "left", "TimeAlign": "right", "QDF": "left"}
    for row in plot_rows:
        system = str(row["system"])
        ax.annotate(
            f"{system}\n{row['trained_model_count']} model{'s' if row['trained_model_count'] != 1 else ''} · "
            f"{float(row['one_epoch_cycle_seconds']):.1f} s/epoch",
            (row["deployed_parameters_million"], row["main_i_macro_mse"]),
            xytext=label_offsets[system],
            textcoords="offset points",
            color=colors[system],
            fontsize=7.5,
            fontweight="semibold",
            ha=label_alignment[system],
            va="center",
        )

    ax.text(
        0.985,
        0.985,
        "Bubble area scales with one-epoch cycle time",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#6A747A",
    )
    ax.set_xlabel("Deployed parameters per dataset (M)")
    ax.set_ylabel("Main-I macro MSE (lower is better)")
    ax.set_xlim(0.0, 12.2)
    ax.set_ylim(0.253, 0.293)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.grid(True, color=GRID, linewidth=0.65, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.annotate(
        "better",
        xy=(0.8, 0.2548),
        xytext=(2.0, 0.2574),
        arrowprops={"arrowstyle": "->", "color": "#6A747A", "lw": 0.8},
        color="#6A747A",
        fontsize=7,
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

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharey=True, constrained_layout=True)
    width = 0.34
    x = np.arange(len(datasets))
    for panel_index, ((backbone, title), ax) in enumerate(zip(panels, axes)):
        rows = [row for row in plot_rows if row["backbone"] == backbone]
        originals = [float(row["original_decoder_mse"]) for row in rows]
        ours = [float(row["iscf_bsca_mse"]) for row in rows]
        reductions = [float(row["relative_mse_reduction_percent"]) for row in rows]
        ax.bar(
            x - width / 2,
            originals,
            width,
            color=GRAY,
            edgecolor="white",
            linewidth=0.8,
            label="Original Decoder",
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            ours,
            width,
            color=TEAL,
            edgecolor="white",
            linewidth=0.8,
            label="ISCF-BSCA",
            zorder=2,
        )
        for index, (left, right, reduction) in enumerate(zip(originals, ours, reductions)):
            top = max(left, right)
            ax.plot(
                [index - width / 2, index - width / 2, index + width / 2, index + width / 2],
                [top + 0.006, top + 0.010, top + 0.010, top + 0.006],
                color=ORANGE,
                linewidth=0.8,
                clip_on=False,
            )
            ax.text(
                index,
                top + 0.013,
                f"{reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=ORANGE,
                fontweight="semibold",
            )
        ax.set_title(title, pad=5)
        ax.set_xticks(x, datasets)
        ax.set_ylim(0.0, 0.405)
        ax.set_yticks(np.arange(0.0, 0.41, 0.1))
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)
        ax.text(
            -0.14,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
    axes[0].set_ylabel("Four-horizon mean MSE (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        ncol=2,
        frameon=False,
    )
    save_figure(fig, "figure_7_decoder_transfer")
    plt.close(fig)


def main() -> None:
    configure_style()
    plot_efficiency()
    plot_transfer()


if __name__ == "__main__":
    main()
