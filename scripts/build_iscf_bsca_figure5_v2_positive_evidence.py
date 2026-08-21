#!/usr/bin/env python3
"""Build the author-review Figure 5 v2 positive-evidence draft."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"


DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather"]
SCOPES = [1, 48, 144, 360, 720]
REGION_LABELS = [
    "1–48",
    "49–96",
    "97–144",
    "145–192",
    "193–288",
    "289–336",
    "337–512",
    "513–720",
]
FULL_ARM = "full_iscf_bsca"
ALLOCATION_CONTROL = "without_target_adaptive_allocation"
BSCA_CONTROL = "without_bsca_prefix_loss_only"
SCOPE_BLUE = "#3E86A8"
ALLOCATION_VIOLET = "#7772B5"
BSCA_ORANGE = "#D36B42"
WINNER_ORANGE = "#C95516"
NEUTRAL = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
            "figure5_mechanism_diagnostics_20260816/raw/full"
        ),
    )
    parser.add_argument(
        "--ablation-means",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
            "core_ablation_20260814/formal_results/"
            "core_ablation_dataset_means.csv"
        ),
    )
    parser.add_argument(
        "--ablation-gates",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
            "core_ablation_20260814/formal_results/"
            "core_ablation_author_corrected_aggregate_gates_20260817.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_section5_6_figure5_v2_positive_evidence_20260821"
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
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


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


def load_scope_competence(
    diagnostic_root: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    matrix_blocks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    winner_positions: list[tuple[int, int]] = []
    row_offset = 0

    for dataset in DATASETS:
        path = diagnostic_root / dataset / "pcsd_validation_diagnostics.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {"arm_row_bin_mse", "scales", "bin_names"}
            missing = required.difference(source.files)
            if missing:
                raise KeyError(f"{path} missing arrays: {sorted(missing)}")
            arm_mse = source["arm_row_bin_mse"].astype(np.float64, copy=False)
            scopes = source["scales"].astype(int).tolist()
            bins = [str(value) for value in source["bin_names"].tolist()]

        if arm_mse.ndim != 3 or arm_mse.shape[1:] != (8, 5):
            raise ValueError(f"unexpected arm_row_bin_mse shape for {dataset}: {arm_mse.shape}")
        if scopes != SCOPES or len(bins) != len(REGION_LABELS):
            raise ValueError(f"scope or region contract mismatch for {dataset}")
        if not np.isfinite(arm_mse).all():
            raise ValueError(f"non-finite validation diagnostics for {dataset}")

        regional_mean = arm_mse.mean(axis=0)
        regional_best = regional_mean.min(axis=1, keepdims=True)
        excess = 100.0 * (regional_mean / regional_best - 1.0)
        block = excess.T
        matrix_blocks.append(block)
        best_indices = np.argmin(regional_mean, axis=1)

        for region_index, region_label in enumerate(REGION_LABELS):
            for scope_index, scope in enumerate(SCOPES):
                rows.append(
                    {
                        "dataset": dataset,
                        "future_region": region_label,
                        "scope": scope,
                        "validation_rows": arm_mse.shape[0],
                        "mean_mse": f"{regional_mean[region_index, scope_index]:.12g}",
                        "excess_mse_percent": f"{excess[region_index, scope_index]:.8f}",
                        "regional_best": int(best_indices[region_index] == scope_index),
                        "split": "validation",
                        "evidence_role": "aggregate_internal_diagnostic",
                    }
                )
            winner_positions.append(
                (row_offset + int(best_indices[region_index]), region_index)
            )
        row_offset += len(SCOPES)

    matrix = np.vstack(matrix_blocks)
    if matrix.shape != (len(DATASETS) * len(SCOPES), len(REGION_LABELS)):
        raise AssertionError(f"unexpected heatmap matrix shape: {matrix.shape}")
    return matrix, rows, np.asarray(winner_positions, dtype=int)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_ablation_gain(
    means_path: Path,
    gates_path: Path,
    control_id: str,
    comparison_label: str,
) -> list[dict[str, Any]]:
    mean_rows = read_csv(means_path)
    gate_rows = read_csv(gates_path)
    by_arm_dataset = {
        (row["arm_id"], row["dataset"]): row for row in mean_rows
    }
    output: list[dict[str, Any]] = []

    for dataset in DATASETS:
        full = float(by_arm_dataset[(FULL_ARM, dataset)]["mean_mse"])
        control = float(by_arm_dataset[(control_id, dataset)]["mean_mse"])
        gain = 100.0 * (control - full) / control
        output.append(
            {
                "dataset": dataset,
                "full_mse": f"{full:.6f}",
                "control_mse": f"{control:.6f}",
                "mse_reduction_percent": f"{gain:.8f}",
                "horizon_count": 4,
                "aggregation": "dataset_four_horizon_mean",
                "comparison": comparison_label,
                "split": "official_test",
                "source_role": "author_corrected_rerun_20260817",
            }
        )

    gate = next(row for row in gate_rows if row["control_id"] == control_id)
    macro_gain = float(gate["full_vs_control_macro_mse_gain_percent"])
    output.append(
        {
            "dataset": "Mean",
            "full_mse": "",
            "control_mse": "",
            "mse_reduction_percent": f"{macro_gain:.8f}",
            "horizon_count": 4,
            "aggregation": "author_corrected_five_dataset_macro_gate",
            "comparison": comparison_label,
            "split": "official_test",
            "source_role": "author_corrected_rerun_20260817",
        }
    )
    gains = np.asarray(
        [float(row["mse_reduction_percent"]) for row in output],
        dtype=float,
    )
    if not np.all(gains > 0):
        raise ValueError(f"non-positive gain found for {control_id}: {gains}")
    return output


def add_panel_label(
    axis: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.035,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_gain_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    color: str,
    title: str,
    subtitle: str,
    panel_label: str,
) -> None:
    labels = [str(row["dataset"]) for row in rows]
    values = np.asarray(
        [float(row["mse_reduction_percent"]) for row in rows],
        dtype=float,
    )
    y_positions = np.arange(len(labels))[::-1]

    for y, value in zip(y_positions, values):
        axis.plot([0, value], [y, y], color=color, linewidth=2.1, alpha=0.42)
        axis.scatter(
            value,
            y,
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.text(
            value + 0.16,
            y,
            f"{value:.1f}%",
            color=color,
            fontsize=6.4,
            fontweight="bold" if labels[len(labels) - 1 - y] == "Mean" else "normal",
            va="center",
        )

    axis.axvline(0, color="#B7B7B7", linewidth=0.8)
    axis.axhline(0.5, color="#D9D9D9", linewidth=0.7)
    axis.set_yticks(y_positions, labels=labels)
    for tick in axis.get_yticklabels():
        if tick.get_text() == "Mean":
            tick.set_fontweight("bold")
    axis.set_xlim(0, 7.0)
    axis.set_xticks([0, 2, 4, 6])
    axis.set_xlabel("MSE reduction (%)")
    axis.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=13)
    axis.text(
        0,
        1.01,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=NEUTRAL,
    )
    axis.grid(axis="x", color="#EDEDED", linewidth=0.6)
    axis.tick_params(axis="both", length=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#A0A0A0")
    add_panel_label(axis, panel_label, x=-0.16)


def build_figure(
    output_dir: Path,
    competence: np.ndarray,
    winners: np.ndarray,
    allocation_rows: list[dict[str, Any]],
    bsca_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    configure_style()
    figure = plt.figure(figsize=(7.087, 4.724))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.78, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.17,
        right=0.975,
        bottom=0.145,
        top=0.92,
        wspace=0.38,
        hspace=0.62,
    )
    axis_a = figure.add_subplot(grid[:, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 1])

    cmap = LinearSegmentedColormap.from_list(
        "scope_excess",
        ["#F5FAFC", "#D8EAF1", "#9AC8D9", "#4C91AE", "#185D78"],
    )
    vmax = float(np.ceil(competence.max()))
    image = axis_a.imshow(
        competence,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    for row, column in winners:
        axis_a.add_patch(
            Rectangle(
                (column - 0.46, row - 0.46),
                0.92,
                0.92,
                fill=False,
                edgecolor=WINNER_ORANGE,
                linewidth=1.0,
            )
        )
    for boundary in range(len(SCOPES), competence.shape[0], len(SCOPES)):
        axis_a.axhline(boundary - 0.5, color="white", linewidth=2.0)

    axis_a.set_xticks(np.arange(len(REGION_LABELS)), labels=REGION_LABELS)
    axis_a.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        pad=3,
    )
    axis_a.set_yticks(
        np.arange(len(DATASETS) * len(SCOPES)),
        labels=[str(scope) for _ in DATASETS for scope in SCOPES],
    )
    axis_a.tick_params(axis="y", length=0, pad=3)
    for index, dataset in enumerate(DATASETS):
        center = index * len(SCOPES) + (len(SCOPES) - 1) / 2
        axis_a.text(
            -0.20,
            center,
            dataset,
            transform=axis_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            clip_on=False,
        )
    axis_a.text(
        -0.20,
        1.035,
        "Dataset",
        transform=axis_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL,
    )
    axis_a.text(
        -0.03,
        1.035,
        "Scope",
        transform=axis_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL,
    )
    axis_a.set_title(
        "Scope performance across future regions",
        loc="left",
        fontsize=8.3,
        fontweight="bold",
        pad=16,
    )
    axis_a.text(
        1.0,
        -0.055,
        "Orange outline: lowest regional MSE",
        transform=axis_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=WINNER_ORANGE,
    )
    for spine in axis_a.spines.values():
        spine.set_visible(False)
    add_panel_label(axis_a, "a", x=-0.29, y=1.07)

    color_axis = axis_a.inset_axes([0.02, -0.105, 0.48, 0.026])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Excess MSE above the regional best (%)", fontsize=6.2, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)

    draw_gain_panel(
        axis_b,
        allocation_rows,
        ALLOCATION_VIOLET,
        "Target-Adaptive Allocation",
        "Full vs equal scope fusion",
        "b",
    )
    draw_gain_panel(
        axis_c,
        bsca_rows,
        BSCA_ORANGE,
        "Balanced Scope Co-Adaptation",
        "Full vs prefix-only training",
        "c",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_5_scope_competence_component_gains"
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

    with Image.open(outputs["tiff"]) as image_tiff:
        image_tiff.save(outputs["tiff"], compression="tiff_lzw", dpi=(600, 600))
    return outputs


def main() -> None:
    args = parse_args()
    competence, competence_rows, winners = load_scope_competence(
        args.diagnostic_root
    )
    allocation_rows = load_ablation_gain(
        args.ablation_means,
        args.ablation_gates,
        ALLOCATION_CONTROL,
        "Full vs w/o Target-Adaptive Allocation",
    )
    bsca_rows = load_ablation_gain(
        args.ablation_means,
        args.ablation_gates,
        BSCA_CONTROL,
        "Full vs w/o BSCA",
    )

    source_dir = args.output_dir / "source_data"
    write_csv(source_dir / "panel_a_scope_competence.csv", competence_rows)
    write_csv(source_dir / "panel_b_allocation_gain.csv", allocation_rows)
    write_csv(source_dir / "panel_c_bsca_gain.csv", bsca_rows)
    outputs = build_figure(
        args.output_dir,
        competence,
        winners,
        allocation_rows,
        bsca_rows,
    )

    print(
        "Figure 5 v2 generated: "
        f"panel_a_rows={len(competence_rows)}, "
        f"panel_b_rows={len(allocation_rows)}, "
        f"panel_c_rows={len(bsca_rows)}, "
        f"outputs={','.join(path.name for path in outputs.values())}"
    )


if __name__ == "__main__":
    main()
