#!/usr/bin/env python3
"""Build Figure 5 v4 from frozen ISCF-BSCA validation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather"]
SCOPES = [1, 48, 144, 360, 720]
REGION_BOUNDS = [0, 48, 96, 144, 192, 288, 336, 512, 720]
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
SCOPE_COLORS = {
    1: "#405F73",
    48: "#648395",
    144: "#91ABB5",
    360: "#9188AD",
    720: "#B47D96",
}
TARGET_COLOR = "#262B30"
NEUTRAL_MID = "#6B7076"
NEUTRAL_LIGHT = "#E7EAED"
ROW_BAND = "#F7F8FA"
ACCENT = "#C96C4A"


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
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/iscf_bsca_section5_6_figure5_v4_scope_allocation_"
            "behavior_20260821"
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
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
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


def text_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.57 else TARGET_COLOR


def add_panel_label(
    axis: plt.Axes,
    label: str,
    x: float = -0.075,
    y: float = 1.08,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=TARGET_COLOR,
    )


def pairwise_scope_disagreement(arms: np.ndarray) -> np.ndarray:
    """Return mean pairwise absolute scope difference for every probe row."""
    differences = [
        np.mean(np.abs(arms[:, left] - arms[:, right]), axis=1)
        for left, right in combinations(range(len(SCOPES)), 2)
    ]
    return np.mean(np.stack(differences, axis=1), axis=1)


def load_diagnostics(
    root: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    sources: dict[str, dict[str, np.ndarray]] = {}
    competence_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        path = root / dataset / "pcsd_validation_diagnostics.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {
                "scales",
                "probe_targets",
                "probe_arms",
                "arm_row_bin_mse",
                "policy_row_bin_usage",
            }
            missing = required.difference(source.files)
            if missing:
                raise KeyError(f"{path} missing arrays: {sorted(missing)}")
            scales = source["scales"].astype(int).tolist()
            if scales != SCOPES:
                raise ValueError(f"scope contract mismatch for {dataset}: {scales}")
            arrays = {
                "targets": source["probe_targets"].astype(np.float64),
                "arms": source["probe_arms"].astype(np.float64),
                "arm_mse": source["arm_row_bin_mse"].astype(np.float64),
                "usage": source["policy_row_bin_usage"].astype(np.float64),
            }
        if arrays["targets"].shape != (256, 720):
            raise ValueError(f"unexpected target shape for {dataset}")
        if arrays["arms"].shape != (256, 5, 720):
            raise ValueError(f"unexpected probe-arm shape for {dataset}")
        if arrays["arm_mse"].ndim != 3 or arrays["arm_mse"].shape[1:] != (8, 5):
            raise ValueError(f"unexpected regional-error shape for {dataset}")
        if arrays["usage"].shape != arrays["arm_mse"].shape:
            raise ValueError(f"usage/error shape mismatch for {dataset}")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"non-finite diagnostics for {dataset}")
        sources[dataset] = arrays

        mean_mse = arrays["arm_mse"].mean(axis=0)
        best_mse = mean_mse.min(axis=1, keepdims=True)
        excess = 100.0 * (mean_mse / best_mse - 1.0)
        best_scope = np.argmin(mean_mse, axis=1)
        mean_usage = arrays["usage"].mean(axis=0)
        if np.max(np.abs(mean_usage.sum(axis=1) - 1.0)) > 2e-5:
            raise ValueError(f"allocation does not sum to one for {dataset}")
        highest_scope = np.argmax(mean_usage, axis=1)

        for region_index, region in enumerate(REGION_LABELS):
            for scope_index, scope in enumerate(SCOPES):
                competence_rows.append(
                    {
                        "dataset": dataset,
                        "future_region": region,
                        "scope": scope,
                        "validation_rows": arrays["arm_mse"].shape[0],
                        "mean_mse": f"{mean_mse[region_index, scope_index]:.12g}",
                        "excess_mse_percent": f"{excess[region_index, scope_index]:.8f}",
                        "regional_best": int(best_scope[region_index] == scope_index),
                        "split": "validation",
                    }
                )
                allocation_rows.append(
                    {
                        "dataset": dataset,
                        "future_region": region,
                        "scope": scope,
                        "validation_rows": arrays["usage"].shape[0],
                        "mean_probability": f"{mean_usage[region_index, scope_index]:.10f}",
                        "highest_weight": int(
                            highest_scope[region_index] == scope_index
                        ),
                        "split": "validation",
                    }
                )

    return sources, competence_rows, allocation_rows


def select_probe(
    sources: dict[str, dict[str, np.ndarray]],
) -> tuple[str, int, list[dict[str, Any]], float, int]:
    """Select by regional-winner diversity, then scope-forecast disagreement."""
    audit_rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, float, str, int]] = []
    for dataset in DATASETS:
        arrays = sources[dataset]
        disagreement = pairwise_scope_disagreement(arrays["arms"])
        winners = np.argmin(arrays["arm_mse"][:256], axis=2)
        for row_index in range(256):
            distinct = len(np.unique(winners[row_index]))
            candidates.append(
                (distinct, float(disagreement[row_index]), dataset, row_index)
            )
            audit_rows.append(
                {
                    "dataset": dataset,
                    "probe_row": row_index,
                    "distinct_regional_winners": distinct,
                    "mean_pairwise_scope_disagreement": (
                        f"{disagreement[row_index]:.12g}"
                    ),
                    "selected": 0,
                    "split": "validation",
                }
            )
    distinct, disagreement, dataset, row_index = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    for row in audit_rows:
        if row["dataset"] == dataset and row["probe_row"] == row_index:
            row["selected"] = 1
    return dataset, row_index, audit_rows, disagreement, distinct


def selected_source_rows(
    sources: dict[str, dict[str, np.ndarray]],
    dataset: str,
    row_index: int,
    disagreement: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = sources[dataset]
    arms = arrays["arms"][row_index]
    targets = arrays["targets"][row_index]
    winners = np.argmin(arrays["arm_mse"][row_index], axis=1)
    trajectory_rows: list[dict[str, Any]] = []
    for step in range(720):
        row: dict[str, Any] = {
            "dataset": dataset,
            "probe_row": row_index,
            "future_step": step + 1,
            "target": f"{targets[step]:.10f}",
            "mean_pairwise_scope_disagreement": f"{disagreement:.12g}",
            "split": "validation",
        }
        for scope_index, scope in enumerate(SCOPES):
            row[f"scope_{scope}_forecast"] = f"{arms[scope_index, step]:.10f}"
        trajectory_rows.append(row)

    preference_rows: list[dict[str, Any]] = []
    for region_index, region in enumerate(REGION_LABELS):
        errors = arrays["arm_mse"][row_index, region_index]
        winner_index = int(winners[region_index])
        preference_rows.append(
            {
                "dataset": dataset,
                "probe_row": row_index,
                "future_region": region,
                "lowest_mse_scope": SCOPES[winner_index],
                "lowest_mse": f"{errors[winner_index]:.12g}",
                "best_to_worst_excess_mse_percent": (
                    f"{100.0 * (errors.max() / errors.min() - 1.0):.8f}"
                ),
                "split": "validation",
            }
        )
    return trajectory_rows, preference_rows


def draw_trajectory_panel(
    axis: plt.Axes,
    strip_axis: plt.Axes,
    sources: dict[str, dict[str, np.ndarray]],
    dataset: str,
    row_index: int,
    disagreement: float,
) -> None:
    arrays = sources[dataset]
    arms = arrays["arms"][row_index]
    winners = np.argmin(arrays["arm_mse"][row_index], axis=1)
    steps = np.arange(1, 721)

    for region_index in range(len(REGION_LABELS)):
        if region_index % 2 == 0:
            axis.axvspan(
                REGION_BOUNDS[region_index] + 1,
                REGION_BOUNDS[region_index + 1],
                color=ROW_BAND,
                zorder=0,
            )
    for boundary in REGION_BOUNDS[1:-1]:
        axis.axvline(boundary, color=NEUTRAL_LIGHT, linewidth=0.55, zorder=0)

    for scope_index, scope in enumerate(SCOPES):
        axis.plot(
            steps,
            arms[scope_index],
            color=SCOPE_COLORS[scope],
            linewidth=1.05,
            alpha=0.92,
            zorder=2,
        )
    axis.set_xlim(1, 720)
    axis.set_ylabel("Forecast (normalized)")
    axis.set_xticks(REGION_BOUNDS[1:], labels=[])
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color="#ECEEF0", linewidth=0.55)
    axis.set_title(
        "Distinct Scope-conditioned Forecasts",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        color=TARGET_COLOR,
        pad=21,
    )
    axis.text(
        0.0,
        1.03,
        (
            f"{dataset} validation probe {row_index}; selected by regional-winner "
            f"diversity, then forecast disagreement (mean pairwise |Δ| = "
            f"{disagreement:.3f})"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=NEUTRAL_MID,
    )
    handles = [
        Line2D(
            [],
            [],
            color=SCOPE_COLORS[scope],
            linewidth=1.2,
            label=rf"$s={scope}$",
        )
        for scope in SCOPES
    ]
    axis.legend(
        handles=handles,
        ncol=5,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.17),
        handlelength=1.75,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    add_panel_label(axis, "a", x=-0.055, y=1.15)

    strip_axis.set_xlim(1, 720)
    strip_axis.set_ylim(0, 1)
    for region_index, winner_index in enumerate(winners):
        start = REGION_BOUNDS[region_index] + 1
        end = REGION_BOUNDS[region_index + 1]
        scope = SCOPES[int(winner_index)]
        strip_axis.add_patch(
            Rectangle(
                (start, 0.02),
                end - start + 1,
                0.66,
                facecolor=SCOPE_COLORS[scope],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        strip_axis.text(
            (start + end) / 2,
            0.35,
            str(scope),
            ha="center",
            va="center",
            fontsize=5.2,
            fontweight="bold",
            color=text_color(SCOPE_COLORS[scope]),
        )
    strip_axis.text(
        -0.008,
        0.35,
        "Best scope",
        transform=strip_axis.transAxes,
        ha="right",
        va="center",
        fontsize=5.8,
        color=NEUTRAL_MID,
    )
    strip_axis.set_xticks(
        [(REGION_BOUNDS[index] + REGION_BOUNDS[index + 1] + 1) / 2
         for index in range(len(REGION_LABELS))],
        labels=REGION_LABELS,
    )
    strip_axis.set_xlabel("Future region", labelpad=2)
    strip_axis.tick_params(axis="x", length=0, pad=1)
    strip_axis.set_yticks([])
    for spine in strip_axis.spines.values():
        spine.set_visible(False)


def marker_area(gap_percent: float) -> float:
    if not np.isfinite(gap_percent) or gap_percent < 0:
        raise ValueError(f"invalid error gap: {gap_percent}")
    return 58.0 + 15.0 * np.sqrt(gap_percent)


def draw_competence_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    winners = [row for row in rows if int(row["regional_best"]) == 1]
    if len(winners) != len(DATASETS) * len(REGION_LABELS):
        raise ValueError("incomplete aggregate competence winners")
    x_lookup = {region: index for index, region in enumerate(REGION_LABELS)}
    y_lookup = {
        dataset: len(DATASETS) - 1 - index
        for index, dataset in enumerate(DATASETS)
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for region in REGION_LABELS:
            grouped[(dataset, region)] = [
                row
                for row in rows
                if row["dataset"] == dataset and row["future_region"] == region
            ]

    for index, dataset in enumerate(DATASETS):
        y = y_lookup[dataset]
        if index % 2 == 0:
            axis.axhspan(y - 0.46, y + 0.46, color=ROW_BAND, zorder=0)
    for boundary in np.arange(-0.5, len(REGION_LABELS), 1.0):
        axis.axvline(boundary, color="#EEF0F2", linewidth=0.55, zorder=0)

    for row in winners:
        dataset = str(row["dataset"])
        region = str(row["future_region"])
        scope = int(row["scope"])
        cell = grouped[(dataset, region)]
        gap = max(float(item["excess_mse_percent"]) for item in cell)
        x = x_lookup[region]
        y = y_lookup[dataset]
        axis.scatter(
            x,
            y,
            s=marker_area(gap),
            color=SCOPE_COLORS[scope],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.text(
            x,
            y,
            str(scope),
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=text_color(SCOPE_COLORS[scope]),
            zorder=4,
        )

    axis.set_xlim(-0.55, len(REGION_LABELS) - 0.45)
    axis.set_ylim(-0.58, len(DATASETS) - 0.42)
    axis.set_xticks(np.arange(len(REGION_LABELS)), labels=REGION_LABELS, rotation=32)
    axis.set_yticks(np.arange(len(DATASETS))[::-1], labels=DATASETS)
    axis.tick_params(axis="both", length=0)
    axis.set_xlabel("Future region", labelpad=4)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Region-wise Scope Competence",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        color=TARGET_COLOR,
        pad=21,
    )
    axis.text(
        0.0,
        1.03,
        "Colour/number = lowest-MSE scope; marker area = best-to-worst MSE gap",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL_MID,
    )
    add_panel_label(axis, "b", x=-0.10, y=1.15)


def draw_allocation_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    highest = [row for row in rows if int(row["highest_weight"]) == 1]
    if len(highest) != len(DATASETS) * len(REGION_LABELS):
        raise ValueError("incomplete aggregate allocation winners")
    x_lookup = {region: index for index, region in enumerate(REGION_LABELS)}
    y_lookup = {
        dataset: len(DATASETS) - 1 - index
        for index, dataset in enumerate(DATASETS)
    }

    for index, dataset in enumerate(DATASETS):
        y = y_lookup[dataset]
        if index % 2 == 0:
            axis.axhspan(y - 0.46, y + 0.46, color=ROW_BAND, zorder=0)
    for boundary in np.arange(-0.5, len(REGION_LABELS), 1.0):
        axis.axvline(boundary, color="#EEF0F2", linewidth=0.55, zorder=0)

    for row in highest:
        x = x_lookup[str(row["future_region"])]
        y = y_lookup[str(row["dataset"])]
        scope = int(row["scope"])
        axis.scatter(
            x,
            y,
            s=83,
            marker="s",
            color=SCOPE_COLORS[scope],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        axis.text(
            x,
            y,
            str(scope),
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=text_color(SCOPE_COLORS[scope]),
            zorder=4,
        )

    axis.set_xlim(-0.55, len(REGION_LABELS) - 0.45)
    axis.set_ylim(-0.58, len(DATASETS) - 0.42)
    axis.set_xticks(np.arange(len(REGION_LABELS)), labels=REGION_LABELS, rotation=32)
    axis.set_yticks(np.arange(len(DATASETS))[::-1], labels=[])
    axis.tick_params(axis="both", length=0)
    axis.set_xlabel("Future region", labelpad=4)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Region-wise Allocation Diversity",
        loc="left",
        fontsize=8.7,
        fontweight="bold",
        color=TARGET_COLOR,
        pad=21,
    )
    axis.text(
        0.0,
        1.03,
        "Colour/number = highest-weight scope of the mean soft allocation",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL_MID,
    )
    add_panel_label(axis, "c", x=-0.06, y=1.15)


def build_figure(
    sources: dict[str, dict[str, np.ndarray]],
    competence_rows: list[dict[str, Any]],
    allocation_rows: list[dict[str, Any]],
    selected_dataset: str,
    selected_row: int,
    disagreement: float,
) -> plt.Figure:
    # 183 x 140 mm, a double-column journal figure.
    figure = plt.figure(figsize=(7.2047244094, 5.5118110236), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.28, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.09,
        right=0.985,
        top=0.91,
        bottom=0.13,
        hspace=0.43,
        wspace=0.19,
    )
    top = outer[0, :].subgridspec(2, 1, height_ratios=[1.0, 0.17], hspace=0.03)
    trajectory_axis = figure.add_subplot(top[0])
    strip_axis = figure.add_subplot(top[1])
    competence_axis = figure.add_subplot(outer[1, 0])
    allocation_axis = figure.add_subplot(outer[1, 1])

    draw_trajectory_panel(
        trajectory_axis,
        strip_axis,
        sources,
        selected_dataset,
        selected_row,
        disagreement,
    )
    draw_competence_panel(competence_axis, competence_rows)
    draw_allocation_panel(allocation_axis, allocation_rows)

    figure.text(
        0.50,
        0.022,
        "Validation-only internal diagnostics · seed 2021",
        ha="center",
        va="center",
        fontsize=5.9,
        color=NEUTRAL_MID,
    )
    return figure


def save_figure(figure: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_5_scope_allocation_behavior"
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    normalize_svg(stem.with_suffix(".svg"))
    with Image.open(stem.with_suffix(".png")) as image:
        if image.width < 1800 or image.height < 1100:
            raise ValueError(f"unexpected PNG size: {image.size}")


def main() -> None:
    args = parse_args()
    configure_style()
    sources, competence_rows, allocation_rows = load_diagnostics(
        args.diagnostic_root
    )
    (
        selected_dataset,
        selected_row,
        selection_audit,
        disagreement,
        distinct_winners,
    ) = select_probe(sources)
    trajectory_rows, selected_preferences = selected_source_rows(
        sources,
        selected_dataset,
        selected_row,
        disagreement,
    )

    source_dir = args.output_dir / "source_data"
    write_csv(source_dir / "scope_trajectory_selected.csv", trajectory_rows)
    write_csv(source_dir / "selected_region_preferences.csv", selected_preferences)
    write_csv(source_dir / "scope_competence.csv", competence_rows)
    write_csv(source_dir / "scope_allocation.csv", allocation_rows)
    write_csv(source_dir / "trajectory_selection_audit.csv", selection_audit)

    competence_winners = [
        int(row["scope"])
        for row in competence_rows
        if int(row["regional_best"]) == 1
    ]
    allocation_winners = [
        int(row["scope"])
        for row in allocation_rows
        if int(row["highest_weight"]) == 1
    ]
    allocation_by_dataset = {
        dataset: [
            int(row["scope"])
            for row in allocation_rows
            if row["dataset"] == dataset and int(row["highest_weight"]) == 1
        ]
        for dataset in DATASETS
    }
    probability_values = np.asarray(
        [float(row["mean_probability"]) for row in allocation_rows],
        dtype=float,
    )
    summary = {
        "split": "validation",
        "seed": 2021,
        "new_training": 0,
        "formal_test": 0,
        "selected_example": {
            "dataset": selected_dataset,
            "probe_row": selected_row,
            "selection_rule": (
                "maximize distinct regional lowest-MSE scopes, then maximize "
                "mean pairwise absolute scope-forecast disagreement"
            ),
            "distinct_regional_winners": distinct_winners,
            "mean_pairwise_scope_disagreement": disagreement,
        },
        "aggregate_competence": {
            "cells": len(competence_winners),
            "scopes_appearing_as_lowest_mse": sorted(set(competence_winners)),
        },
        "aggregate_allocation": {
            "cells": len(allocation_winners),
            "scopes_appearing_as_highest_weight": sorted(set(allocation_winners)),
            "datasets_with_regionwise_highest_weight_change": sum(
                len(set(values)) > 1 for values in allocation_by_dataset.values()
            ),
            "mean_probability_min": float(probability_values.min()),
            "mean_probability_max": float(probability_values.max()),
        },
        "claim_boundary": (
            "Supports scope-forecast diversity, regional scope-error heterogeneity, "
            "and non-identical soft allocation profiles. Does not establish "
            "oracle scope recovery, hard routing, or causal specialization."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    figure = build_figure(
        sources,
        competence_rows,
        allocation_rows,
        selected_dataset,
        selected_row,
        disagreement,
    )
    save_figure(figure, args.output_dir)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
