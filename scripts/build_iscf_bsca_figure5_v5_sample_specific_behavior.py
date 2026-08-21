#!/usr/bin/env python3
"""Build Figure 5 v5 from frozen sample-specific validation diagnostics."""

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
from matplotlib.colors import LinearSegmentedColormap
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
DARK = "#262B30"
MID = "#676D73"
LIGHT = "#E6E9EC"
REGION_BAND = "#F7F8FA"
ACCENT = "#C96C4A"
PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "allocation_probability",
    ["#ECEAF3", "#C5C1D8", "#928BB4", "#5B537F"],
)


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
            "analysis/iscf_bsca_section5_6_figure5_v5_sample_specific_"
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
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
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
    content = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in content.splitlines()) + "\n",
        encoding="utf-8",
    )


def contrasting_text(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (
        int(color[0:2], 16) / 255.0,
        int(color[2:4], 16) / 255.0,
        int(color[4:6], 16) / 255.0,
    )
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.58 else DARK


def add_panel_label(
    axis: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.08,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
    )


def pairwise_scope_disagreement(arms: np.ndarray) -> np.ndarray:
    """Return mean pairwise absolute scope difference for every probe."""
    differences = [
        np.mean(np.abs(arms[:, left] - arms[:, right]), axis=1)
        for left, right in combinations(range(len(SCOPES)), 2)
    ]
    return np.mean(np.stack(differences, axis=1), axis=1)


def load_diagnostics(root: Path) -> dict[str, dict[str, np.ndarray]]:
    sources: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASETS:
        path = root / dataset / "pcsd_validation_diagnostics.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {
                "scales",
                "probe_fused",
                "probe_targets",
                "probe_arms",
                "probe_direct_policy",
            }
            missing = required.difference(source.files)
            if missing:
                raise KeyError(f"{path} missing arrays: {sorted(missing)}")
            scales = source["scales"].astype(int).tolist()
            if scales != SCOPES:
                raise ValueError(f"scope contract mismatch for {dataset}: {scales}")
            arrays = {
                "fused": source["probe_fused"].astype(np.float64),
                "targets": source["probe_targets"].astype(np.float64),
                "arms": source["probe_arms"].astype(np.float64),
                "probabilities": source["probe_direct_policy"].astype(np.float64),
            }
        expected_shapes = {
            "fused": (256, 720),
            "targets": (256, 720),
            "arms": (256, 5, 720),
            "probabilities": (256, 720, 5),
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"unexpected {name} shape for {dataset}: {arrays[name].shape}"
                )
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"non-finite {name} values for {dataset}")
        probability_error = np.max(
            np.abs(arrays["probabilities"].sum(axis=2) - 1.0)
        )
        if probability_error > 2e-5:
            raise ValueError(
                f"allocation probabilities do not sum to one for {dataset}: "
                f"{probability_error}"
            )
        sources[dataset] = arrays
    return sources


def regional_probability_means(probabilities: np.ndarray) -> np.ndarray:
    """Aggregate per-step probabilities within the eight displayed regions."""
    return np.stack(
        [
            probabilities[REGION_BOUNDS[index] : REGION_BOUNDS[index + 1]].mean(
                axis=0
            )
            for index in range(len(REGION_LABELS))
        ],
        axis=0,
    )


def select_probe(
    sources: dict[str, dict[str, np.ndarray]],
) -> tuple[str, int, list[dict[str, Any]], float, int]:
    """Select by dominant-scope diversity, then forecast disagreement."""
    candidates: list[tuple[int, float, str, int]] = []
    audit_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        arms = sources[dataset]["arms"]
        probabilities = sources[dataset]["probabilities"]
        disagreements = pairwise_scope_disagreement(arms)
        for probe_row in range(arms.shape[0]):
            region_means = regional_probability_means(probabilities[probe_row])
            dominant_indices = np.argmax(region_means, axis=1)
            distinct = len(np.unique(dominant_indices))
            candidates.append(
                (
                    distinct,
                    float(disagreements[probe_row]),
                    dataset,
                    probe_row,
                )
            )
            audit_rows.append(
                {
                    "dataset": dataset,
                    "probe_row": probe_row,
                    "distinct_regional_dominant_scopes": distinct,
                    "mean_pairwise_scope_disagreement": (
                        f"{disagreements[probe_row]:.12g}"
                    ),
                    "selected": 0,
                    "split": "validation",
                }
            )
    distinct, disagreement, dataset, probe_row = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    for row in audit_rows:
        if row["dataset"] == dataset and row["probe_row"] == probe_row:
            row["selected"] = 1
    return dataset, probe_row, audit_rows, disagreement, distinct


def build_source_rows(
    arrays: dict[str, np.ndarray],
    dataset: str,
    probe_row: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    fused = arrays["fused"][probe_row]
    targets = arrays["targets"][probe_row]
    arms = arrays["arms"][probe_row]
    probabilities = arrays["probabilities"][probe_row]
    region_means = regional_probability_means(probabilities)
    dominant_indices = np.argmax(region_means, axis=1)

    trajectory_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for step_index in range(720):
        region_index = int(
            np.searchsorted(REGION_BOUNDS[1:], step_index, side="right")
        )
        dominant_scope = SCOPES[int(dominant_indices[region_index])]
        trajectory_row: dict[str, Any] = {
            "dataset": dataset,
            "probe_row": probe_row,
            "future_step": step_index + 1,
            "future_region": REGION_LABELS[region_index],
            "fused_forecast": f"{fused[step_index]:.12g}",
            "target": f"{targets[step_index]:.12g}",
            "regional_dominant_scope": dominant_scope,
            "split": "validation",
        }
        probability_row: dict[str, Any] = {
            "dataset": dataset,
            "probe_row": probe_row,
            "future_step": step_index + 1,
            "future_region": REGION_LABELS[region_index],
            "split": "validation",
        }
        for scope_index, scope in enumerate(SCOPES):
            forecast = arms[scope_index, step_index]
            trajectory_row[f"scope_{scope}_forecast"] = f"{forecast:.12g}"
            trajectory_row[f"scope_{scope}_deviation"] = (
                f"{forecast - fused[step_index]:.12g}"
            )
            probability_row[f"scope_{scope}_probability"] = (
                f"{probabilities[step_index, scope_index]:.12g}"
            )
        trajectory_rows.append(trajectory_row)
        probability_rows.append(probability_row)

    regional_rows: list[dict[str, Any]] = []
    for region_index, region_label in enumerate(REGION_LABELS):
        dominant_index = int(dominant_indices[region_index])
        row: dict[str, Any] = {
            "dataset": dataset,
            "probe_row": probe_row,
            "future_region": region_label,
            "start_step": REGION_BOUNDS[region_index] + 1,
            "end_step": REGION_BOUNDS[region_index + 1],
            "dominant_scope": SCOPES[dominant_index],
            "dominant_probability": f"{region_means[region_index, dominant_index]:.12g}",
            "split": "validation",
        }
        for scope_index, scope in enumerate(SCOPES):
            row[f"scope_{scope}_mean_probability"] = (
                f"{region_means[region_index, scope_index]:.12g}"
            )
        regional_rows.append(row)
    return trajectory_rows, probability_rows, regional_rows


def shade_regions(axis: plt.Axes) -> None:
    for region_index in range(len(REGION_LABELS)):
        if region_index % 2 == 0:
            axis.axvspan(
                REGION_BOUNDS[region_index] + 1,
                REGION_BOUNDS[region_index + 1],
                color=REGION_BAND,
                zorder=0,
            )
    for boundary in REGION_BOUNDS[1:-1]:
        axis.axvline(boundary, color=LIGHT, linewidth=0.55, zorder=0)


def draw_fused_panel(
    axis: plt.Axes,
    strip_axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    probe_row: int,
) -> None:
    fused = arrays["fused"][probe_row]
    probabilities = arrays["probabilities"][probe_row]
    region_means = regional_probability_means(probabilities)
    dominant_indices = np.argmax(region_means, axis=1)
    dominant_scopes = [SCOPES[int(index)] for index in dominant_indices]
    steps = np.arange(1, 721)

    shade_regions(axis)
    for region_index, scope in enumerate(dominant_scopes):
        start = REGION_BOUNDS[region_index]
        end = REGION_BOUNDS[region_index + 1]
        plot_start = max(0, start - 1)
        axis.plot(
            steps[plot_start:end],
            fused[plot_start:end],
            color=SCOPE_COLORS[scope],
            linewidth=1.55,
            solid_capstyle="round",
            zorder=3,
        )
    axis.set_xlim(1, 720)
    axis.set_ylabel("Fused forecast\n(normalized)")
    axis.set_xticks(REGION_BOUNDS[1:], labels=[])
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color="#EBEDF0", linewidth=0.55)
    axis.set_title(
        "Region-adaptive Fused Forecast",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
        pad=21,
    )
    axis.text(
        0.0,
        1.10,
        "Line colour denotes the highest-weight scope of the region-mean soft allocation",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.15,
        color=MID,
    )
    handles = [
        Line2D(
            [],
            [],
            color=SCOPE_COLORS[scope],
            linewidth=1.6,
            label=rf"$s={scope}$",
        )
        for scope in SCOPES
    ]
    axis.legend(
        handles=handles,
        ncol=5,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.045),
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    add_panel_label(axis, "a", x=-0.055, y=1.15)

    strip_axis.set_xlim(1, 720)
    strip_axis.set_ylim(0, 1)
    for region_index, scope in enumerate(dominant_scopes):
        start = REGION_BOUNDS[region_index] + 1
        end = REGION_BOUNDS[region_index + 1]
        strip_axis.add_patch(
            Rectangle(
                (start, 0.05),
                end - start + 1,
                0.63,
                facecolor=SCOPE_COLORS[scope],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        strip_axis.text(
            (start + end) / 2,
            0.365,
            str(scope),
            ha="center",
            va="center",
            fontsize=5.2,
            fontweight="bold",
            color=contrasting_text(SCOPE_COLORS[scope]),
        )
    strip_axis.text(
        -0.008,
        0.365,
        "Dominant scope",
        transform=strip_axis.transAxes,
        ha="right",
        va="center",
        fontsize=5.8,
        color=MID,
    )
    strip_axis.set_xticks(
        [
            (REGION_BOUNDS[index] + REGION_BOUNDS[index + 1] + 1) / 2
            for index in range(len(REGION_LABELS))
        ],
        labels=REGION_LABELS,
    )
    strip_axis.set_xlabel("Future region", labelpad=2)
    strip_axis.tick_params(axis="x", length=0, pad=1)
    strip_axis.set_yticks([])
    for spine in strip_axis.spines.values():
        spine.set_visible(False)


def draw_scope_deviation_panel(
    figure: plt.Figure,
    grid: mpl.gridspec.SubplotSpec,
    arrays: dict[str, np.ndarray],
    probe_row: int,
) -> None:
    fused = arrays["fused"][probe_row]
    arms = arrays["arms"][probe_row]
    deviations = arms - fused[np.newaxis, :]
    absolute_limit = float(np.quantile(np.abs(deviations), 0.995))
    absolute_limit = max(absolute_limit, 1e-6)
    steps = np.arange(1, 721)
    subgrid = grid.subgridspec(5, 1, hspace=0.08)
    axes: list[plt.Axes] = []

    for scope_index, scope in enumerate(SCOPES):
        axis = figure.add_subplot(subgrid[scope_index])
        axes.append(axis)
        shade_regions(axis)
        axis.axhline(0.0, color="#B7BCC1", linewidth=0.55, zorder=1)
        axis.plot(
            steps,
            deviations[scope_index],
            color=SCOPE_COLORS[scope],
            linewidth=0.9,
            zorder=2,
        )
        axis.fill_between(
            steps,
            0.0,
            deviations[scope_index],
            color=SCOPE_COLORS[scope],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        axis.set_xlim(1, 720)
        axis.set_ylim(-absolute_limit, absolute_limit)
        axis.set_yticks([])
        axis.tick_params(axis="x", length=0, pad=1)
        axis.text(
            -0.018,
            0.5,
            rf"$s={scope}$",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=SCOPE_COLORS[scope],
        )
        if scope_index < len(SCOPES) - 1:
            axis.set_xticks([])
        else:
            axis.set_xticks(
                [1, 96, 192, 336, 512, 720],
                labels=["1", "96", "192", "336", "512", "720"],
            )
            axis.set_xlabel("Future step", labelpad=2)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#AEB3B8")
        axis.spines["bottom"].set_linewidth(0.55)

    top_axis = axes[0]
    top_axis.set_title(
        "Distinct Scope-conditioned Forecast Signals",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
        pad=21,
    )
    top_axis.text(
        0.0,
        1.42,
        (
            r"Deviation from the fused forecast, "
            r"$\mathcal{F}^{(s)}_{\tau}-\widehat{y}_{\tau}$; shared scale"
        ),
        transform=top_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.05,
        color=MID,
    )
    add_panel_label(top_axis, "b", x=-0.14, y=2.02)
    figure.text(
        0.018,
        0.285,
        "Scope-specific deviation",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
        color=DARK,
    )


def draw_probability_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    probe_row: int,
) -> None:
    probabilities = arrays["probabilities"][probe_row].T
    observed_min = float(probabilities.min())
    observed_max = float(probabilities.max())
    lower = np.floor(observed_min * 1000.0) / 1000.0
    upper = np.ceil(observed_max * 1000.0) / 1000.0
    if upper <= lower:
        upper = lower + 0.001
    image = axis.imshow(
        probabilities,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[1, 720, len(SCOPES) - 0.5, -0.5],
        cmap=PROBABILITY_CMAP,
        vmin=lower,
        vmax=upper,
        rasterized=True,
    )
    for boundary in REGION_BOUNDS[1:-1]:
        axis.axvline(boundary, color="white", linewidth=0.75, alpha=0.92)
    for boundary in np.arange(0.5, len(SCOPES), 1.0):
        axis.axhline(boundary, color="white", linewidth=0.55, alpha=0.82)
    axis.set_xlim(1, 720)
    axis.set_yticks(np.arange(len(SCOPES)), labels=[rf"$s={scope}$" for scope in SCOPES])
    axis.set_xticks(
        [1, 96, 192, 336, 512, 720],
        labels=["1", "96", "192", "336", "512", "720"],
    )
    axis.set_xlabel("Future step", labelpad=3)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Sample-specific Scope Probabilities",
        loc="left",
        fontsize=8.8,
        fontweight="bold",
        color=DARK,
        pad=21,
    )
    axis.text(
        0.0,
        1.17,
        "Per-step soft allocation; white lines mark future-region boundaries",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.05,
        color=MID,
    )
    add_panel_label(axis, "c", x=-0.12, y=1.22)
    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.075,
        pad=0.22,
        aspect=22,
    )
    colorbar.set_label("Allocation probability", labelpad=2)
    colorbar.set_ticks(np.linspace(lower, upper, 6))
    colorbar.ax.tick_params(length=2, width=0.55, pad=1)
    colorbar.outline.set_linewidth(0.55)


def build_figure(
    sources: dict[str, dict[str, np.ndarray]],
    dataset: str,
    probe_row: int,
) -> plt.Figure:
    """Create a 183 x 142 mm asymmetric three-panel figure."""
    figure = plt.figure(
        figsize=(7.2047244094, 5.5905511811),
        constrained_layout=False,
    )
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.10, 1.0],
        width_ratios=[1.12, 0.88],
        left=0.09,
        right=0.975,
        top=0.91,
        bottom=0.11,
        hspace=0.47,
        wspace=0.24,
    )
    top = outer[0, :].subgridspec(2, 1, height_ratios=[1.0, 0.18], hspace=0.03)
    fused_axis = figure.add_subplot(top[0])
    strip_axis = figure.add_subplot(top[1])
    probability_axis = figure.add_subplot(outer[1, 1])

    arrays = sources[dataset]
    draw_fused_panel(fused_axis, strip_axis, arrays, probe_row)
    draw_scope_deviation_panel(figure, outer[1, 0], arrays, probe_row)
    draw_probability_panel(figure, probability_axis, arrays, probe_row)
    figure.text(
        0.52,
        0.022,
        "Validation-only sample-specific diagnostics · seed 2021",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MID,
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
        if image.width < 1800 or image.height < 1200:
            raise ValueError(f"unexpected PNG size: {image.size}")


def main() -> None:
    args = parse_args()
    configure_style()
    sources = load_diagnostics(args.diagnostic_root)
    dataset, probe_row, selection_audit, disagreement, distinct = select_probe(
        sources
    )
    trajectory_rows, probability_rows, regional_rows = build_source_rows(
        sources[dataset],
        dataset,
        probe_row,
    )
    source_dir = args.output_dir / "source_data"
    write_csv(source_dir / "selected_fused_and_scope_forecasts.csv", trajectory_rows)
    write_csv(source_dir / "selected_scope_probabilities.csv", probability_rows)
    write_csv(source_dir / "selected_regional_allocation.csv", regional_rows)
    write_csv(source_dir / "sample_selection_audit.csv", selection_audit)

    probabilities = sources[dataset]["probabilities"][probe_row]
    region_means = regional_probability_means(probabilities)
    dominant_scopes = [SCOPES[int(index)] for index in np.argmax(region_means, axis=1)]
    summary = {
        "split": "validation",
        "seed": 2021,
        "new_training": 0,
        "formal_test": 0,
        "selected_example": {
            "dataset": dataset,
            "probe_row": probe_row,
            "selection_pool": len(DATASETS) * 256,
            "selection_rule": (
                "maximize the number of distinct region-dominant scopes under "
                "mean soft allocation, then maximize mean pairwise absolute "
                "scope-forecast disagreement"
            ),
            "distinct_regional_dominant_scopes": distinct,
            "regional_dominant_scopes": dominant_scopes,
            "mean_pairwise_scope_disagreement": disagreement,
            "per_step_probability_min": float(probabilities.min()),
            "per_step_probability_max": float(probabilities.max()),
            "regional_mean_probability_min": float(region_means.min()),
            "regional_mean_probability_max": float(region_means.max()),
        },
        "panel_semantics": {
            "a": (
                "actual fused forecast; segment colour identifies the scope with "
                "the highest mean soft-allocation probability in that region"
            ),
            "b": (
                "each scope-conditioned forecast minus the actual fused forecast; "
                "all five rows use one shared deviation scale"
            ),
            "c": "actual per-step soft-allocation probabilities for the same probe",
        },
        "claim_boundary": (
            "The selected validation example illustrates non-identical scope "
            "forecasts and region-dependent soft reweighting. It does not estimate "
            "prevalence, establish sparse routing, or demonstrate oracle scope "
            "selection or causal specialization."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    figure = build_figure(sources, dataset, probe_row)
    save_figure(figure, args.output_dir)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
