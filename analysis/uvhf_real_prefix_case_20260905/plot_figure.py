"""Render a real-data matched-history forecasting case with auditable metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
HORIZONS = (96, 192, 336, 720)
COLORS = {96: "#B75A3A", 192: "#C6922C", 336: "#4D78A8", 720: "#82718C"}
MARKERS = {96: "o", 192: "s", 336: "^", 720: "D"}
TRUTH = "#252A31"
UVHF = "#087F79"


def main() -> None:
    frame = pd.read_csv(OUT / "source_data.csv")
    data = frame[frame.step > 0]
    audit = json.loads((OUT / "selection_audit.json").read_text())
    metrics = pd.read_csv(OUT / "selected_metrics.csv")
    pairs = pd.read_csv(OUT / "selected_pair_disagreement.csv")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    width_in, height_in = 183 / 25.4, 157 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.98,
        bottom=0.145,
        top=0.82,
        hspace=0.44,
        wspace=0.25,
        height_ratios=[1, 1.34],
    )
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[0, 1], sharex=a, sharey=a)
    zoom_grid = grid[1, 0].subgridspec(
        2, 1, height_ratios=[1.65, 1], hspace=0.16
    )
    c = fig.add_subplot(zoom_grid[0])
    delta = fig.add_subplot(zoom_grid[1], sharex=c)
    d = fig.add_subplot(grid[1, 1])
    fig.text(
        0.09,
        0.965,
        "One forecast origin, four requested horizons",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.09,
        0.928,
        f"ETTh2 · {audit['channel_name']} · same 720-step input · selected validation example",
        fontsize=7.5,
        color="#50565E",
    )
    handles = [
        Line2D([], [], color=TRUTH, lw=1.2, label="Ground truth"),
        Line2D([], [], color=UVHF, lw=1.4, label="UVHF"),
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLORS[h],
            lw=1,
            marker=MARKERS[h],
            markersize=3,
            label=f"DLinear H={h}",
        )
        for h in HORIZONS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.083, 0.905),
        ncol=6,
        fontsize=6.5,
        handlelength=1.7,
        columnspacing=1.15,
    )

    hist = frame[(frame.step <= 0) & (frame.step >= -95)]
    future = data.step.to_numpy()
    all_values = np.r_[
        hist.history.to_numpy(),
        data.ground_truth.to_numpy(),
        data.uvhf.to_numpy(),
        *[data[f"dlinear_h{h}"].to_numpy()[:h] for h in HORIZONS],
    ]
    lo, hi = float(all_values.min()), float(all_values.max())
    padding = (hi - lo) * 0.12
    for axis in (a, b):
        axis.axvspan(-95, 0, color="#F0F1F2", zorder=0)
        axis.axvspan(1, 96, color="#EAF1F7", alpha=0.75, zorder=0)
        axis.axvline(0, color="#959BA1", linestyle="--", lw=0.7)
        axis.plot(hist.step, hist.history, color="#90969D", lw=0.9)
        axis.plot(future, data.ground_truth, color=TRUTH, lw=1.1, zorder=7)
        axis.set_xlim(-95, 730)
        axis.set_ylim(lo - padding, hi + padding)
        axis.set_xticks([0, 96, 336, 720])
        axis.set_xlabel("Forecast step (hours)")
        axis.grid(axis="y", color="#E5E7EB", lw=0.4)
    for h in reversed(HORIZONS):
        vals = data[f"dlinear_h{h}"].to_numpy()[:h]
        a.plot(
            future[:h],
            vals,
            color=COLORS[h],
            lw=0.95,
            marker=MARKERS[h],
            markersize=2.5,
            markevery=[h - 1],
            zorder=4,
        )
    b.plot(future, data.uvhf, color=UVHF, lw=1.3, zorder=8)
    for h in HORIZONS:
        b.plot(
            h,
            data.uvhf.iloc[h - 1],
            marker="o",
            markersize=3,
            color=UVHF,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=9,
        )
    a.set_title(
        "a  DLinear: four independent models", loc="left", weight="bold", pad=7
    )
    b.set_title(
        "b  UVHF: one unified trajectory", loc="left", weight="bold", pad=7
    )
    a.set_ylabel("Oil temperature (°C)")
    a.text(
        0.98,
        0.95,
        "96 / 192 / 336 / 720-step forecasts",
        transform=a.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#5D626A",
    )
    b.text(
        0.98,
        0.95,
        "Four prefixes of the same forecast",
        transform=b.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=UVHF,
    )
    b.tick_params(labelleft=False)

    common = data.iloc[:96]
    x = common.step.to_numpy()
    baseline = np.stack([common[f"dlinear_h{h}"].to_numpy() for h in HORIZONS])
    # The shaded region is a min-max forecast envelope, not uncertainty.
    c.fill_between(
        x,
        baseline.min(axis=0),
        baseline.max(axis=0),
        color="#B4BCC8",
        alpha=0.22,
    )
    for j, h in enumerate(HORIZONS):
        c.plot(
            x,
            baseline[j],
            color=COLORS[h],
            lw=0.95,
            marker=MARKERS[h],
            markevery=(j * 3 + 5, 24),
            markersize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
    c.plot(x, common.ground_truth, color=TRUTH, lw=1.2, zorder=7)
    c.plot(x, common.uvhf, color=UVHF, lw=1.45, zorder=8)
    c.set_xlim(1, 96)
    c.set_xticks([1, 24, 48, 72, 96])
    c.tick_params(labelbottom=False)
    c.set_ylabel("Oil temperature (°C)")
    c.set_title(
        "c  A closer look at the shared first 96 steps",
        loc="left",
        weight="bold",
        pad=7,
    )
    c.grid(axis="y", color="#E5E7EB", lw=0.4)
    for j, h in enumerate(HORIZONS[:-1]):
        delta.plot(
            x,
            baseline[j] - baseline[-1],
            color=COLORS[h],
            lw=0.95,
            marker=MARKERS[h],
            markevery=(j * 3 + 5, 24),
            markersize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
    delta.axhline(0, color=UVHF, lw=1.2)
    delta.set_ylabel("Within-system Δ\nvs H720 (°C)", fontsize=6.5)
    delta.set_xlabel("Shared forecast step (hours)")
    delta.text(
        0.97,
        0.91,
        "UVHF: Δ = 0",
        transform=delta.transAxes,
        ha="right",
        va="top",
        color=UVHF,
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    delta.grid(axis="y", color="#E5E7EB", lw=0.4)

    dm = (
        metrics[metrics.model == "DLinear"]
        .set_index("horizon")
        .loc[list(HORIZONS), "mse_scaled"]
        .to_numpy()
    )
    um = (
        metrics[metrics.model == "UVHF"]
        .set_index("horizon")
        .loc[list(HORIZONS), "mse_scaled"]
        .to_numpy()
    )
    for j, (dv, uv) in enumerate(zip(dm, um)):
        d.plot([j, j], [uv, dv], color="#BDC4CB", lw=1.1, zorder=1)
    d.plot(
        np.arange(4),
        dm,
        marker="s",
        markersize=4,
        color="#B75A3A",
        ls="none",
        label="DLinear",
    )
    d.plot(
        np.arange(4),
        um,
        marker="o",
        markersize=4,
        color=UVHF,
        ls="none",
        label="UVHF",
    )
    for j, (dv, uv) in enumerate(zip(dm, um)):
        d.text(
            j + 0.10,
            (dv + uv) / 2,
            f"−{100 * (1 - uv / dv):.1f}%",
            fontsize=7,
            color=UVHF,
            va="center",
        )
    d.set_xticks(np.arange(4), HORIZONS)
    d.set_xlim(-0.4, 3.8)
    d.set_ylim(0, max(dm.max(), um.max()) * 1.24)
    d.set_xlabel("Requested horizon")
    d.set_ylabel("MSE (train-standardized)")
    d.set_title(
        "d  Lower error at all four horizons", loc="left", weight="bold", pad=7
    )
    d.grid(axis="y", color="#E5E7EB", lw=0.4)
    d.legend(loc="upper left", fontsize=6.5, ncol=2, handletextpad=0.3)
    chpd = pairs.chpd_raw.mean()
    fig.text(
        0.09,
        0.068,
        f"Cross-horizon disagreement (mean of 6 pairs): DLinear {chpd:.2f} °C  |  UVHF 0 (prefix identity)",
        fontsize=7,
        weight="bold",
    )
    fig.text(
        0.09,
        0.035,
        f"Selected from {audit['candidate_count']:,} validation origin–variable pairs; illustrative, not a population estimate. No curve smoothing.",
        fontsize=6.5,
        color="#656B73",
    )
    for axis in (a, b, c, delta, d):
        axis.tick_params(length=2.5, width=0.6)
    fig.savefig(OUT / "uvhf_real_prefix_case.pdf")
    fig.savefig(OUT / "uvhf_real_prefix_case.svg")
    svg_path = OUT / "uvhf_real_prefix_case.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
        + "\n"
    )
    fig.savefig(OUT / "uvhf_real_prefix_case.png", dpi=300)
    fig.savefig(
        OUT / "uvhf_real_prefix_case.tiff",
        dpi=1000,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
