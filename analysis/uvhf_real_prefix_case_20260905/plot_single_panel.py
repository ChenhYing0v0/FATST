"""Fuse real forecasts and evidence annotations into one linear coordinate plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import pandas as pd

OUT = Path(__file__).resolve().parent
HORIZONS = (96, 192, 336, 720)
COLORS = {96: "#B75A3A", 192: "#C6922C", 336: "#4D78A8", 720: "#82718C"}
MARKERS = {96: "o", 192: "s", 336: "^", 720: "D"}
TRUTH, UVHF = "#252A31", "#087F79"


def main() -> None:
    source = pd.read_csv(OUT / "source_data.csv")
    future = source[source.step > 0].set_index("step")
    history = source[(source.step >= -47) & (source.step <= 0)]
    scores = pd.read_csv(OUT / "selected_metrics.csv").pivot(
        index="horizon", columns="model", values="mse_scaled"
    )
    pairs = pd.read_csv(OUT / "selected_pair_disagreement.csv")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    width_in, height_in = 183 / 25.4, 110 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.235, top=0.79)
    fig.text(
        0.08,
        0.962,
        "Consistent prefixes, lower forecast error",
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.08,
        0.919,
        "ETTh2 · oil temperature · identical 720-step history · selected validation example",
        fontsize=7.5,
        color="#626A73",
    )
    handles = [
        Line2D([], [], color=TRUTH, lw=1.15, label="Ground truth"),
        Line2D([], [], color=UVHF, lw=1.45, label="UVHF (one model)"),
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLORS[h],
            marker=MARKERS[h],
            lw=1,
            markersize=3,
            label=f"DLinear H={h}",
        )
        for h in HORIZONS
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.071, 0.899),
        fontsize=7,
        handlelength=2.1,
        columnspacing=2.8,
        labelspacing=0.7,
    )

    ax.axvspan(-47, 0, color="#ECEEEF", zorder=0)
    ax.axvspan(1, 96, color="#ECF2F8", zorder=0)
    ax.axvline(0, color="#9BA3AB", ls="--", lw=0.7, zorder=1)
    for h in HORIZONS:
        ax.axvline(h, color="#D4D9DD", ls=(0, (2, 3)), lw=0.6, zorder=0)
    ax.plot(history.step, history.history, color="#A0A6AD", lw=0.95)
    ax.plot(future.index, future.ground_truth, color=TRUTH, lw=1.05, zorder=3)
    for j, h in enumerate(reversed(HORIZONS)):
        values = future[f"dlinear_h{h}"].iloc[:h]
        ax.plot(
            values.index,
            values,
            color=COLORS[h],
            lw=0.95,
            marker=MARKERS[h],
            markersize=2.5,
            markevery=[min(16 + j * 17, h - 1), h - 1],
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=4 + j,
            path_effects=[
                pe.Stroke(linewidth=1.3, foreground="white"),
                pe.Normal(),
            ],
        )
    ax.plot(future.index, future.uvhf, color=UVHF, lw=1.4, zorder=8)
    ax.scatter(
        HORIZONS,
        future.loc[list(HORIZONS), "uvhf"],
        s=14,
        color=UVHF,
        edgecolor="white",
        linewidth=0.45,
        zorder=9,
    )
    for h in HORIZONS:
        ax.text(
            h,
            54.3,
            f"H={h}",
            fontsize=7,
            color="#6F7881",
            ha="center",
            va="bottom",
        )
    ax.text(
        48,
        52.9,
        "Shared prefix",
        ha="center",
        va="bottom",
        color="#68829C",
        fontsize=6.5,
        bbox={
            "facecolor": "#ECF2F8",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 1,
        },
    )

    # This real pointwise range is an illustration, not an uncertainty interval.
    step = 68
    values = future.loc[step, [f"dlinear_h{h}" for h in HORIZONS]]
    low, high = float(values.min()), float(values.max())
    ax.vlines(step, low, high, color="#505963", linewidth=0.9, zorder=11)
    ax.hlines(
        [low, high],
        step - 3,
        step + 3,
        color="#505963",
        linewidth=0.9,
        zorder=11,
    )
    ax.annotate(
        f"Same future step, different values\nDLinear spread at step {step}: {high - low:.2f} °C",
        xy=(step, low),
        xytext=(130, 20.6),
        fontsize=6.8,
        color="#505963",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "color": "#85909B",
            "lw": 0.65,
            "connectionstyle": "angle,angleA=0,angleB=-90,rad=3",
        },
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        zorder=12,
    )
    ax.text(
        735,
        future.loc[720, "uvhf"],
        "UVHF",
        color=UVHF,
        fontsize=7,
        weight="bold",
        va="center",
    )
    ax.text(
        735,
        future.loc[720, "dlinear_h720"],
        "DLinear",
        color=COLORS[720],
        fontsize=7,
        va="center",
    )
    ax.set_xlim(-47, 802)
    ax.set_ylim(18.5, 56.5)
    ax.set_xticks([0, 96, 192, 336, 480, 600, 720])
    ax.set_yticks([20, 30, 40, 50])
    ax.set_xlabel("Forecast step (hours)", labelpad=5)
    ax.set_ylabel("Oil temperature (°C)")
    ax.grid(axis="y", color="#E5E8EB", lw=0.45)
    ax.tick_params(length=3, width=0.65)

    gains = 100 * (1 - scores.UVHF / scores.DLinear)
    gain_text = "    ".join(f"H{h}: −{gains.loc[h]:.1f}%" for h in HORIZONS)
    fig.text(
        0.08, 0.12, "MSE vs DLinear", color=TRUTH, weight="bold", fontsize=7
    )
    fig.text(0.27, 0.12, gain_text, color=UVHF, weight="bold", fontsize=7)
    fig.text(
        0.08,
        0.077,
        f"Mean cross-horizon disagreement: DLinear {pairs.chpd_raw.mean():.2f} °C   |   UVHF 0 (identical prefixes)",
        fontsize=7,
    )
    fig.text(
        0.08,
        0.033,
        "MSE uses each complete horizon; disagreement averages 6 pairs over their full overlaps. Selected case, not a population estimate.",
        fontsize=6.2,
        color="#69717A",
    )
    assert len(fig.axes) == 1
    fig.savefig(OUT / "uvhf_real_prefix_single.pdf")
    fig.savefig(OUT / "uvhf_real_prefix_single.svg")
    svg = OUT / "uvhf_real_prefix_single.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines())
        + "\n"
    )
    fig.savefig(OUT / "uvhf_real_prefix_single.png", dpi=300)
    fig.savefig(
        OUT / "uvhf_real_prefix_single.tiff",
        dpi=1000,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
