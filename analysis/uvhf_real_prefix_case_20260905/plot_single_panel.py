"""Fuse real forecasts and evidence annotations into one linear coordinate plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch, Rectangle
import pandas as pd

OUT = Path(__file__).resolve().parent
HORIZONS = (96, 192, 336, 720)
COLORS = {96: "#B75A3A", 192: "#C6922C", 336: "#4D78A8", 720: "#82718C"}
MARKERS = {96: "o", 192: "s", 336: "^", 720: "D"}
TRUTH, UVHF = "#252A31", "#087F79"


def main(zoom: bool = False, output: Path = OUT) -> None:
    settings_path = output / "figure_settings.json"
    settings = (
        json.loads(settings_path.read_text()) if settings_path.exists() else {}
    )
    source = pd.read_csv(output / "source_data.csv")
    future = source[source.step > 0].set_index("step")
    history = source[(source.step >= -47) & (source.step <= 0)]
    scores = pd.read_csv(output / "selected_metrics.csv").pivot(
        index="horizon", columns="model", values="mse_scaled"
    )
    pairs = pd.read_csv(output / "selected_pair_disagreement.csv")
    unit = settings.get("unit", "°C")
    baseline_label = settings.get("baseline", "DLinear")
    baseline_prefix = settings.get("baseline_prefix", "dlinear")
    prefix_ylim = settings.get("prefix_ylim", [26, 53])
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
    width_in, height_in = 183 / 25.4, (135 if zoom else 110) / 25.4
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
        settings.get(
            "subtitle",
            "ETTh2 · oil temperature · identical 720-step history · selected validation example",
        ),
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
            label=f"{baseline_label} H={h}",
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
        values = future[f"{baseline_prefix}_h{h}"].iloc[:h]
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
            h - 24 if zoom and h == 336 else h,
            settings.get("horizon_y", {}).get(
                str(h), 46 if zoom and h == 720 else 54.3
            ),
            f"H={h}",
            fontsize=7,
            color="#6F7881",
            ha="center",
            va="bottom",
        )
    ax.text(
        48,
        settings.get("prefix_label_y", 52.9),
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
    step = settings.get("annotation_step", 68)
    values = future.loc[step, [f"{baseline_prefix}_h{h}" for h in HORIZONS]]
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
        f"Same future step, different values\n{baseline_label} spread at step {step}: {high - low:.2f} {unit}",
        xy=(step, low),
        xytext=(130, settings.get("annotation_y", 20.6)),
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
        settings.get("endpoint_label_y", {}).get("UVHF", future.loc[720, "uvhf"]),
        "UVHF",
        color=UVHF,
        fontsize=7,
        weight="bold",
        va="center",
    )
    ax.text(
        735,
        settings.get("endpoint_label_y", {}).get(
            baseline_label, future.loc[720, f"{baseline_prefix}_h720"]
        ),
        baseline_label,
        color=COLORS[720],
        fontsize=7,
        va="center",
    )
    for label, col, color in [
        ("UVHF", "uvhf", UVHF),
        (baseline_label, f"{baseline_prefix}_h720", COLORS[720]),
    ]:
        if label in settings.get("endpoint_label_y", {}):
            ax.plot(
                [722, 733],
                [future.loc[720, col], settings["endpoint_label_y"][label]],
                color=color,
                lw=0.5,
            )
    ax.set_xlim(-47, 802)
    ax.set_ylim(settings.get("main_ylim", [18.5, 83.5 if zoom else 56.5]))
    ax.set_xticks([0, 96, 192, 336, 480, 600, 720])
    ax.set_yticks(settings.get("main_yticks", [20, 30, 40, 50]))
    ax.set_xlabel("Forecast step (hours)", labelpad=5)
    ax.set_ylabel(settings.get("ylabel", "Oil temperature (°C)"))
    ax.grid(axis="y", color="#E5E8EB", lw=0.45)
    ax.tick_params(length=3, width=0.65)

    if zoom:
        # Repeat all six unmodified prefixes on explicit linear inset axes.
        inset = ax.inset_axes([0.49, 0.50, 0.49, 0.47], zorder=15)
        prefix = future.loc[1:96]
        inset.set_facecolor("#F8FAFC")
        inset.plot(prefix.index, prefix.ground_truth, color=TRUTH, lw=0.9)
        for j, h in enumerate(HORIZONS):
            inset.plot(
                prefix.index,
                prefix[f"{baseline_prefix}_h{h}"],
                color=COLORS[h],
                lw=0.85,
                marker=MARKERS[h],
                markevery=[10 + j * 4, step - 1, 90 - j * 4],
                markersize=2.8,
                markeredgecolor="white",
                markeredgewidth=0.3,
            )
        inset.plot(prefix.index, prefix.uvhf, color=UVHF, lw=1.25)
        inset.axvline(step, color="#AAB2BA", ls=":", lw=0.6)
        inset.vlines(step, low, high, color="#505963", lw=0.85)
        inset.hlines(
            [low, high], step - 1.5, step + 1.5, color="#505963", lw=0.85
        )
        inset.set(
            xlim=(1, 96),
            ylim=prefix_ylim,
            xticks=[1, 24, 48, 72, 96],
            yticks=settings.get("prefix_yticks", [30, 40, 50]),
        )
        inset.set_title(
            "Shared prefix enlarged · steps 1–96",
            fontsize=7.5,
            loc="left",
            pad=5,
            weight="bold",
        )
        inset.set_xlabel("Forecast step (hours)", fontsize=6.5, labelpad=2)
        inset.set_ylabel(
            settings.get("ylabel", "Temperature (°C)"),
            fontsize=6.5,
            labelpad=2,
        )
        inset.tick_params(labelsize=6.5, length=2.5)
        inset.grid(axis="y", color="#E1E6EB", lw=0.4)
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#9BAABD")
            spine.set_linewidth(0.6)
        ax.add_patch(
            Rectangle(
                (1, prefix_ylim[0]),
                95,
                prefix_ylim[1] - prefix_ylim[0],
                fill=False,
                edgecolor="#8FA3B8",
                lw=0.8,
                zorder=10,
            )
        )
        connector = ConnectionPatch(
            xyA=(96, prefix_ylim[1]),
            coordsA=ax.transData,
            xyB=settings.get("connector_corner", [0, 1]),
            coordsB=inset.transAxes,
            color="#8FA3B8",
            lw=0.7,
            ls="--",
            zorder=14,
        )
        fig.add_artist(connector)
        assert len(ax.child_axes) == 1
        for line in inset.lines[:6]:
            assert len(line.get_xdata()) == 96

    gains = 100 * (1 - scores.UVHF / scores[baseline_label])
    gain_text = "    ".join(f"H{h}: −{gains.loc[h]:.1f}%" for h in HORIZONS)
    fig.text(
        0.08,
        0.12,
        f"MSE vs {baseline_label}",
        color=TRUTH,
        weight="bold",
        fontsize=7,
    )
    fig.text(0.27, 0.12, gain_text, color=UVHF, weight="bold", fontsize=7)
    fig.text(
        0.08,
        0.077,
        f"Mean cross-horizon disagreement: {baseline_label} {pairs.chpd_raw.mean():.2f} {unit}   |   UVHF 0 (identical prefixes)",
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
    stem = "uvhf_real_prefix_zoom" if zoom else "uvhf_real_prefix_single"
    fig.savefig(output / f"{stem}.pdf")
    fig.savefig(output / f"{stem}.svg")
    svg = output / f"{stem}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines())
        + "\n"
    )
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(
        output / f"{stem}.tiff",
        dpi=1000,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Add an inset of the complete shared prefix.",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    main(zoom=args.zoom, output=args.output)
