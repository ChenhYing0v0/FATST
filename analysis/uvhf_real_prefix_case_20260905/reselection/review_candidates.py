"""Render shared prefixes for the five time-separated review candidates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    rows = pd.read_csv(OUT / "review_candidates.csv")
    baseline = {
        h: dict(np.load(BASE / f"raw/dlinear/h{h}.npz"))
        for h in (96, 192, 336, 720)
    }
    uvhf = np.load(BASE / "raw/uvhf_all_channels.npz")["prediction_scaled"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 12))
    for ax, row in zip(axes, rows.itertuples()):
        c, o = int(row.channel), int(row.origin)
        scale = baseline[720]["train_std"].reshape(-1)[c]
        mean = baseline[720]["train_mean"].reshape(-1)[c]
        ax.plot(
            np.arange(1, 97),
            baseline[720]["true"][o, :96, c] * scale + mean,
            color="black",
            label="GT",
        )
        ax.plot(
            np.arange(1, 97),
            uvhf[o, :96, c] * scale + mean,
            color="teal",
            label="UVHF",
        )
        for h, color in zip(
            (96, 192, 336, 720), ("#B75A3A", "#C6922C", "#4D78A8", "#82718C")
        ):
            ax.plot(
                np.arange(1, 97),
                baseline[h]["pred"][o, :96, c] * scale + mean,
                color=color,
                label=f"H{h}",
            )
        ax.set_title(
            f"Origin {o}, channel {c}, visibility {row.visibility96:.3f}, minimum gain {row.min_gain:.1%}"
        )
    axes[0].legend(ncol=6)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_review.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
