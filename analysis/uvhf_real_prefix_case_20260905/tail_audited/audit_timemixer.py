"""Audit native TimeMixer predictions under the unchanged 96-step gates."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
HORIZONS = (96, 192, 336, 720)


def main() -> None:
    old = pd.read_csv(OUT / "etth1_all_candidate_audit.csv")
    u = np.load(BASE / "raw/etth1_all_uvhf.npy").astype(float)
    exports = {
        h: dict(
            np.load(
                BASE
                / f"matched_checkpoints/timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HORIZONS
    }
    mean, scale = exports[720]["train_mean"], exports[720]["train_std"]
    raw = (
        pd.read_csv(
            "/Users/river/PaperResearch/Project/datasets/ETT-small/ETTh1.csv"
        )
        .iloc[:, 1:]
        .to_numpy()
    )
    y = np.stack(
        [(raw[8640 + o : 8640 + o + 720] - mean) / scale for o in range(2161)]
    )
    rows = []
    for c in range(7):
        pred = {h: exports[h]["pred"][:, :, c].astype(float) for h in HORIZONS}
        common = np.stack([pred[h][:, :96] for h in HORIZONS], axis=1)
        span = np.ptp(common, axis=1)
        values = np.concatenate(
            [common, u[:, None, :96, c], y[:, None, :96, c]], axis=1
        )
        extent = np.ptp(values.reshape(2161, -1), axis=1)
        gains = np.stack(
            [
                1
                - np.mean((u[:, :h, c] - y[:, :h, c]) ** 2, axis=1)
                / np.mean((pred[h] - y[:, :h, c]) ** 2, axis=1)
                for h in HORIZONS
            ],
            axis=1,
        )
        common_d = np.stack(
            [
                np.mean((pred[h][:, :96] - y[:, :96, c]) ** 2, axis=1)
                for h in HORIZONS
            ],
            axis=1,
        )
        common_u = np.mean((u[:, :96, c] - y[:, :96, c]) ** 2, axis=1)
        for o in range(2161):
            rows.append(
                {
                    "origin": o,
                    "channel": c,
                    "min_gain": gains[o].min(),
                    "visibility96": span[o].mean() / extent[o],
                    "persistent_fraction96": np.mean(
                        span[o] > 0.1 * extent[o]
                    ),
                    "accuracy_eligible": bool(
                        gains[o].min() > 0
                        and y[o, :, c].std() >= 0.25
                        and common_u[o] < common_d[o].min()
                    ),
                    **{
                        f"gain_h{h}": gains[o, j]
                        for j, h in enumerate(HORIZONS)
                    },
                }
            )
    new = pd.DataFrame(rows)
    keep = ["origin", "channel", "fit_eligible"] + [
        x for x in old if x.startswith(("full_", "tail_", "last192_"))
    ]
    result = new.merge(
        old[keep], on=["origin", "channel"], validate="one_to_one"
    )
    result["audited_eligible"] = (
        result.accuracy_eligible
        & result.fit_eligible
        & (result.visibility96 >= 0.075)
    )
    result.to_csv(OUT / "timemixer_all_candidate_audit.csv", index=False)
    ranked = result[result.audited_eligible].sort_values(
        ["visibility96", "tail_r2"], ascending=False
    )
    ranked.to_csv(OUT / "timemixer_eligible.csv", index=False)
    candidates = []
    for _, row in ranked.iterrows():
        if all(
            row.channel != r.channel or abs(row.origin - r.origin) >= 96
            for r in candidates
        ):
            candidates.append(row)
        if len(candidates) == 5:
            break
    pd.DataFrame(candidates).to_csv(
        OUT / "timemixer_review_candidates.csv", index=False
    )
    print("Eligible", len(ranked))
    print(
        pd.DataFrame(candidates)[
            [
                "origin",
                "channel",
                "visibility96",
                "min_gain",
                "full_r2",
                "tail_r2",
                "last192_r2",
            ]
        ].to_string(index=False)
        if candidates
        else "No eligible case"
    )


if __name__ == "__main__":
    main()
