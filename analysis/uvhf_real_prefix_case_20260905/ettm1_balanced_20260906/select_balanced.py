"""Choose a central eligible example with competent baseline forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HS = (96, 192, 336, 720)


def main() -> None:
    old = pd.read_csv(BASE / "ettm1_segments_20260906/all_candidate_audit.csv")
    ex = {
        h: dict(
            np.load(
                BASE
                / f"matched_checkpoints/ettm1_timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HS
    }
    raw = (
        pd.read_csv("/Users/river/PaperResearch/Project/datasets/ETT-small/ETTm1.csv")
        .iloc[:, 1:]
        .to_numpy()
    )
    rows = []
    for c in range(7):
        y = np.lib.stride_tricks.sliding_window_view(
            (raw[:, c] - ex[720]["train_mean"][c]) / ex[720]["train_std"][c], 720
        )[34560:45361]
        data = {"origin": np.arange(10801), "channel": c}
        for h in HS:
            p = ex[h]["pred"][:, :, c].astype(float)
            for label, a, b in [("full", 0, h), ("prefix", 0, 96)] + (
                [("tail", 336, 720), ("last192", 528, 720)] if h == 720 else []
            ):
                v = y[:, a:b].var(1)
                v = np.where(v > 0, v, np.nan)
                data[f"timemixer_{label}_r2_h{h}"] = (
                    1 - ((p[:, a:b] - y[:, a:b]) ** 2).mean(1) / v
                )
        rows.append(pd.DataFrame(data))
    table = old.merge(
        pd.concat(rows, ignore_index=True),
        on=["origin", "channel"],
        validate="one_to_one",
    )
    table["baseline_competent"] = (
        (table[[f"timemixer_full_r2_h{h}" for h in HS]].min(axis=1) >= 0.5)
        & (table[[f"timemixer_prefix_r2_h{h}" for h in HS]].min(axis=1) >= 0.3)
        & (table.timemixer_tail_r2_h720 >= 0.4)
        & (table.timemixer_last192_r2_h720 >= 0.4)
    )
    table["balanced_gain"] = (table[[f"gain_h{h}" for h in HS]].max(axis=1) <= 0.6) & (
        table[[f"mse_gain_1_96_h{h}" for h in HS]].max(axis=1) <= 0.7
    )
    table["balanced_prefix"] = (
        (table.visibility96 >= 0.10)
        & (table.persistent_fraction96 >= 0.5)
        & (table.min_prefix_mse_gain >= 0.10)
    )
    table["balanced_eligible"] = (
        table.fit_eligible
        & table.segment_eligible
        & table.baseline_competent
        & table.balanced_gain
        & table.balanced_prefix
    )
    newcols = ["origin", "channel"] + [c for c in table if c not in old.columns]
    table[newcols].to_csv(OUT / "baseline_audit.csv", index=False)
    eligible = table[table.balanced_eligible].copy()
    cols = [f"gain_h{h}" for h in HS] + [
        "visibility96",
        "timemixer_full_r2_h720",
        "full_r2",
    ]
    eligible["centrality_distance"] = (
        (eligible[cols].rank(pct=True) - 0.5).abs().sum(axis=1)
    )
    eligible = eligible.sort_values(["centrality_distance", "origin", "channel"])
    eligible.to_csv(OUT / "eligible.csv", index=False)
    selected = []
    for _, r in eligible.iterrows():
        if all(
            r.channel != s.channel or abs(r.origin - s.origin) >= 96 for s in selected
        ):
            selected.append(r)
        if len(selected) == 3:
            break
    # Recorded visual-review extension after the first case's late reversal.
    # Inspect the next unreviewed eligible case whose final 192 steps improve.
    if selected[0].last192_r2 < selected[0].timemixer_last192_r2_h720:
        for _, r in eligible.iterrows():
            already_selected = any(
                r.channel == s.channel and r.origin == s.origin for s in selected
            )
            if not already_selected and r.last192_r2 >= r.timemixer_last192_r2_h720:
                selected.append(r)
                break
    pd.DataFrame(selected).to_csv(OUT / "review_candidates.csv", index=False)
    summary = {
        k: int(table[k].sum())
        for k in [
            "baseline_competent",
            "balanced_gain",
            "balanced_prefix",
            "balanced_eligible",
        ]
    }
    summary["baseline_full720_r2_quantiles"] = table.timemixer_full_r2_h720.quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9]
    ).to_dict()
    (OUT / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
    if selected:
        print(
            pd.DataFrame(selected)[
                [
                    "origin",
                    "channel",
                    "visibility96",
                    "min_prefix_mse_gain",
                    "gain_h96",
                    "gain_h192",
                    "gain_h336",
                    "gain_h720",
                    "full_r2",
                    "tail_r2",
                    "timemixer_full_r2_h720",
                    "centrality_distance",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
