"""Compare full-horizon readability within the previously audited case pool."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HORIZONS = (96, 192, 336, 720)


def main() -> None:
    audited = pd.read_csv(BASE / "tail_audited/timemixer_all_candidate_audit.csv")
    eligible = audited[audited.audited_eligible].copy()
    uvhf = np.load(BASE / "raw/etth1_all_uvhf.npy", mmap_mode="r")
    exports = {
        h: dict(
            np.load(
                BASE / f"matched_checkpoints/timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HORIZONS
    }
    raw = pd.read_csv("/Users/river/PaperResearch/Project/datasets/ETT-small/ETTh1.csv")
    values = raw.iloc[:, 1:].to_numpy()
    mean, scale = exports[720]["train_mean"], exports[720]["train_std"]
    rows = []
    for record in eligible.itertuples():
        origin, channel = int(record.origin), int(record.channel)
        y = values[8640 + origin : 8640 + origin + 720, channel]
        u = uvhf[origin, :, channel].astype(float) * scale[channel] + mean[channel]
        pred = {
            h: exports[h]["pred"][origin, :, channel].astype(float) * scale[channel]
            + mean[channel]
            for h in HORIZONS
        }
        extent = np.ptp(np.concatenate([y, u, *pred.values()]))
        eu, eb = abs(u - y), abs(pred[720] - y)
        win = (eu <= 0.08 * extent) & (eb - eu >= 0.04 * extent)
        loss = (eb <= 0.08 * extent) & (eu - eb >= 0.04 * extent)
        power = abs(np.fft.rfft(y - y.mean())) ** 2
        power[0] = 0
        dominant_period = 720 / np.argmax(power)
        frequencies = np.fft.rfftfreq(720)
        rows.append(
            {
                "origin": origin,
                "channel": channel,
                "channel_name": raw.columns[channel + 1],
                "dominant_period": dominant_period,
                "prominent_peaks": len(
                    find_peaks(y, prominence=0.3 * extent, distance=8)[0]
                ),
                "high_frequency_share": power[frequencies > 1 / 48].sum() / power.sum(),
                "normalized_mae_advantage": (eb - eu).mean() / extent,
                "visible_win_fraction": win.mean(),
                "visible_loss_fraction": loss.mean(),
                "visible_net": win.mean() - loss.mean(),
                "tail_visible_net": win[336:].mean() - loss[336:].mean(),
            }
        )
    result = eligible.merge(
        pd.DataFrame(rows), on=["origin", "channel"], validate="one_to_one"
    )
    current = result[(result.origin == 947) & (result.channel == 2)].iloc[0]
    result["less_dense"] = (
        (result.prominent_peaks <= 0.75 * current.prominent_peaks)
        & (result.visible_net >= current.visible_net)
        & (result.gain_h720 >= current.gain_h720)
    )
    result["clearer_same_density"] = (
        (result.visible_net >= current.visible_net + 0.05)
        & (result.gain_h720 >= current.gain_h720 + 0.1)
        & (result.full_r2 >= current.full_r2 - 0.05)
        & (result.tail_r2 >= current.tail_r2 - 0.05)
    )
    result.to_csv(OUT / "eligible_readability_audit.csv", index=False)
    selections = [current.to_dict()]
    for direction, columns, ascending in [
        (
            "less_dense",
            ["prominent_peaks", "visible_net", "tail_visible_net"],
            [True, False, False],
        ),
        ("clearer_same_density", ["visible_net", "tail_visible_net"], [False, False]),
    ]:
        ranked = result[result[direction]].sort_values(columns, ascending=ascending)
        ranked.to_csv(OUT / f"{direction}_candidates.csv", index=False)
        count = 0
        for _, row in ranked.iterrows():
            if all(
                row.channel != r["channel"] or abs(row.origin - r["origin"]) >= 96
                for r in selections
            ):
                entry = row.to_dict()
                entry["review_direction"] = direction
                selections.append(entry)
                count += 1
                if count == 3:
                    break
    pd.DataFrame(selections).to_csv(OUT / "review_candidates.csv", index=False)
    summary = {
        "existing_audit_cells": len(audited),
        "unchanged_gates_eligible": len(result),
        "channel_counts": result.channel_name.value_counts().to_dict(),
        "dominant_period_range": [
            float(result.dominant_period.min()),
            float(result.dominant_period.max()),
        ],
        "prominent_peaks_range": [
            int(result.prominent_peaks.min()),
            int(result.prominent_peaks.max()),
        ],
        "less_dense_count": int(result.less_dense.sum()),
        "clearer_same_density_count": int(result.clearer_same_density.sum()),
        "current_case": current.to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(
        pd.DataFrame(selections)[
            [
                "origin",
                "channel_name",
                "visible_net",
                "tail_visible_net",
                "gain_h720",
                "full_r2",
                "tail_r2",
                "last192_r2",
                "prominent_peaks",
                "visibility96",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
