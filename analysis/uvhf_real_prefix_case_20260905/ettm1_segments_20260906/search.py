"""Screen existing ETTm1 forecasts for segment-wise skill and visible prefixes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HS = (96, 192, 336, 720)


def main() -> None:
    uvhf = np.load(BASE / "raw/ettm1_all_uvhf.npy", mmap_mode="r")
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
        u = uvhf[:, :, c].astype(float)
        p = {h: ex[h]["pred"][:, :, c].astype(float) for h in HS}
        common = np.stack([p[h][:, :96] for h in HS], axis=1)
        allprefix = np.concatenate([common, u[:, None, :96], y[:, None, :96]], axis=1)
        extent = np.ptp(allprefix.reshape(10801, -1), axis=1)
        span = np.ptp(common, axis=1)
        data = {
            "origin": np.arange(10801),
            "channel": np.full(10801, c),
            "visibility96": span.mean(1) / extent,
            "persistent_fraction96": (span > 0.1 * extent[:, None]).mean(1),
        }
        gains = []
        mg = []
        ag = []
        for h in HS:
            gain = 1 - ((u[:, :h] - y[:, :h]) ** 2).mean(1) / (
                (p[h] - y[:, :h]) ** 2
            ).mean(1)
            data[f"gain_h{h}"] = gain
            gains.append(gain)
        for a, b in [(0, 96), (96, 192), (192, 336), (336, 720)]:
            for h in HS:
                if h < b:
                    continue
                eu = u[:, a:b] - y[:, a:b]
                eb = p[h][:, a:b] - y[:, a:b]
                m = 1 - (eu**2).mean(1) / (eb**2).mean(1)
                t = 1 - abs(eu).mean(1) / abs(eb).mean(1)
                data[f"mse_gain_{a+1}_{b}_h{h}"] = m
                data[f"mae_gain_{a+1}_{b}_h{h}"] = t
                mg.append(m)
                ag.append(t)
        for label, a in [("full", 0), ("tail", 336), ("last192", 528)]:
            x, z = u[:, a:], y[:, a:]
            xc = x - x.mean(1)[:, None]
            zc = z - z.mean(1)[:, None]
            std = z.std(1)
            std = np.where(std > 0, std, np.nan)
            data[f"{label}_r2"] = 1 - ((x - z) ** 2).mean(1) / std**2
            data[f"{label}_corr"] = (xc * zc).sum(1) / np.maximum(
                np.linalg.norm(xc, axis=1) * np.linalg.norm(zc, axis=1), 1e-12
            )
            data[f"{label}_amplitude_ratio"] = x.std(1) / std
            data[f"{label}_bias_sigma"] = abs((x - z).mean(1)) / std
        data["min_segment_mse_gain"] = np.min(mg, axis=0)
        data["min_segment_mae_gain"] = np.min(ag, axis=0)
        data["min_prefix_mse_gain"] = np.min(
            [data[f"mse_gain_1_96_h{h}"] for h in HS], axis=0
        )
        data["fit_eligible"] = (
            (data["full_r2"] >= 0.7)
            & (data["tail_r2"] >= 0.65)
            & (data["last192_r2"] >= 0.5)
            & (data["tail_corr"] >= 0.7)
            & (data["tail_amplitude_ratio"] >= 0.5)
            & (data["tail_amplitude_ratio"] <= 1.5)
            & (data["tail_bias_sigma"] <= 0.35)
            & (y.std(1) >= 0.25)
        )
        data["segment_eligible"] = (
            (np.min(gains, axis=0) > 0)
            & (data["min_segment_mse_gain"] > 0)
            & (data["min_segment_mae_gain"] > 0)
        )
        data["prefix_eligible"] = (
            (data["visibility96"] >= 0.12)
            & (data["persistent_fraction96"] >= 0.5)
            & (data["min_prefix_mse_gain"] >= 0.2)
        )
        data["eligible"] = (
            data["fit_eligible"] & data["segment_eligible"] & data["prefix_eligible"]
        )
        rows.append(pd.DataFrame(data))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "all_candidate_audit.csv", index=False)
    ranked = table[table.eligible].sort_values(
        ["visibility96", "min_segment_mse_gain"], ascending=False
    )
    ranked.to_csv(OUT / "eligible.csv", index=False)
    selected = []
    for _, r in ranked.iterrows():
        if all(
            r.channel != s.channel or abs(r.origin - s.origin) >= 96 for s in selected
        ):
            selected.append(r)
        if len(selected) == 5:
            break
    pd.DataFrame(selected).to_csv(OUT / "review_candidates.csv", index=False)
    summary = {
        k: int(table[k].sum())
        for k in ["fit_eligible", "segment_eligible", "prefix_eligible", "eligible"]
    }
    (OUT / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)
    if selected:
        print(
            pd.DataFrame(selected)[
                [
                    "origin",
                    "channel",
                    "visibility96",
                    "min_segment_mse_gain",
                    "min_prefix_mse_gain",
                    "full_r2",
                    "tail_r2",
                    "last192_r2",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
