"""Rank existing validation cases by visible shared-prefix disagreement."""

from pathlib import Path
import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HORIZONS = (96, 192, 336, 720)


def main() -> None:
    baseline = {
        h: dict(np.load(BASE / f"raw/dlinear/h{h}.npz")) for h in HORIZONS
    }
    all_uvhf = np.load(BASE / "raw/uvhf_all_channels.npz")["prediction_scaled"]
    rows, pools = [], {}
    for c in range(7):
        scale = float(baseline[720]["train_std"].reshape(-1)[c])
        mean = float(baseline[720]["train_mean"].reshape(-1)[c])
        u = all_uvhf[:, :, c].astype(float)
        y = baseline[720]["true"][:, :, c].astype(float)
        pools[c] = {
            "prediction": u * scale + mean,
            "ground_truth": y * scale + mean,
            "prediction_scaled": u,
            "ground_truth_scaled": y,
        }
        d_mse = np.stack(
            [
                np.mean((baseline[h]["pred"][:, :, c] - y[:, :h]) ** 2, axis=1)
                for h in HORIZONS
            ],
            axis=1,
        )
        u_mse = np.stack(
            [np.mean((u[:, :h] - y[:, :h]) ** 2, axis=1) for h in HORIZONS],
            axis=1,
        )
        common_d = np.stack(
            [
                np.mean(
                    (baseline[h]["pred"][:, :96, c] - y[:, :96]) ** 2, axis=1
                )
                for h in HORIZONS
            ],
            axis=1,
        )
        common_u = np.mean((u[:, :96] - y[:, :96]) ** 2, axis=1)
        gain = 1 - u_mse / d_mse
        pred = np.stack(
            [baseline[h]["pred"][:, :96, c] for h in HORIZONS], axis=1
        ).astype(float)
        truth = pools[c]["ground_truth_scaled"][:, :96]
        uvhf = pools[c]["prediction_scaled"][:, :96]
        limits = np.concatenate([pred, truth[:, None], uvhf[:, None]], axis=1)
        extent = limits.max(axis=(1, 2)) - limits.min(axis=(1, 2))
        span = np.ptp(pred, axis=1)
        for o in range(len(pred)):
            rows.append(
                {
                    "origin": o,
                    "channel": c,
                    "min_gain": gain[o].min(),
                    "target_std_scaled": y[o].std(),
                    "uvhf_common96_mse": common_u[o],
                    **{
                        f"dlinear_common96_mse_h{h}": common_d[o, j]
                        for j, h in enumerate(HORIZONS)
                    },
                    **{
                        f"relative_gain_h{h}": gain[o, j]
                        for j, h in enumerate(HORIZONS)
                    },
                    "mean_range96": span[o].mean() * scale,
                    "q25_range96": np.quantile(span[o], 0.25) * scale,
                    "visibility96": span[o].mean() / extent[o],
                    "persistent_fraction96": np.mean(
                        span[o] > 0.1 * extent[o]
                    ),
                    "mean_sorted_gap96": span[o].mean() * scale / 3,
                }
            )
    table = pd.DataFrame(rows)
    table["accuracy_eligible"] = (
        (table.min_gain > 0)
        & (table.target_std_scaled >= 0.25)
        & (
            table.uvhf_common96_mse
            < table[[f"dlinear_common96_mse_h{h}" for h in HORIZONS]].min(
                axis=1
            )
        )
    )
    table.to_csv(OUT / "all_candidate_scores.csv", index=False)
    ranked = table[table.accuracy_eligible].sort_values(
        ["visibility96", "persistent_fraction96", "min_gain"], ascending=False
    )
    ranked.to_csv(OUT / "eligible_candidates.csv", index=False)
    diverse = []
    for _, row in ranked.iterrows():
        if all(
            row.channel != r.channel or abs(row.origin - r.origin) >= 96
            for r in diverse
        ):
            diverse.append(row)
        if len(diverse) == 5:
            break
    pd.DataFrame(diverse).to_csv(OUT / "review_candidates.csv", index=False)
    chosen = ranked.iloc[0]
    o, c = int(chosen.origin), int(chosen.channel)
    ref = baseline[720]
    mean, scale = [
        float(ref[k].reshape(-1)[c]) for k in ("train_mean", "train_std")
    ]
    y, u = pools[c]["ground_truth"][o], pools[c]["prediction"][o]
    pred = {
        h: baseline[h]["pred"][o, :, c].astype(float) * scale + mean
        for h in HORIZONS
    }
    source = pd.DataFrame(
        {
            "step": np.arange(-719, 721),
            "history": np.r_[
                ref["history"][o, :, c].astype(float) * scale + mean,
                np.full(720, np.nan),
            ],
            "ground_truth": np.r_[np.full(720, np.nan), y],
            "uvhf": np.r_[np.full(720, np.nan), u],
        }
    )
    for h in HORIZONS:
        source[f"dlinear_h{h}"] = np.r_[
            np.full(720, np.nan), pred[h], np.full(720 - h, np.nan)
        ]
    source.to_csv(OUT / "source_data.csv", index=False)
    metrics = []
    for h in HORIZONS:
        for model, values in [("DLinear", pred[h]), ("UVHF", u[:h])]:
            delta = values - y[:h]
            metrics.append(
                {
                    "model": model,
                    "horizon": h,
                    "mse_raw": np.mean(delta**2),
                    "mae_raw": np.mean(abs(delta)),
                    "mse_scaled": np.mean((delta / scale) ** 2),
                    "mae_scaled": np.mean(abs(delta / scale)),
                }
            )
    pd.DataFrame(metrics).to_csv(OUT / "selected_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "short_horizon": a,
                "long_horizon": b,
                "chpd_raw": np.mean(abs(pred[a] - pred[b][:a])),
                "nchpd": np.mean(abs(pred[a] - pred[b][:a])) / scale,
            }
            for a, b in combinations(HORIZONS, 2)
        ]
    ).to_csv(OUT / "selected_pair_disagreement.csv", index=False)
    audit = json.loads((BASE / "selection_audit.json").read_text())
    audit["all_channel_uvhf_input_sha256"] = hashlib.sha256(
        (BASE / "raw/uvhf_all_channels.npz").read_bytes()
    ).hexdigest()
    raw = pd.read_csv(
        "/Users/river/PaperResearch/Project/datasets/ETT-small/ETTh2.csv"
    )
    np.testing.assert_allclose(
        y, raw.iloc[8640 + o : 8640 + o + 720, c + 1], atol=1e-5, rtol=0
    )
    audit.update(
        {
            "selected": chosen.to_dict(),
            "searched_channels": list(range(7)),
            "candidate_count": len(table),
            "eligible_count": len(ranked),
            "raw_forecast_origin": 8639 + o,
            "forecast_origin_time": str(raw.iloc[8639 + o, 0]),
            "channel_name": str(raw.columns[c + 1]),
            "mean": mean,
            "train_std": scale,
            "new_baseline_trainings": 0,
            "chpc_status": "new selected origin request check pending",
            "selection_rule": "post-hoc visible common96 disagreement; see protocol.md",
        }
    )
    (OUT / "selection_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    cols = [
        "origin",
        "channel",
        "visibility96",
        "persistent_fraction96",
        "mean_range96",
        "min_gain",
    ]
    print(pd.DataFrame(diverse)[cols].to_string(index=False))
    print(
        "Old:",
        table[(table.origin == 1239) & (table.channel == 6)][cols].to_string(
            index=False
        ),
    )
    print(f"Eligible: {len(ranked)}/{len(table)}")


if __name__ == "__main__":
    main()
