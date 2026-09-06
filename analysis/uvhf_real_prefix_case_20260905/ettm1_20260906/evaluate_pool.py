"""Audit the frozen ETTm1 HUFL pool against four native TimeMixer forecasts."""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
HORIZONS = (96, 192, 336, 720)
POOL = ROOT / "analysis/iscf_bsca_appendix_c_prediction_export_20260825/ETTm1"


def fit_metrics(u: np.ndarray, y: np.ndarray) -> dict:
    result = {}
    for name, start in [("full", 0), ("tail", 336), ("last192", 528)]:
        a, b = u[start:], y[start:]
        std = b.std()
        result[f"{name}_r2"] = 1 - np.mean((a - b) ** 2) / std**2
        result[f"{name}_corr"] = np.corrcoef(a, b)[0, 1]
        result[f"{name}_amplitude_ratio"] = a.std() / std
        result[f"{name}_bias_sigma"] = abs((a - b).mean()) / std
    return result


def main() -> None:
    pool = dict(np.load(POOL / "candidate_pool.npz"))
    metadata = json.loads((POOL / "metadata.json").read_text())
    raw = pd.read_csv("/Users/river/PaperResearch/Project/datasets/ETT-small/ETTm1.csv")
    exports = {
        h: dict(
            np.load(
                BASE
                / f"matched_checkpoints/ettm1_timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HORIZONS
    }
    mean, std = exports[720]["train_mean"], exports[720]["train_std"]
    channel = int(pool["channel"])
    assert channel == 0
    for h in HORIZONS:
        assert exports[h]["pred"].shape == (10801, h, 7)
        assert np.isfinite(exports[h]["pred"]).all()
        np.testing.assert_array_equal(exports[h]["train_mean"], mean)
        np.testing.assert_array_equal(exports[h]["train_std"], std)
    rows = []
    for i, origin in enumerate(pool["validation_window_index"]):
        origin = int(origin)
        y = raw.iloc[34560 + origin : 34560 + origin + 720, channel + 1].to_numpy()
        u = pool["prediction"][i].astype(float)
        np.testing.assert_allclose(y, pool["ground_truth"][i], atol=1e-5, rtol=0)
        pred = {
            h: exports[h]["pred"][origin, :, channel].astype(float) * std[channel]
            + mean[channel]
            for h in HORIZONS
        }
        fit = fit_metrics(u, y)
        gain = {
            h: 1 - np.mean((u[:h] - y[:h]) ** 2) / np.mean((pred[h] - y[:h]) ** 2)
            for h in HORIZONS
        }
        common = np.stack([pred[h][:96] for h in HORIZONS])
        prefix_extent = np.ptp(np.concatenate([common.ravel(), u[:96], y[:96]]))
        visibility = np.ptp(common, axis=0).mean() / prefix_extent
        extent = np.ptp(np.concatenate([u, y, *pred.values()]))
        eu, eb = abs(u - y), abs(pred[720] - y)
        win = (eu <= 0.08 * extent) & (eb - eu >= 0.04 * extent)
        loss = (eb <= 0.08 * extent) & (eu - eb >= 0.04 * extent)
        row = {
            "pool_index": i,
            "origin": origin,
            "channel": channel,
            **fit,
            **{f"gain_h{h}": gain[h] for h in HORIZONS},
            "min_gain": min(gain.values()),
            "visibility96": visibility,
            "visible_net": float(win.mean() - loss.mean()),
            "tail_visible_net": float(win[336:].mean() - loss[336:].mean()),
            "accuracy_eligible": min(gain.values()) > 0
            and y.std() / std[channel] >= 0.25
            and all(
                np.mean((u[:96] - y[:96]) ** 2) < np.mean((v[:96] - y[:96]) ** 2)
                for v in pred.values()
            ),
            "fit_eligible": fit["full_r2"] >= 0.35
            and fit["tail_r2"] >= 0.25
            and fit["tail_corr"] >= 0.7
            and fit["last192_r2"] >= 0
            and 0.5 <= fit["tail_amplitude_ratio"] <= 1.5
            and fit["tail_bias_sigma"] <= 0.35,
        }
        row["audited_eligible"] = (
            row["accuracy_eligible"] and row["fit_eligible"] and visibility >= 0.075
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "all_pool_audit.csv", index=False)
    eligible = table[table.audited_eligible].sort_values(
        ["visible_net", "min_gain", "visibility96"], ascending=False
    )
    eligible.to_csv(OUT / "eligible_candidates.csv", index=False)
    selected = []
    for _, row in eligible.iterrows():
        if all(abs(row.origin - r.origin) >= 96 for r in selected):
            selected.append(row)
        if len(selected) == 5:
            break
    review = pd.DataFrame(selected)
    review.to_csv(OUT / "review_candidates.csv", index=False)
    summary = {
        "pool_size": len(table),
        "accuracy_pass": int(table.accuracy_eligible.sum()),
        "fit_pass": int(table.fit_eligible.sum()),
        "all_pass": len(eligible),
        "review_count": len(review),
    }
    (OUT / "gate_counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    for rank, row in enumerate(selected):
        i, o = int(row.pool_index), int(row.origin)
        y = raw.iloc[34560 + o : 34560 + o + 720, channel + 1].to_numpy()
        u = pool["prediction"][i].astype(float)
        pred = {
            h: exports[h]["pred"][o, :, channel].astype(float) * std[channel]
            + mean[channel]
            for h in HORIZONS
        }
        case = OUT / f"review_case_{rank}"
        case.mkdir(exist_ok=True)
        source = pd.DataFrame(
            {
                "step": np.arange(-719, 721),
                "history": np.r_[
                    raw.iloc[34560 + o - 720 : 34560 + o, channel + 1].to_numpy(),
                    np.full(720, np.nan),
                ],
                "ground_truth": np.r_[np.full(720, np.nan), y],
                "uvhf": np.r_[np.full(720, np.nan), u],
            }
        )
        for h in HORIZONS:
            source[f"timemixer_h{h}"] = np.r_[
                np.full(720, np.nan), pred[h], np.full(720 - h, np.nan)
            ]
        source.to_csv(case / "source_data.csv", index=False)
        metrics = []
        for h in HORIZONS:
            for model, values in [("UVHF", u[:h]), ("TimeMixer", pred[h])]:
                error = values - y[:h]
                metrics.append(
                    {
                        "model": model,
                        "horizon": h,
                        "mse_raw": np.mean(error**2),
                        "mae_raw": np.mean(abs(error)),
                        "mse_scaled": np.mean(error**2) / std[channel] ** 2,
                        "mae_scaled": np.mean(abs(error)) / std[channel],
                    }
                )
        pd.DataFrame(metrics).to_csv(case / "selected_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "short_horizon": a,
                    "long_horizon": b,
                    "chpd_raw": np.mean(abs(pred[a] - pred[b][:a])),
                    "nchpd": np.mean(abs(pred[a] - pred[b][:a])) / std[channel],
                }
                for a, b in combinations(HORIZONS, 2)
            ]
        ).to_csv(case / "selected_pair_disagreement.csv", index=False)
        audit = {
            "dataset": "ETTm1",
            "split": "validation",
            "selected": row.to_dict(),
            "channel_name": "HUFL",
            "raw_forecast_origin": 34559 + o,
            "forecast_origin_time": str(raw.iloc[34559 + o, 0]),
            "uvhf_checkpoint_sha256": metadata["checkpoint_sha256"],
            "baseline": "TimeMixer",
            "post_hoc_selection": True,
            "historical_pool_preselected_by_uvhf_fidelity": True,
            "new_test_access": False,
            "status": "pending independent numeric and visual audit",
        }
        (case / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary)
    if len(review):
        print(
            review[
                [
                    "origin",
                    "min_gain",
                    "gain_h720",
                    "visibility96",
                    "visible_net",
                    "full_r2",
                    "tail_r2",
                    "last192_r2",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
