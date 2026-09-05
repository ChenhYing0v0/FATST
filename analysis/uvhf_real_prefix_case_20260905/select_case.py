"""Audit aligned real forecasts and select a disclosed validation case."""

from __future__ import annotations

import hashlib
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RAW = (
    ROOT
    / "UVHF_CN_patent_PDT_style_20260901/work/figure5_prefix_consistency_20260904/raw"
)
HORIZONS = (96, 192, 336, 720)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    baseline = {
        h: dict(np.load(args.baseline_root / f"h{h}.npz")) for h in HORIZONS
    }
    reference = baseline[720]
    count = len(reference["pred"])
    assert count == 2161
    assert (
        reference["history"].shape[1] == 720
    ), "Matched-history case requires L720"
    for h, data in baseline.items():
        assert np.array_equal(data["origin_index"][:count], np.arange(count))
        np.testing.assert_array_equal(
            data["history"][:count], reference["history"]
        )
        np.testing.assert_array_equal(
            data["true"][:count], reference["true"][:, :h]
        )
        np.testing.assert_array_equal(
            data["train_mean"], reference["train_mean"]
        )
        np.testing.assert_array_equal(
            data["train_std"], reference["train_std"]
        )
        assert np.isfinite(data["pred"]).all()

    rows = []
    uvhf = {}
    target_gap = 0.0
    for c in (0, 6):
        data = dict(np.load(RAW / f"uvhf_ch{c}/candidate_pool.npz"))
        order = np.argsort(data["validation_window_index"])
        assert np.array_equal(
            data["validation_window_index"][order], np.arange(count)
        )
        assert np.array_equal(
            data["raw_forecast_origin"][order], 8639 + np.arange(count)
        )
        assert int(data["channel"]) == c
        uvhf[c] = {
            k: data[k][order]
            for k in (
                "prediction",
                "ground_truth",
                "prediction_scaled",
                "ground_truth_scaled",
            )
        }
        u = uvhf[c]["prediction_scaled"].astype(np.float64)
        y = reference["true"][:, :, c].astype(np.float64)
        target_gap = max(
            target_gap,
            float(np.max(np.abs(y - uvhf[c]["ground_truth_scaled"]))),
        )
        np.testing.assert_allclose(
            y, uvhf[c]["ground_truth_scaled"], atol=5e-6, rtol=0
        )
        preds = {
            h: baseline[h]["pred"][:count, :, c].astype(np.float64)
            for h in HORIZONS
        }
        d_mse = np.stack(
            [np.mean((preds[h] - y[:, :h]) ** 2, axis=1) for h in HORIZONS],
            axis=1,
        )
        u_mse = np.stack(
            [np.mean((u[:, :h] - y[:, :h]) ** 2, axis=1) for h in HORIZONS],
            axis=1,
        )
        gain = 1 - u_mse / np.maximum(d_mse, 1e-12)
        pairs = np.stack(
            [
                np.mean(np.abs(preds[a] - preds[b][:, :a]), axis=1)
                for a, b in combinations(HORIZONS, 2)
            ],
            axis=1,
        )
        disagreement = pairs.mean(axis=1)
        percentile = (
            pd.Series(disagreement).rank(method="average", pct=True).to_numpy()
        )
        common_d = np.stack(
            [
                np.mean((preds[h][:, :96] - y[:, :96]) ** 2, axis=1)
                for h in HORIZONS
            ],
            axis=1,
        )
        common_u = np.mean((u[:, :96] - y[:, :96]) ** 2, axis=1)
        centered_u = u - u.mean(axis=1, keepdims=True)
        centered_y = y - y.mean(axis=1, keepdims=True)
        corr = np.sum(centered_u * centered_y, axis=1) / np.maximum(
            np.linalg.norm(centered_u, axis=1)
            * np.linalg.norm(centered_y, axis=1),
            1e-12,
        )
        target_std = y.std(axis=1)
        eligible = (
            (gain.min(axis=1) > 0)
            & (common_u < common_d.min(axis=1))
            & (percentile >= 0.75)
            & (target_std >= 0.25)
        )
        score = (
            0.5 * percentile + 0.3 * gain.min(axis=1) + 0.2 * (corr + 1) / 2
        )
        for o in range(count):
            row = {
                "origin": o,
                "channel": c,
                "eligible": bool(eligible[o]),
                "score": score[o],
                "mean_nchpd": disagreement[o],
                "disagreement_percentile": percentile[o],
                "min_gain": gain[o].min(),
                "uvhf_corr720": corr[o],
                "target_std_scaled": target_std[o],
                "uvhf_common96_mse": common_u[o],
            }
            for j, h in enumerate(HORIZONS):
                row.update(
                    {
                        f"dlinear_mse_h{h}": d_mse[o, j],
                        f"uvhf_mse_h{h}": u_mse[o, j],
                        f"relative_gain_h{h}": gain[o, j],
                        f"dlinear_common96_mse_h{h}": common_d[o, j],
                    }
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "all_candidate_scores.csv", index=False)
    ranked = table[table.eligible].sort_values(
        ["score", "channel", "origin"], ascending=[False, True, True]
    )
    ranked.to_csv(OUT / "eligible_candidates.csv", index=False)
    if ranked.empty:
        raise RuntimeError("No case meets the frozen selection rule")
    best = ranked.iloc[0].to_dict()
    o, c = int(best["origin"]), int(best["channel"])
    mean, scale = (
        float(reference[k].reshape(-1)[c]) for k in ("train_mean", "train_std")
    )
    y, u = uvhf[c]["ground_truth"][o], uvhf[c]["prediction"][o]
    pred = {
        h: baseline[h]["pred"][o, :, c].astype(np.float64) * scale + mean
        for h in HORIZONS
    }
    history = reference["history"][o, :, c].astype(np.float64) * scale + mean
    source = []
    for t in range(-719, 721):
        row = {
            "step": t,
            "history": history[t + 719] if t <= 0 else np.nan,
            "ground_truth": y[t - 1] if t > 0 else np.nan,
            "uvhf": u[t - 1] if t > 0 else np.nan,
        }
        for h in HORIZONS:
            row[f"dlinear_h{h}"] = pred[h][t - 1] if 1 <= t <= h else np.nan
        source.append(row)
    pd.DataFrame(source).to_csv(OUT / "source_data.csv", index=False)
    metrics = []
    for h in HORIZONS:
        for model, values in (("DLinear", pred[h]), ("UVHF", u[:h])):
            delta = values - y[:h]
            metrics.append(
                {
                    "model": model,
                    "horizon": h,
                    "mse_raw": float(np.mean(delta**2)),
                    "mae_raw": float(np.mean(np.abs(delta))),
                    "mse_scaled": float(np.mean((delta / scale) ** 2)),
                    "mae_scaled": float(np.mean(np.abs(delta / scale))),
                }
            )
    pd.DataFrame(metrics).to_csv(OUT / "selected_metrics.csv", index=False)
    pair_rows = []
    for a, b in combinations(HORIZONS, 2):
        pair_rows.append(
            {
                "short_horizon": a,
                "long_horizon": b,
                "chpd_raw": float(np.mean(np.abs(pred[a] - pred[b][:a]))),
                "nchpd": float(np.mean(np.abs(pred[a] - pred[b][:a])) / scale),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        OUT / "selected_pair_disagreement.csv", index=False
    )
    raw_path = Path(
        "/Users/river/PaperResearch/Project/datasets/ETT-small/ETTh2.csv"
    )
    raw_df = pd.read_csv(raw_path)
    raw_y = raw_df.iloc[8640 + o : 8640 + o + 720, c + 1].to_numpy()
    np.testing.assert_allclose(raw_y, y, rtol=0, atol=1e-5)
    manifest = {
        "dataset": "ETTh2",
        "split": "validation",
        "seed": 2021,
        "baseline_role": "source-audited L720 DLinear visualization protocol; four independent checkpoints",
        "uvhf_role": "frozen Main-I/II selected profile h2_lr5e4",
        "uvhf_checkpoint_sha256": "bcfbc9955754a9825d1dd33015610a049551e3441dabd0ad98982c0fc2285d3e",
        "searched_channels": [0, 6],
        "origins_per_channel": count,
        "candidate_count": len(table),
        "eligible_count": len(ranked),
        "selected": best,
        "raw_forecast_origin": 8639 + o,
        "forecast_origin_time": str(raw_df.iloc[8639 + o, 0]),
        "channel_name": str(raw_df.columns[c + 1]),
        "mean": mean,
        "train_std": scale,
        "max_scaled_target_alignment_gap": target_gap,
        "new_baseline_trainings": 4,
        "new_uvhf_training": False,
        "new_test_access": False,
        "chpc_status": "single-trajectory prefixes by construction; separate request check pending",
        "uvhf_input_sha256": {
            f"uvhf_ch{c}": sha256(RAW / f"uvhf_ch{c}/candidate_pool.npz")
            for c in (0, 6)
        },
        "baseline_input_sha256": {
            p.name: sha256(p) for p in sorted(args.baseline_root.glob("*.npz"))
        },
        "raw_dataset_sha256": sha256(raw_path),
    }
    (OUT / "selection_audit.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        ranked.head(8)[
            [
                "origin",
                "channel",
                "score",
                "min_gain",
                "mean_nchpd",
                "uvhf_corr720",
            ]
        ].to_string(index=False)
    )
    print(f"Eligible: {len(ranked)}/{len(table)}")


if __name__ == "__main__":
    main()
