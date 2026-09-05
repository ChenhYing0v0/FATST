"""Build a reviewable real-data figure source from an audited TimeMixer case."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HORIZONS = (96, 192, 336, 720)


def main(rank: int = 0) -> None:
    candidate = pd.read_csv(OUT / "timemixer_review_candidates.csv").iloc[rank]
    o, c = int(candidate.origin), int(candidate.channel)
    assert candidate.audited_eligible
    exports = {
        h: dict(
            np.load(
                BASE
                / f"matched_checkpoints/timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HORIZONS
    }
    mean, scale = exports[720]["train_mean"][c], exports[720]["train_std"][c]
    raw = pd.read_csv(
        "/Users/river/PaperResearch/Project/datasets/ETT-small/ETTh1.csv"
    )
    y = raw.iloc[8640 + o : 8640 + o + 720, c + 1].to_numpy()
    history = raw.iloc[8640 + o - 720 : 8640 + o, c + 1].to_numpy()
    u = (
        np.load(BASE / "raw/etth1_all_uvhf.npy", mmap_mode="r")[
            o, :, c
        ].astype(float)
        * scale
        + mean
    )
    pred = {
        h: exports[h]["pred"][o, :, c].astype(float) * scale + mean
        for h in HORIZONS
    }
    dest = OUT / f"review_case_{rank}"
    dest.mkdir(exist_ok=True)
    source = pd.DataFrame(
        {
            "step": np.arange(-719, 721),
            "history": np.r_[history, np.full(720, np.nan)],
            "ground_truth": np.r_[np.full(720, np.nan), y],
            "uvhf": np.r_[np.full(720, np.nan), u],
        }
    )
    for h in HORIZONS:
        source[f"timemixer_h{h}"] = np.r_[
            np.full(720, np.nan), pred[h], np.full(720 - h, np.nan)
        ]
    source.to_csv(dest / "source_data.csv", index=False)
    metrics = []
    for h in HORIZONS:
        for model, values in [("UVHF", u[:h]), ("TimeMixer", pred[h])]:
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
    pd.DataFrame(metrics).to_csv(dest / "selected_metrics.csv", index=False)
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
    ).to_csv(dest / "selected_pair_disagreement.csv", index=False)
    full = np.concatenate([y, u, history[-48:], *pred.values()])
    low, high = full.min(), full.max()
    width = high - low
    prefix = np.concatenate(
        [y[:96], u[:96], *[values[:96] for values in pred.values()]]
    )
    lo, hi = prefix.min(), prefix.max()
    pwidth = hi - lo
    span = np.ptp(np.stack([pred[h][23:80] for h in HORIZONS]), axis=0)
    step = int(np.argmax(span) + 24)
    plim = [float(lo - 0.06 * pwidth), float(hi + 0.06 * pwidth)]
    settings = {
        "baseline": "TimeMixer",
        "baseline_prefix": "timemixer",
        "subtitle": f"ETTh1 · {raw.columns[c+1]} · selected validation example",
        "unit": "units",
        "ylabel": f"{raw.columns[c+1]} (original scale)",
        "main_ylim": [float(low - 0.15 * width), float(low + 2.2 * width)],
        "main_yticks": np.linspace(low, high, 4).round(1).tolist(),
        "prefix_ylim": plim,
        "prefix_yticks": np.linspace(lo, hi, 4).round(1).tolist(),
        "prefix_label_y": plim[1] + 0.02 * width,
        "horizon_y": {str(h): float(high + 0.06 * width) for h in HORIZONS},
        "annotation_y": float(low - 0.115 * width),
        "annotation_step": step,
        "connector_corner": [0, 0],
    }
    # Preserve the reviewed layout when rebuilding the same source data.
    settings_path = dest / "figure_settings.json"
    if not settings_path.exists():
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    audit = {
        "dataset": "ETTh1",
        "split": "validation",
        "selected": candidate.to_dict(),
        "channel_name": str(raw.columns[c + 1]),
        "forecast_origin_time": str(raw.iloc[8639 + o, 0]),
        "raw_forecast_origin": 8639 + o,
        "mean": float(mean),
        "train_std": float(scale),
        "baseline": "TimeMixer",
        "baseline_role": "existing native L96 TimeMixer; seq_len treated as tunable hyperparameter",
        "uvhf_checkpoint_sha256": "1e4021071ac805b8941d9b4ab49da13ee4d8b520c03f16c4f4e458e6dec98814",
        "post_hoc_selection": True,
        "new_test_access": False,
        "status": "pending full-trajectory visual review and independent request audit",
    }
    (dest / "selection_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=0)
    main(parser.parse_args().rank)
