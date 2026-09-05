"""Replay matched Weather controls and audit all frozen UVHF candidate origins."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "baselines/dlinear"))
from dataset import ForecastDataset
from model import DLinear

HORIZONS = (96, 192, 336, 720)


def fit_metrics(u: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for name, start in [("full", 0), ("tail", 336), ("last192", 528)]:
        pred, target = u[:, start:], y[:, start:]
        yc = target - target.mean(1, keepdims=True)
        uc = pred - pred.mean(1, keepdims=True)
        std = target.std(1)
        # Constant targets have undefined centered fit and fail the fit gate.
        std = np.where(std > 0, std, np.nan)
        result[f"{name}_r2"] = (
            1 - np.mean((pred - target) ** 2, axis=1) / std**2
        )
        result[f"{name}_corr"] = np.sum(yc * uc, axis=1) / np.maximum(
            np.linalg.norm(yc, axis=1) * np.linalg.norm(uc, axis=1), 1e-12
        )
        result[f"{name}_amplitude_ratio"] = pred.std(1) / std
        result[f"{name}_bias_sigma"] = abs((pred - target).mean(1)) / std
    return result


def main() -> None:
    torch.set_num_threads(4)
    pool_dir = (
        ROOT
        / "analysis/iscf_bsca_appendix_c_prediction_export_20260825/Weather"
    )
    pool = dict(np.load(pool_dir / "candidate_pool.npz"))
    order = np.argsort(pool["validation_window_index"])
    origins = pool["validation_window_index"][order]
    u = pool["prediction_scaled"][order].astype(float)
    y = pool["ground_truth_scaled"][order].astype(float)
    channel = int(pool["channel"])
    ds = ForecastDataset(
        "/Users/river/PaperResearch/Project/datasets",
        "Weather",
        "val",
        608,
        720,
    )
    x = torch.stack([ds[int(i)][0] for i in origins])
    true = torch.stack([ds[int(i)][1] for i in origins]).numpy()
    np.testing.assert_allclose(true[:, :, channel], y, atol=5e-6, rtol=0)
    pred = {}
    provenance = {}
    for h in HORIZONS:
        run = BASE / f"matched_checkpoints/weather/h{h}"
        config = json.loads((run / "effective_config.json").read_text())
        assert (
            config["seq_len"] == 608
            and config["pred_len"] == h
            and config["skip_test"]
        )
        model = DLinear(608, h, 21, init_mode=config["init_mode"]).eval()
        model.load_state_dict(
            torch.load(
                run / "checkpoint.pt", map_location="cpu", weights_only=True
            )
        )
        with torch.no_grad():
            pred[h] = np.concatenate(
                [
                    model(x[i : i + 32]).numpy()[:, :, channel]
                    for i in range(0, len(x), 32)
                ]
            ).astype(float)
        log = pd.read_csv(run / "training_log.csv")
        provenance[str(h)] = {
            "checkpoint_sha256": hashlib.sha256(
                (run / "checkpoint.pt").read_bytes()
            ).hexdigest(),
            "epochs": len(log),
            "config": config,
            "environment": json.loads((run / "environment.json").read_text()),
            "validation_metrics": json.loads(
                (run / "metrics_val.json").read_text()
            ),
        }
    fit = fit_metrics(u, y)
    common = np.stack([pred[h][:, :96] for h in HORIZONS], axis=1)
    ranges = np.ptp(common, axis=1)
    together = np.concatenate(
        [common, u[:, None, :96], y[:, None, :96]], axis=1
    )
    extent = np.ptp(together.reshape(len(u), -1), axis=1)
    rows = []
    for i, origin in enumerate(origins):
        gain = {
            h: 1
            - np.mean((u[i, :h] - y[i, :h]) ** 2)
            / np.mean((pred[h][i] - y[i, :h]) ** 2)
            for h in HORIZONS
        }
        accuracy = (
            min(gain.values()) > 0
            and y[i].std() >= 0.25
            and all(
                np.mean((u[i, :96] - y[i, :96]) ** 2)
                < np.mean((pred[h][i, :96] - y[i, :96]) ** 2)
                for h in HORIZONS
            )
        )
        row = {
            "origin": int(origin),
            "channel": channel,
            "accuracy_eligible": accuracy,
            "min_gain": min(gain.values()),
            "visibility96": ranges[i].mean() / extent[i],
            "persistent_fraction96": np.mean(ranges[i] > 0.1 * extent[i]),
            **{f"gain_h{h}": gain[h] for h in HORIZONS},
            **{k: v[i] for k, v in fit.items()},
        }
        row["fit_eligible"] = (
            row["full_r2"] >= 0.35
            and row["tail_r2"] >= 0.25
            and row["tail_corr"] >= 0.7
            and row["last192_r2"] >= 0
            and 0.5 <= row["tail_amplitude_ratio"] <= 1.5
            and row["tail_bias_sigma"] <= 0.35
        )
        row["audited_eligible"] = (
            accuracy and row["fit_eligible"] and row["visibility96"] >= 0.075
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "weather_candidate_audit.csv", index=False)
    ranked = table[table.audited_eligible].sort_values(
        ["visibility96", "tail_r2"], ascending=False
    )
    ranked.to_csv(OUT / "weather_eligible_candidates.csv", index=False)
    np.savez_compressed(
        BASE / "raw/weather_audited_candidates.npz",
        origins=origins,
        uvhf=u,
        true=y,
        history=x.numpy()[:, :, channel],
        train_mean=ds.scaler.mean.reshape(-1)[channel],
        train_std=ds.scaler.std.reshape(-1)[channel],
        **{f"dlinear_h{h}": pred[h] for h in HORIZONS},
    )
    (OUT / "weather_baseline_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Weather eligible", len(ranked), "/", len(table))
    print(
        ranked.head(8)[
            [
                "origin",
                "visibility96",
                "min_gain",
                "full_r2",
                "tail_r2",
                "tail_corr",
                "last192_r2",
            ]
        ].to_string(index=False)
    )
    print(
        "Max visibility",
        table.visibility96.max(),
        "accuracy pass",
        table.accuracy_eligible.sum(),
    )


if __name__ == "__main__":
    main()
