"""Scan all Weather validation cells under unchanged late-fidelity gates."""

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "baselines/dlinear"))
import export_iscf_bsca_appendix_c_predictions as exporter
from dataset import ForecastDataset
from model import DLinear
from evaluate_weather import fit_metrics

HORIZONS = (96, 192, 336, 720)


def main() -> None:
    torch.set_num_threads(4)
    args = exporter.load_effective_args(
        BASE / "checkpoints/weather/effective_config.json",
        "Weather",
        Path("/Users/river/PaperResearch/Project/datasets"),
        OUT,
        "cpu",
    )
    official = exporter.train_repo.build_official_args(
        args, exporter.train_repo.OFFICIAL_PRESETS["Weather"][720]
    )
    model = exporter.train_repo.TimeAlign.Model(official).float().eval()
    model.load_state_dict(
        torch.load(
            BASE / "checkpoints/weather/checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    uvhf_ds, _ = exporter.train_repo.data_provider(official, "val")
    ds = ForecastDataset(
        "/Users/river/PaperResearch/Project/datasets",
        "Weather",
        "val",
        608,
        720,
    )
    count = len(ds)
    assert count == len(uvhf_ds) == 4551
    dest = BASE / "raw/weather_all_uvhf.npy"
    values = np.lib.format.open_memmap(
        dest, mode="w+", dtype=np.float32, shape=(count, 720, 21)
    )
    y = np.empty((count, 720, 21), dtype=np.float32)
    prefix = {h: np.empty((count, 96, 21), dtype=np.float32) for h in HORIZONS}
    errors = {h: np.empty((count, 21), dtype=np.float64) for h in HORIZONS}
    baselines = {}
    for h in HORIZONS:
        b = DLinear(608, h, 21, init_mode="pytorch_default").eval()
        b.load_state_dict(
            torch.load(
                BASE / f"matched_checkpoints/weather/h{h}/checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        baselines[h] = b
    started = time.monotonic()
    max_gap = 0.0
    with torch.no_grad():
        for start in range(0, count, 16):
            stop = min(start + 16, count)
            x = torch.stack(
                [
                    torch.as_tensor(uvhf_ds[i][0]).float()
                    for i in range(start, stop)
                ]
            )
            base_x = torch.stack([ds[i][0] for i in range(start, stop)])
            max_gap = max(
                max_gap, float(np.max(abs(x.numpy() - base_x.numpy())))
            )
            assert max_gap < 1e-5
            values[start:stop] = model(
                x, torch.zeros((stop - start, 720, 21)), is_training=False
            )[0].numpy()
            y[start:stop] = torch.stack(
                [ds[i][1] for i in range(start, stop)]
            ).numpy()
            for h, b in baselines.items():
                pred = b(base_x).numpy()
                prefix[h][start:stop] = pred[:, :96]
                errors[h][start:stop] = np.mean(
                    (pred.astype(float) - y[start:stop, :h]) ** 2, axis=1
                )
            if start % 512 == 0:
                print(
                    f"{stop}/{count} origins replayed, {time.monotonic()-started:.1f}s",
                    flush=True,
                )
    values.flush()
    rows = []
    for c in range(21):
        u = values[:, :, c].astype(float)
        target = y[:, :, c].astype(float)
        fit = fit_metrics(u, target)
        common = np.stack([prefix[h][:, :, c] for h in HORIZONS], axis=1)
        ranges = np.ptp(common, axis=1)
        together = np.concatenate(
            [common, u[:, None, :96], target[:, None, :96]], axis=1
        )
        extent = np.ptp(together.reshape(count, -1), axis=1)
        gains = np.stack(
            [
                1
                - np.mean((u[:, :h] - target[:, :h]) ** 2, axis=1)
                / np.maximum(errors[h][:, c], 1e-12)
                for h in HORIZONS
            ],
            axis=1,
        )
        common_mse = np.stack(
            [
                np.mean((prefix[h][:, :, c] - target[:, :96]) ** 2, axis=1)
                for h in HORIZONS
            ],
            axis=1,
        )
        accuracy = (
            (gains.min(1) > 0)
            & (target.std(1) >= 0.25)
            & (
                np.mean((u[:, :96] - target[:, :96]) ** 2, axis=1)
                < common_mse.min(1)
            )
        )
        for o in range(count):
            row = {
                "origin": o,
                "channel": c,
                "accuracy_eligible": bool(accuracy[o]),
                "min_gain": gains[o].min(),
                "visibility96": ranges[o].mean() / max(extent[o], 1e-12),
                "persistent_fraction96": np.mean(ranges[o] > 0.1 * extent[o]),
                **{f"gain_h{h}": gains[o, j] for j, h in enumerate(HORIZONS)},
                **{k: v[o] for k, v in fit.items()},
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
                accuracy[o]
                and row["fit_eligible"]
                and row["visibility96"] >= 0.075
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "weather_all_candidate_audit.csv", index=False)
    ranked = table[table.audited_eligible].sort_values(
        ["visibility96", "tail_r2"], ascending=False
    )
    ranked.to_csv(OUT / "weather_all_eligible.csv", index=False)
    (OUT / "weather_all_replay_audit.json").write_text(
        json.dumps(
            {
                "origins": count,
                "channels": 21,
                "max_input_gap": max_gap,
                "seconds": time.monotonic() - started,
                "eligible": len(ranked),
                "future_labels": "zeros",
                "new_test_access": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        ranked.head(10)[
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
    )


if __name__ == "__main__":
    main()
