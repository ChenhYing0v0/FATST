"""Replay frozen DLinear checkpoints locally without training or test access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "baselines/dlinear"))
from dataset import ForecastDataset
from model import DLinear


def main() -> None:
    torch.set_num_threads(4)
    reports = []
    output = OUT / "raw/dlinear"
    output.mkdir(parents=True, exist_ok=True)
    for h in (96, 192, 336, 720):
        run = OUT / "matched_checkpoints" / f"h{h}"
        config = json.loads((run / "effective_config.json").read_text())
        assert config["seq_len"] == 720 and config["pred_len"] == h
        assert config["skip_test"] and config["seed"] == 2021
        model = DLinear(720, h, 7, init_mode=config["init_mode"]).eval()
        model.load_state_dict(
            torch.load(
                run / "checkpoint.pt", map_location="cpu", weights_only=True
            )
        )
        ds = ForecastDataset(
            "/Users/river/PaperResearch/Project/datasets",
            "ETTh2",
            "val",
            720,
            h,
        )
        predictions, targets, histories = [], [], []
        with torch.no_grad():
            for start in range(0, len(ds), 64):
                batch = [ds[i] for i in range(start, min(start + 64, len(ds)))]
                x = torch.stack([item[0] for item in batch])
                y = torch.stack([item[1] for item in batch])
                predictions.append(model(x).numpy())
                targets.append(y.numpy())
                histories.append(x.numpy())
        pred = np.concatenate(predictions)
        true = np.concatenate(targets)
        history = np.concatenate(histories)
        mse = float(np.mean((pred.astype(np.float64) - true) ** 2))
        remote = json.loads((run / "metrics_val.json").read_text())
        assert abs(mse - remote["mse"]) < 1e-5
        np.savez_compressed(
            output / f"h{h}.npz",
            pred=pred[:2161],
            true=true[:2161],
            history=history[:2161],
            origin_index=np.arange(2161),
            train_mean=ds.scaler.mean,
            train_std=ds.scaler.std,
        )
        reports.append(
            {
                "horizon": h,
                "validation_count": len(ds),
                "saved_common_origins": 2161,
                "local_mse": mse,
                "remote_mse": remote["mse"],
                "difference": abs(mse - remote["mse"]),
            }
        )
        print(
            f"Replayed H{h}: MSE gap {abs(mse - remote['mse']):.3g}",
            flush=True,
        )
    (OUT / "baseline_replay_audit.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
