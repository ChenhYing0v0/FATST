"""Audit actual 192-step overlap without extending the H96 forecast."""

from pathlib import Path
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


def main() -> None:
    torch.set_num_threads(4)
    tables = []
    for dataset, length, channels in [("Weather", 608, 21), ("ETTh1", 720, 7)]:
        key = dataset.lower()
        table = pd.read_csv(OUT / f"{key}_all_candidate_audit.csv")
        table = table[table.accuracy_eligible & table.fit_eligible].copy()
        ds = ForecastDataset(
            "/Users/river/PaperResearch/Project/datasets",
            dataset,
            "val",
            length,
            720,
        )
        uvhf = np.load(BASE / f"raw/{key}_all_uvhf.npy", mmap_mode="r")
        models = {}
        for h in [96, 192, 336, 720]:
            m = DLinear(
                length, h, channels, init_mode="pytorch_default"
            ).eval()
            m.load_state_dict(
                torch.load(
                    BASE / f"matched_checkpoints/{key}/h{h}/checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            models[h] = m
        stats = []
        for start in range(0, len(table), 32):
            batch = table.iloc[start : start + 32]
            o = batch.origin.to_numpy()
            c = batch.channel.to_numpy()
            x = torch.stack([ds[int(i)][0] for i in o])
            y = np.stack(
                [ds[int(i)][1].numpy()[:192, int(j)] for i, j in zip(o, c)]
            )
            u = np.stack([uvhf[i, :192, j] for i, j in zip(o, c)])
            pred = np.full((len(batch), 4, 192), np.nan)
            with torch.no_grad():
                for j, (h, m) in enumerate(models.items()):
                    full = m(x).numpy()
                    for i, channel in enumerate(c):
                        pred[i, j, : min(h, 192)] = full[i, :192, channel]
            span = np.nanmax(pred, axis=1) - np.nanmin(pred, axis=1)
            allvals = np.concatenate([pred, y[:, None], u[:, None]], axis=1)
            extent = np.nanmax(allvals, axis=(1, 2)) - np.nanmin(
                allvals, axis=(1, 2)
            )
            stats.extend(
                [
                    {
                        "origin": int(oo),
                        "channel": int(cc),
                        "visibility192": span[i].mean() / extent[i],
                        "persistent_fraction192": np.mean(
                            span[i] > 0.1 * extent[i]
                        ),
                    }
                    for i, (oo, cc) in enumerate(zip(o, c))
                ]
            )
        if stats:
            table = table.merge(pd.DataFrame(stats), on=["origin", "channel"])
            table["dataset"] = dataset
            tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(OUT / "overlap192_all_audit.csv", index=False)
    ranked = result[result.visibility192 >= 0.075].sort_values(
        ["visibility192", "tail_r2"], ascending=False
    )
    ranked.to_csv(OUT / "overlap192_eligible.csv", index=False)
    print("Eligible", len(ranked), "of", len(result))
    print(
        ranked.head(10)[
            [
                "dataset",
                "origin",
                "channel",
                "visibility96",
                "visibility192",
                "min_gain",
                "full_r2",
                "tail_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
