from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: str
    channels: int
    split: str


DATASETS: dict[str, DatasetSpec] = {
    "ETTh2": DatasetSpec("ETTh2", "ETT-small/ETTh2.csv", 7, "ett_hour"),
    "ETTm1": DatasetSpec("ETTm1", "ETT-small/ETTm1.csv", 7, "ett_minute"),
    "Weather": DatasetSpec("Weather", "weather/weather.csv", 21, "ratio"),
}


class StandardScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fit before transform.")
        return (values - self.mean) / self.std


class ForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset_root: str | Path,
        dataset: str,
        flag: str,
        seq_len: int,
        pred_len: int,
        scale: bool = True,
    ) -> None:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split flag: {flag}")

        self.spec = DATASETS[dataset]
        self.seq_len = seq_len
        self.pred_len = pred_len
        path = Path(dataset_root) / self.spec.relative_path
        if not path.exists():
            raise FileNotFoundError(path)
        df_raw = pd.read_csv(path)
        values = df_raw.iloc[:, 1:].to_numpy(dtype=np.float32)

        border1s, border2s = self._borders(len(values), seq_len)
        split_index = {"train": 0, "val": 1, "test": 2}[flag]

        scaler = StandardScaler()
        if scale:
            scaler.fit(values[border1s[0] : border2s[0]])
            values = scaler.transform(values).astype(np.float32)
        self.scaler = scaler
        self.data = values[border1s[split_index] : border2s[split_index]]

    def _borders(self, n_rows: int, seq_len: int) -> tuple[list[int], list[int]]:
        if self.spec.split == "ett_hour":
            return (
                [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len],
                [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
            )
        if self.spec.split == "ett_minute":
            return (
                [
                    0,
                    12 * 30 * 24 * 4 - seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
                ],
                [
                    12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                ],
            )
        train = int(n_rows * 0.7)
        test = int(n_rows * 0.2)
        val = n_rows - train - test
        return ([0, train - seq_len, train + val - seq_len], [train, train + val, n_rows])

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_end = index + self.seq_len
        y_end = x_end + self.pred_len
        return torch.from_numpy(self.data[index:x_end]), torch.from_numpy(self.data[x_end:y_end])
