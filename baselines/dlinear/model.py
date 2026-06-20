from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        moving_avg: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_avg)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(channels)]
            )
            for seasonal, trend in zip(self.linear_seasonal, self.linear_trend):
                seasonal.weight = nn.Parameter((1 / seq_len) * torch.ones(pred_len, seq_len))
                trend.weight = nn.Parameter((1 / seq_len) * torch.ones(pred_len, seq_len))
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal.weight = nn.Parameter(
                (1 / seq_len) * torch.ones(pred_len, seq_len)
            )
            self.linear_trend.weight = nn.Parameter((1 / seq_len) * torch.ones(pred_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                seasonal.size(0), seasonal.size(1), self.pred_len, dtype=seasonal.dtype,
                device=seasonal.device
            )
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.channels):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        return (seasonal_out + trend_out).permute(0, 2, 1)
