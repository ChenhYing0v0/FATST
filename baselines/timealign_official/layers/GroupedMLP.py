"""Nonlinear future-group sharing head for the StageC D14-A1 diagnostic."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedMLPReadout(nn.Module):
    """Map one history state to future groups with group-private hidden banks."""

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        scale: int = 144,
        point_hidden_width: int = 4,
        partition: str = "canonical",
        partition_seed: int = 14101,
    ) -> None:
        super().__init__()
        if readout_dim <= 0 or series_length <= 0:
            raise ValueError("readout_dim and series_length must be positive")
        if scale <= 0 or series_length % scale:
            raise ValueError("scale must be a positive divisor of series_length")
        if point_hidden_width < 2:
            raise ValueError("point_hidden_width must be at least two")
        if partition not in {"canonical", "random"}:
            raise ValueError("partition must be canonical or random")
        if partition == "random" and scale in {1, series_length}:
            raise ValueError("random endpoint partitions are not meaningful")
        self.readout_dim = int(readout_dim)
        self.series_length = int(series_length)
        self.scale = int(scale)
        self.group_count = self.series_length // self.scale
        self.point_hidden_width = int(point_hidden_width)
        self.partition = partition
        self.partition_seed = int(partition_seed)
        self.target_decoder_parameters = self._parameter_count(
            scale=1,
            hidden_width=self.point_hidden_width,
        )
        denominator = self.group_count * (
            self.readout_dim + self.scale + 1
        )
        self.hidden_width = max(
            1,
            round(
                (self.target_decoder_parameters - self.series_length)
                / denominator
            ),
        )
        minimum_affine_width = 2 * min(self.readout_dim, self.scale)
        if self.hidden_width < minimum_affine_width:
            raise ValueError(
                "matched hidden width cannot contain a full affine block: "
                f"observed={self.hidden_width}, required={minimum_affine_width}"
            )
        order = torch.arange(self.series_length, dtype=torch.long)
        if self.partition == "random":
            generator = torch.Generator(device="cpu").manual_seed(
                self.partition_seed + self.scale * 101
            )
            order = order[torch.randperm(self.series_length, generator=generator)]
        self.register_buffer(
            "group_indices",
            order.reshape(self.group_count, self.scale),
            persistent=True,
        )
        self.input_weight = nn.Parameter(
            torch.empty(
                self.group_count,
                self.readout_dim,
                self.hidden_width,
            )
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.group_count, self.hidden_width)
        )
        self.output_weight = nn.Parameter(
            torch.empty(
                self.group_count,
                self.hidden_width,
                self.scale,
            )
        )
        self.output_bias = nn.Parameter(
            torch.zeros(self.group_count, self.scale)
        )
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.output_weight, a=math.sqrt(5.0))

    def _parameter_count(self, scale: int, hidden_width: int) -> int:
        groups = self.series_length // scale
        return groups * (
            self.readout_dim * hidden_width
            + hidden_width
            + hidden_width * scale
            + scale
        )

    @property
    def decoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_relative_gap(self) -> float:
        return abs(
            self.decoder_parameters - self.target_decoder_parameters
        ) / self.target_decoder_parameters

    def target_group_labels(self) -> torch.Tensor:
        labels = torch.empty(
            self.series_length,
            dtype=torch.long,
            device=self.group_indices.device,
        )
        group_ids = torch.arange(
            self.group_count,
            dtype=torch.long,
            device=self.group_indices.device,
        ).unsqueeze(1)
        labels.scatter_(
            0,
            self.group_indices.flatten(),
            group_ids.expand_as(self.group_indices).flatten(),
        )
        return labels

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        preactivation = torch.einsum(
            "bcr,grk->bcgk",
            hidden,
            self.input_weight,
        )
        features = F.gelu(
            preactivation
            + self.hidden_bias.view(1, 1, self.group_count, self.hidden_width)
        )
        grouped = torch.einsum(
            "bcgk,gks->bcgs",
            features,
            self.output_weight,
        )
        grouped = grouped + self.output_bias.view(
            1, 1, self.group_count, self.scale
        )
        source = grouped.flatten(start_dim=-2)
        indices = self.group_indices.flatten().view(1, 1, self.series_length)
        indices = indices.expand(hidden.shape[0], hidden.shape[1], -1)
        output = torch.zeros(
            hidden.shape[0],
            hidden.shape[1],
            self.series_length,
            dtype=hidden.dtype,
            device=hidden.device,
        ).scatter(-1, indices, source)
        return output[:, :, :horizon].permute(0, 2, 1)
