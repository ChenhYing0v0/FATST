"""Projective multi-resolution forecast readouts for StageC Step 7A."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PMFO_BLOCK_SIZES = (90, 30, 10, 5, 1)
PMFO_RADICES = (3, 3, 2, 5)
PMFO_SERIES_LENGTH = 720


def _helmert_detail(radix: int) -> torch.Tensor:
    """Return an orthonormal complement of the normalized constant vector."""
    detail = torch.zeros(radix, radix - 1, dtype=torch.float32)
    for column in range(radix - 1):
        count = column + 1
        scale = math.sqrt(float(count * (count + 1)))
        detail[:count, column] = 1.0 / scale
        detail[count, column] = -float(count) / scale
    return detail


def _validate_horizon(horizon: int) -> None:
    if horizon <= 0 or horizon > PMFO_SERIES_LENGTH:
        raise ValueError("target_prefix must be in [1, 720]")


def _active_group_sizes(horizon: int) -> tuple[int, tuple[int, ...]]:
    _validate_horizon(horizon)
    coarse = math.ceil(horizon / PMFO_BLOCK_SIZES[0])
    details = tuple(
        math.ceil(horizon / PMFO_BLOCK_SIZES[level]) * (radix - 1)
        for level, radix in enumerate(PMFO_RADICES)
    )
    return coarse, details


class ConservativeTreeSynthesis(nn.Module):
    """Fixed mixed-radix synthesis shared by PMFO and its controls."""

    def __init__(self, conservative: bool = True) -> None:
        super().__init__()
        self.conservative = conservative
        for level, radix in enumerate(PMFO_RADICES):
            self.register_buffer(
                f"scaling_{level}",
                torch.full((radix,), 1.0 / math.sqrt(float(radix))),
                persistent=False,
            )
            self.register_buffer(
                f"contrast_{level}",
                _helmert_detail(radix),
                persistent=False,
            )

    def refine(
        self,
        parent: torch.Tensor,
        update: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Refine parent coefficients into children at one tree level."""
        scaling = getattr(self, f"scaling_{level}").to(
            dtype=parent.dtype,
            device=parent.device,
        )
        children = parent.unsqueeze(-1) * scaling
        if self.conservative:
            contrast = getattr(self, f"contrast_{level}").to(
                dtype=parent.dtype,
                device=parent.device,
            )
            children = children + torch.einsum("...d,rd->...r", update, contrast)
        else:
            children = children + update
        return children

    def forward(
        self,
        coarse: torch.Tensor,
        details: tuple[torch.Tensor, ...],
        horizon: int,
    ) -> torch.Tensor:
        """Synthesize active coefficients into `[B, C, H]` leaves."""
        _validate_horizon(horizon)
        if len(details) != len(PMFO_RADICES):
            raise ValueError("PMFO synthesis expects four detail groups")
        values = coarse
        for level, (radix, detail) in enumerate(
            zip(PMFO_RADICES, details, strict=True)
        ):
            active_children = math.ceil(horizon / PMFO_BLOCK_SIZES[level + 1])
            children = self.refine(values, detail, level)
            values = children.flatten(start_dim=-2)[..., :active_children]
        return values[..., :horizon]


class PMFORCTReadout(nn.Module):
    """Shared parent-to-child state tree with conservative synthesis."""

    def __init__(
        self,
        readout_dim: int,
        state_dim: int = 32,
        conservative: bool = True,
    ) -> None:
        super().__init__()
        if readout_dim <= 0 or state_dim <= 0:
            raise ValueError("readout_dim and state_dim must be positive")
        self.readout_dim = readout_dim
        self.state_dim = state_dim
        self.conservative = conservative
        self.coarse_count = PMFO_SERIES_LENGTH // PMFO_BLOCK_SIZES[0]
        self.seed = nn.Linear(readout_dim, self.coarse_count * state_dim)
        self.coarse_head = nn.Linear(state_dim, 1)
        self.split_layers = nn.ModuleList(
            nn.Linear(state_dim, radix * state_dim) for radix in PMFO_RADICES
        )
        update_dims = (
            radix - 1 if conservative else radix for radix in PMFO_RADICES
        )
        self.detail_heads = nn.ModuleList(
            nn.Linear((radix + 1) * state_dim, update_dim)
            for radix, update_dim in zip(
                PMFO_RADICES,
                update_dims,
                strict=True,
            )
        )
        self.synthesis = ConservativeTreeSynthesis(conservative=conservative)

    def tree_coefficients(
        self,
        hidden: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Produce active coarse/detail groups without horizon-valued inputs."""
        active_coarse, _active_details = _active_group_sizes(horizon)
        batch, channels, _readout_dim = hidden.shape
        states = F.gelu(self.seed(hidden)).reshape(
            batch,
            channels,
            self.coarse_count,
            self.state_dim,
        )
        states = states[:, :, :active_coarse, :]
        coarse = self.coarse_head(states).squeeze(-1)
        details = []
        for level, (radix, split_layer, detail_head) in enumerate(
            zip(
                PMFO_RADICES,
                self.split_layers,
                self.detail_heads,
                strict=True,
            )
        ):
            active_children = math.ceil(
                horizon / PMFO_BLOCK_SIZES[level + 1]
            )
            children = F.gelu(split_layer(states)).reshape(
                batch,
                channels,
                states.shape[2],
                radix,
                self.state_dim,
            )
            context = torch.cat(
                [states, children.flatten(start_dim=-2)],
                dim=-1,
            )
            details.append(detail_head(context))
            states = children.flatten(start_dim=2, end_dim=3)
            states = states[:, :, :active_children, :]
        return coarse, tuple(details)

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = PMFO_SERIES_LENGTH if target_prefix is None else int(target_prefix)
        coarse, details = self.tree_coefficients(hidden, horizon)
        leaves = self.synthesis(coarse, details, horizon)
        return leaves.permute(0, 2, 1)


class PMFONoTransitionReadout(nn.Module):
    """Direct history-to-scale control with the same conservative synthesis."""

    def __init__(self, readout_dim: int) -> None:
        super().__init__()
        if readout_dim <= 0:
            raise ValueError("readout_dim must be positive")
        self.readout_dim = readout_dim
        self.conservative = True
        self.group_sizes = (8, 16, 48, 72, 576)
        self.coefficient_heads = nn.ModuleList(
            nn.Linear(readout_dim, group_size) for group_size in self.group_sizes
        )
        self.synthesis = ConservativeTreeSynthesis(conservative=True)

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = PMFO_SERIES_LENGTH if target_prefix is None else int(target_prefix)
        active_coarse, active_details = _active_group_sizes(horizon)
        groups = tuple(head(hidden) for head in self.coefficient_heads)
        coarse = groups[0][..., :active_coarse]
        details = tuple(
            group[..., :active_size].reshape(
                *hidden.shape[:2],
                active_size // (radix - 1),
                radix - 1,
            )
            for group, active_size, radix in zip(
                groups[1:],
                active_details,
                PMFO_RADICES,
                strict=True,
            )
        )
        leaves = self.synthesis(coarse, details, horizon)
        return leaves.permute(0, 2, 1)


class DenseMLPMatchedReadout(nn.Module):
    """Nonlinear dense future head matched to the PMFO decoder budget."""

    def __init__(self, readout_dim: int, hidden_dim: int = 144) -> None:
        super().__init__()
        if readout_dim <= 0 or hidden_dim <= 0:
            raise ValueError("readout_dim and hidden_dim must be positive")
        self.readout_dim = readout_dim
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(readout_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, PMFO_SERIES_LENGTH)

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = PMFO_SERIES_LENGTH if target_prefix is None else int(target_prefix)
        _validate_horizon(horizon)
        output = self.output_layer(F.gelu(self.input_layer(hidden)))
        return output[..., :horizon].permute(0, 2, 1)
