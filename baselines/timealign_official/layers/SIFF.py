"""Scale-indexed forecast field readouts for projective forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from layers.PCSD import PCSDCouplingFieldReadout, pcsd_parameter_count


SIFF_SCALE_BASIS_MODES = frozenset(
    {"ordered", "constant", "permuted", "independent"}
)


def siff_parameter_count(
    readout_dim: int,
    series_length: int = 720,
    scale_count: int = 5,
    coordinate_dim: int = 4,
    mode_rank: int = 256,
    scale_components: int = 2,
    policy_history_dim: int = 32,
    policy_hidden_dim: int = 64,
) -> int:
    """Return the full SIFF readout parameter count."""
    base = pcsd_parameter_count(
        readout_dim=readout_dim,
        series_length=series_length,
        scale_count=scale_count,
        coordinate_dim=coordinate_dim,
        mode_rank=mode_rank,
        policy_history_dim=policy_history_dim,
        policy_hidden_dim=policy_hidden_dim,
    )
    one_field = coordinate_dim * readout_dim * mode_rank
    one_field += coordinate_dim * mode_rank
    return base + (int(scale_components) - 1) * one_field


class SIFFCouplingFieldReadout(PCSDCouplingFieldReadout):
    """Condition one shared coupling field on an internal scale coordinate.

    ``hidden [B,C,R]`` first produces component modes ``[B,C,Q,D,K]``.
    A fixed scale basis ``[S,Q]`` then constructs scale-indexed history modes
    ``[B,C,S,D,K]``.  Each scale-specific mode is consumed by the unchanged
    PCSD scope pooling and shared synthesis path.
    """

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        scales: Sequence[int] = (1, 48, 144, 360, 720),
        coordinate_dim: int = 4,
        mode_rank: int = 256,
        scale_components: int = 2,
        scale_basis_mode: str = "ordered",
        policy_history_dim: int = 32,
        policy_hidden_dim: int = 64,
        policy_mode: str = "direct",
        fixed_scale: int = 720,
        partition: str = "canonical",
        partition_seed: int = 15101,
        group_chunk_size: int = 64,
        target_chunk_size: int = 128,
    ) -> None:
        self.scale_components = int(scale_components)
        self.scale_basis_mode = str(scale_basis_mode)
        if self.scale_components <= 0:
            raise ValueError("scale_components must be positive")
        if self.scale_basis_mode not in SIFF_SCALE_BASIS_MODES:
            raise ValueError(
                f"unsupported SIFF scale basis: {self.scale_basis_mode}"
            )
        if self.scale_basis_mode == "independent":
            if self.scale_components != len(scales):
                raise ValueError(
                    "independent SIFF requires one component per scale"
                )
        elif self.scale_components not in {1, 2}:
            raise ValueError("ordered SIFF controls require Q in {1, 2}")

        super().__init__(
            readout_dim=readout_dim,
            series_length=series_length,
            scales=scales,
            coordinate_dim=coordinate_dim,
            mode_rank=mode_rank,
            policy_history_dim=policy_history_dim,
            policy_hidden_dim=policy_hidden_dim,
            policy_mode=policy_mode,
            fixed_scale=fixed_scale,
            partition=partition,
            partition_seed=partition_seed,
            group_chunk_size=group_chunk_size,
            target_chunk_size=target_chunk_size,
        )

        self.mode_weight = nn.Parameter(
            torch.empty(
                self.scale_components,
                self.coordinate_dim,
                self.readout_dim,
                self.mode_rank,
            )
        )
        self.mode_bias = nn.Parameter(
            torch.empty(
                self.scale_components,
                self.coordinate_dim,
                self.mode_rank,
            )
        )
        self.register_buffer(
            "scale_basis",
            self._build_scale_basis(
                self.scales,
                self.series_length,
                self.scale_components,
                self.scale_basis_mode,
            ),
        )
        self._reset_scale_field_parameters()

    @staticmethod
    def _build_scale_basis(
        scales: Sequence[int],
        series_length: int,
        components: int,
        mode: str,
    ) -> torch.Tensor:
        scale_count = len(scales)
        if mode == "independent":
            return torch.eye(scale_count, dtype=torch.float64)
        if components == 1:
            return torch.ones(scale_count, 1, dtype=torch.float64)

        coordinate = torch.log(
            torch.tensor(scales, dtype=torch.float64)
        ) / torch.log(torch.tensor(float(series_length), dtype=torch.float64))
        coordinate = coordinate - coordinate.mean()
        coordinate = coordinate / coordinate.square().mean().sqrt()
        if mode == "constant":
            coordinate = torch.ones_like(coordinate)
        elif mode == "permuted":
            coordinate = torch.flip(coordinate, dims=(0,))
        return torch.stack((torch.ones_like(coordinate), coordinate), dim=-1)

    def _reset_scale_field_parameters(self) -> None:
        bound = self.readout_dim**-0.5
        nn.init.uniform_(self.mode_weight, -bound, bound)
        nn.init.uniform_(self.mode_bias, -bound, bound)

    def component_history_modes(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[B,C,R]`` to component modes ``[B,C,Q,D,K]``."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.readout_dim:
            raise ValueError(
                f"expected hidden [B,C,{self.readout_dim}], got {tuple(hidden.shape)}"
            )
        return (
            torch.einsum("bcr,qdrk->bcqdk", hidden, self.mode_weight)
            + self.mode_bias.view(
                1,
                1,
                self.scale_components,
                self.coordinate_dim,
                self.mode_rank,
            )
        )

    def history_modes(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return scale-indexed modes with shape ``[B,C,S,D,K]``."""
        components = self.component_history_modes(hidden)
        return torch.einsum(
            "sq,bcqdk->bcsdk",
            self.scale_basis.to(dtype=hidden.dtype),
            components,
        )

    def arm_forecasts(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return scale arms with shape ``[B,C,S,T]``."""
        modes = self.history_modes(hidden)
        arms = [
            self._scope_forecast(modes[:, :, scale_index], scale_index)
            for scale_index in range(len(self.scales))
        ]
        return torch.stack(arms, dim=2)

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        if self.policy_mode == "fixed":
            horizon = (
                self.series_length if target_prefix is None else int(target_prefix)
            )
            if horizon <= 0 or horizon > self.series_length:
                raise ValueError("target_prefix must lie in [1, series_length]")
            scale_index = self.scales.index(self.fixed_scale)
            modes = self.history_modes(hidden)
            output = self._scope_forecast(modes[:, :, scale_index], scale_index)
            return output[:, :, :horizon].permute(0, 2, 1)
        return super().forward(hidden, target_prefix)

    def map_a6_parameters_(
        self,
        coefficient_weight: torch.Tensor,
        coefficient_bias: torch.Tensor,
        temporal_basis: torch.Tensor,
        temporal_bias: torch.Tensor,
    ) -> None:
        """Place the A6 witness in the constant component when available."""
        expected = {
            "coefficient_weight": (self.mode_rank, self.readout_dim),
            "coefficient_bias": (self.mode_rank,),
            "temporal_basis": (self.series_length, self.mode_rank),
            "temporal_bias": (self.series_length,),
        }
        actual = {
            "coefficient_weight": tuple(coefficient_weight.shape),
            "coefficient_bias": tuple(coefficient_bias.shape),
            "temporal_basis": tuple(temporal_basis.shape),
            "temporal_bias": tuple(temporal_bias.shape),
        }
        if actual != expected:
            raise ValueError(
                f"invalid A6 mapping shapes: expected {expected}, got {actual}"
            )
        if not torch.allclose(
            self.scale_basis[:, 0],
            torch.ones_like(self.scale_basis[:, 0]),
        ):
            raise ValueError("A6 mapping requires a constant first scale component")
        with torch.no_grad():
            self.mode_weight.zero_()
            self.mode_bias.zero_()
            self.mode_weight[0, 0].copy_(
                coefficient_weight.transpose(0, 1).to(self.mode_weight)
            )
            self.mode_bias[0, 0].copy_(coefficient_bias.to(self.mode_bias))
            self.identity_synthesis.copy_(temporal_basis.to(self.identity_synthesis))
            self.nonlinear_synthesis.zero_()
            self.temporal_bias.copy_(temporal_bias.to(self.temporal_bias))
