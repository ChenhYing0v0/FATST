"""Native projective coupling-spectrum decoder readout."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def pcsd_parameter_count(
    readout_dim: int,
    series_length: int = 720,
    scale_count: int = 5,
    coordinate_dim: int = 4,
    mode_rank: int = 256,
    policy_history_dim: int = 32,
    policy_hidden_dim: int = 64,
) -> int:
    """Return the trainable parameter count of the full PCSD-CF readout."""
    field = coordinate_dim * readout_dim * mode_rank
    field += coordinate_dim * mode_rank
    field += 2 * series_length * mode_rank + series_length
    policy = readout_dim * policy_history_dim + policy_history_dim
    policy += (
        (policy_history_dim + coordinate_dim) * policy_hidden_dim
        + policy_hidden_dim
    )
    policy += policy_hidden_dim * scale_count + scale_count
    return field + policy


class PCSDM0Readout(nn.Module):
    """Exact A6-equivalent morphism control with paired initialization order."""

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        mode_rank: int = 256,
    ) -> None:
        super().__init__()
        self.readout_dim = int(readout_dim)
        self.series_length = int(series_length)
        self.mode_rank = int(mode_rank)
        if self.readout_dim <= 0 or self.series_length <= 0 or self.mode_rank <= 0:
            raise ValueError("PCSD M0 dimensions must be positive")
        self.coefficient = nn.Linear(self.readout_dim, self.mode_rank)
        self.identity_synthesis = nn.Parameter(
            torch.empty(self.series_length, self.mode_rank)
        )
        self.temporal_bias = nn.Parameter(torch.zeros(self.series_length))
        nn.init.normal_(
            self.identity_synthesis,
            mean=0.0,
            std=self.mode_rank**-0.5,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        coefficients = self.coefficient(hidden)
        full = torch.einsum(
            "tk,bck->bct",
            self.identity_synthesis.to(dtype=hidden.dtype),
            coefficients,
        )
        full = full + self.temporal_bias.to(dtype=hidden.dtype).view(1, 1, -1)
        return full[:, :, :horizon].permute(0, 2, 1)


class PCSDDenseMatchedReadout(nn.Module):
    """Generic dense nonlinear control matched to full PCSD parameter storage."""

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        target_parameters: int | None = None,
    ) -> None:
        super().__init__()
        self.readout_dim = int(readout_dim)
        self.series_length = int(series_length)
        self.target_parameters = int(
            pcsd_parameter_count(self.readout_dim, self.series_length)
            if target_parameters is None
            else target_parameters
        )
        if self.readout_dim <= 0 or self.series_length <= 0:
            raise ValueError("dense matched dimensions must be positive")
        denominator = self.readout_dim + self.series_length + 1
        real_width = (self.target_parameters - self.series_length) / denominator
        candidates = {
            max(1, int(real_width)),
            max(1, int(real_width) + 1),
        }
        self.hidden_dim = min(
            candidates,
            key=lambda width: abs(
                width * denominator
                + self.series_length
                - self.target_parameters
            ),
        )
        self.input = nn.Linear(self.readout_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.series_length)

    @property
    def decoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_relative_gap(self) -> float:
        return abs(self.decoder_parameters - self.target_parameters) / float(
            self.target_parameters
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        full = self.output(F.gelu(self.input(hidden)))
        return full[:, :, :horizon].permute(0, 2, 1)


class PCSDCouplingFieldReadout(nn.Module):
    """Generate multiple future-output coupling scopes from one parameter field.

    The module always constructs the complete ``series_length`` forecast. A
    requested prefix is applied only after the direct policy has fused the
    scope arms, so the requested horizon cannot affect decoder computation.
    """

    POLICY_MODES = {
        "direct",
        "equal",
        "static-target",
        "fixed",
        "target-scale-field",
        "target-scale-field-permuted",
        "target-scale-global",
    }
    PARTITIONS = {"canonical", "random"}

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        scales: Sequence[int] = (1, 48, 144, 360, 720),
        coordinate_dim: int = 4,
        mode_rank: int = 256,
        policy_history_dim: int = 32,
        policy_hidden_dim: int = 64,
        policy_mode: str = "direct",
        fixed_scale: int = 720,
        partition: str = "canonical",
        partition_seed: int = 15101,
        group_chunk_size: int = 64,
        target_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        self.readout_dim = int(readout_dim)
        self.series_length = int(series_length)
        self.scales = tuple(int(scale) for scale in scales)
        self.coordinate_dim = int(coordinate_dim)
        self.mode_rank = int(mode_rank)
        self.policy_history_dim = int(policy_history_dim)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.policy_mode = str(policy_mode)
        self.fixed_scale = int(fixed_scale)
        self.partition = str(partition)
        self.partition_seed = int(partition_seed)
        self.group_chunk_size = int(group_chunk_size)
        self.target_chunk_size = int(target_chunk_size)
        self._validate_contract()

        coordinate_field = self._build_coordinate_field(
            self.series_length,
            self.coordinate_dim,
        )
        self.register_buffer("coordinate_field", coordinate_field)
        self._group_index_names: list[str] = []
        self._pooled_coordinate_names: list[str] = []
        for scale_index, scale in enumerate(self.scales):
            group_indices = self._build_group_indices(scale, scale_index)
            pooled_coordinates = coordinate_field[group_indices].mean(dim=1)
            index_name = f"group_indices_{scale_index}"
            coordinate_name = f"pooled_coordinates_{scale_index}"
            self.register_buffer(index_name, group_indices)
            self.register_buffer(coordinate_name, pooled_coordinates)
            self._group_index_names.append(index_name)
            self._pooled_coordinate_names.append(coordinate_name)

        self.mode_weight = nn.Parameter(
            torch.empty(
                self.coordinate_dim,
                self.readout_dim,
                self.mode_rank,
            )
        )
        self.mode_bias = nn.Parameter(
            torch.zeros(self.coordinate_dim, self.mode_rank)
        )
        self.identity_synthesis = nn.Parameter(
            torch.empty(self.series_length, self.mode_rank)
        )
        self.nonlinear_synthesis = nn.Parameter(
            torch.empty(self.series_length, self.mode_rank)
        )
        self.temporal_bias = nn.Parameter(torch.zeros(self.series_length))

        self.history_projection = nn.Linear(
            self.readout_dim,
            self.policy_history_dim,
        )
        self.policy_hidden = nn.Linear(
            self.policy_history_dim + self.coordinate_dim,
            self.policy_hidden_dim,
        )
        self.policy_output = nn.Linear(
            self.policy_hidden_dim,
            len(self.scales),
        )
        self.reset_parameters()

    def _validate_contract(self) -> None:
        if self.readout_dim <= 0 or self.series_length <= 0:
            raise ValueError("readout_dim and series_length must be positive")
        if self.coordinate_dim < 2:
            raise ValueError("coordinate_dim must include constant and nonconstant modes")
        if self.mode_rank <= 0:
            raise ValueError("mode_rank must be positive")
        if self.policy_history_dim <= 0 or self.policy_hidden_dim <= 0:
            raise ValueError("policy dimensions must be positive")
        if self.policy_mode not in self.POLICY_MODES:
            raise ValueError(f"unsupported PCSD policy mode: {self.policy_mode}")
        if self.partition not in self.PARTITIONS:
            raise ValueError(f"unsupported PCSD partition: {self.partition}")
        if not self.scales or len(set(self.scales)) != len(self.scales):
            raise ValueError("PCSD scales must be non-empty and unique")
        if tuple(sorted(self.scales)) != self.scales:
            raise ValueError("PCSD scales must be strictly increasing")
        if any(
            scale <= 0 or self.series_length % scale
            for scale in self.scales
        ):
            raise ValueError("every PCSD scale must divide series_length")
        if self.fixed_scale not in self.scales:
            raise ValueError("fixed_scale must be one of the PCSD scales")
        if self.group_chunk_size <= 0 or self.target_chunk_size <= 0:
            raise ValueError("PCSD chunk sizes must be positive")

    @staticmethod
    def _build_coordinate_field(length: int, dimension: int) -> torch.Tensor:
        steps = torch.arange(length, dtype=torch.float64) + 0.5
        frequencies = torch.arange(dimension, dtype=torch.float64)
        coordinates = torch.cos(
            torch.pi * torch.outer(steps, frequencies) / float(length)
        )
        coordinates[:, 0] = 1.0
        if dimension > 1:
            coordinates[:, 1:] *= 2.0**0.5
            coordinates[:, 1:] -= coordinates[:, 1:].mean(dim=0, keepdim=True)
        return coordinates.to(torch.float32)

    def _build_group_indices(
        self,
        scale: int,
        scale_index: int,
    ) -> torch.Tensor:
        indices = torch.arange(self.series_length, dtype=torch.long)
        if self.partition == "random" and scale not in {1, self.series_length}:
            generator = torch.Generator(device="cpu").manual_seed(
                self.partition_seed + 1009 * scale_index + scale
            )
            indices = indices[torch.randperm(self.series_length, generator=generator)]
        return indices.reshape(self.series_length // scale, scale)

    def reset_parameters(self) -> None:
        bound = self.readout_dim**-0.5
        nn.init.uniform_(
            self.mode_weight,
            -bound,
            bound,
        )
        nn.init.normal_(
            self.identity_synthesis,
            mean=0.0,
            std=self.mode_rank**-0.5,
        )
        nn.init.normal_(
            self.nonlinear_synthesis,
            mean=0.0,
            std=self.mode_rank**-0.5,
        )
        nn.init.uniform_(self.mode_bias, -bound, bound)
        nn.init.zeros_(self.temporal_bias)
        self.history_projection.reset_parameters()
        self.policy_hidden.reset_parameters()
        nn.init.zeros_(self.policy_output.weight)
        nn.init.zeros_(self.policy_output.bias)

    def group_indices(self, scale_index: int) -> torch.Tensor:
        return getattr(self, self._group_index_names[scale_index])

    def pooled_coordinates(self, scale_index: int) -> torch.Tensor:
        return getattr(self, self._pooled_coordinate_names[scale_index])

    def target_group_labels(self, scale_index: int) -> torch.Tensor:
        indices = self.group_indices(scale_index)
        labels = torch.empty(
            self.series_length,
            dtype=torch.long,
            device=indices.device,
        )
        group_ids = torch.arange(
            indices.shape[0],
            device=indices.device,
        ).unsqueeze(1).expand_as(indices)
        return labels.scatter(0, indices.flatten(), group_ids.flatten())

    def target_scope_descriptors(self, scale_index: int) -> torch.Tensor:
        labels = self.target_group_labels(scale_index)
        return self.pooled_coordinates(scale_index)[labels]

    def history_modes(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[B,C,R]`` history states to ``[B,C,Dq,K]`` modes."""
        return (
            torch.einsum("bcr,drk->bcdk", hidden, self.mode_weight)
            + self.mode_bias.view(1, 1, self.coordinate_dim, self.mode_rank)
        )

    def _scope_forecast(
        self,
        modes: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        indices = self.group_indices(scale_index)
        pooled_coordinates = self.pooled_coordinates(scale_index).to(
            dtype=modes.dtype
        )
        grouped_outputs: list[torch.Tensor] = []
        flattened_indices: list[torch.Tensor] = []
        for start in range(0, indices.shape[0], self.group_chunk_size):
            end = min(start + self.group_chunk_size, indices.shape[0])
            chunk_indices = indices[start:end]
            chunk_coordinates = pooled_coordinates[start:end]
            states = torch.einsum("bcdk,gd->bcgk", modes, chunk_coordinates)
            identity_rows = self.identity_synthesis[chunk_indices].to(
                dtype=modes.dtype
            )
            nonlinear_rows = self.nonlinear_synthesis[chunk_indices].to(
                dtype=modes.dtype
            )
            values = torch.einsum("bcgk,gsk->bcgs", states, identity_rows)
            values = values + torch.einsum(
                "bcgk,gsk->bcgs",
                F.gelu(states),
                nonlinear_rows,
            )
            grouped_outputs.append(values.flatten(start_dim=-2))
            flattened_indices.append(chunk_indices.flatten())
        source = torch.cat(grouped_outputs, dim=-1)
        target_indices = torch.cat(flattened_indices).view(1, 1, -1)
        target_indices = target_indices.expand(source.shape[0], source.shape[1], -1)
        output = source.new_zeros(source.shape[0], source.shape[1], self.series_length)
        output = output.scatter(-1, target_indices, source)
        return output + self.temporal_bias.to(dtype=modes.dtype).view(1, 1, -1)

    def arm_forecasts(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return scope arms with shape ``[B,C,S,T]``."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.readout_dim:
            raise ValueError(
                f"expected hidden [B,C,{self.readout_dim}], got {tuple(hidden.shape)}"
            )
        modes = self.history_modes(hidden)
        arms = [
            self._scope_forecast(modes, scale_index)
            for scale_index in range(len(self.scales))
        ]
        return torch.stack(arms, dim=2)

    def _learned_policy_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.policy_mode.startswith("target-scale-"):
            raise RuntimeError(
                "target-scale allocation policies require SIFF semantics"
            )
        history_state = self.history_projection(hidden)
        if self.policy_mode == "static-target":
            history_state = torch.zeros_like(history_state)
        coordinate_field = self.coordinate_field.to(dtype=hidden.dtype)
        chunks: list[torch.Tensor] = []
        for start in range(0, self.series_length, self.target_chunk_size):
            end = min(start + self.target_chunk_size, self.series_length)
            target_count = end - start
            history_chunk = history_state.unsqueeze(2).expand(
                -1,
                -1,
                target_count,
                -1,
            )
            target_chunk = coordinate_field[start:end].view(
                1,
                1,
                target_count,
                self.coordinate_dim,
            ).expand(hidden.shape[0], hidden.shape[1], -1, -1)
            policy_input = torch.cat((history_chunk, target_chunk), dim=-1)
            chunks.append(
                self.policy_output(F.gelu(self.policy_hidden(policy_input)))
            )
        return torch.cat(chunks, dim=2)

    def policy_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return per-target scope probabilities with shape ``[B,C,T,S]``."""
        shape = (
            hidden.shape[0],
            hidden.shape[1],
            self.series_length,
            len(self.scales),
        )
        if self.policy_mode == "equal":
            return hidden.new_full(shape, 1.0 / len(self.scales))
        if self.policy_mode == "fixed":
            weights = hidden.new_zeros(shape)
            weights[..., self.scales.index(self.fixed_scale)] = 1.0
            return weights
        return torch.softmax(self._learned_policy_logits(hidden), dim=-1)

    def forward_with_diagnostics(
        self,
        hidden: torch.Tensor,
        target_prefix: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        arms = self.arm_forecasts(hidden)
        weights = self.policy_weights(hidden)
        full = (arms * weights.permute(0, 1, 3, 2)).sum(dim=2)
        return full[:, :, :horizon].permute(0, 2, 1), arms, weights

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
            modes = self.history_modes(hidden)
            output = self._scope_forecast(
                modes,
                self.scales.index(self.fixed_scale),
            )
            return output[:, :, :horizon].permute(0, 2, 1)
        output, _arms, _weights = self.forward_with_diagnostics(
            hidden,
            target_prefix,
        )
        return output

    def map_a6_parameters_(
        self,
        coefficient_weight: torch.Tensor,
        coefficient_bias: torch.Tensor,
        temporal_basis: torch.Tensor,
        temporal_bias: torch.Tensor,
    ) -> None:
        """Construct the exact A6-LBF subspace witness in place."""
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
            raise ValueError(f"invalid A6 mapping shapes: expected {expected}, got {actual}")
        with torch.no_grad():
            self.mode_weight.zero_()
            self.mode_bias.zero_()
            self.mode_weight[0].copy_(
                coefficient_weight.transpose(0, 1).to(self.mode_weight)
            )
            self.mode_bias[0].copy_(coefficient_bias.to(self.mode_bias))
            self.identity_synthesis.copy_(temporal_basis.to(self.identity_synthesis))
            self.nonlinear_synthesis.zero_()
            self.temporal_bias.copy_(temporal_bias.to(self.temporal_bias))

    @property
    def coupling_field_parameters(self) -> int:
        names = {
            "mode_weight",
            "mode_bias",
            "identity_synthesis",
            "nonlinear_synthesis",
            "temporal_bias",
        }
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name in names
        )

    @property
    def policy_parameters(self) -> int:
        prefixes = (
            "history_projection.",
            "policy_hidden.",
            "policy_output.",
            "target_allocation_projection.",
            "scale_allocation_projection.",
            "target_scale_allocation_bias",
            "target_scale_allocation_output.",
        )
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
        )
