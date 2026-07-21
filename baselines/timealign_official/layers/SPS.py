"""Scope-projected synthesis for independent coupling fields."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from layers.SIFF import SIFFCouplingFieldReadout


SPS_PROJECTION_MODES = frozenset({"scope", "global", "identity"})


class ScopeProjectedSynthesisReadout(SIFFCouplingFieldReadout):
    """Filter each ISCF arm through a scope-native output projector.

    The parent produces raw arms ``[B,C,S,T]``.  Before policy fusion, each
    arm is gathered into its native groups ``[B,C,G_s,s]`` and projected onto
    the first ``r_s`` orthonormal local-DCT modes.  The projection acts on the
    forward forecast and, by transposition, on the gradient received by the
    corresponding independent history map.
    """

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        scales: Sequence[int] = (1, 48, 144, 360, 720),
        coordinate_dim: int = 4,
        mode_rank: int = 256,
        projection_mode: str = "scope",
        policy_history_dim: int = 32,
        policy_hidden_dim: int = 64,
        policy_mode: str = "direct",
        fixed_scale: int = 720,
        partition: str = "canonical",
        partition_seed: int = 15101,
        group_chunk_size: int = 64,
        target_chunk_size: int = 128,
    ) -> None:
        projection_mode = str(projection_mode)
        if projection_mode not in SPS_PROJECTION_MODES:
            raise ValueError(f"unsupported SPS projection: {projection_mode}")
        if policy_mode != "direct":
            raise ValueError("SPS v0 requires the frozen ISCF direct policy")
        super().__init__(
            readout_dim=readout_dim,
            series_length=series_length,
            scales=scales,
            coordinate_dim=coordinate_dim,
            mode_rank=mode_rank,
            scale_components=len(scales),
            scale_basis_mode="independent",
            policy_history_dim=policy_history_dim,
            policy_hidden_dim=policy_hidden_dim,
            policy_mode=policy_mode,
            fixed_scale=fixed_scale,
            partition=partition,
            partition_seed=partition_seed,
            group_chunk_size=group_chunk_size,
            target_chunk_size=target_chunk_size,
        )
        self.projection_mode = projection_mode
        self._projection_basis_names: list[str] = []
        self._projection_ranks: list[int] = []
        for scale_index, scale in enumerate(self.scales):
            basis_length = (
                self.series_length
                if self.projection_mode == "global"
                else scale
            )
            rank = self._projection_rank(scale, basis_length)
            name = f"sps_projection_basis_{scale_index}"
            self.register_buffer(name, self._local_dct_basis(basis_length, rank))
            self._projection_basis_names.append(name)
            self._projection_ranks.append(rank)

    def _projection_rank(self, scale: int, basis_length: int) -> int:
        if self.projection_mode == "identity":
            return basis_length
        if self.projection_mode == "global":
            return min(self.series_length, self.mode_rank)
        return min(
            scale,
            max(1, round(self.mode_rank * scale / self.series_length)),
        )

    @staticmethod
    def _local_dct_basis(length: int, rank: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float64).unsqueeze(1)
        frequencies = torch.arange(rank, dtype=torch.float64).unsqueeze(0)
        basis = torch.cos(
            math.pi * (positions + 0.5) * frequencies / float(length)
        )
        basis[:, 0] *= math.sqrt(1.0 / float(length))
        if rank > 1:
            basis[:, 1:] *= math.sqrt(2.0 / float(length))
        return basis

    @property
    def projection_ranks(self) -> tuple[int, ...]:
        """Return retained DCT ranks in scale order."""
        return tuple(self._projection_ranks)

    def projection_basis(self, scale_index: int) -> torch.Tensor:
        """Return the registered orthonormal basis for one scope."""
        return getattr(self, self._projection_basis_names[scale_index])

    def _project_scope_arm(
        self,
        arm: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        if arm.ndim != 3 or arm.shape[-1] != self.series_length:
            raise ValueError(
                "SPS expects one arm [B,C,T] with the configured series length"
            )
        basis = self.projection_basis(scale_index).to(dtype=arm.dtype)
        if self.projection_mode == "global":
            coefficients = torch.einsum("bct,tr->bcr", arm, basis)
            return torch.einsum("bcr,tr->bct", coefficients, basis)

        indices = self.group_indices(scale_index)
        grouped = arm[..., indices]
        coefficients = torch.einsum("bcgs,sr->bcgr", grouped, basis)
        reconstructed = torch.einsum("bcgr,sr->bcgs", coefficients, basis)
        source = reconstructed.flatten(start_dim=-2)
        flattened_indices = indices.flatten().view(1, 1, -1)
        flattened_indices = flattened_indices.expand(
            source.shape[0], source.shape[1], -1
        )
        output = source.new_zeros(source.shape[0], source.shape[1], self.series_length)
        return output.scatter(-1, flattened_indices, source)

    def _arm_forecasts_from_components(
        self,
        components: torch.Tensor,
    ) -> torch.Tensor:
        modes = self._history_modes_from_components(components)
        arms = []
        for scale_index in range(len(self.scales)):
            raw = self._scope_forecast(modes[:, :, scale_index], scale_index)
            arms.append(self._project_scope_arm(raw, scale_index))
        return torch.stack(arms, dim=2)

    def projection_diagnostics(
        self,
        hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return raw/projected arms and removed energy for local audits."""
        components = self.component_history_modes(hidden)
        modes = self._history_modes_from_components(components)
        raw_arms = torch.stack(
            [
                self._scope_forecast(modes[:, :, scale_index], scale_index)
                for scale_index in range(len(self.scales))
            ],
            dim=2,
        )
        projected_arms = torch.stack(
            [
                self._project_scope_arm(
                    raw_arms[:, :, scale_index], scale_index
                )
                for scale_index in range(len(self.scales))
            ],
            dim=2,
        )
        removed = raw_arms - projected_arms
        return {
            "raw_arms": raw_arms,
            "projected_arms": projected_arms,
            "removed_arms": removed,
            "removed_rms": removed.square().mean(dim=(0, 1, 3)).sqrt(),
        }
