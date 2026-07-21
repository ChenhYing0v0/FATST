"""Common-private scope interaction readouts built on ISCF."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SIFF import SIFFCouplingFieldReadout


CPSI_INTERACTION_MODES = frozenset(
    {"common-private", "self", "linear", "common", "post-synthesis"}
)
CPSI_READOUT_CONFIG = {
    "iscf-v1-cpsi": "common-private",
    "iscf-v1-cpsi-self": "self",
    "iscf-v1-cpsi-linear": "linear",
    "iscf-v1-cpsi-common": "common",
    "iscf-v1-cpsi-post": "post-synthesis",
}
CPSI_READOUT_MODES = frozenset(CPSI_READOUT_CONFIG)


def cpsi_interaction_parameter_count(
    coordinate_dim: int,
    mode_rank: int,
    interaction_rank: int,
) -> int:
    """Return the exact pre-synthesis CPSI parameter count."""
    flattened_dim = int(coordinate_dim) * int(mode_rank)
    return 3 * flattened_dim * int(interaction_rank)


class CPSIReadout(SIFFCouplingFieldReadout):
    """Interact independent scope modes before or after scope synthesis.

    The parent ISCF parameters are initialized by ``SIFFCouplingFieldReadout``
    before the three interaction matrices are created.  This preserves paired
    parent initialization for every CPSI arm under the same random seed.
    """

    def __init__(
        self,
        readout_dim: int,
        series_length: int = 720,
        scales: Sequence[int] = (1, 48, 144, 360, 720),
        coordinate_dim: int = 4,
        mode_rank: int = 256,
        interaction_rank: int = 32,
        interaction_mode: str = "common-private",
        policy_history_dim: int = 32,
        policy_hidden_dim: int = 64,
        policy_mode: str = "direct",
        fixed_scale: int = 720,
        partition: str = "canonical",
        partition_seed: int = 15101,
        group_chunk_size: int = 64,
        target_chunk_size: int = 128,
    ) -> None:
        interaction_mode = str(interaction_mode)
        if interaction_mode not in CPSI_INTERACTION_MODES:
            raise ValueError(f"unsupported CPSI interaction: {interaction_mode}")
        if interaction_rank <= 0:
            raise ValueError("CPSI interaction rank must be positive")
        if policy_mode != "direct":
            raise ValueError("CPSI v1 requires the frozen direct policy")

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

        self.interaction_mode = interaction_mode
        self.interaction_rank = int(interaction_rank)
        self.mode_width = self.coordinate_dim * self.mode_rank
        self.interaction_width = (
            self.series_length
            if self.interaction_mode == "post-synthesis"
            else self.mode_width
        )
        self.effective_interaction_rank = (
            max(
                1,
                round(
                    self.mode_width
                    * self.interaction_rank
                    / self.series_length
                ),
            )
            if self.interaction_mode == "post-synthesis"
            else self.interaction_rank
        )
        self.common_projection = nn.Parameter(
            torch.empty(
                self.effective_interaction_rank,
                self.interaction_width,
            )
        )
        self.private_projection = nn.Parameter(
            torch.empty(
                self.effective_interaction_rank,
                self.interaction_width,
            )
        )
        self.interaction_output = nn.Parameter(
            torch.zeros(
                self.interaction_width,
                self.effective_interaction_rank,
            )
        )
        nn.init.xavier_uniform_(self.common_projection)
        nn.init.xavier_uniform_(self.private_projection)

    @property
    def interaction_parameters(self) -> int:
        """Return the number of parameters in the interaction path."""
        return sum(
            parameter.numel()
            for parameter in (
                self.common_projection,
                self.private_projection,
                self.interaction_output,
            )
        )

    def _interaction_terms(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return updated ``[B,C,S,W]`` values and diagnostic tensors."""
        if values.ndim != 4 or values.shape[2] != len(self.scales):
            raise ValueError(
                "CPSI expects values [B,C,S,W] with one slot per scope"
            )
        if values.shape[-1] != self.interaction_width:
            raise ValueError(
                f"CPSI expected width {self.interaction_width}, "
                f"got {values.shape[-1]}"
            )
        common = values.mean(dim=2, keepdim=True)
        private = values - common

        if self.interaction_mode == "self":
            left = F.gelu(F.linear(values, self.common_projection))
            right = F.gelu(F.linear(values, self.private_projection))
            latent = left * right
        elif self.interaction_mode == "linear":
            left = F.linear(common, self.common_projection)
            right = F.linear(private, self.private_projection)
            latent = left + right
        elif self.interaction_mode == "common":
            left = F.gelu(F.linear(common, self.common_projection))
            right = F.gelu(F.linear(common, self.private_projection))
            latent = (left * right).expand(-1, -1, values.shape[2], -1)
        else:
            left = F.gelu(F.linear(common, self.common_projection))
            right = F.gelu(F.linear(private, self.private_projection))
            latent = left * right

        message = F.linear(latent, self.interaction_output)
        updated = values + message
        return updated, {
            "common": common,
            "private": private,
            "left": left,
            "right": right,
            "latent": latent,
            "message": message,
        }

    def interact_modes(
        self,
        modes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Interact modes ``[B,C,S,D,K]`` before scope synthesis."""
        if modes.ndim != 5:
            raise ValueError("CPSI expects modes [B,C,S,D,K]")
        if modes.shape[-2:] != (self.coordinate_dim, self.mode_rank):
            raise ValueError(
                "CPSI mode coordinate/rank dimensions do not match readout"
            )
        flattened = modes.flatten(start_dim=-2)
        updated, details = self._interaction_terms(flattened)
        return updated.unflatten(
            -1,
            (self.coordinate_dim, self.mode_rank),
        ), details

    def interact_forecasts(
        self,
        arms: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Interact arm forecasts ``[B,C,S,T]`` after scope synthesis."""
        return self._interaction_terms(arms)

    def _arm_forecasts_from_components(
        self,
        components: torch.Tensor,
    ) -> torch.Tensor:
        modes = self._history_modes_from_components(components)
        if self.interaction_mode != "post-synthesis":
            modes, _details = self.interact_modes(modes)
        arms = torch.stack(
            [
                self._scope_forecast(modes[:, :, scale_index], scale_index)
                for scale_index in range(len(self.scales))
            ],
            dim=2,
        )
        if self.interaction_mode == "post-synthesis":
            arms, _details = self.interact_forecasts(arms)
        return arms

    def interaction_diagnostics(
        self,
        hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return internal interaction tensors without changing inference."""
        components = self.component_history_modes(hidden)
        modes = self._history_modes_from_components(components)
        if self.interaction_mode == "post-synthesis":
            arms = torch.stack(
                [
                    self._scope_forecast(
                        modes[:, :, scale_index],
                        scale_index,
                    )
                    for scale_index in range(len(self.scales))
                ],
                dim=2,
            )
            _updated, details = self.interact_forecasts(arms)
            return details
        _updated, details = self.interact_modes(modes)
        return details
