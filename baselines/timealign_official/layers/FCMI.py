"""Future-coordinate main-interaction readout and matched controls."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


FCMI_MODE = "fcmi"
STANDARD_QUERY_MODE = "fcmi-standard-query"
STANDARD_DUAL_MODE = "fcmi-standard-dual-matched"
GENERIC_DUAL_MODE = "fcmi-generic-dual-matched"
ORDER_SHUFFLED_MODE = "fcmi-order-shuffled"
TARGET_SHUFFLED_MODE = "fcmi-target-shuffled-query"

FCMI_READOUT_MODES = {
    FCMI_MODE,
    STANDARD_QUERY_MODE,
    STANDARD_DUAL_MODE,
    GENERIC_DUAL_MODE,
    ORDER_SHUFFLED_MODE,
    TARGET_SHUFFLED_MODE,
}
FCMI_DUAL_MODES = FCMI_READOUT_MODES - {STANDARD_QUERY_MODE}


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    """Return fixed positions with shape [length, dimension]."""
    if length <= 0 or dimension <= 0 or dimension % 2:
        raise ValueError(
            "sinusoidal positions require positive length and even dimension"
        )
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding


class FutureCoordinateMainInteractionReadout(nn.Module):
    """Decode ordered memory through identifiable main and interaction states."""

    def __init__(
        self,
        memory_dim: int,
        prediction_length: int,
        patch_count: int,
        n_heads: int,
        dropout: float,
        mode: str,
        permutation_seed: int,
    ) -> None:
        super().__init__()
        if mode not in FCMI_READOUT_MODES:
            raise ValueError(f"unsupported FCMI mode: {mode}")
        if memory_dim <= 0 or memory_dim % n_heads:
            raise ValueError("memory_dim must be positive and divisible by n_heads")
        if patch_count <= 0 or prediction_length <= 0:
            raise ValueError("patch_count and prediction_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.memory_dim = int(memory_dim)
        self.prediction_length = int(prediction_length)
        self.patch_count = int(patch_count)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.mode = mode

        self.query_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            memory_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        if mode == STANDARD_QUERY_MODE:
            self.standard_projection = nn.Linear(
                memory_dim,
                memory_dim,
                bias=False,
            )
        else:
            self.main_projection = nn.Linear(
                memory_dim,
                memory_dim,
                bias=False,
            )
            self.interaction_projection = nn.Linear(
                memory_dim,
                memory_dim,
                bias=False,
            )
            self.interaction_projection.load_state_dict(
                self.main_projection.state_dict()
            )
        self.output_projection = nn.Linear(memory_dim, 1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(permutation_seed)
        memory_permutation = torch.randperm(patch_count, generator=generator)
        target_permutation = torch.randperm(
            prediction_length,
            generator=generator,
        )
        self.register_buffer(
            "memory_positions",
            sinusoidal_positions(patch_count, memory_dim),
            persistent=True,
        )
        self.register_buffer(
            "target_positions",
            sinusoidal_positions(prediction_length, memory_dim),
            persistent=True,
        )
        self.register_buffer(
            "memory_permutation",
            memory_permutation,
            persistent=True,
        )
        self.register_buffer(
            "target_permutation",
            target_permutation,
            persistent=True,
        )

    @property
    def is_dual(self) -> bool:
        return self.mode in FCMI_DUAL_MODES

    @property
    def decoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def compose_context(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        """Compose context coordinates before the shared nonlinear readout."""
        effective_mode = self.mode if mode is None else mode
        if effective_mode not in FCMI_READOUT_MODES:
            raise ValueError(f"unsupported FCMI composition mode: {effective_mode}")
        if context.shape != query.shape or context.ndim != 3:
            raise ValueError("context and query must have equal [N,T,D] shapes")

        main = context.mean(dim=1, keepdim=True)
        interaction = context - main
        if effective_mode == STANDARD_QUERY_MODE:
            if not hasattr(self, "standard_projection"):
                raise ValueError(
                    "standard-query composition requires the single branch"
                )
            evidence = self.standard_projection(context)
            interaction_used = False
        elif effective_mode == STANDARD_DUAL_MODE:
            evidence = 0.5 * (
                self.main_projection(context)
                + self.interaction_projection(context)
            )
            interaction_used = False
        elif effective_mode == GENERIC_DUAL_MODE:
            evidence = 0.5 * (
                self.main_projection(main)
                + self.interaction_projection(main)
            )
            evidence = evidence.expand_as(context)
            interaction_used = False
        else:
            evidence = self.main_projection(main)
            evidence = evidence + self.interaction_projection(interaction)
            interaction_used = True
        state = evidence + query
        details: dict[str, torch.Tensor | bool] = {
            "main": main,
            "interaction": interaction,
            "evidence": evidence,
            "state": state,
            "interaction_used": interaction_used,
        }
        return state, details

    def forward(
        self,
        memory: torch.Tensor,
        target_prefix: int | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        """Map memory [B,C,P,D] to a normalized trajectory [B,H,C]."""
        if memory.ndim != 4:
            raise ValueError("memory must have shape [B,C,P,D]")
        batch, channels, patch_count, dimension = memory.shape
        if patch_count != self.patch_count or dimension != self.memory_dim:
            raise ValueError(
                "unexpected memory shape: "
                f"{tuple(memory.shape)} expected P={self.patch_count}, "
                f"D={self.memory_dim}"
            )
        horizon = (
            self.prediction_length
            if target_prefix is None
            else int(target_prefix)
        )
        if horizon <= 0 or horizon > self.prediction_length:
            raise ValueError("target_prefix must lie in [1, prediction_length]")

        content = memory.reshape(
            batch * channels,
            patch_count,
            dimension,
        )
        if self.mode == ORDER_SHUFFLED_MODE:
            content = content.index_select(1, self.memory_permutation)
        positions = self.memory_positions.to(dtype=content.dtype).unsqueeze(0)
        attended_memory = content + positions

        target_positions = self.target_positions
        if self.mode == TARGET_SHUFFLED_MODE:
            target_positions = target_positions.index_select(
                0,
                self.target_permutation,
            )
        query = self.query_encoder(
            target_positions.to(dtype=content.dtype)
        ).unsqueeze(0)
        query = query.expand(batch * channels, -1, -1)
        context, attention = self.cross_attention(
            query,
            attended_memory,
            attended_memory,
            need_weights=return_details,
            average_attn_weights=True,
        )
        state, details = self.compose_context(context, query)
        output = self.output_projection(F.gelu(state)).squeeze(-1)
        output = output.reshape(
            batch,
            channels,
            self.prediction_length,
        ).permute(0, 2, 1)
        output = output[:, :horizon, :]
        if not return_details:
            return output

        details.update(
            {
                "memory_content": content,
                "attended_memory": attended_memory,
                "query": query,
                "context": context,
                "attention": attention,
                "output": output,
            }
        )
        return output, details
