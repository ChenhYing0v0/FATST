from __future__ import annotations

import torch
from torch import nn


class NeutralSharingExtentForecaster(nn.Module):
    """Single-scale diagnostic forecaster with matched parameters across scales."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        sharing_extent: int,
        history_dim: int = 64,
        step_dim: int = 32,
        hidden_dim: int = 128,
        state_dim: int = 64,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if sharing_extent <= 0 or sharing_extent > pred_len:
            raise ValueError("sharing_extent must be in [1, pred_len]")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sharing_extent = sharing_extent
        self.history_dim = history_dim
        self.step_dim = step_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.history_encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, history_dim),
        )
        self.step_embedding = nn.Parameter(torch.empty(pred_len, step_dim))
        self.history_to_hidden = nn.Linear(history_dim, hidden_dim, bias=False)
        self.step_to_hidden = nn.Linear(step_dim, hidden_dim, bias=True)
        self.hidden_to_state = nn.Linear(hidden_dim, state_dim)
        self.pooled_state_norm = nn.LayerNorm(state_dim)
        self.synthesis = nn.Parameter(torch.empty(pred_len, state_dim))
        self.output_bias = nn.Parameter(torch.zeros(pred_len))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.synthesis)

    def candidate_states(self, x: torch.Tensor) -> torch.Tensor:
        """Return U with shape [B, C, T, D_z]."""
        if x.ndim != 3:
            raise ValueError(f"expected x [B,L,C], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"expected seq_len={self.seq_len}, got {x.shape[1]}"
            )
        history = self.history_encoder(x.transpose(1, 2))
        history_hidden = self.history_to_hidden(history).unsqueeze(2)
        step_hidden = self.step_to_hidden(self.step_embedding)
        joint_hidden = torch.nn.functional.gelu(
            history_hidden + step_hidden.unsqueeze(0).unsqueeze(0)
        )
        return self.hidden_to_state(joint_hidden)

    def pooled_states(self, candidate_states: torch.Tensor) -> torch.Tensor:
        """Pool U within fixed future blocks and broadcast back to [B,C,T,D_z]."""
        if candidate_states.ndim != 4:
            raise ValueError(
                "candidate_states must have shape [B,C,T,D_z], got "
                f"{tuple(candidate_states.shape)}"
            )
        if candidate_states.shape[2] != self.pred_len:
            raise ValueError(
                f"expected pred_len={self.pred_len}, got {candidate_states.shape[2]}"
            )

        pooled = torch.empty_like(candidate_states)
        for start in range(0, self.pred_len, self.sharing_extent):
            end = min(start + self.sharing_extent, self.pred_len)
            state = candidate_states[:, :, start:end, :].mean(dim=2)
            state = self.pooled_state_norm(state)
            pooled[:, :, start:end, :] = state.unsqueeze(2)
        return pooled

    def forward_with_states(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = self.candidate_states(x)
        pooled = self.pooled_states(candidate)
        prediction = (
            pooled * self.synthesis.unsqueeze(0).unsqueeze(0)
        ).sum(dim=-1)
        prediction = prediction + self.output_bias.view(1, 1, -1)
        return prediction.transpose(1, 2), candidate, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _, _ = self.forward_with_states(x)
        return prediction
