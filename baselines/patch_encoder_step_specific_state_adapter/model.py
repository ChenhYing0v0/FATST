from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RevIN(nn.Module):
    def __init__(self, channels: int, affine: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, channels))
            self.bias = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.weight + self.bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            return x * self.std + self.mean
        raise ValueError(f"Unknown RevIN mode: {mode}")


class SegmentAdapterBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        memory_norm = self.memory_norm(memory)
        attn_out, _ = self.cross_attn(
            self.query_norm(query),
            memory_norm,
            memory_norm,
            need_weights=False,
        )
        query = query + self.attn_dropout(attn_out)
        return query + self.ffn(self.ffn_norm(query))


class PatchEncoderStepSpecificStateAdapter(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.0,
        revin: bool = True,
        segment_len: int = 48,
        adapter_layers: int = 1,
        adapter_heads: int = 8,
        adapter_d_ff: int = 256,
    ) -> None:
        super().__init__()
        if segment_len <= 0:
            raise ValueError("segment_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.segment_len = segment_len
        self.num_segments = math.ceil(pred_len / segment_len)
        self.revin = RevIN(channels, affine=False) if revin else None

        self.padding_patch = nn.ReplicationPad1d((0, stride))
        self.patch_embedding = nn.Linear(patch_len, d_model)
        n_patches = int((seq_len + stride - patch_len) / stride + 1)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(n_patches * d_model, pred_len)

        self.segment_queries = nn.Parameter(torch.zeros(1, self.num_segments, d_model))
        self.adapter = nn.ModuleList(
            [
                SegmentAdapterBlock(
                    d_model=d_model,
                    n_heads=adapter_heads,
                    d_ff=adapter_d_ff,
                    dropout=dropout,
                )
                for _ in range(adapter_layers)
            ]
        )
        self.modulation_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.segment_queries, std=0.02)
        final = self.modulation_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, length, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch * channels, 1, length)
        x = self.padding_patch(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        z = self.patch_embedding(patches) + self.pos_embedding
        return self.encoder(z), batch, channels

    def _segment_states(self, z: torch.Tensor) -> torch.Tensor:
        query = self.segment_queries.expand(z.shape[0], -1, -1)
        for block in self.adapter:
            query = block(query, z)
        return query

    def _fixed_head(self, z: torch.Tensor) -> torch.Tensor:
        return self.head_linear(torch.flatten(self.head_dropout(z), start_dim=1))

    def _state_adapter(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        segment_states = self._segment_states(z)
        affine = self.modulation_head(segment_states).reshape(z.shape[0], self.num_segments, 2, -1)
        gamma = affine[:, :, 0, :]
        beta = affine[:, :, 1, :]
        return segment_states, gamma, beta

    def _segment_readout(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        z_dropped = self.head_dropout(z)
        outputs = []
        for segment_index in range(self.num_segments):
            start = segment_index * self.segment_len
            end = min(start + self.segment_len, self.pred_len)
            z_mod = z_dropped * (1.0 + gamma[:, segment_index, None, :]) + beta[:, segment_index, None, :]
            flat = torch.flatten(z_mod, start_dim=1)
            outputs.append(
                F.linear(
                    flat,
                    self.head_linear.weight[start:end],
                    self.head_linear.bias[start:end],
                )
            )
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor, return_components: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        base = self._fixed_head(z)
        segment_states, gamma, beta = self._state_adapter(z)
        y_norm = self._segment_readout(z, gamma, beta)

        base_y = base.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        y = y_norm.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        gamma_state = gamma.reshape(batch, channels, self.num_segments, -1)
        beta_state = beta.reshape(batch, channels, self.num_segments, -1)
        segment_states = segment_states.reshape(batch, channels, self.num_segments, -1)

        if self.revin is not None:
            y = self.revin(y, "denorm")
            base_y = self.revin(base_y, "denorm")

        if return_components:
            return {
                "prediction": y,
                "base_prediction": base_y,
                "gamma": gamma_state,
                "beta": beta_state,
                "segment_states": segment_states,
            }
        return y
