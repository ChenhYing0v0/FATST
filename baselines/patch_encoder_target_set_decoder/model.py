from __future__ import annotations

import math

import torch
from torch import nn


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


class TargetDecoderBlock(nn.Module):
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


class PatchEncoderTargetSetDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
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
        max_pred_len: int = 720,
        target_layers: int = 1,
        target_heads: int = 8,
        target_d_ff: int = 256,
        readout_dim: int = 256,
    ) -> None:
        super().__init__()
        if segment_len <= 0:
            raise ValueError("segment_len must be positive.")
        if max_pred_len <= 0:
            raise ValueError("max_pred_len must be positive.")

        self.seq_len = seq_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.segment_len = segment_len
        self.max_pred_len = max_pred_len
        self.max_segments = math.ceil(max_pred_len / segment_len)
        self.readout_dim = readout_dim
        self.revin = RevIN(channels, affine=False) if revin else None

        self.padding_patch = nn.ReplicationPad1d((0, stride))
        self.patch_embedding = nn.Linear(patch_len, d_model)
        n_patches = int((seq_len + stride - patch_len) / stride + 1)
        self.n_patches = n_patches
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

        self.target_feature_embedding = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.target_pos_embedding = nn.Parameter(torch.zeros(1, self.max_segments, d_model))
        self.target_decoder = nn.ModuleList(
            [
                TargetDecoderBlock(
                    d_model=d_model,
                    n_heads=target_heads,
                    d_ff=target_d_ff,
                    dropout=dropout,
                )
                for _ in range(target_layers)
            ]
        )

        self.history_projector = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(n_patches * d_model, readout_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.condition_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * readout_dim),
        )
        self.segment_output = nn.Linear(readout_dim, segment_len)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.target_pos_embedding, std=0.02)

    def _validate_pred_len(self, pred_len: int) -> int:
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if pred_len > self.max_pred_len:
            raise ValueError(f"pred_len={pred_len} exceeds max_pred_len={self.max_pred_len}.")
        return math.ceil(pred_len / self.segment_len)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, length, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch * channels, 1, length)
        x = self.padding_patch(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        z = self.patch_embedding(patches) + self.pos_embedding
        return self.encoder(z), batch, channels

    def _target_features(self, pred_len: int, segment_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        index = torch.arange(segment_count, device=device, dtype=dtype)
        start = index * self.segment_len + 1.0
        end = torch.clamp((index + 1.0) * self.segment_len, max=float(pred_len))
        width = end - start + 1.0
        center = (start + end) * 0.5
        denom = float(self.max_pred_len)
        reserved = torch.zeros_like(center)
        segment_denom = max(float(self.max_segments - 1), 1.0)
        segment_index = index / segment_denom
        phase = 2.0 * math.pi * center / denom
        return torch.stack(
            [
                start / denom,
                end / denom,
                center / denom,
                width / denom,
                reserved,
                segment_index,
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=-1,
        )

    def _target_states(self, z: torch.Tensor, pred_len: int, segment_count: int) -> torch.Tensor:
        features = self._target_features(pred_len, segment_count, z.device, z.dtype)
        query = self.target_feature_embedding(features).unsqueeze(0)
        query = query + self.target_pos_embedding[:, :segment_count, :]
        query = query.expand(z.shape[0], -1, -1)
        for block in self.target_decoder:
            query = block(query, z)
        return query

    def forward(
        self,
        x: torch.Tensor,
        pred_len: int,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        segment_count = self._validate_pred_len(pred_len)
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        target_states = self._target_states(z, pred_len, segment_count)
        history_readout = self.history_projector(z)
        affine = self.condition_head(target_states).reshape(z.shape[0], segment_count, 2, self.readout_dim)
        gamma = affine[:, :, 0, :]
        beta = affine[:, :, 1, :]

        conditioned = history_readout[:, None, :] * (1.0 + gamma) + beta
        segment_values = self.segment_output(conditioned).reshape(z.shape[0], segment_count * self.segment_len)
        y_norm = segment_values[:, :pred_len]
        y = y_norm.reshape(batch, channels, pred_len).permute(0, 2, 1)

        target_states_view = target_states.reshape(batch, channels, segment_count, -1)
        gamma_view = gamma.reshape(batch, channels, segment_count, -1)
        beta_view = beta.reshape(batch, channels, segment_count, -1)
        history_readout_view = history_readout.reshape(batch, channels, -1)

        if self.revin is not None:
            y = self.revin(y, "denorm")

        if return_components:
            return {
                "prediction": y,
                "target_states": target_states_view,
                "gamma": gamma_view,
                "beta": beta_view,
                "history_readout": history_readout_view,
            }
        return y
