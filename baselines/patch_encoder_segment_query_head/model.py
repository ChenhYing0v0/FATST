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


class SegmentDecoderBlock(nn.Module):
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
        attn_out, _ = self.cross_attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + self.attn_dropout(attn_out)
        return query + self.ffn(self.ffn_norm(query))


class PatchEncoderSegmentQueryHead(nn.Module):
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
        decoder_layers: int = 1,
        decoder_heads: int = 8,
        decoder_d_ff: int = 256,
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
        self.segment_queries = nn.Parameter(torch.zeros(1, self.num_segments, d_model))
        self.decoder = nn.ModuleList(
            [
                SegmentDecoderBlock(
                    d_model=d_model,
                    n_heads=decoder_heads,
                    d_ff=decoder_d_ff,
                    dropout=dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.segment_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, segment_len),
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.segment_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")

        batch, length, channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch * channels, 1, length)
        x = self.padding_patch(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        z = self.patch_embedding(patches) + self.pos_embedding
        z = self.encoder(z)

        query = self.segment_queries.expand(batch * channels, -1, -1)
        for block in self.decoder:
            query = block(query, z)
        y = self.segment_head(query).reshape(batch * channels, self.num_segments * self.segment_len)
        y = y[:, : self.pred_len].reshape(batch, channels, self.pred_len).permute(0, 2, 1)

        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
