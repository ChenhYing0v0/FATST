from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


class PatchEncoderFutureAwareAdapter(nn.Module):
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
        teacher_layers: int = 1,
        teacher_heads: int = 8,
        teacher_d_ff: int = 256,
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
        self.fixed_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(n_patches * d_model, pred_len),
        )

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
        self.adapter_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * segment_len),
        )

        self.teacher_segment_embedding = nn.Linear(segment_len, d_model)
        teacher_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=teacher_heads,
            dim_feedforward=teacher_d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.teacher_encoder = nn.TransformerEncoder(teacher_layer, num_layers=teacher_layers)
        self.student_align = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.teacher_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, segment_len),
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.segment_queries, std=0.02)
        final = self.adapter_head[-1]
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

    def _segment_adapter(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.segment_queries.expand(z.shape[0], -1, -1)
        for block in self.adapter:
            query = block(query, z)
        affine = self.adapter_head(query).reshape(z.shape[0], self.num_segments, 2, self.segment_len)
        gamma = affine[:, :, 0, :].reshape(z.shape[0], self.num_segments * self.segment_len)
        beta = affine[:, :, 1, :].reshape(z.shape[0], self.num_segments * self.segment_len)
        return query, gamma[:, : self.pred_len], beta[:, : self.pred_len]

    def _normalize_future(self, y: torch.Tensor) -> torch.Tensor:
        if self.revin is None:
            return y
        return (y - self.revin.mean) / (self.revin.std + self.revin.eps)

    def _teacher_state(self, y_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, channels = y_norm.shape
        y_flat = y_norm.permute(0, 2, 1).reshape(batch * channels, horizon)
        pad_len = self.num_segments * self.segment_len - horizon
        if pad_len > 0:
            y_flat = F.pad(y_flat, (0, pad_len))
        segments = y_flat.reshape(batch * channels, self.num_segments, self.segment_len)
        teacher_state = self.teacher_segment_embedding(segments) + self.segment_queries
        teacher_state = self.teacher_encoder(teacher_state)
        recon = self.teacher_reconstruction_head(teacher_state).reshape(
            batch * channels,
            self.num_segments * self.segment_len,
        )
        return teacher_state, recon[:, : self.pred_len]

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        base = self.fixed_head(z)
        student_state, gamma, beta = self._segment_adapter(z)
        y_norm = base * (1.0 + gamma) + beta

        base_y = base.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        y_pred = y_norm.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        gamma_y = gamma.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        beta_y = beta.reshape(batch, channels, self.pred_len).permute(0, 2, 1)

        output: dict[str, torch.Tensor] = {
            "prediction_norm": y_pred,
            "base_prediction_norm": base_y,
            "gamma": gamma_y,
            "beta_norm": beta_y,
            "student_state": student_state,
        }

        if y is not None:
            y_target_norm = self._normalize_future(y)
            teacher_state, recon_norm = self._teacher_state(y_target_norm)
            student_aligned = self.student_align(student_state)
            align_loss = 1.0 - F.cosine_similarity(
                F.normalize(student_aligned, dim=-1),
                F.normalize(teacher_state.detach(), dim=-1),
                dim=-1,
            ).mean()
            target_flat = y_target_norm.permute(0, 2, 1).reshape(batch * channels, self.pred_len)
            recon_loss = F.mse_loss(recon_norm, target_flat)
            output.update(
                {
                    "teacher_state": teacher_state,
                    "student_aligned_state": student_aligned,
                    "teacher_reconstruction_norm": recon_norm.reshape(batch, channels, self.pred_len).permute(0, 2, 1),
                    "alignment_loss": align_loss,
                    "reconstruction_loss": recon_loss,
                }
            )

        if self.revin is not None:
            y_pred = self.revin(y_pred, "denorm")
            base_y = self.revin(base_y, "denorm")
            beta_y = beta_y * self.revin.std

        output.update(
            {
                "prediction": y_pred,
                "base_prediction": base_y,
                "beta": beta_y,
            }
        )
        if return_components:
            return output
        return y_pred
