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


class PatchEncoderTrajectoryBasisResidual(nn.Module):
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
        basis_count: int = 8,
        residual_gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if basis_count <= 0:
            raise ValueError("basis_count must be positive.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.basis_count = basis_count
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

        self.basis_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, basis_count),
        )
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, basis_count),
        )
        self.residual_gate_logits = nn.Parameter(torch.full((pred_len,), residual_gate_init))

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        final = self.coefficient_head[-1]
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

    def _future_position_features(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        steps = torch.arange(1, self.pred_len + 1, device=device, dtype=dtype)
        normalized = steps / float(self.pred_len)
        return torch.stack(
            [
                normalized,
                normalized.square(),
                torch.sin(2.0 * math.pi * normalized),
                torch.cos(2.0 * math.pi * normalized),
            ],
            dim=-1,
        )

    def _trajectory_residual(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = z.mean(dim=1)
        coefficients = self.coefficient_head(pooled)
        position_features = self._future_position_features(z.device, z.dtype)
        basis = self.basis_mlp(position_features)
        residual = coefficients @ basis.T
        gate = torch.sigmoid(self.residual_gate_logits).to(device=z.device, dtype=z.dtype)
        return residual * gate.unsqueeze(0), residual, gate, coefficients

    def forward(self, x: torch.Tensor, return_components: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        base = self.fixed_head(z)
        gated_residual, raw_residual, gate, coefficients = self._trajectory_residual(z)
        y_norm = base + gated_residual

        base_y = base.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        residual_y = gated_residual.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        raw_residual_y = raw_residual.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        y = y_norm.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        coefficients_y = coefficients.reshape(batch, channels, self.basis_count)
        gate_y = gate.reshape(1, self.pred_len, 1).expand(batch, self.pred_len, channels)

        if self.revin is not None:
            y = self.revin(y, "denorm")
            base_y = self.revin(base_y, "denorm")
            residual_y = residual_y * self.revin.std
            raw_residual_y = raw_residual_y * self.revin.std

        if return_components:
            return {
                "prediction": y,
                "base_prediction": base_y,
                "residual": residual_y,
                "raw_residual": raw_residual_y,
                "residual_norm": gated_residual.reshape(batch, channels, self.pred_len).permute(0, 2, 1),
                "raw_residual_norm": raw_residual.reshape(batch, channels, self.pred_len).permute(0, 2, 1),
                "residual_gate": gate_y,
                "coefficients": coefficients_y,
            }
        return y
