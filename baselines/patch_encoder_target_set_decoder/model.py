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


class CausalTargetInteractionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, target_states: torch.Tensor) -> torch.Tensor:
        segment_count = target_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(segment_count, segment_count, device=target_states.device, dtype=torch.bool),
            diagonal=1,
        )
        target_norm = self.attn_norm(target_states)
        attn_out, _ = self.self_attn(
            target_norm,
            target_norm,
            target_norm,
            attn_mask=causal_mask,
            need_weights=False,
        )
        target_states = target_states + self.attn_dropout(attn_out)
        return target_states + self.ffn(self.ffn_norm(target_states))


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
        target_interaction_layers: int = 0,
        target_interaction_heads: int | None = None,
        target_interaction_d_ff: int | None = None,
        readout_dim: int = 256,
        prefix_residual_segments: int = 0,
        prefix_residual_dropout: float = 0.0,
        future_teacher_layers: int = 0,
        future_teacher_heads: int | None = None,
        future_teacher_d_ff: int | None = None,
        future_state_dim: int | None = None,
        future_recon_normalization: str = "none",
        future_align_weighting: str = "uniform",
        future_confidence_temperature: float = 1.0,
        future_confidence_floor: float = 0.0,
        future_recon_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if segment_len <= 0:
            raise ValueError("segment_len must be positive.")
        if max_pred_len <= 0:
            raise ValueError("max_pred_len must be positive.")
        if future_recon_normalization not in {"none", "target_energy"}:
            raise ValueError("future_recon_normalization must be one of: none, target_energy.")
        if future_align_weighting not in {"uniform", "reconstruction_confidence"}:
            raise ValueError("future_align_weighting must be one of: uniform, reconstruction_confidence.")
        if future_confidence_temperature <= 0:
            raise ValueError("future_confidence_temperature must be positive.")
        if not 0 <= future_confidence_floor < 1:
            raise ValueError("future_confidence_floor must be in [0, 1).")
        if future_recon_eps <= 0:
            raise ValueError("future_recon_eps must be positive.")

        self.seq_len = seq_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.segment_len = segment_len
        self.max_pred_len = max_pred_len
        self.max_segments = math.ceil(max_pred_len / segment_len)
        self.readout_dim = readout_dim
        if prefix_residual_segments < 0:
            raise ValueError("prefix_residual_segments must be non-negative.")
        self.prefix_residual_segments = min(prefix_residual_segments, self.max_segments)
        if target_interaction_layers < 0:
            raise ValueError("target_interaction_layers must be non-negative.")
        if target_interaction_heads is None:
            target_interaction_heads = target_heads
        if target_interaction_d_ff is None:
            target_interaction_d_ff = target_d_ff
        if future_teacher_layers < 0:
            raise ValueError("future_teacher_layers must be non-negative.")
        if future_teacher_heads is None:
            future_teacher_heads = target_heads
        if future_teacher_d_ff is None:
            future_teacher_d_ff = target_d_ff
        if future_state_dim is None:
            future_state_dim = d_model
        self.future_recon_normalization = future_recon_normalization
        self.future_align_weighting = future_align_weighting
        self.future_confidence_temperature = future_confidence_temperature
        self.future_confidence_floor = future_confidence_floor
        self.future_recon_eps = future_recon_eps
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
        self.target_interaction = nn.ModuleList(
            [
                CausalTargetInteractionBlock(
                    d_model=d_model,
                    n_heads=target_interaction_heads,
                    d_ff=target_interaction_d_ff,
                    dropout=dropout,
                )
                for _ in range(target_interaction_layers)
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
        self.future_teacher_layers = future_teacher_layers
        self.future_segment_embedding: nn.Linear | None = None
        self.future_feature_embedding: nn.Sequential | None = None
        self.future_pos_embedding: nn.Parameter | None = None
        self.future_teacher_encoder: nn.TransformerEncoder | None = None
        self.future_student_projection: nn.Sequential | None = None
        self.future_teacher_projection: nn.Sequential | None = None
        self.future_reconstruction_head: nn.Sequential | None = None
        if future_teacher_layers > 0:
            self.future_segment_embedding = nn.Linear(segment_len, d_model)
            self.future_feature_embedding = nn.Sequential(
                nn.Linear(8, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.future_pos_embedding = nn.Parameter(torch.zeros(1, self.max_segments, d_model))
            future_teacher_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=future_teacher_heads,
                dim_feedforward=future_teacher_d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.future_teacher_encoder = nn.TransformerEncoder(
                future_teacher_layer,
                num_layers=future_teacher_layers,
            )
            self.future_student_projection = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, future_state_dim),
            )
            self.future_teacher_projection = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, future_state_dim),
            )
            self.future_reconstruction_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, segment_len),
            )
        self.prefix_residual_head: nn.Sequential | None = None
        if self.prefix_residual_segments > 0:
            self.prefix_residual_head = nn.Sequential(
                nn.Dropout(prefix_residual_dropout),
                nn.Flatten(start_dim=1),
                nn.Linear(n_patches * d_model, self.prefix_residual_segments * segment_len),
            )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.target_pos_embedding, std=0.02)
        if self.future_pos_embedding is not None:
            nn.init.trunc_normal_(self.future_pos_embedding, std=0.02)
        if self.prefix_residual_head is not None:
            final = self.prefix_residual_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

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
        for block in self.target_interaction:
            query = block(query)
        return query

    def _normalize_future(self, y: torch.Tensor) -> torch.Tensor:
        if self.revin is None:
            return y
        return (y - self.revin.mean) / (self.revin.std + self.revin.eps)

    def _future_teacher_state(
        self,
        y_norm: torch.Tensor,
        pred_len: int,
        segment_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.future_segment_embedding is None
            or self.future_feature_embedding is None
            or self.future_pos_embedding is None
            or self.future_teacher_encoder is None
            or self.future_teacher_projection is None
            or self.future_reconstruction_head is None
        ):
            raise RuntimeError("future teacher branch is disabled.")

        batch, horizon, channels = y_norm.shape
        y_flat = y_norm.permute(0, 2, 1).reshape(batch * channels, horizon)
        pad_len = segment_count * self.segment_len - horizon
        if pad_len > 0:
            y_flat = F.pad(y_flat, (0, pad_len))
        segments = y_flat.reshape(batch * channels, segment_count, self.segment_len)
        features = self._target_features(pred_len, segment_count, y_norm.device, y_norm.dtype)
        teacher_state = self.future_segment_embedding(segments)
        teacher_state = teacher_state + self.future_feature_embedding(features).unsqueeze(0)
        teacher_state = teacher_state + self.future_pos_embedding[:, :segment_count, :]
        teacher_state = self.future_teacher_encoder(teacher_state)
        teacher_projected = self.future_teacher_projection(teacher_state)
        reconstruction = self.future_reconstruction_head(teacher_state).reshape(
            batch * channels,
            segment_count * self.segment_len,
        )
        return teacher_projected, reconstruction[:, :pred_len]

    def _future_alignment_components(
        self,
        target_states: torch.Tensor,
        y_norm: torch.Tensor,
        pred_len: int,
        segment_count: int,
    ) -> dict[str, torch.Tensor]:
        if self.future_student_projection is None:
            raise RuntimeError("future teacher branch is disabled.")

        student_state = self.future_student_projection(target_states)
        teacher_state, reconstruction = self._future_teacher_state(y_norm, pred_len, segment_count)
        student_norm = F.normalize(student_state, dim=-1)
        teacher_norm = F.normalize(teacher_state.detach(), dim=-1)
        batch, horizon, channels = y_norm.shape
        target_flat = y_norm.permute(0, 2, 1).reshape(batch * channels, horizon)
        pad_len = segment_count * self.segment_len - horizon
        reconstruction_padded = F.pad(reconstruction, (0, pad_len)) if pad_len > 0 else reconstruction
        target_padded = F.pad(target_flat, (0, pad_len)) if pad_len > 0 else target_flat
        valid_mask = target_padded.new_zeros(1, segment_count * self.segment_len)
        valid_mask[:, :horizon] = 1.0
        valid_mask = valid_mask.reshape(1, segment_count, self.segment_len)
        reconstruction_segments = reconstruction_padded.reshape(-1, segment_count, self.segment_len)
        target_segments = target_padded.reshape(-1, segment_count, self.segment_len)
        squared_error = (reconstruction_segments - target_segments).pow(2) * valid_mask
        valid_counts = valid_mask.sum(dim=-1).clamp_min(1.0)
        segment_mse = squared_error.sum(dim=-1) / valid_counts
        segment_energy = (target_segments.detach().pow(2) * valid_mask).sum(dim=-1) / valid_counts
        normalized_segment_mse = segment_mse / segment_energy.clamp_min(self.future_recon_eps)
        raw_reconstruction_loss = F.mse_loss(reconstruction, target_flat)
        target_energy = target_flat.detach().pow(2).mean().clamp_min(self.future_recon_eps)
        if self.future_recon_normalization == "target_energy":
            reconstruction_loss = raw_reconstruction_loss / target_energy
        else:
            reconstruction_loss = raw_reconstruction_loss

        alignment_confidence = torch.ones_like(segment_mse)
        if self.future_align_weighting == "reconstruction_confidence":
            alignment_confidence = torch.exp(
                -normalized_segment_mse.detach() / self.future_confidence_temperature
            )
            if self.future_confidence_floor > 0:
                alignment_confidence = alignment_confidence.clamp_min(self.future_confidence_floor)

        local_alignment = 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        local_alignment_loss = (
            local_alignment * alignment_confidence
        ).sum() / alignment_confidence.sum().clamp_min(self.future_recon_eps)
        student_relation = torch.bmm(student_norm, student_norm.transpose(1, 2))
        teacher_relation = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))
        relation_error = (student_relation - teacher_relation).pow(2)
        if self.future_align_weighting == "reconstruction_confidence":
            relation_weight = torch.sqrt(
                alignment_confidence[:, :, None] * alignment_confidence[:, None, :]
            )
            relation_alignment_loss = (
                relation_error * relation_weight
            ).sum() / relation_weight.sum().clamp_min(self.future_recon_eps)
        else:
            relation_alignment_loss = relation_error.mean()
        return {
            "future_student_state": student_state,
            "future_teacher_state": teacher_state,
            "future_reconstruction_norm": reconstruction,
            "future_local_alignment_loss": local_alignment_loss,
            "future_relation_alignment_loss": relation_alignment_loss,
            "future_reconstruction_loss": reconstruction_loss,
            "future_raw_reconstruction_loss": raw_reconstruction_loss,
            "future_normalized_reconstruction_loss": raw_reconstruction_loss / target_energy,
            "future_alignment_confidence_mean": alignment_confidence.mean(),
            "future_alignment_confidence_min": alignment_confidence.min(),
            "future_alignment_confidence_max": alignment_confidence.max(),
        }

    def forward(
        self,
        x: torch.Tensor,
        pred_len: int,
        future_y: torch.Tensor | None = None,
        return_components: bool = False,
        window_index_norm: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del window_index_norm
        segment_count = self._validate_pred_len(pred_len)
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        target_states = self._target_states(z, pred_len, segment_count)
        future_components = None
        if future_y is not None:
            y_norm_target = self._normalize_future(future_y)
            future_components = self._future_alignment_components(
                target_states,
                y_norm_target,
                pred_len,
                segment_count,
            )
        history_readout = self.history_projector(z)
        affine = self.condition_head(target_states).reshape(z.shape[0], segment_count, 2, self.readout_dim)
        gamma = affine[:, :, 0, :]
        beta = affine[:, :, 1, :]

        conditioned = history_readout[:, None, :] * (1.0 + gamma) + beta
        segment_values = self.segment_output(conditioned).reshape(z.shape[0], segment_count * self.segment_len)
        prefix_residual = segment_values.new_zeros(z.shape[0], segment_count * self.segment_len)
        if self.prefix_residual_head is not None:
            active_prefix_segments = min(segment_count, self.prefix_residual_segments)
            active_width = active_prefix_segments * self.segment_len
            residual_values = self.prefix_residual_head(z)[:, :active_width]
            prefix_residual[:, :active_width] = residual_values
            segment_values = segment_values + prefix_residual
        y_norm = segment_values[:, :pred_len]
        y = y_norm.reshape(batch, channels, pred_len).permute(0, 2, 1)
        prefix_residual_y = prefix_residual[:, :pred_len].reshape(batch, channels, pred_len).permute(0, 2, 1)

        target_states_view = target_states.reshape(batch, channels, segment_count, -1)
        gamma_view = gamma.reshape(batch, channels, segment_count, -1)
        beta_view = beta.reshape(batch, channels, segment_count, -1)
        history_readout_view = history_readout.reshape(batch, channels, -1)

        if self.revin is not None:
            y = self.revin(y, "denorm")

        if return_components:
            output = {
                "prediction": y,
                "target_states": target_states_view,
                "gamma": gamma_view,
                "beta": beta_view,
                "history_readout": history_readout_view,
                "prefix_residual_norm": prefix_residual_y,
            }
            if future_components is not None:
                output.update(
                    {
                        "future_student_state": future_components["future_student_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_teacher_state": future_components["future_teacher_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_reconstruction_norm": future_components["future_reconstruction_norm"]
                        .reshape(batch, channels, pred_len)
                        .permute(0, 2, 1),
                        "future_local_alignment_loss": future_components["future_local_alignment_loss"],
                        "future_relation_alignment_loss": future_components["future_relation_alignment_loss"],
                        "future_reconstruction_loss": future_components["future_reconstruction_loss"],
                        "future_raw_reconstruction_loss": future_components["future_raw_reconstruction_loss"],
                        "future_normalized_reconstruction_loss": future_components[
                            "future_normalized_reconstruction_loss"
                        ],
                        "future_alignment_confidence_mean": future_components[
                            "future_alignment_confidence_mean"
                        ],
                        "future_alignment_confidence_min": future_components[
                            "future_alignment_confidence_min"
                        ],
                        "future_alignment_confidence_max": future_components[
                            "future_alignment_confidence_max"
                        ],
                    }
                )
            return output
        return y


class PatchEncoderRegimeSegmentTargetOperator(PatchEncoderTargetSetDecoder):
    def __init__(
        self,
        *args,
        regime_hidden_dim: int = 64,
        regime_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if regime_hidden_dim <= 0:
            raise ValueError("regime_hidden_dim must be positive.")
        if regime_dropout < 0:
            raise ValueError("regime_dropout must be non-negative.")
        self.regime_feature_dim = 10
        self.regime_encoder = nn.Sequential(
            nn.LayerNorm(self.regime_feature_dim),
            nn.Linear(self.regime_feature_dim, regime_hidden_dim),
            nn.GELU(),
            nn.Dropout(regime_dropout),
            nn.Linear(regime_hidden_dim, self.patch_embedding.out_features),
            nn.GELU(),
        )
        self.regime_segment_operator = nn.Sequential(
            nn.LayerNorm(2 * self.patch_embedding.out_features),
            nn.Linear(2 * self.patch_embedding.out_features, regime_hidden_dim),
            nn.GELU(),
            nn.Dropout(regime_dropout),
            nn.Linear(regime_hidden_dim, 2 * self.readout_dim),
        )
        final = self.regime_segment_operator[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _history_regime_features(
        self,
        x_norm: torch.Tensor,
        batch: int,
        channels: int,
        window_index_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        x_channel = x_norm.permute(0, 2, 1).reshape(batch * channels, x_norm.shape[1])
        first_half = x_channel[:, : x_channel.shape[1] // 2]
        second_half = x_channel[:, x_channel.shape[1] // 2 :]
        recent = x_channel[:, -48:]
        previous = x_channel[:, -96:-48]
        time = torch.linspace(-1.0, 1.0, x_channel.shape[1], device=x_norm.device, dtype=x_norm.dtype)
        centered_time = time - time.mean()
        denom = torch.sum(centered_time * centered_time).clamp_min(1e-12)
        centered_x = x_channel - x_channel.mean(dim=1, keepdim=True)
        slope = torch.sum(centered_x * centered_time.unsqueeze(0), dim=1) / denom
        if window_index_norm is None:
            position = torch.zeros(batch, device=x_norm.device, dtype=x_norm.dtype)
        else:
            position = window_index_norm.to(device=x_norm.device, dtype=x_norm.dtype).reshape(batch)
        position = position[:, None].expand(batch, channels).reshape(batch * channels)
        return torch.stack(
            [
                torch.mean(x_channel, dim=1),
                torch.std(x_channel, dim=1, unbiased=False),
                torch.mean(torch.abs(x_channel), dim=1),
                torch.abs(x_channel[:, -1]),
                torch.mean(recent, dim=1),
                torch.std(recent, dim=1, unbiased=False),
                torch.mean(recent, dim=1) - torch.mean(previous, dim=1),
                torch.mean(second_half, dim=1) - torch.mean(first_half, dim=1),
                torch.abs(slope),
                position,
            ],
            dim=-1,
        )

    def _apply_regime_segment_operator(
        self,
        conditioned: torch.Tensor,
        target_states: torch.Tensor,
        pred_len: int,
        segment_count: int,
        regime_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        regime_token = self.regime_encoder(regime_features)
        target_features = self._target_features(
            pred_len,
            segment_count,
            target_states.device,
            target_states.dtype,
        )
        target_token = self.target_feature_embedding(target_features).unsqueeze(0)
        target_token = target_token + self.target_pos_embedding[:, :segment_count, :]
        target_token = target_token.expand(target_states.shape[0], -1, -1)
        regime_token = regime_token[:, None, :].expand(-1, segment_count, -1)
        operator_input = torch.cat([regime_token, target_token], dim=-1)
        affine = self.regime_segment_operator(operator_input).reshape(
            target_states.shape[0],
            segment_count,
            2,
            self.readout_dim,
        )
        scale = 0.1 * torch.tanh(affine[:, :, 0, :])
        shift = 0.1 * torch.tanh(affine[:, :, 1, :])
        return conditioned * (1.0 + scale) + shift, scale, shift, regime_features

    def forward(
        self,
        x: torch.Tensor,
        pred_len: int,
        future_y: torch.Tensor | None = None,
        return_components: bool = False,
        window_index_norm: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        segment_count = self._validate_pred_len(pred_len)
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        target_states = self._target_states(z, pred_len, segment_count)
        future_components = None
        if future_y is not None:
            y_norm_target = self._normalize_future(future_y)
            future_components = self._future_alignment_components(
                target_states,
                y_norm_target,
                pred_len,
                segment_count,
            )
        history_readout = self.history_projector(z)
        affine = self.condition_head(target_states).reshape(z.shape[0], segment_count, 2, self.readout_dim)
        gamma = affine[:, :, 0, :]
        beta = affine[:, :, 1, :]

        conditioned = history_readout[:, None, :] * (1.0 + gamma) + beta
        regime_features = self._history_regime_features(x, batch, channels, window_index_norm)
        conditioned, regime_scale, regime_shift, regime_features = self._apply_regime_segment_operator(
            conditioned,
            target_states,
            pred_len,
            segment_count,
            regime_features,
        )
        segment_values = self.segment_output(conditioned).reshape(z.shape[0], segment_count * self.segment_len)
        prefix_residual = segment_values.new_zeros(z.shape[0], segment_count * self.segment_len)
        if self.prefix_residual_head is not None:
            active_prefix_segments = min(segment_count, self.prefix_residual_segments)
            active_width = active_prefix_segments * self.segment_len
            residual_values = self.prefix_residual_head(z)[:, :active_width]
            prefix_residual[:, :active_width] = residual_values
            segment_values = segment_values + prefix_residual
        y_norm = segment_values[:, :pred_len]
        y = y_norm.reshape(batch, channels, pred_len).permute(0, 2, 1)
        prefix_residual_y = prefix_residual[:, :pred_len].reshape(batch, channels, pred_len).permute(0, 2, 1)

        target_states_view = target_states.reshape(batch, channels, segment_count, -1)
        gamma_view = gamma.reshape(batch, channels, segment_count, -1)
        beta_view = beta.reshape(batch, channels, segment_count, -1)
        history_readout_view = history_readout.reshape(batch, channels, -1)
        regime_scale_view = regime_scale.reshape(batch, channels, segment_count, -1)
        regime_shift_view = regime_shift.reshape(batch, channels, segment_count, -1)
        regime_features_view = regime_features.reshape(batch, channels, -1)

        if self.revin is not None:
            y = self.revin(y, "denorm")

        if return_components:
            output = {
                "prediction": y,
                "target_states": target_states_view,
                "gamma": gamma_view,
                "beta": beta_view,
                "history_readout": history_readout_view,
                "prefix_residual_norm": prefix_residual_y,
                "regime_operator_scale": regime_scale_view,
                "regime_operator_shift": regime_shift_view,
                "regime_features": regime_features_view,
            }
            if future_components is not None:
                output.update(
                    {
                        "future_student_state": future_components["future_student_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_teacher_state": future_components["future_teacher_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_reconstruction_norm": future_components["future_reconstruction_norm"]
                        .reshape(batch, channels, pred_len)
                        .permute(0, 2, 1),
                        "future_local_alignment_loss": future_components["future_local_alignment_loss"],
                        "future_relation_alignment_loss": future_components["future_relation_alignment_loss"],
                        "future_reconstruction_loss": future_components["future_reconstruction_loss"],
                        "future_raw_reconstruction_loss": future_components["future_raw_reconstruction_loss"],
                        "future_normalized_reconstruction_loss": future_components[
                            "future_normalized_reconstruction_loss"
                        ],
                        "future_alignment_confidence_mean": future_components[
                            "future_alignment_confidence_mean"
                        ],
                        "future_alignment_confidence_min": future_components[
                            "future_alignment_confidence_min"
                        ],
                        "future_alignment_confidence_max": future_components[
                            "future_alignment_confidence_max"
                        ],
                    }
                )
            return output
        return y


class PatchEncoderErrorProcessDecoder(PatchEncoderTargetSetDecoder):
    def __init__(
        self,
        *args,
        error_process_dim: int = 64,
        error_process_layers: int = 1,
        error_residual_gate_init: float = -4.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if error_process_dim <= 0:
            raise ValueError("error_process_dim must be positive.")
        if error_process_layers != 1:
            raise ValueError("Only error_process_layers=1 is currently supported.")
        self.error_process_dim = error_process_dim
        self.error_process_layers = error_process_layers
        self.error_state_cell = nn.GRUCell(2 * self.patch_embedding.out_features, error_process_dim)
        self.error_residual_head = nn.Sequential(
            nn.LayerNorm(2 * self.patch_embedding.out_features + error_process_dim),
            nn.Linear(2 * self.patch_embedding.out_features + error_process_dim, self.patch_embedding.out_features),
            nn.GELU(),
            nn.Linear(self.patch_embedding.out_features, self.segment_len),
        )
        self.error_residual_gate_logit = nn.Parameter(torch.tensor(float(error_residual_gate_init)))
        final = self.error_residual_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _error_process_residual(
        self,
        target_states: torch.Tensor,
        pred_len: int,
        segment_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._target_features(pred_len, segment_count, target_states.device, target_states.dtype)
        target_query = self.target_feature_embedding(features).unsqueeze(0)
        target_query = target_query + self.target_pos_embedding[:, :segment_count, :]
        target_query = target_query.expand(target_states.shape[0], -1, -1)
        state = target_states.new_zeros(target_states.shape[0], self.error_process_dim)
        residual_segments = []
        process_states = []
        gate = torch.sigmoid(self.error_residual_gate_logit).to(
            device=target_states.device,
            dtype=target_states.dtype,
        )
        for segment_index in range(segment_count):
            cell_input = torch.cat(
                [target_states[:, segment_index, :], target_query[:, segment_index, :]],
                dim=-1,
            )
            state = self.error_state_cell(cell_input, state)
            head_input = torch.cat([cell_input, state], dim=-1)
            residual_segments.append(self.error_residual_head(head_input) * gate)
            process_states.append(state)
        residual = torch.stack(residual_segments, dim=1).reshape(
            target_states.shape[0],
            segment_count * self.segment_len,
        )
        states = torch.stack(process_states, dim=1)
        return residual[:, :pred_len], states, gate

    def forward(
        self,
        x: torch.Tensor,
        pred_len: int,
        future_y: torch.Tensor | None = None,
        return_components: bool = False,
        window_index_norm: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del window_index_norm
        segment_count = self._validate_pred_len(pred_len)
        if self.revin is not None:
            x = self.revin(x, "norm")

        z, batch, channels = self._encode(x)
        target_states = self._target_states(z, pred_len, segment_count)
        future_components = None
        if future_y is not None:
            y_norm_target = self._normalize_future(future_y)
            future_components = self._future_alignment_components(
                target_states,
                y_norm_target,
                pred_len,
                segment_count,
            )
        history_readout = self.history_projector(z)
        affine = self.condition_head(target_states).reshape(z.shape[0], segment_count, 2, self.readout_dim)
        gamma = affine[:, :, 0, :]
        beta = affine[:, :, 1, :]

        conditioned = history_readout[:, None, :] * (1.0 + gamma) + beta
        segment_values = self.segment_output(conditioned).reshape(z.shape[0], segment_count * self.segment_len)
        prefix_residual = segment_values.new_zeros(z.shape[0], segment_count * self.segment_len)
        if self.prefix_residual_head is not None:
            active_prefix_segments = min(segment_count, self.prefix_residual_segments)
            active_width = active_prefix_segments * self.segment_len
            residual_values = self.prefix_residual_head(z)[:, :active_width]
            prefix_residual[:, :active_width] = residual_values
            segment_values = segment_values + prefix_residual

        base_norm_flat = segment_values[:, :pred_len]
        error_residual_norm_flat, error_process_states, error_gate = self._error_process_residual(
            target_states,
            pred_len,
            segment_count,
        )
        y_norm_flat = base_norm_flat + error_residual_norm_flat

        base_y = base_norm_flat.reshape(batch, channels, pred_len).permute(0, 2, 1)
        error_residual_norm = error_residual_norm_flat.reshape(batch, channels, pred_len).permute(0, 2, 1)
        y = y_norm_flat.reshape(batch, channels, pred_len).permute(0, 2, 1)
        prefix_residual_y = prefix_residual[:, :pred_len].reshape(batch, channels, pred_len).permute(0, 2, 1)

        target_states_view = target_states.reshape(batch, channels, segment_count, -1)
        gamma_view = gamma.reshape(batch, channels, segment_count, -1)
        beta_view = beta.reshape(batch, channels, segment_count, -1)
        history_readout_view = history_readout.reshape(batch, channels, -1)
        error_process_states_view = error_process_states.reshape(batch, channels, segment_count, -1)

        error_residual = error_residual_norm
        if self.revin is not None:
            y = self.revin(y, "denorm")
            base_y = self.revin(base_y, "denorm")
            error_residual = error_residual_norm * self.revin.std

        if return_components:
            output = {
                "prediction": y,
                "base_prediction": base_y,
                "error_residual": error_residual,
                "error_residual_norm": error_residual_norm,
                "error_process_states": error_process_states_view,
                "error_residual_gate": error_gate.reshape(1),
                "target_states": target_states_view,
                "gamma": gamma_view,
                "beta": beta_view,
                "history_readout": history_readout_view,
                "prefix_residual_norm": prefix_residual_y,
            }
            if future_components is not None:
                output.update(
                    {
                        "future_student_state": future_components["future_student_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_teacher_state": future_components["future_teacher_state"].reshape(
                            batch,
                            channels,
                            segment_count,
                            -1,
                        ),
                        "future_reconstruction_norm": future_components["future_reconstruction_norm"]
                        .reshape(batch, channels, pred_len)
                        .permute(0, 2, 1),
                        "future_local_alignment_loss": future_components["future_local_alignment_loss"],
                        "future_relation_alignment_loss": future_components["future_relation_alignment_loss"],
                        "future_reconstruction_loss": future_components["future_reconstruction_loss"],
                        "future_raw_reconstruction_loss": future_components["future_raw_reconstruction_loss"],
                        "future_normalized_reconstruction_loss": future_components[
                            "future_normalized_reconstruction_loss"
                        ],
                        "future_alignment_confidence_mean": future_components[
                            "future_alignment_confidence_mean"
                        ],
                        "future_alignment_confidence_min": future_components[
                            "future_alignment_confidence_min"
                        ],
                        "future_alignment_confidence_max": future_components[
                            "future_alignment_confidence_max"
                        ],
                    }
                )
            return output
        return y
