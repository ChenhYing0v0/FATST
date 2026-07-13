from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, length: int, patch_num: int, d_model: int, use_positional: bool = True) -> None:
        super().__init__()
        if length % patch_num != 0:
            raise ValueError(f"length={length} must be divisible by patch_num={patch_num}.")
        self.patch_num = patch_num
        self.patch_len = length // patch_num
        self.proj = nn.Linear(self.patch_len, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 1, patch_num, d_model)) if use_positional else None
        if self.positional is not None:
            nn.init.normal_(self.positional, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, C, patch_num, d_model]
        patches = x.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        embedded = self.proj(patches)
        if self.positional is not None:
            embedded = embedded + self.positional
        return embedded


class GlocalAlignment(nn.Module):
    def __init__(
        self,
        local_margin: float = 0.0,
        global_margin: float = 0.0,
        use_local: bool = True,
        use_global: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.local_margin = local_margin
        self.global_margin = global_margin
        self.use_local = use_local
        self.use_global = use_global
        self.eps = eps

    def _dynamic_loss(self, losses: list[torch.Tensor]) -> torch.Tensor:
        if not losses:
            raise RuntimeError("At least one alignment component must be enabled.")
        detached_mean = sum(loss.detach() for loss in losses) / len(losses)
        return sum(detached_mean * loss / loss.detach().clamp_min(self.eps) for loss in losses)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # pred/target: [B, C, patch_num, d_model].
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        losses = []
        stats: dict[str, torch.Tensor] = {}
        if self.use_local:
            local = F.gelu(1.0 - torch.abs(pred * target) - self.local_margin).mean()
            losses.append(local)
            stats["local_alignment_loss"] = local.detach()
        if self.use_global:
            pred_gram = torch.matmul(pred, pred.transpose(-1, -2))
            target_gram = torch.matmul(target, target.transpose(-1, -2))
            global_loss = F.gelu(torch.abs(pred_gram - target_gram) - self.global_margin).mean()
            losses.append(global_loss)
            stats["global_alignment_loss"] = global_loss.detach()
        loss = self._dynamic_loss(losses)
        stats["alignment_loss"] = loss.detach()
        return loss, stats


class TimeAlignCarrier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_num: int = 48,
        d_model: int = 32,
        d_ff: int = 32,
        e_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_positional: bool = True,
        use_local_align: bool = True,
        use_global_align: bool = True,
        local_margin: float = 0.0,
        global_margin: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_num = patch_num
        self.d_model = d_model
        self.e_layers = e_layers
        self.eps = eps

        self.patch_x = PatchEmbedding(seq_len, patch_num, d_model, use_positional)
        self.patch_y = PatchEmbedding(pred_len, patch_num, d_model, use_positional)
        self.encoder = nn.ModuleList([self._block(d_model, d_ff, dropout) for _ in range(e_layers)])
        self.autoencoder = nn.ModuleList([self._block(d_model, d_ff, dropout) for _ in range(e_layers)])
        self.student_projection = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(e_layers)])
        self.norm_x = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])
        self.norm_y = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])
        self.use_layer_norm = use_layer_norm
        self.alignment = GlocalAlignment(
            local_margin=local_margin,
            global_margin=global_margin,
            use_local=use_local_align,
            use_global=use_global_align,
        )
        self.head_x = nn.Linear(d_model * patch_num, pred_len)
        self.head_y = nn.Linear(d_model * patch_num, pred_len)

    @staticmethod
    def _block(d_model: int, d_ff: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(self.eps)
        return (x - mean) / std, mean, std

    @staticmethod
    def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def _readout(self, states: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        # states: [B, C, patch_num, d_model] -> [B, pred_len, C]
        batch, channels, _, _ = states.shape
        out = head(states.reshape(batch, channels, -1))
        return out.permute(0, 2, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x_norm, x_mean, x_std = self._normalize(x)
        x_state = self.patch_x(x_norm)
        y_state = None
        y_mean = y_std = None
        if y is not None:
            y_norm, y_mean, y_std = self._normalize(y)
            y_state = self.patch_y(y_norm)

        align_losses = []
        align_stats: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.e_layers):
            x_state = x_state + self.encoder[layer_idx](x_state)
            if self.use_layer_norm:
                x_state = self.norm_x[layer_idx](x_state)
            if y_state is not None:
                student = self.student_projection[layer_idx](x_state)
                y_state = y_state + self.autoencoder[layer_idx](y_state)
                if self.use_layer_norm:
                    y_state = self.norm_y[layer_idx](y_state)
                align_loss, stats = self.alignment(student, y_state.detach())
                align_losses.append(align_loss)
                align_stats.append(stats)

        pred_norm = self._readout(x_state, self.head_x)
        pred = self._denormalize(pred_norm, x_mean, x_std)
        output = {"prediction": pred}

        if y_state is not None:
            if y_mean is None or y_std is None:
                raise RuntimeError("Future normalization stats are missing.")
            recon_norm = self._readout(y_state, self.head_y)
            recon = self._denormalize(recon_norm, y_mean, y_std)
            alignment_loss = sum(align_losses) / len(align_losses)
            output["reconstruction"] = recon
            output["alignment_loss"] = alignment_loss
            output["local_alignment_loss"] = torch.stack(
                [stats.get("local_alignment_loss", alignment_loss.detach()) for stats in align_stats]
            ).mean()
            output["global_alignment_loss"] = torch.stack(
                [stats.get("global_alignment_loss", alignment_loss.detach()) for stats in align_stats]
            ).mean()
        return output
