from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class OnlineInstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=-1, keepdim=True).detach()
            self.std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.std
        if mode == "denorm":
            return x * self.std + self.mean
        raise ValueError(f"Unknown norm mode: {mode}")


class PatchEmbedding(nn.Module):
    def __init__(self, patch_width: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Conv1d(1, d_model, kernel_size=patch_width, stride=patch_width, bias=False)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        x = x.reshape(batch * channels, 1, length)
        x = self.embed(x)
        x = self.dropout(self.norm(x))
        return x.permute(0, 2, 1).contiguous()


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.theta <= 0:
            return x
        seq_len = x.shape[1]
        device = x.device
        dtype = x.dtype
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        cos = freqs.cos()[None, :, None, :].to(dtype)
        sin = freqs.sin()[None, :, None, :].to(dtype)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bsz, seq_len, n_kv, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bsz, seq_len, n_kv, self.n_rep, head_dim)
            .reshape(bsz, seq_len, n_kv * self.n_rep, head_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        y = y.transpose(1, 2).reshape(bsz, seq_len, d_model)
        return self.out_proj(y)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseTSFTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        drop_path: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout, rope_theta)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = RMSNorm(d_model)
        self.ffn = DenseFFN(d_model, d_ff, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class SegTSFTDenseFixedHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_width: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        encoder_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        rope_theta: float = 10000.0,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        if seq_len % patch_width != 0:
            raise ValueError("seq_len must be divisible by patch_width for SegTSFTDenseFixedHead")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.input_norm = OnlineInstanceNorm() if use_input_norm else None
        self.patch_embedding = PatchEmbedding(patch_width, d_model, dropout)
        n_patches = seq_len // patch_width
        rates = torch.linspace(0, drop_path, encoder_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                DenseTSFTBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, rates[i], rope_theta)
                for i in range(encoder_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as (B, L, C); Seg-MoE TSFT convention is (B, C, L).
        x = x.permute(0, 2, 1)
        if self.input_norm is not None:
            x = self.input_norm(x, "norm")
        batch, channels, _ = x.shape
        z = self.patch_embedding(x)
        for block in self.blocks:
            z = block(z)
        z = self.final_norm(z)
        y = self.head(z.flatten(start_dim=1))
        y = y.reshape(batch, channels, self.pred_len)
        if self.input_norm is not None:
            y = self.input_norm(y, "denorm")
        return y.permute(0, 2, 1)
