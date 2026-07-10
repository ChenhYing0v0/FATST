import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Alignment import glocal_align_ablation
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize

LEARNED_BASIS_READOUTS = {
    "learned-basis-forecast-operator",
    "stage-native-coefficient-field",
    "stage-native-coefficient-field-no-stage",
    "basis-conditioned-coefficient-field",
    "basis-conditioned-coefficient-field-no-basis",
    "basis-conditioned-coefficient-field-shuffled-basis",
    "basis-conditioned-coefficient-field-constant-slot",
}

STBO_READOUTS = {
    "subspace-tiled-basis-operator-shared",
    "subspace-tiled-basis-operator-bank",
    "subspace-tiled-basis-operator-dct",
    "subspace-tiled-basis-operator-independent",
}

ENCODER_MODES = {
    "timealign-token-mlp",
    "contextual-patch-transformer",
    "hierarchical-patch-memory",
}


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            self.pe = PositionalEmbedding(dim, 10000)

    def forward(self, x):
        # x: [B, C, L] -> [B, C * N, D]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)
        if self.pos:
            x += self.pe(x)
        return x


class TokenBatchNorm(nn.Module):
    """Batch-normalize the feature axis of [N, P, D] patch tokens."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualPatchAttention(nn.Module):
    """Multi-head self-attention with optional pre-softmax residual scores."""

    def __init__(self, dim, heads, attn_dropout, proj_dropout):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("history d_model must be divisible by history n_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))

    def forward(self, x, previous_scores=None):
        batch, patch_num, dim = x.shape
        query = self.query(x).reshape(
            batch, patch_num, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(x).reshape(
            batch, patch_num, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).reshape(
            batch, patch_num, self.heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if previous_scores is not None:
            scores = scores + previous_scores
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).contiguous().reshape(batch, patch_num, dim)
        return self.output(context), scores


class ContextualPatchEncoderLayer(nn.Module):
    """PatchTST-derived post-BatchNorm residual attention block."""

    def __init__(self, dim, heads, d_ff, dropout, attn_dropout):
        super().__init__()
        self.attention = ResidualPatchAttention(dim, heads, attn_dropout, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = TokenBatchNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = TokenBatchNorm(dim)

    def forward(self, x, previous_scores=None):
        attended, scores = self.attention(x, previous_scores)
        x = self.attention_norm(x + self.attention_dropout(attended))
        x = self.feed_forward_norm(x + self.feed_forward_dropout(self.feed_forward(x)))
        return x, scores


class ContextualPatchEncoder(nn.Module):
    """Channel-independent overlapping patch encoder returning [B, C, P, D]."""

    def __init__(
        self,
        seq_len,
        patch_len,
        stride,
        dim,
        heads,
        d_ff,
        layers,
        dropout,
        attn_dropout,
        residual_attention,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("history patch length and stride must be positive")
        if patch_len > seq_len + stride:
            raise ValueError("history patch length cannot exceed padded sequence length")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.dim = dim
        self.residual_attention = residual_attention
        self.end_padding = nn.ReplicationPad1d((0, stride))
        self.patch_projection = nn.Linear(patch_len, dim)
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_num, dim))
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                ContextualPatchEncoderLayer(dim, heads, d_ff, dropout, attn_dropout)
                for _ in range(layers)
            ]
        )
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)

    def forward(self, x):
        # x: [B, C, L] -> memory: [B, C, P, D]
        batch, channels, _length = x.shape
        patches = self.end_padding(x).unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_projection(patches)
        tokens = tokens.reshape(batch * channels, self.patch_num, self.dim)
        tokens = self.input_dropout(tokens + self.position_embedding)
        scores = None
        for layer in self.layers:
            tokens, scores = layer(tokens, scores if self.residual_attention else None)
        return tokens.reshape(batch, channels, self.patch_num, self.dim)


class CanonicalPatchMemory(nn.Module):
    """Parameter-free overlapping patches for a stable retrieval interface."""

    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("history patch length and stride must be positive")
        if patch_len > seq_len + stride:
            raise ValueError("history patch length cannot exceed padded sequence length")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len + stride - patch_len) // stride + 1
        self.end_padding = nn.ReplicationPad1d((0, stride))

    def forward(self, x):
        # x: [B, C, L] -> memory: [B, C, P, K]
        return self.end_padding(x).unfold(-1, self.patch_len, self.stride)


class Model(nn.Module):
    """Legacy TimeAlign baseline plus clean A6 history/readout candidates."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.encoder_mode = getattr(configs, "encoder_mode", "timealign-token-mlp")
        if self.encoder_mode not in ENCODER_MODES:
            raise ValueError(f"Unsupported encoder mode: {self.encoder_mode}")
        self.readout_mode = getattr(configs, "readout_mode", "official")
        if self.readout_mode not in {"official", *LEARNED_BASIS_READOUTS, *STBO_READOUTS}:
            raise ValueError(
                "Clean TimeAlign supports only 'official' and "
                "learned-basis/stage-native/basis-conditioned/STBO readout modes"
            )
        self.has_future_recon_branch = self.readout_mode == "official"
        if (
            self.encoder_mode == "contextual-patch-transformer"
            and self.readout_mode != "learned-basis-forecast-operator"
        ):
            raise ValueError(
                "The contextual patch encoder is a clean A6 prerequisite and requires "
                "readout_mode=learned-basis-forecast-operator"
            )

        self.e_layers = configs.e_layers
        if self.encoder_mode in {
            "timealign-token-mlp",
            "hierarchical-patch-memory",
        }:
            self.patch_num = configs.patch_num
            self.d_model = configs.d_model
            self.patch_emb_x = PatchEmbed(
                configs.d_model,
                self.seq_len // self.patch_num,
                pos=configs.pos,
            )
            self.encoder = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(configs.d_model, configs.d_ff),
                        nn.GELU(),
                        nn.Dropout(configs.dropout),
                        nn.Linear(configs.d_ff, configs.d_model),
                    )
                    for _ in range(configs.e_layers)
                ]
            )
        else:
            self.d_model = configs.d_model
            self.history_encoder = ContextualPatchEncoder(
                seq_len=configs.seq_len,
                patch_len=configs.history_patch_len,
                stride=configs.history_patch_stride,
                dim=configs.d_model,
                heads=configs.n_heads,
                d_ff=configs.d_ff,
                layers=configs.e_layers,
                dropout=configs.dropout,
                attn_dropout=configs.history_attn_dropout,
                residual_attention=configs.history_res_attention,
            )
            self.patch_num = self.history_encoder.patch_num

        if self.encoder_mode == "hierarchical-patch-memory":
            self.retrieval_memory = CanonicalPatchMemory(
                seq_len=configs.seq_len,
                patch_len=configs.history_patch_len,
                stride=configs.history_patch_stride,
            )

        if self.has_future_recon_branch:
            self.patch_emb_y = PatchEmbed(configs.d_model, self.pred_len // self.patch_num, pos=configs.pos)

        if self.has_future_recon_branch:
            self.ffn = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model) for _ in range(configs.e_layers)])
            self.autoencoder = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(configs.d_model, configs.d_ff),
                        nn.GELU(),
                        nn.Dropout(configs.dropout),
                        nn.Linear(configs.d_ff, configs.d_model),
                    )
                    for _ in range(configs.e_layers)
                ]
            )
            self.align = glocal_align_ablation(configs.local_margin, configs.global_margin, configs.loc, configs.glo)

        self.layer_norm = (
            configs.layer_norm
            if self.encoder_mode
            in {"timealign-token-mlp", "hierarchical-patch-memory"}
            else False
        )
        if self.layer_norm:
            self.norm_x = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])
            if self.has_future_recon_branch:
                self.norm_y = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])

        readout_dim = configs.d_model * self.patch_num
        if self.has_future_recon_branch or self.encoder_mode in {
            "timealign-token-mlp",
            "hierarchical-patch-memory",
        }:
            self.proj_x = nn.Linear(readout_dim, configs.pred_len)
        if self.has_future_recon_branch:
            self.proj_y = nn.Linear(readout_dim, configs.pred_len)

        if self.readout_mode in LEARNED_BASIS_READOUTS:
            self.basis_rank = int(getattr(configs, "basis_rank", 256))
            self.learned_basis_coeff = nn.Linear(readout_dim, self.basis_rank)
            self.learned_temporal_basis = nn.Parameter(torch.empty(configs.pred_len, self.basis_rank))
            self.learned_temporal_bias = nn.Parameter(torch.zeros(configs.pred_len))
            nn.init.normal_(self.learned_temporal_basis, mean=0.0, std=self.basis_rank ** -0.5)
            if self.readout_mode.startswith("stage-native-coefficient-field"):
                self.stage_boundaries = self._build_stage_boundaries(configs)
                self.stage_count = len(self.stage_boundaries)
                self.stage_token_dim = int(getattr(configs, "stage_token_dim", 32))
                self.stage_field_rank = int(getattr(configs, "stage_field_rank", 32))
                self.stage_tokens = nn.Parameter(torch.zeros(self.stage_count, self.stage_token_dim))
                self.stage_coeff_norm = nn.LayerNorm(self.basis_rank)
                self.stage_coeff_down = nn.Linear(self.basis_rank + self.stage_token_dim, self.stage_field_rank)
                self.stage_coeff_up = nn.Linear(self.stage_field_rank, self.basis_rank)
                self.stage_gate_logits = nn.Parameter(
                    torch.full((self.stage_count, 1), float(getattr(configs, "stage_gate_init", -5.0)))
                )
                nn.init.normal_(self.stage_tokens, mean=0.0, std=0.02)
                nn.init.zeros_(self.stage_coeff_up.weight)
                nn.init.zeros_(self.stage_coeff_up.bias)
            if self.readout_mode.startswith("basis-conditioned-coefficient-field"):
                self.basis_field_window_len = int(getattr(configs, "basis_field_window_len", 96))
                self.basis_field_stride = int(getattr(configs, "basis_field_stride", 48))
                self.basis_field_rank = int(getattr(configs, "basis_field_rank", 32))
                self.basis_field_tau = float(getattr(configs, "basis_field_tau", 1.0))
                if self.basis_field_window_len <= 0 or self.basis_field_stride <= 0:
                    raise ValueError("basis field window length and stride must be positive")
                if self.basis_field_rank <= 0:
                    raise ValueError("basis field rank must be positive")
                if self.basis_field_tau <= 0.0:
                    raise ValueError("basis field tau must be positive")
                starts = self._build_basis_field_window_starts(configs.pred_len)
                self.register_buffer(
                    "basis_field_window_starts",
                    torch.tensor(starts, dtype=torch.long),
                    persistent=False,
                )
                self.basis_field_window_count = len(starts)
                self.basis_field_desc_norm = nn.LayerNorm(self.basis_rank)
                self.basis_field_desc_proj = nn.Linear(self.basis_rank, self.basis_field_rank)
                self.basis_field_state_proj = nn.Linear(readout_dim, self.basis_field_rank)
                self.basis_field_delta = nn.Linear(self.basis_field_rank, self.basis_rank)
                self.basis_field_gate_logit = nn.Parameter(
                    torch.tensor(float(getattr(configs, "basis_field_gate_init", -5.0)))
                )
                self.basis_field_no_basis_rows = nn.Parameter(
                    torch.empty(configs.pred_len, self.basis_field_rank)
                )
                self.basis_field_no_basis_slots = nn.Parameter(
                    torch.empty(self.basis_field_window_count, self.basis_field_rank)
                )
                nn.init.normal_(self.basis_field_no_basis_rows, mean=0.0, std=0.02)
                nn.init.normal_(self.basis_field_no_basis_slots, mean=0.0, std=0.02)
                nn.init.zeros_(self.basis_field_delta.weight)
                nn.init.zeros_(self.basis_field_delta.bias)
        if self.readout_mode in STBO_READOUTS:
            self.stbo_tile_len = int(getattr(configs, "stbo_tile_len", 48))
            self.stbo_rank = int(getattr(configs, "stbo_rank", 16))
            self.stbo_bank_count = int(getattr(configs, "stbo_bank_count", 4))
            if self.stbo_tile_len <= 0:
                raise ValueError("stbo_tile_len must be positive")
            if configs.pred_len % self.stbo_tile_len != 0:
                raise ValueError("stbo_tile_len must divide pred_len")
            if self.stbo_rank <= 0 or self.stbo_rank > self.stbo_tile_len:
                raise ValueError("stbo_rank must be in [1, stbo_tile_len]")
            if self.stbo_bank_count < 2:
                raise ValueError("stbo_bank_count must be at least 2")
            self.stbo_tile_count = configs.pred_len // self.stbo_tile_len
            self.stbo_coeff = nn.Linear(readout_dim, self.stbo_tile_count * self.stbo_rank)
            self.stbo_temporal_bias = nn.Parameter(torch.zeros(configs.pred_len))
            init_std = float(getattr(configs, "stbo_basis_init_std", self.stbo_rank ** -0.5))
            if self.readout_mode == "subspace-tiled-basis-operator-shared":
                self.stbo_shared_basis = nn.Parameter(torch.empty(self.stbo_tile_len, self.stbo_rank))
                nn.init.normal_(self.stbo_shared_basis, mean=0.0, std=init_std)
            elif self.readout_mode == "subspace-tiled-basis-operator-bank":
                self.stbo_basis_bank = nn.Parameter(
                    torch.empty(self.stbo_bank_count, self.stbo_tile_len, self.stbo_rank)
                )
                self.stbo_tile_bank_logits = nn.Parameter(torch.zeros(self.stbo_tile_count, self.stbo_bank_count))
                nn.init.normal_(self.stbo_basis_bank, mean=0.0, std=init_std)
            elif self.readout_mode == "subspace-tiled-basis-operator-independent":
                self.stbo_tile_basis = nn.Parameter(
                    torch.empty(self.stbo_tile_count, self.stbo_tile_len, self.stbo_rank)
                )
                nn.init.normal_(self.stbo_tile_basis, mean=0.0, std=init_std)
            elif self.readout_mode == "subspace-tiled-basis-operator-dct":
                self.register_buffer(
                    "stbo_dct_basis",
                    self._build_dct_basis(self.stbo_tile_len, self.stbo_rank),
                    persistent=False,
                )

        self.normalization_x = Normalize(configs.enc_in, affine=False)
        if self.has_future_recon_branch:
            self.normalization_y = Normalize(configs.enc_in, affine=False)

    def _encode_normalized_history(self, x):
        # x: [B, L, C] -> memory: [B, C, P, D]
        batch, seq_len, channels = x.shape
        if self.encoder_mode == "contextual-patch-transformer":
            return self.history_encoder(x.permute(0, 2, 1))
        tokens = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
        for layer_idx in range(self.e_layers):
            tokens = tokens + self.encoder[layer_idx](tokens)
            if self.layer_norm:
                tokens = self.norm_x[layer_idx](tokens)
        return tokens.reshape(batch, channels, self.patch_num, self.d_model)

    def encode_history(self, x):
        """Return normalized history memory for diagnostics and retrieval."""
        return self._encode_normalized_history(self.normalization_x(x, "norm"))

    def encode_retrieval_memory(self, x):
        """Return the canonical local memory without changing the forecast path."""
        normalized = self.normalization_x(x, "norm")
        if self.encoder_mode == "hierarchical-patch-memory":
            return self.retrieval_memory(normalized.permute(0, 2, 1))
        return self._encode_normalized_history(normalized)

    def _build_stage_boundaries(self, configs):
        horizons = sorted({int(horizon) for horizon in getattr(configs, "target_horizons", [])})
        if not horizons:
            horizons = [configs.pred_len]
        if horizons[-1] != configs.pred_len:
            horizons.append(configs.pred_len)
        if horizons[-1] > configs.pred_len:
            raise ValueError("stage boundaries cannot exceed pred_len")
        return horizons

    def _build_basis_field_window_starts(self, pred_len):
        starts = list(range(0, pred_len - self.basis_field_window_len + 1, self.basis_field_stride))
        if not starts or starts[-1] + self.basis_field_window_len < pred_len:
            starts.append(max(0, pred_len - self.basis_field_window_len))
        return sorted(set(starts))

    def _build_dct_basis(self, length, rank):
        steps = torch.arange(length, dtype=torch.float32) + 0.5
        freqs = torch.arange(rank, dtype=torch.float32)
        basis = torch.cos(torch.pi * torch.outer(steps, freqs) / float(length))
        basis[:, 0] *= (1.0 / float(length)) ** 0.5
        if rank > 1:
            basis[:, 1:] *= (2.0 / float(length)) ** 0.5
        return basis

    def _learned_basis_forecast_operator(self, hidden, target_prefix):
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        coeff = self.learned_basis_coeff(hidden)
        basis = self.learned_temporal_basis[:horizon].to(dtype=hidden.dtype)
        bias = self.learned_temporal_bias[:horizon].to(dtype=hidden.dtype)
        output = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
        return output.permute(0, 2, 1)

    def _stage_segments(self, horizon):
        segments = []
        start = 0
        for stage_idx, boundary in enumerate(self.stage_boundaries):
            end = min(boundary, horizon)
            if end > start:
                segments.append((stage_idx, start, end))
            if boundary >= horizon:
                break
            start = boundary
        if not segments or segments[-1][2] != horizon:
            segments.append((len(self.stage_boundaries) - 1, segments[-1][2] if segments else 0, horizon))
        return segments

    def _stage_coeff_field(self, coeff):
        # coeff: [B, C, K] -> coeff_field: [B, C, S, K]
        z = self.stage_coeff_norm(coeff)
        if self.readout_mode == "stage-native-coefficient-field-no-stage":
            stage_tokens = self.stage_tokens.mean(dim=0, keepdim=True).expand(self.stage_count, -1)
            gate_logits = self.stage_gate_logits.mean(dim=0, keepdim=True).expand(self.stage_count, -1)
        else:
            stage_tokens = self.stage_tokens
            gate_logits = self.stage_gate_logits

        z = z.unsqueeze(2).expand(-1, -1, self.stage_count, -1)
        tokens = stage_tokens.to(dtype=coeff.dtype).view(1, 1, self.stage_count, self.stage_token_dim)
        tokens = tokens.expand(coeff.shape[0], coeff.shape[1], -1, -1)
        stage_input = torch.cat([z, tokens], dim=-1)
        delta = torch.tanh(self.stage_coeff_up(F.gelu(self.stage_coeff_down(stage_input))))
        gate = torch.sigmoid(gate_logits.to(dtype=coeff.dtype)).view(1, 1, self.stage_count, 1)
        return coeff.unsqueeze(2) * (1.0 + gate * delta)

    def _stage_native_coefficient_field_operator(self, hidden, target_prefix):
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        coeff = self.learned_basis_coeff(hidden)
        coeff_field = self._stage_coeff_field(coeff)
        outputs = []
        for stage_idx, start, end in self._stage_segments(horizon):
            basis = self.learned_temporal_basis[start:end].to(dtype=hidden.dtype)
            bias = self.learned_temporal_bias[start:end].to(dtype=hidden.dtype)
            stage_coeff = coeff_field[:, :, stage_idx, :]
            stage_output = torch.einsum("hk,bck->bch", basis, stage_coeff) + bias.view(1, 1, -1)
            outputs.append(stage_output)
        output = torch.cat(outputs, dim=-1)
        return output.permute(0, 2, 1)

    def _basis_field_descriptors(self, horizon, dtype):
        if self.readout_mode == "basis-conditioned-coefficient-field-no-basis":
            row_desc = self.basis_field_no_basis_rows[:horizon].to(dtype=dtype)
            window_desc = self.basis_field_no_basis_slots.to(dtype=dtype)
            return row_desc, window_desc

        descriptor_basis = self.learned_temporal_basis
        if self.readout_mode == "basis-conditioned-coefficient-field-shuffled-basis":
            descriptor_basis = descriptor_basis.flip(dims=[0])

        descriptor_basis = descriptor_basis.to(dtype=dtype)
        row_desc = self.basis_field_desc_proj(self.basis_field_desc_norm(descriptor_basis[:horizon]))
        window_means = []
        for start_tensor in self.basis_field_window_starts:
            start = int(start_tensor.item())
            end = min(start + self.basis_field_window_len, self.pred_len)
            window_means.append(descriptor_basis[start:end].mean(dim=0))
        window_mean = torch.stack(window_means, dim=0)
        window_desc = self.basis_field_desc_proj(self.basis_field_desc_norm(window_mean))
        return row_desc, window_desc

    def _basis_field_alpha(self, horizon, dtype):
        row_desc, window_desc = self._basis_field_descriptors(horizon, dtype)
        row_desc = F.normalize(row_desc, dim=-1)
        window_desc = F.normalize(window_desc, dim=-1)
        scores = torch.matmul(row_desc, window_desc.transpose(0, 1)) / self.basis_field_tau
        alpha = torch.softmax(scores, dim=-1)
        if self.readout_mode == "basis-conditioned-coefficient-field-constant-slot":
            alpha = alpha.mean(dim=0, keepdim=True).expand(horizon, -1)
        return alpha, window_desc

    def _basis_conditioned_coefficient_field_operator(self, hidden, target_prefix):
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        coeff = self.learned_basis_coeff(hidden)
        alpha, window_desc = self._basis_field_alpha(horizon, hidden.dtype)
        state = self.basis_field_state_proj(hidden)
        field_state = state.unsqueeze(2) + window_desc.view(1, 1, self.basis_field_window_count, -1)
        delta = torch.tanh(self.basis_field_delta(F.gelu(field_state)))
        gate = torch.sigmoid(self.basis_field_gate_logit.to(dtype=hidden.dtype))
        coeff_slots = coeff.unsqueeze(2) + gate * delta
        basis = self.learned_temporal_basis[:horizon].to(dtype=hidden.dtype)
        bias = self.learned_temporal_bias[:horizon].to(dtype=hidden.dtype)
        output = torch.einsum("hm,hk,bcmk->bch", alpha, basis, coeff_slots) + bias.view(1, 1, -1)
        return output.permute(0, 2, 1)

    def _stbo_basis_tiles(self, dtype):
        if self.readout_mode == "subspace-tiled-basis-operator-shared":
            basis = self.stbo_shared_basis.to(dtype=dtype)
            return basis.unsqueeze(0).expand(self.stbo_tile_count, -1, -1)
        if self.readout_mode == "subspace-tiled-basis-operator-bank":
            weights = torch.softmax(self.stbo_tile_bank_logits.to(dtype=dtype), dim=-1)
            bank = self.stbo_basis_bank.to(dtype=dtype)
            return torch.einsum("mq,qlr->mlr", weights, bank)
        if self.readout_mode == "subspace-tiled-basis-operator-dct":
            basis = self.stbo_dct_basis.to(dtype=dtype)
            return basis.unsqueeze(0).expand(self.stbo_tile_count, -1, -1)
        if self.readout_mode == "subspace-tiled-basis-operator-independent":
            return self.stbo_tile_basis.to(dtype=dtype)
        raise ValueError(f"Unsupported STBO readout mode: {self.readout_mode}")

    def _subspace_tiled_basis_operator(self, hidden, target_prefix):
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.pred_len:
            raise ValueError("target_prefix must be in [1, pred_len]")
        needed_tiles = (horizon + self.stbo_tile_len - 1) // self.stbo_tile_len
        coeff = self.stbo_coeff(hidden)
        coeff = coeff.view(hidden.shape[0], hidden.shape[1], self.stbo_tile_count, self.stbo_rank)
        coeff = coeff[:, :, :needed_tiles, :]
        basis = self._stbo_basis_tiles(hidden.dtype)[:needed_tiles]
        output = torch.einsum("mlr,bcmr->bcml", basis, coeff)
        output = output.reshape(hidden.shape[0], hidden.shape[1], needed_tiles * self.stbo_tile_len)
        bias = self.stbo_temporal_bias[: needed_tiles * self.stbo_tile_len].to(dtype=hidden.dtype)
        output = output + bias.view(1, 1, -1)
        output = output[:, :, :horizon]
        return output.permute(0, 2, 1)

    def forward(self, x, y, is_training=True, target_prefix=None):
        # x: [B, seq_len, C], y: [B, pred_len, C]
        batch, seq_len, channels = x.shape
        _batch_y, pred_len, _channels_y = y.shape

        x = self.normalization_x(x, "norm")

        recon = None
        if self.has_future_recon_branch and is_training:
            x = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
            y = self.normalization_y(y, "norm")
            y = self.patch_emb_y(y.permute(0, 2, 1).reshape(-1, channels * pred_len))
            align_loss = x.new_zeros(())
            for layer_idx in range(self.e_layers):
                x = x + self.encoder[layer_idx](x)
                if self.layer_norm:
                    x = self.norm_x[layer_idx](x)
                x_aligned = self.ffn[layer_idx](x)
                y = y + self.autoencoder[layer_idx](y)
                if self.layer_norm:
                    y = self.norm_y[layer_idx](y)
                align_loss = align_loss + self.align(x_aligned, y.detach())
            align_loss = align_loss / self.e_layers
            memory = x.reshape(batch, channels, self.patch_num, self.d_model)
        else:
            memory = self._encode_normalized_history(x)
            align_loss = memory.new_zeros(())

        hidden = memory.flatten(start_dim=-2)
        if self.readout_mode == "official":
            output = self.proj_x(hidden).permute(0, 2, 1)
        elif self.readout_mode == "learned-basis-forecast-operator":
            output = self._learned_basis_forecast_operator(hidden, target_prefix)
        elif self.readout_mode.startswith("stage-native-coefficient-field"):
            output = self._stage_native_coefficient_field_operator(hidden, target_prefix)
        elif self.readout_mode.startswith("basis-conditioned-coefficient-field"):
            output = self._basis_conditioned_coefficient_field_operator(hidden, target_prefix)
        elif self.readout_mode in STBO_READOUTS:
            output = self._subspace_tiled_basis_operator(hidden, target_prefix)
        else:
            raise ValueError(f"Unsupported readout mode: {self.readout_mode}")
        output = self.normalization_x(output, "denorm")

        if self.has_future_recon_branch and is_training:
            recon = self.proj_y(y.reshape(batch, channels, self.patch_num, self.d_model).flatten(start_dim=-2))
            recon = recon.permute(0, 2, 1)
            recon = self.normalization_y(recon, "denorm")

        return output[:, -self.pred_len :, :], recon, align_loss
