import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Alignment import glocal_align_ablation
from layers.CCSF import CCSFCouplingFieldReadout
from layers.Embed import PositionalEmbedding
from layers.GroupedMLP import GroupedMLPReadout
from layers.ImplicitForecast import (
    DirectNonlinearMatchedReadout,
    ImplicitFrequencyReadout,
)
from layers.PMFO import (
    DenseMLPMatchedReadout,
    PMFONoTransitionReadout,
    PMFORCTReadout,
)
from layers.PLGO import JAPOReadout, PLGOPAFReadout
from layers.PCSD import (
    PCSDCouplingFieldReadout,
    PCSDDenseMatchedReadout,
    PCSDM0Readout,
)
from layers.SIFF import SIFFCouplingFieldReadout, siff_parameter_count
from layers.StandardNorm import Normalize

LEARNED_BASIS_READOUTS = {
    "learned-basis-forecast-operator",
    "learned-basis-compact-history-statistic",
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

PMFO_READOUTS = {
    "pmfo-rct",
    "pmfo-rct-no-transition",
    "pmfo-rct-no-conservation",
    "dense-mlp-matched",
}

PLGO_PAF_READOUTS = {
    "plgo-paf-geo-c256",
    "plgo-paf-perm-c256",
    "plgo-paf-random-c256",
    "plgo-paf-geo-m694",
    "plgo-paf-perm-m694",
    "plgo-paf-random-m694",
}

JAPO_READOUTS = {
    "japo-joint-geo",
    "japo-uniform",
    "japo-history",
    "japo-atom",
    "japo-joint-perm",
    "japo-joint-random",
}

JAPO_READOUT_CONFIG = {
    "japo-joint-geo": ("joint", "geo"),
    "japo-uniform": ("uniform", "geo"),
    "japo-history": ("history", "geo"),
    "japo-atom": ("atom", "geo"),
    "japo-joint-perm": ("joint", "perm"),
    "japo-joint-random": ("joint", "random"),
}

ENCODER_MODES = {
    "raw-history-identity",
    "timealign-token-mlp",
    "contextual-patch-transformer",
    "global-anchored-patch-transformer",
    "hierarchical-patch-memory",
}

GROUPED_MLP_READOUTS = {"grouped-mlp"}
PCSD_READOUTS = {"pcsd-coupling-field"}
PCSD_CONTROL_READOUTS = {
    "pcsd-coupling-field-m0",
    "pcsd-dense-nonlinear-matched",
}
SIFF_READOUT_CONFIG = {
    "siff-coupling-field": (2, "ordered"),
    "siff-constant-control": (2, "constant"),
    "siff-permuted-scale-control": (2, "permuted"),
    "siff-q1-wide-control": (1, "ordered"),
    "siff-independent-scope-control": (5, "independent"),
}
SIFF_READOUTS = set(SIFF_READOUT_CONFIG)
SIFF_CONTROL_READOUTS = {"siff-dense-nonlinear-matched"}
CCSF_READOUT_CONFIG = {
    "ccsf-coupling-field": (2, "ordered", "true"),
    "ccsf-no-contrast-control": (2, "ordered", "zero"),
    "ccsf-permuted-contrast-control": (2, "ordered", "permuted"),
    "ccsf-independent-scope-control": (5, "independent", "true"),
}
CCSF_READOUTS = set(CCSF_READOUT_CONFIG)
COUPLING_READOUTS = PCSD_READOUTS | SIFF_READOUTS | CCSF_READOUTS
D19_IMPLICIT_READOUTS = {
    "implicit-frequency-readout",
    "implicit-frequency-noskip-control",
}
D19_DIRECT_READOUTS = {"implicit-direct-nonlinear-matched"}
D19_READOUTS = D19_IMPLICIT_READOUTS | D19_DIRECT_READOUTS
D20_READOUTS = {"learned-basis-compact-history-statistic"}


def _real_fourier_projection(length, dimension):
    if dimension <= 0 or dimension % 2:
        raise ValueError("history statistic dimension must be positive and even")
    max_frequency = dimension // 2
    steps = torch.arange(length, dtype=torch.float64)
    scale = (2.0 / float(length)) ** 0.5
    columns = []
    for frequency in range(1, max_frequency + 1):
        angle = 2.0 * torch.pi * float(frequency) * steps / float(length)
        columns.extend((scale * torch.cos(angle), scale * torch.sin(angle)))
    return torch.stack(columns, dim=1).float()


def _random_orthogonal_projection(length, dimension, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        length,
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    projection, upper = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(
        torch.diagonal(upper) < 0.0,
        -torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    return (projection * signs.unsqueeze(0)).float()


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


class GlobalAnchoredPatchEncoderLayer(nn.Module):
    """Pre-norm patch mixer with independently controlled dropout sites."""

    def __init__(
        self,
        dim,
        heads,
        d_ff,
        attn_dropout,
        attn_residual_dropout,
        ffn_dropout,
        ffn_residual_dropout,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = ResidualPatchAttention(
            dim,
            heads,
            attn_dropout,
            proj_dropout=0.0,
        )
        self.attention_residual_dropout = nn.Dropout(attn_residual_dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ff, dim),
        )
        self.feed_forward_residual_dropout = nn.Dropout(ffn_residual_dropout)

    def forward(self, x, previous_scores=None):
        attended, scores = self.attention(
            self.attention_norm(x),
            previous_scores,
        )
        x = x + self.attention_residual_dropout(attended)
        x = x + self.feed_forward_residual_dropout(
            self.feed_forward(self.feed_forward_norm(x))
        )
        return x, scores


class GlobalAnchoredPatchEncoder(nn.Module):
    """Full-window anchor plus contextual valid local patches."""

    def __init__(
        self,
        seq_len,
        patch_len,
        stride,
        dim,
        heads,
        d_ff,
        layers,
        token_dropout,
        attn_dropout,
        attn_residual_dropout,
        ffn_dropout,
        ffn_residual_dropout,
        residual_attention,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("history patch length and stride must be positive")
        if patch_len > seq_len:
            raise ValueError("history patch length cannot exceed sequence length")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len - patch_len) // stride + 1
        self.dim = dim
        self.residual_attention = residual_attention
        self.token_dropout_p = float(token_dropout)
        self.attn_dropout_p = float(attn_dropout)
        self.attn_residual_dropout_p = float(attn_residual_dropout)
        self.ffn_dropout_p = float(ffn_dropout)
        self.ffn_residual_dropout_p = float(ffn_residual_dropout)
        self.global_projection = nn.Linear(seq_len, dim)
        self.local_projection = nn.Linear(patch_len, dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.patch_num + 1, dim)
        )
        self.token_dropout = nn.Dropout(token_dropout)
        self.layers = nn.ModuleList(
            [
                GlobalAnchoredPatchEncoderLayer(
                    dim=dim,
                    heads=heads,
                    d_ff=d_ff,
                    attn_dropout=attn_dropout,
                    attn_residual_dropout=attn_residual_dropout,
                    ffn_dropout=ffn_dropout,
                    ffn_residual_dropout=ffn_residual_dropout,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)

    def forward(self, x):
        # x: [B, C, L] -> global: [B, C, D], local: [B, C, P, D]
        batch, channels, _length = x.shape
        global_token = self.global_projection(x).unsqueeze(2)
        local_patches = x.unfold(-1, self.patch_len, self.stride)
        local_tokens = self.local_projection(local_patches)
        tokens = torch.cat([global_token, local_tokens], dim=2)
        tokens = tokens.reshape(
            batch * channels,
            self.patch_num + 1,
            self.dim,
        )
        tokens = self.token_dropout(tokens + self.position_embedding)
        scores = None
        for layer in self.layers:
            tokens, scores = layer(
                tokens,
                scores if self.residual_attention else None,
            )
        tokens = self.output_norm(tokens).reshape(
            batch,
            channels,
            self.patch_num + 1,
            self.dim,
        )
        return tokens[:, :, 0, :], tokens[:, :, 1:, :]


class CanonicalPatchMemory(nn.Module):
    """Parameter-free valid patches for a stable retrieval interface."""

    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("history patch length and stride must be positive")
        if patch_len > seq_len:
            raise ValueError("history patch length cannot exceed sequence length")
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len - patch_len) // stride + 1

    def forward(self, x):
        # x: [B, C, L] -> memory: [B, C, P, K]
        return x.unfold(-1, self.patch_len, self.stride)


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
        if self.readout_mode not in {
            "official",
            *LEARNED_BASIS_READOUTS,
            *STBO_READOUTS,
            *PMFO_READOUTS,
            *PLGO_PAF_READOUTS,
            *JAPO_READOUTS,
            *GROUPED_MLP_READOUTS,
            *PCSD_READOUTS,
            *PCSD_CONTROL_READOUTS,
            *SIFF_READOUTS,
            *SIFF_CONTROL_READOUTS,
            *CCSF_READOUTS,
            *D19_READOUTS,
        }:
            raise ValueError(
                "Clean TimeAlign supports only 'official' and "
                "learned-basis/stage-native/basis-conditioned/STBO/PMFO "
                "PLGO/JAPO/grouped-MLP/PCSD readout modes"
            )
        if (
            self.readout_mode
            in PMFO_READOUTS
            | PLGO_PAF_READOUTS
            | JAPO_READOUTS
            | PCSD_READOUTS
            | PCSD_CONTROL_READOUTS
            | SIFF_READOUTS
            | SIFF_CONTROL_READOUTS
            | CCSF_READOUTS
            | D19_READOUTS
            | D20_READOUTS
            and self.pred_len != 720
        ):
            raise ValueError("StageC projective readouts require pred_len=720")
        self.has_future_recon_branch = self.readout_mode == "official"
        if (
            self.encoder_mode
            in {
                "contextual-patch-transformer",
                "global-anchored-patch-transformer",
            }
            and self.readout_mode != "learned-basis-forecast-operator"
        ):
            raise ValueError(
                "The contextual patch encoder is a clean A6 prerequisite and requires "
                "readout_mode=learned-basis-forecast-operator"
            )

        self.e_layers = configs.e_layers
        if self.encoder_mode == "raw-history-identity":
            self.patch_num = 1
            self.d_model = self.seq_len
        elif self.encoder_mode in {
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
        elif self.encoder_mode == "contextual-patch-transformer":
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
        else:
            self.d_model = configs.d_model
            self.history_encoder = GlobalAnchoredPatchEncoder(
                seq_len=configs.seq_len,
                patch_len=configs.history_patch_len,
                stride=configs.history_patch_stride,
                dim=configs.d_model,
                heads=configs.n_heads,
                d_ff=configs.d_ff,
                layers=configs.e_layers,
                token_dropout=configs.history_token_dropout,
                attn_dropout=configs.history_attn_dropout,
                attn_residual_dropout=configs.history_attn_residual_dropout,
                ffn_dropout=configs.history_ffn_dropout,
                ffn_residual_dropout=configs.history_ffn_residual_dropout,
                residual_attention=configs.history_res_attention,
            )
            self.patch_num = 1

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

        readout_dim = self.d_model * self.patch_num
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
            if self.readout_mode in D20_READOUTS:
                self.history_statistic_dim = int(
                    getattr(configs, "history_statistic_dim", 64)
                )
                self.history_statistic_mode = str(
                    getattr(
                        configs,
                        "history_statistic_mode",
                        "fixed-real-fourier-low32",
                    )
                )
                self.history_statistic_random_seed = int(
                    getattr(configs, "history_statistic_random_seed", 20260719)
                )
                if self.seq_len != 720 or self.pred_len != 720:
                    raise ValueError("D20 requires matched history/output length 720")
                if self.basis_rank != 256 or self.history_statistic_dim != 64:
                    raise ValueError("D20 requires basis_rank=256 and statistic_dim=64")
                if self.history_statistic_mode == "fixed-real-fourier-low32":
                    projection = _real_fourier_projection(
                        self.seq_len,
                        self.history_statistic_dim,
                    )
                elif self.history_statistic_mode == "fixed-gaussian-qr":
                    projection = _random_orthogonal_projection(
                        self.seq_len,
                        self.history_statistic_dim,
                        self.history_statistic_random_seed,
                    )
                else:
                    raise ValueError(
                        "Unsupported history statistic mode: "
                        f"{self.history_statistic_mode}"
                    )
                self.register_buffer(
                    "history_statistic_projection",
                    projection,
                )
                self.history_statistic_coeff = nn.Linear(
                    self.history_statistic_dim,
                    self.basis_rank,
                    bias=False,
                )
                nn.init.zeros_(self.history_statistic_coeff.weight)
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
        if self.readout_mode == "pmfo-rct":
            self.pmfo_readout = PMFORCTReadout(
                readout_dim,
                state_dim=int(getattr(configs, "pmfo_state_dim", 32)),
                conservative=True,
            )
        elif self.readout_mode == "pmfo-rct-no-transition":
            self.pmfo_readout = PMFONoTransitionReadout(readout_dim)
        elif self.readout_mode == "pmfo-rct-no-conservation":
            self.pmfo_readout = PMFORCTReadout(
                readout_dim,
                state_dim=int(getattr(configs, "pmfo_state_dim", 32)),
                conservative=False,
            )
        elif self.readout_mode == "dense-mlp-matched":
            self.pmfo_readout = DenseMLPMatchedReadout(
                readout_dim,
                hidden_dim=int(getattr(configs, "pmfo_dense_hidden_dim", 144)),
            )

        if self.readout_mode in PLGO_PAF_READOUTS:
            descriptor_name = self.readout_mode.split("-")[2]
            width_code = self.readout_mode.split("-")[3]
            trunk_width = 256 if width_code == "c256" else 694
            self.plgo_paf_readout = PLGOPAFReadout(
                readout_dim=readout_dim,
                descriptor_name=descriptor_name,
                trunk_width=trunk_width,
                series_length=self.pred_len,
                global_rank=int(getattr(configs, "plgo_global_rank", 16)),
                latent_width=int(getattr(configs, "plgo_latent_width", 256)),
                permutation_seed=int(getattr(configs, "plgo_permutation_seed", 7101)),
                random_seed=int(getattr(configs, "plgo_random_descriptor_seed", 7102)),
            )
        if self.readout_mode in JAPO_READOUTS:
            gate_mode, descriptor_name = JAPO_READOUT_CONFIG[self.readout_mode]
            self.japo_readout = JAPOReadout(
                readout_dim=readout_dim,
                gate_mode=gate_mode,
                descriptor_name=descriptor_name,
                series_length=self.pred_len,
                global_rank=int(getattr(configs, "plgo_global_rank", 16)),
                expert_count=int(getattr(configs, "japo_expert_count", 2)),
                expert_rank=int(getattr(configs, "japo_expert_rank", 256)),
                router_width=int(getattr(configs, "japo_router_width", 32)),
                router_output_init_std=float(
                    getattr(configs, "japo_router_output_init_std", 0.01)
                ),
                permutation_seed=int(getattr(configs, "plgo_permutation_seed", 7101)),
                random_seed=int(getattr(configs, "plgo_random_descriptor_seed", 7102)),
            )
        if self.readout_mode in GROUPED_MLP_READOUTS:
            self.grouped_mlp_readout = GroupedMLPReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                scale=int(getattr(configs, "grouped_mlp_scale", 144)),
                point_hidden_width=int(
                    getattr(configs, "grouped_mlp_point_hidden_width", 4)
                ),
                partition=str(
                    getattr(configs, "grouped_mlp_partition", "canonical")
                ),
                partition_seed=int(
                    getattr(configs, "grouped_mlp_partition_seed", 14101)
                ),
            )
        if self.readout_mode in PCSD_READOUTS:
            self.pcsd_readout = PCSDCouplingFieldReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                coordinate_dim=int(getattr(configs, "pcsd_coordinate_dim", 4)),
                mode_rank=int(getattr(configs, "pcsd_mode_rank", 256)),
                policy_history_dim=int(
                    getattr(configs, "pcsd_policy_history_dim", 32)
                ),
                policy_hidden_dim=int(
                    getattr(configs, "pcsd_policy_hidden_dim", 64)
                ),
                policy_mode=str(getattr(configs, "pcsd_policy_mode", "direct")),
                fixed_scale=int(getattr(configs, "pcsd_fixed_scale", 720)),
                partition=str(getattr(configs, "pcsd_partition", "canonical")),
                partition_seed=int(getattr(configs, "pcsd_partition_seed", 15101)),
                group_chunk_size=int(
                    getattr(configs, "pcsd_group_chunk_size", 64)
                ),
                target_chunk_size=int(
                    getattr(configs, "pcsd_target_chunk_size", 128)
                ),
            )
        if self.readout_mode in SIFF_READOUTS:
            scale_components, scale_basis_mode = SIFF_READOUT_CONFIG[
                self.readout_mode
            ]
            self.pcsd_readout = SIFFCouplingFieldReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                coordinate_dim=int(getattr(configs, "pcsd_coordinate_dim", 4)),
                mode_rank=int(getattr(configs, "pcsd_mode_rank", 256)),
                scale_components=scale_components,
                scale_basis_mode=scale_basis_mode,
                policy_history_dim=int(
                    getattr(configs, "pcsd_policy_history_dim", 32)
                ),
                policy_hidden_dim=int(
                    getattr(configs, "pcsd_policy_hidden_dim", 64)
                ),
                policy_mode=str(getattr(configs, "pcsd_policy_mode", "direct")),
                fixed_scale=int(getattr(configs, "pcsd_fixed_scale", 720)),
                partition=str(getattr(configs, "pcsd_partition", "canonical")),
                partition_seed=int(getattr(configs, "pcsd_partition_seed", 15101)),
                group_chunk_size=int(
                    getattr(configs, "pcsd_group_chunk_size", 64)
                ),
                target_chunk_size=int(
                    getattr(configs, "pcsd_target_chunk_size", 128)
                ),
            )
        if self.readout_mode in CCSF_READOUTS:
            scale_components, scale_basis_mode, correction_mode = (
                CCSF_READOUT_CONFIG[self.readout_mode]
            )
            self.pcsd_readout = CCSFCouplingFieldReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                coordinate_dim=int(getattr(configs, "pcsd_coordinate_dim", 4)),
                mode_rank=int(getattr(configs, "pcsd_mode_rank", 256)),
                scale_components=scale_components,
                scale_basis_mode=scale_basis_mode,
                policy_history_dim=int(
                    getattr(configs, "pcsd_policy_history_dim", 32)
                ),
                policy_hidden_dim=int(
                    getattr(configs, "pcsd_policy_hidden_dim", 64)
                ),
                policy_mode=str(getattr(configs, "pcsd_policy_mode", "direct")),
                fixed_scale=int(getattr(configs, "pcsd_fixed_scale", 720)),
                partition=str(getattr(configs, "pcsd_partition", "canonical")),
                partition_seed=int(getattr(configs, "pcsd_partition_seed", 15101)),
                group_chunk_size=int(
                    getattr(configs, "pcsd_group_chunk_size", 64)
                ),
                target_chunk_size=int(
                    getattr(configs, "pcsd_target_chunk_size", 128)
                ),
                correction_mode=correction_mode,
                correction_hidden_dim=int(
                    getattr(configs, "ccsf_correction_hidden_dim", 64)
                ),
            )
        if self.readout_mode == "pcsd-coupling-field-m0":
            self.pcsd_m0_readout = PCSDM0Readout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                mode_rank=int(getattr(configs, "pcsd_mode_rank", 256)),
            )
        if self.readout_mode == "pcsd-dense-nonlinear-matched":
            self.pcsd_dense_readout = PCSDDenseMatchedReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
            )
        if self.readout_mode == "siff-dense-nonlinear-matched":
            self.pcsd_dense_readout = PCSDDenseMatchedReadout(
                readout_dim=readout_dim,
                series_length=self.pred_len,
                target_parameters=siff_parameter_count(
                    readout_dim=readout_dim,
                    series_length=self.pred_len,
                    coordinate_dim=int(
                        getattr(configs, "pcsd_coordinate_dim", 4)
                    ),
                    mode_rank=256,
                    scale_components=2,
                    policy_history_dim=int(
                        getattr(configs, "pcsd_policy_history_dim", 32)
                    ),
                    policy_hidden_dim=int(
                        getattr(configs, "pcsd_policy_hidden_dim", 64)
                    ),
                ),
            )
        if self.readout_mode in D19_IMPLICIT_READOUTS:
            self.implicit_frequency_readout = ImplicitFrequencyReadout(
                readout_dim=readout_dim,
                history_length=self.seq_len,
                series_length=self.pred_len,
                hidden_width=int(getattr(configs, "if_hidden_width", 2048)),
                dropout=float(getattr(configs, "if_head_dropout", 0.1)),
                fourier_norm=str(getattr(configs, "if_fourier_norm", "ortho")),
                use_input_spectrum=(
                    self.readout_mode == "implicit-frequency-readout"
                ),
            )
        if self.readout_mode in D19_DIRECT_READOUTS:
            self.implicit_direct_readout = DirectNonlinearMatchedReadout(
                readout_dim=readout_dim,
                hidden_width=int(
                    getattr(configs, "if_direct_hidden_width", 4143)
                ),
                history_length=self.seq_len,
                series_length=self.pred_len,
                dropout=float(getattr(configs, "if_head_dropout", 0.1)),
                fourier_norm=str(getattr(configs, "if_fourier_norm", "ortho")),
            )

        self.normalization_x = Normalize(configs.enc_in, affine=False)
        if self.has_future_recon_branch:
            self.normalization_y = Normalize(configs.enc_in, affine=False)

    def _encode_normalized_history(self, x):
        # x: [B, L, C] -> memory: [B, C, P, D]
        batch, seq_len, channels = x.shape
        if self.encoder_mode == "raw-history-identity":
            return x.permute(0, 2, 1).unsqueeze(2)
        if self.encoder_mode == "contextual-patch-transformer":
            return self.history_encoder(x.permute(0, 2, 1))
        if self.encoder_mode == "global-anchored-patch-transformer":
            global_state, _local_memory = self.history_encoder(
                x.permute(0, 2, 1)
            )
            return global_state.unsqueeze(2)
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
        if self.encoder_mode == "global-anchored-patch-transformer":
            _global_state, local_memory = self.history_encoder(
                normalized.permute(0, 2, 1)
            )
            return local_memory
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

    def _compact_history_statistic_operator(
        self,
        hidden,
        normalized_history,
        target_prefix,
    ):
        # hidden: [B,C,R], normalized_history: [B,T,C] -> output: [B,H,C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        projection = self.history_statistic_projection.to(
            dtype=normalized_history.dtype
        )
        summary = torch.einsum(
            "btc,tq->bcq",
            normalized_history,
            projection,
        )
        coeff = self.learned_basis_coeff(hidden)
        coeff = coeff + self.history_statistic_coeff(summary)
        basis = self.learned_temporal_basis[:horizon].to(dtype=hidden.dtype)
        bias = self.learned_temporal_bias[:horizon].to(dtype=hidden.dtype)
        output = torch.einsum("hk,bck->bch", basis, coeff)
        output = output + bias.view(1, 1, -1)
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

    def forward(
        self,
        x,
        y,
        is_training=True,
        target_prefix=None,
        return_pcsd_training_details=False,
    ):
        # x: [B, seq_len, C], y: [B, pred_len, C]
        batch, seq_len, channels = x.shape
        _batch_y, pred_len, _channels_y = y.shape

        x = self.normalization_x(x, "norm")
        normalized_history = x

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
        elif self.readout_mode in D20_READOUTS:
            output = self._compact_history_statistic_operator(
                hidden,
                normalized_history,
                target_prefix,
            )
        elif self.readout_mode.startswith("stage-native-coefficient-field"):
            output = self._stage_native_coefficient_field_operator(hidden, target_prefix)
        elif self.readout_mode.startswith("basis-conditioned-coefficient-field"):
            output = self._basis_conditioned_coefficient_field_operator(hidden, target_prefix)
        elif self.readout_mode in STBO_READOUTS:
            output = self._subspace_tiled_basis_operator(hidden, target_prefix)
        elif self.readout_mode in PMFO_READOUTS:
            output = self.pmfo_readout(hidden, target_prefix)
        elif self.readout_mode in PLGO_PAF_READOUTS:
            output = self.plgo_paf_readout(hidden, target_prefix)
        elif self.readout_mode in JAPO_READOUTS:
            output = self.japo_readout(hidden, target_prefix)
        elif self.readout_mode in GROUPED_MLP_READOUTS:
            output = self.grouped_mlp_readout(hidden, target_prefix)
        elif self.readout_mode in COUPLING_READOUTS:
            if return_pcsd_training_details:
                if hasattr(
                    self.pcsd_readout,
                    "forward_with_ccsf_diagnostics",
                ):
                    output, pcsd_arms, pcsd_policy, ccsf_details = (
                        self.pcsd_readout.forward_with_ccsf_diagnostics(
                            hidden,
                            target_prefix,
                        )
                    )
                else:
                    output, pcsd_arms, pcsd_policy = (
                        self.pcsd_readout.forward_with_diagnostics(
                            hidden,
                            target_prefix,
                        )
                    )
                    ccsf_details = None
            else:
                output = self.pcsd_readout(hidden, target_prefix)
        elif self.readout_mode == "pcsd-coupling-field-m0":
            output = self.pcsd_m0_readout(hidden, target_prefix)
        elif self.readout_mode in {
            "pcsd-dense-nonlinear-matched",
            "siff-dense-nonlinear-matched",
        }:
            output = self.pcsd_dense_readout(hidden, target_prefix)
        elif self.readout_mode in D19_IMPLICIT_READOUTS:
            output = self.implicit_frequency_readout(
                hidden,
                normalized_history,
                target_prefix,
            )
        elif self.readout_mode in D19_DIRECT_READOUTS:
            output = self.implicit_direct_readout(
                hidden,
                normalized_history,
                target_prefix,
            )
        else:
            raise ValueError(f"Unsupported readout mode: {self.readout_mode}")
        output = self.normalization_x(output, "denorm")

        if self.has_future_recon_branch and is_training:
            recon = self.proj_y(y.reshape(batch, channels, self.patch_num, self.d_model).flatten(start_dim=-2))
            recon = recon.permute(0, 2, 1)
            recon = self.normalization_y(recon, "denorm")

        result = (output[:, -self.pred_len :, :], recon, align_loss)
        if not return_pcsd_training_details:
            return result
        if self.readout_mode not in COUPLING_READOUTS:
            raise ValueError(
                "coupling training details require a PCSD/SIFF coupling readout"
            )
        if self.normalization_x.affine:
            raise RuntimeError("scoped PCSD denormalization expects affine=False")
        if self.normalization_x.non_norm:
            denormalized_arms = pcsd_arms
        else:
            scale = (
                self.normalization_x.stdev.squeeze(1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            if self.normalization_x.subtract_last:
                center = self.normalization_x.last.squeeze(1)
            else:
                center = self.normalization_x.mean.squeeze(1)
            center = center.unsqueeze(-1).unsqueeze(-1)
            denormalized_arms = pcsd_arms * scale + center
        details = {
            "arm_forecasts": denormalized_arms,
            "policy": pcsd_policy,
        }
        if ccsf_details is not None:
            details.update(ccsf_details)
        return (*result, details)
