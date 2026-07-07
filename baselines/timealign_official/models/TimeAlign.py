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


class Model(nn.Module):
    """Official TimeAlign carrier with only the accepted A6-LBF unified head."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_num = configs.patch_num
        self.d_model = configs.d_model
        self.readout_mode = getattr(configs, "readout_mode", "official")
        if self.readout_mode not in {"official", *LEARNED_BASIS_READOUTS}:
            raise ValueError(
                "Clean TimeAlign supports only 'official' and "
                "learned-basis/stage-native readout modes"
            )
        self.has_future_recon_branch = self.readout_mode == "official"

        self.patch_emb_x = PatchEmbed(configs.d_model, self.seq_len // self.patch_num, pos=configs.pos)
        if self.has_future_recon_branch:
            self.patch_emb_y = PatchEmbed(configs.d_model, self.pred_len // self.patch_num, pos=configs.pos)

        self.e_layers = configs.e_layers
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

        self.layer_norm = configs.layer_norm
        if self.layer_norm:
            self.norm_x = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])
            if self.has_future_recon_branch:
                self.norm_y = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])

        readout_dim = configs.d_model * self.patch_num
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

        self.normalization_x = Normalize(configs.enc_in, affine=False)
        if self.has_future_recon_branch:
            self.normalization_y = Normalize(configs.enc_in, affine=False)

    def _build_stage_boundaries(self, configs):
        horizons = sorted({int(horizon) for horizon in getattr(configs, "target_horizons", [])})
        if not horizons:
            horizons = [configs.pred_len]
        if horizons[-1] != configs.pred_len:
            horizons.append(configs.pred_len)
        if horizons[-1] > configs.pred_len:
            raise ValueError("stage boundaries cannot exceed pred_len")
        return horizons

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

    def forward(self, x, y, is_training=True, target_prefix=None):
        # x: [B, seq_len, C], y: [B, pred_len, C]
        batch, seq_len, channels = x.shape
        _batch_y, pred_len, _channels_y = y.shape

        x = self.normalization_x(x, "norm")
        x = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))

        recon = None
        if self.has_future_recon_branch and is_training:
            y = self.normalization_y(y, "norm")
            y = self.patch_emb_y(y.permute(0, 2, 1).reshape(-1, channels * pred_len))

        align_loss = x.new_zeros(())
        for layer_idx in range(self.e_layers):
            x = x + self.encoder[layer_idx](x)
            if self.layer_norm:
                x = self.norm_x[layer_idx](x)
            if self.has_future_recon_branch and is_training:
                x_aligned = self.ffn[layer_idx](x)
                y = y + self.autoencoder[layer_idx](y)
                if self.layer_norm:
                    y = self.norm_y[layer_idx](y)
                align_loss = align_loss + self.align(x_aligned, y.detach())
        align_loss = align_loss / self.e_layers

        hidden = x.reshape(batch, channels, self.patch_num, self.d_model).flatten(start_dim=-2)
        if self.readout_mode == "official":
            output = self.proj_x(hidden).permute(0, 2, 1)
        elif self.readout_mode == "learned-basis-forecast-operator":
            output = self._learned_basis_forecast_operator(hidden, target_prefix)
        elif self.readout_mode.startswith("stage-native-coefficient-field"):
            output = self._stage_native_coefficient_field_operator(hidden, target_prefix)
        else:
            raise ValueError(f"Unsupported readout mode: {self.readout_mode}")
        output = self.normalization_x(output, "denorm")

        if self.has_future_recon_branch and is_training:
            recon = self.proj_y(y.reshape(batch, channels, self.patch_num, self.d_model).flatten(start_dim=-2))
            recon = recon.permute(0, 2, 1)
            recon = self.normalization_y(recon, "denorm")

        return output[:, -self.pred_len :, :], recon, align_loss
