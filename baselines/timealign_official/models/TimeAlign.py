import torch
import torch.nn as nn

from layers.Alignment import glocal_align_ablation
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize


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
        if self.readout_mode not in {"official", "learned-basis-forecast-operator"}:
            raise ValueError(
                "Clean TimeAlign supports only 'official' and "
                "'learned-basis-forecast-operator' readout modes"
            )

        self.patch_emb_x = PatchEmbed(configs.d_model, self.seq_len // self.patch_num, pos=configs.pos)
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
            self.norm_y = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])

        readout_dim = configs.d_model * self.patch_num
        self.proj_x = nn.Linear(readout_dim, configs.pred_len)
        self.proj_y = nn.Linear(readout_dim, configs.pred_len)

        if self.readout_mode == "learned-basis-forecast-operator":
            self.basis_rank = int(getattr(configs, "basis_rank", 256))
            self.learned_basis_coeff = nn.Linear(readout_dim, self.basis_rank)
            self.learned_temporal_basis = nn.Parameter(torch.empty(configs.pred_len, self.basis_rank))
            self.learned_temporal_bias = nn.Parameter(torch.zeros(configs.pred_len))
            nn.init.normal_(self.learned_temporal_basis, mean=0.0, std=self.basis_rank ** -0.5)

        self.normalization_x = Normalize(configs.enc_in, affine=False)
        self.normalization_y = Normalize(configs.enc_in, affine=False)

    def _learned_basis_forecast_operator(self, hidden, target_prefix):
        # hidden: [B, C, R] -> output: [B, H, C]
        horizon = self.pred_len if target_prefix is None else int(target_prefix)
        coeff = self.learned_basis_coeff(hidden)
        basis = self.learned_temporal_basis[:horizon].to(dtype=hidden.dtype)
        bias = self.learned_temporal_bias[:horizon].to(dtype=hidden.dtype)
        output = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
        return output.permute(0, 2, 1)

    def forward(self, x, y, is_training=True, target_prefix=None):
        # x: [B, seq_len, C], y: [B, pred_len, C]
        batch, seq_len, channels = x.shape
        _batch_y, pred_len, _channels_y = y.shape

        x = self.normalization_x(x, "norm")
        x = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))

        recon = y
        if is_training:
            y = self.normalization_y(y, "norm")
            y = self.patch_emb_y(y.permute(0, 2, 1).reshape(-1, channels * pred_len))

        align_loss = x.new_zeros(())
        for layer_idx in range(self.e_layers):
            x = x + self.encoder[layer_idx](x)
            if self.layer_norm:
                x = self.norm_x[layer_idx](x)
            if is_training:
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
        else:
            raise ValueError(f"Unsupported readout mode: {self.readout_mode}")
        output = self.normalization_x(output, "denorm")

        if is_training:
            recon = self.proj_y(y.reshape(batch, channels, self.patch_num, self.d_model).flatten(start_dim=-2))
            recon = recon.permute(0, 2, 1)
            recon = self.normalization_y(recon, "denorm")

        return output[:, -self.pred_len :, :], recon, align_loss
