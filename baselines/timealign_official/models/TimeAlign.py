import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Alignment import orth_align, glocal_align, glocal_align_ablation
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)

        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    def forward(self, x):
        # x: [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, C*N, P]
        x = self.patch_proj(x) # [B, C*N, D]

        if self.pos:
            x += self.pe(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_num = configs.patch_num
        self.d_model = configs.d_model
        # embedding
        self.patch_emb_x = PatchEmbed(configs.d_model, self.seq_len // self.patch_num, pos=configs.pos)
        self.patch_emb_y = PatchEmbed(configs.d_model, self.pred_len // self.patch_num, pos=configs.pos)

        # Encoder
        self.e_layers = configs.e_layers
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
            )
            for _ in range(configs.e_layers)
        ])

        # self.align = glocal_align(configs.local_margin, configs.global_margin)
        self.align = glocal_align_ablation(configs.local_margin, configs.global_margin, configs.loc, configs.glo)

        self.ffn = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model) for _ in range(configs.e_layers)])

        self.autoencoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
            )
            for _ in range(configs.e_layers)
        ])

        self.layer_norm = configs.layer_norm
        if self.layer_norm:
            self.norm_x = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])
            self.norm_y = nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)])

        # Decoder
        self.readout_mode = getattr(configs, "readout_mode", "official")
        self.proj_x = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.proj_y = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        self.conditioned_projection_modes = {"prefix-conditioned-head", "target-set-decoder"}
        self.variable_prefix_modes = {"target-set-prefix-head", "prefix-token-decoder"}
        if self.readout_mode in self.conditioned_projection_modes:
            readout_dim = configs.d_model * self.patch_num
            self.prefix_condition = nn.Sequential(
                nn.Linear(1, readout_dim),
                nn.GELU(),
                nn.Linear(readout_dim, readout_dim),
            )
            nn.init.zeros_(self.prefix_condition[-1].weight)
            nn.init.zeros_(self.prefix_condition[-1].bias)
        if self.readout_mode == "target-set-prefix-head":
            readout_dim = configs.d_model * self.patch_num
            self.prefix_step_weight = nn.Sequential(
                nn.Linear(2, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, readout_dim),
            )
            self.prefix_step_bias = nn.Linear(2, 1)
            self.readout_scale = readout_dim ** -0.5
        if self.readout_mode == "prefix-token-decoder":
            self.step_query = nn.Sequential(
                nn.Linear(2, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model),
            )
            self.token_key = nn.Linear(configs.d_model, configs.d_model)
            self.token_value = nn.Linear(configs.d_model, configs.d_model)
            self.token_out = nn.Linear(configs.d_model, 1)
            self.readout_scale = configs.d_model ** -0.5

        self.normalization_x = Normalize(configs.enc_in, affine=False)
        self.normalization_y = Normalize(configs.enc_in, affine=False)

    def _condition_readout(self, hidden, target_prefix):
        if self.readout_mode not in self.conditioned_projection_modes:
            return hidden
        if target_prefix is None:
            target_prefix = self.pred_len
        prefix_value = hidden.new_tensor([[float(target_prefix) / float(self.pred_len)]])
        condition = torch.tanh(self.prefix_condition(prefix_value)).view(1, 1, -1)
        return hidden + hidden * condition

    def _prefix_features(self, target_prefix, device, dtype):
        if target_prefix is None:
            target_prefix = self.pred_len
        horizon = int(target_prefix)
        steps = torch.arange(1, horizon + 1, device=device, dtype=dtype) / float(self.pred_len)
        prefix = torch.full_like(steps, float(horizon) / float(self.pred_len))
        return torch.stack([steps, prefix], dim=-1)

    def _target_set_prefix_head(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, H, C]
        features = self._prefix_features(target_prefix, hidden.device, hidden.dtype)
        weights = self.prefix_step_weight(features)
        bias = self.prefix_step_bias(features).squeeze(-1)
        output = torch.einsum("bcr,hr->bch", hidden, weights) * self.readout_scale
        output = output + bias.view(1, 1, -1)
        return output.permute(0, 2, 1)

    def _prefix_token_decoder(self, hidden, target_prefix):
        # hidden: [B, C, N, D], output: [B, H, C]
        features = self._prefix_features(target_prefix, hidden.device, hidden.dtype)
        query = self.step_query(features)
        key = self.token_key(hidden)
        value = self.token_value(hidden)
        scores = torch.einsum("bcnd,hd->bchn", key, query) * self.readout_scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bchn,bcnd->bchd", weights, value)
        output = self.token_out(context).squeeze(-1)
        return output.permute(0, 2, 1)

    def forward(self, x, y, is_training=True, target_prefix=None):
        # [B, L, C]   [B, T, C]
        B, T, C = x.shape
        _, L, C = y.shape

        x = self.normalization_x(x, 'norm')
        x = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, C*T))

        if is_training:
            y = self.normalization_y(y, 'norm')
            y = self.patch_emb_y(y.permute(0, 2, 1).reshape(-1, C*L))

        # [B, C, D]
        align_loss = 0.0
        for i in range(self.e_layers):
            x = x + self.encoder[i](x)
            if self.layer_norm:
                x = self.norm_x[i](x)
            if is_training:
                x_ = self.ffn[i](x)
                y = y + self.autoencoder[i](y)
                if self.layer_norm:
                    y = self.norm_y[i](y)
                # align_loss += self.align(x_, y)
                align_loss += self.align(x_, y.detach())
        align_loss /= self.e_layers

        # [B, C, N, D]
        # print(x.reshape(-1, C, self.patch_num, self.d_model).shape)

        x = x.reshape(-1, C, self.patch_num, self.d_model)
        if self.readout_mode == "target-set-prefix-head":
            x = self._target_set_prefix_head(x.flatten(start_dim=-2), target_prefix)
        elif self.readout_mode == "prefix-token-decoder":
            x = self._prefix_token_decoder(x, target_prefix)
        else:
            x = x.flatten(start_dim=-2)
            x = self._condition_readout(x, target_prefix)
            x = self.proj_x(x) # [B, C, T]
            x = x.permute(0, 2, 1)
        x = self.normalization_x(x, 'denorm')

        if is_training:
            y = self.proj_y(y.reshape(-1, C, self.patch_num, self.d_model).flatten(start_dim=-2)) # [B, C, T]
            y = y.permute(0, 2, 1)
            y = self.normalization_y(y, 'denorm')

        return x[:, -self.pred_len :, :], y, align_loss
