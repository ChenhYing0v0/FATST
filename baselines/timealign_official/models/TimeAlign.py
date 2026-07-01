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
        self.direct_prefix_modes = {
            "target-set-prefix-head",
            "prefix-token-decoder",
            "dense-row-initialized-prefix-decoder",
            "nested-segment-decoder",
            "dense-initialized-nested-segment-decoder",
            "target-conditioned-nested-residual-decoder",
            "checkpoint-initialized-nested-segment-decoder",
        }
        self.capacity_preserving_modes = {
            "dense-prefix-residual-adapter",
            "row-gated-dense-head",
            "prefix-adapter-shared-dense",
            "target-conditioned-nested-residual-decoder",
        }
        readout_dim = configs.d_model * self.patch_num
        adapter_dim = min(64, readout_dim)
        if self.readout_mode in self.conditioned_projection_modes:
            self.prefix_condition = nn.Sequential(
                nn.Linear(1, readout_dim),
                nn.GELU(),
                nn.Linear(readout_dim, readout_dim),
            )
            nn.init.zeros_(self.prefix_condition[-1].weight)
            nn.init.zeros_(self.prefix_condition[-1].bias)
        if self.readout_mode == "target-set-prefix-head":
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
        if self.readout_mode == "dense-prefix-residual-adapter":
            self.residual_down = nn.Linear(readout_dim, adapter_dim)
            self.residual_condition = nn.Linear(1, adapter_dim)
            self.residual_up = nn.Linear(adapter_dim, configs.pred_len)
            nn.init.zeros_(self.residual_up.weight)
            nn.init.zeros_(self.residual_up.bias)
        if self.readout_mode == "row-gated-dense-head":
            self.row_gate = nn.Sequential(
                nn.Linear(2, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, 1),
            )
            nn.init.zeros_(self.row_gate[-1].weight)
            nn.init.zeros_(self.row_gate[-1].bias)
        if self.readout_mode == "prefix-adapter-shared-dense":
            self.hidden_adapter_down = nn.Linear(readout_dim, adapter_dim)
            self.hidden_adapter_condition = nn.Linear(1, adapter_dim)
            self.hidden_adapter_up = nn.Linear(adapter_dim, readout_dim)
            nn.init.zeros_(self.hidden_adapter_up.weight)
            nn.init.zeros_(self.hidden_adapter_up.bias)
        if self.readout_mode == "dense-row-initialized-prefix-decoder":
            self.prefix_row_delta_down = nn.Linear(readout_dim, adapter_dim)
            self.prefix_row_delta_condition = nn.Linear(1, adapter_dim)
            self.prefix_row_delta_up = nn.Linear(adapter_dim, configs.pred_len)
            nn.init.zeros_(self.prefix_row_delta_up.weight)
            nn.init.zeros_(self.prefix_row_delta_up.bias)
        self.nested_readout_modes = {
            "nested-segment-decoder",
            "dense-initialized-nested-segment-decoder",
            "target-conditioned-nested-residual-decoder",
            "checkpoint-initialized-nested-segment-decoder",
        }
        if self.readout_mode in self.nested_readout_modes:
            boundaries = sorted(set(getattr(configs, "target_horizons", [configs.pred_len])))
            boundaries = [value for value in boundaries if 0 < value <= configs.pred_len]
            if configs.pred_len not in boundaries:
                boundaries.append(configs.pred_len)
            self.nested_boundaries = boundaries
            previous = 0
            if self.readout_mode == "target-conditioned-nested-residual-decoder":
                self.nested_residual_down = nn.Linear(readout_dim, adapter_dim)
                self.nested_residual_condition = nn.Linear(1, adapter_dim)
                self.nested_segment_heads = nn.ModuleList()
                for boundary in self.nested_boundaries:
                    head = nn.Linear(adapter_dim, boundary - previous)
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
                    self.nested_segment_heads.append(head)
                    previous = boundary
            else:
                self.nested_segment_heads = nn.ModuleList()
                for boundary in self.nested_boundaries:
                    head = nn.Linear(readout_dim, boundary - previous)
                    if self.readout_mode == "dense-initialized-nested-segment-decoder":
                        with torch.no_grad():
                            head.weight.copy_(self.proj_x.weight[previous:boundary])
                            head.bias.copy_(self.proj_x.bias[previous:boundary])
                    self.nested_segment_heads.append(head)
                    previous = boundary

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

    def _prefix_scalar_feature(self, target_prefix, hidden):
        if target_prefix is None:
            target_prefix = self.pred_len
        return hidden.new_tensor([[float(target_prefix) / float(self.pred_len)]])

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

    def _dense_prefix_residual_adapter(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, C, pred_len]
        base = self.proj_x(hidden)
        condition = torch.tanh(self.residual_condition(self._prefix_scalar_feature(target_prefix, hidden))).view(1, 1, -1)
        adapted = torch.nn.functional.gelu(self.residual_down(hidden)) * (1.0 + condition)
        residual = self.residual_up(adapted)
        return base + residual

    def _row_gated_dense_head(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, C, pred_len]
        base = self.proj_x(hidden)
        features = self._prefix_features(self.pred_len, hidden.device, hidden.dtype)
        if target_prefix is None:
            target_prefix = self.pred_len
        features[:, 1] = float(target_prefix) / float(self.pred_len)
        gate = 1.0 + 0.1 * torch.tanh(self.row_gate(features).squeeze(-1))
        return base * gate.view(1, 1, -1)

    def _prefix_adapter_shared_dense(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, C, pred_len]
        condition = torch.tanh(self.hidden_adapter_condition(self._prefix_scalar_feature(target_prefix, hidden))).view(1, 1, -1)
        adapted = torch.nn.functional.gelu(self.hidden_adapter_down(hidden)) * (1.0 + condition)
        hidden = hidden + self.hidden_adapter_up(adapted)
        return self.proj_x(hidden)

    def _dense_row_initialized_prefix_decoder(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, H, C]
        if target_prefix is None:
            target_prefix = self.pred_len
        horizon = int(target_prefix)
        base = torch.nn.functional.linear(hidden, self.proj_x.weight[:horizon], self.proj_x.bias[:horizon])
        condition = torch.tanh(self.prefix_row_delta_condition(self._prefix_scalar_feature(target_prefix, hidden))).view(1, 1, -1)
        adapted = torch.nn.functional.gelu(self.prefix_row_delta_down(hidden)) * (1.0 + condition)
        delta = self.prefix_row_delta_up(adapted)[:, :, :horizon]
        return (base + delta).permute(0, 2, 1)

    def _nested_segment_decoder(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, H, C]
        if target_prefix is None:
            target_prefix = self.pred_len
        horizon = int(target_prefix)
        segments = []
        previous = 0
        for boundary, head in zip(self.nested_boundaries, self.nested_segment_heads):
            if previous >= horizon:
                break
            segment = head(hidden)
            take = min(boundary, horizon) - previous
            if take > 0:
                segments.append(segment[:, :, :take])
            previous = boundary
        return torch.cat(segments, dim=-1).permute(0, 2, 1)

    def _target_conditioned_nested_residual_decoder(self, hidden, target_prefix):
        # hidden: [B, C, R], output: [B, H, C]
        if target_prefix is None:
            target_prefix = self.pred_len
        horizon = int(target_prefix)
        base = self.proj_x(hidden)[:, :, :horizon]
        condition = torch.tanh(self.nested_residual_condition(self._prefix_scalar_feature(target_prefix, hidden))).view(1, 1, -1)
        adapted = F.gelu(self.nested_residual_down(hidden)) * (1.0 + condition)
        segments = []
        previous = 0
        for boundary, head in zip(self.nested_boundaries, self.nested_segment_heads):
            if previous >= horizon:
                break
            segment = head(adapted)
            take = min(boundary, horizon) - previous
            if take > 0:
                segments.append(segment[:, :, :take])
            previous = boundary
        residual = torch.cat(segments, dim=-1)
        return (base + residual).permute(0, 2, 1)

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
        elif self.readout_mode == "dense-row-initialized-prefix-decoder":
            x = self._dense_row_initialized_prefix_decoder(x.flatten(start_dim=-2), target_prefix)
        elif self.readout_mode in {
            "nested-segment-decoder",
            "dense-initialized-nested-segment-decoder",
            "checkpoint-initialized-nested-segment-decoder",
        }:
            x = self._nested_segment_decoder(x.flatten(start_dim=-2), target_prefix)
        elif self.readout_mode == "target-conditioned-nested-residual-decoder":
            x = self._target_conditioned_nested_residual_decoder(x.flatten(start_dim=-2), target_prefix)
        else:
            x = x.flatten(start_dim=-2)
            if self.readout_mode == "dense-prefix-residual-adapter":
                x = self._dense_prefix_residual_adapter(x, target_prefix)
            elif self.readout_mode == "row-gated-dense-head":
                x = self._row_gated_dense_head(x, target_prefix)
            elif self.readout_mode == "prefix-adapter-shared-dense":
                x = self._prefix_adapter_shared_dense(x, target_prefix)
            else:
                x = self._condition_readout(x, target_prefix)
                x = self.proj_x(x) # [B, C, T]
            x = x.permute(0, 2, 1)
        x = self.normalization_x(x, 'denorm')

        if is_training:
            y = self.proj_y(y.reshape(-1, C, self.patch_num, self.d_model).flatten(start_dim=-2)) # [B, C, T]
            y = y.permute(0, 2, 1)
            y = self.normalization_y(y, 'denorm')

        return x[:, -self.pred_len :, :], y, align_loss
