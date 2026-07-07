# Phase5 StageB B9-FSN-SCF Code Explanation

本文档解释 `B9-FSN-SCF` 的最小实现。对应代码：

- `baselines/timealign_official/models/TimeAlign.py`
- `baselines/timealign_official/train_repo.py`

## Research Role

`B9-FSN-SCF` 是 StageB 的 Step 7 最小实现候选。它来自 Step 4-6 设计门：

```text
coeff_base = learned_basis_coeff(hidden)
coeff_s = StageCoefficientField(coeff_base, hidden, stage_token_s)
y[t in stage_s,c] = learned_temporal_basis[t] @ coeff_s[b,c] + bias[t]
```

它不是 residual correction，不计算 `true - pred`，也不在 A6 output 后追加 correction。

## Readout Modes

新增两个 readout mode：

| Mode | Role |
| --- | --- |
| `stage-native-coefficient-field` | B9-FSN-SCF candidate |
| `stage-native-coefficient-field-no-stage` | no-stage capacity control |

二者都属于 prefix-native learned-basis readout：

- 不实例化 TimeAlign future reconstruction/alignment branch；
- `w_recon=0.0`；
- `w_align=0.0`；
- 使用 `pred_loss_mode=multi-prefix`。

## Forward Tensor Flow

以 ETT / `pred_len=720` / `basis_rank=256` / `target_horizons=96,192,336,720` 为例：

```text
x: [B, 720, C]
x_norm = Normalize(x)
tokens = PatchEmbed(x_norm)                         # [B*C, patch_num, d_model]
encoded = TimeAlign encoder(tokens)
hidden = reshape(encoded)                           # [B, C, R]

coeff_base = learned_basis_coeff(hidden)            # [B, C, 256]
coeff_field = _stage_coeff_field(coeff_base)        # [B, C, 4, 256]
```

四个 stage boundaries 来自 `target_horizons`：

```text
S0 = [0, 96)
S1 = [96, 192)
S2 = [192, 336)
S3 = [336, 720)
```

每个 stage 使用自己的 coefficient：

```text
pred[:, 0:96, :]    = learned_temporal_basis[0:96]    @ coeff_field[:, :, 0, :]
pred[:, 96:192, :]  = learned_temporal_basis[96:192]  @ coeff_field[:, :, 1, :]
pred[:, 192:336, :] = learned_temporal_basis[192:336] @ coeff_field[:, :, 2, :]
pred[:, 336:720, :] = learned_temporal_basis[336:720] @ coeff_field[:, :, 3, :]
```

最后输出仍是 `[B, H, C]`，并经过 `normalization_x(..., "denorm")`。

## Stage Coefficient Field

`_stage_coeff_field(coeff)` 的输入输出：

```text
coeff:       [B, C, K]
coeff_field: [B, C, S, K]
```

最小参数化：

```text
z = LayerNorm(coeff_base)
stage_input = concat(z, stage_token_s)
delta_s = tanh(stage_coeff_up(gelu(stage_coeff_down(stage_input))))
gate_s = sigmoid(stage_gate_logits_s)
coeff_s = coeff_base * (1 + gate_s * delta_s)
```

其中：

- `stage_tokens: [S, stage_token_dim]`;
- `stage_coeff_down: K + stage_token_dim -> stage_field_rank`;
- `stage_coeff_up: stage_field_rank -> K`;
- `stage_gate_logits: [S, 1]`。

## Function-Preserving Fallback

`stage_coeff_up.weight` 和 `stage_coeff_up.bias` 初始化为 0。因此初始时：

```text
delta_s = 0
coeff_s = coeff_base
```

这使 `stage-native-coefficient-field` 和 `stage-native-coefficient-field-no-stage` 的初始 forward 与
`learned-basis-forecast-operator` 精确一致。若 warm-start 到已训练 A6 checkpoint，只有 shared A6 参数来自
trained checkpoint 才能声称 learned capacity transfer；随机初始化复制仍不能称为 learned capacity
preservation。

## No-Stage Control

`stage-native-coefficient-field-no-stage` 保留同样的 module 和参数量，但在 forward 中把所有 stage tokens 和
gates 求均值后共享给每个 stage：

```text
stage_token_s = mean(stage_tokens)
gate_s = mean(stage_gate_logits)
```

这样它能测试“额外参数量”是否足以解释收益。如果 no-stage control 与 B9-FSN-SCF 等价或更强，则不能 claim
native future-stage mechanism。

## Exported Diagnostics

`train_repo.py` 在每个 run 完成训练后导出 `model_diagnostics.json`：

| Field | Meaning |
| --- | --- |
| `total_parameters` | model parameter count |
| `trainable_parameters` | trainable parameter count |
| `readout_mode` | active readout mode |
| `stage_count` | number of stage boundaries, present only for B9 modes |
| `stage_boundaries` | stage end positions such as `[96, 192, 336, 720]` |
| `stage_gate_sigmoid` | current scalar gate value per stage |
| `stage_gate_mean/min/max` | gate summary |
| `stage_token_l2` | stage token norm |
| `stage_coeff_down_l2` | down projection weight norm |
| `stage_coeff_up_l2` | up projection weight norm |

这些字段用于 small gate 判断 stage module 是否完全塌缩，以及 B9/no-stage 是否只是参数量差异。

## Prefix Contract

`_stage_segments(target_prefix)` 只拼接 requested prefix 覆盖的 stage slices：

- `target_prefix=96` 只使用 `S0`;
- `target_prefix=192` 使用 `S0+S1`;
- `target_prefix=720` 使用全部 `S0..S3`。

因为没有 future placeholder 之间的信息流，请求更长 horizon 不应改写已有 prefix。small gate 必须检查：

```text
pred_H96 == pred_H720[:, :96]
pred_H192 == pred_H720[:, :192]
pred_H336 == pred_H720[:, :336]
```

在 eval mode 下，初始/训练后都应接近浮点误差。

## Code-Theory Consistency

[Intended Theory] B9 解决 A6 shared coefficient state 的 stage-gradient conflict。

[Code Realization] 代码把 `coeff_base: [B,C,K]` 扩展为 `coeff_field: [B,C,S,K]`，使 stage 信息在
prediction 前进入 coefficient/operator path。

[Proxy Boundary] 当前实现只做 stage-level coefficient field，不做 per-step query、cross-attention decoder 或
future placeholder encoder。因此它验证的是 stage-native coefficient routing，而不是完整 future-query
architecture。

[Falsification] 若 no-stage control 同样有效、prefix consistency 破坏、或 stage gates/coeff variance 全部塌缩，
则 B9-FSN-SCF 不能作为 paper-core architecture 继续推进。
