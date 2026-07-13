# Phase5 StageB B10-TSI-D Failure Attribution Code Explanation

## 诊断位置

| 字段 | 内容 |
| --- | --- |
| `script` | `scripts/analyze_phase5_stage_b_b10_tsi_failure_attribution.py` |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-D` |
| `current_step` | StageB Step 3：failure attribution and diagnostic redesign |
| `input` | clean A6 checkpoint、train/val/test split、frozen encoder memory、`coeff`、`learned_temporal_basis` |
| `output` | `analysis/phase5_stage_b_b10_tsi_failure_attribution_20260708/`; rank16 control at `analysis/phase5_stage_b_b10_tsi_failure_attribution_rank16_20260708/` |
| `scope` | Frozen-A6 offline diagnostic；不训练 forecasting model；不能单独用于方向级拒绝 |

## Reader Path

B10-TSI-C 的问题是把 target-set 信息放在太晚的位置：

```text
frozen coeff -> Linear_s(coeff) -> delta_s
```

这个诊断把两个因素混在了一起：

1. target-set 信息是否有用；
2. late readout / head 设计是否病态。

B10-TSI-D 因此把 feature source 拆成三类，并加入稳定性控制：

```text
coeff_late         = A6 coeff                         # [N, 256]
memory_pool        = mean/last/std encoder memory     # [N, 3D]
memory_plus_coeff  = concat(memory_pool, coeff)
```

其中 `N = batch * channel`。对 ETT 数据，`D=32` 时 `memory_pool` 为 `[N, 96]`；对 ETTm1，
`D=256` 时 `memory_pool` 为 `[N, 768]`；Weather 为 `[N, 384]`。

## Tensor Flow

脚本复用 clean A6 的 forward path，但在 encoder 后截取 memory：

```text
batch_x                      # [B, 720, C]
x_norm = normalization_x(x)  # [B, 720, C]
patch_emb_x                  # [B, C * patch_num, D]
encoder layers               # [B, C * patch_num, D]
memory                       # [B, C, patch_num, D]
hidden = flatten(memory)     # [B, C, patch_num * D]
coeff = learned_basis_coeff  # [B, C, 256]
pred_norm                    # [B, 720, C]
residual = target_norm - pred_norm
```

随后按 channel 展平成 row：

```text
coeff_rows       # [B*C, 256]
memory_pool_rows # [B*C, 3D]
residual_rows    # [B*C, 720]
```

## Row-Space Target

为避免 B10-TSI-C 的 full coefficient inverse 病态，本脚本不再拟合完整 coefficient delta。对每个
future segment：

```text
basis_s = learned_temporal_basis[start:end]       # [L_s, 256]
U_s, S_s, V_s = svd(basis_s)
target_s = residual_s @ U_s[:, :rank]             # [N, rank]
correction_s = readout(feature) @ U_s[:, :rank].T  # [N, L_s]
```

这只测试 residual 在 learned-basis segment row-space 中是否能被 feature source 稳定解释。

## Controls

每个 feature source 都使用同一组 controls：

- `shared_control`: 所有 segments 共享一个 readout；
- `pooled_multihead_control`: 多个 pooled heads，不按 target set 选择；
- `target_set_aware`: 每个 segment 一个独立 readout；
- `wrong_target_control`: 用错误 segment 的 readout，检查 target label 是否有意义；
- `stabilized_target_set`: 从 shared readout 出发，只允许 validation 选择小幅 target-specific deviation。

`stabilized_target_set` 的形式是：

```text
W_s(beta) = W_shared + beta * (W_target_s - W_shared)
beta in {0, 0.05, 0.1, 0.25, 0.5, 1.0}
```

它用于区分“target-set 信息无效”和“独立 target head 因数据切分/容量过大而过拟合”。

## Current Result

rank64 主诊断：

| feature_source | stabilized target vs pooled mean | stabilized pathology datasets |
| --- | ---: | ---: |
| `coeff_late` | `-12.3695%` | `1` |
| `memory_pool` | `-40.7499%` | `2` |
| `memory_plus_coeff` | `-44.4687%` | `2` |

rank16 稳定性对照：

| feature_source | stabilized target vs pooled mean | stabilized pathology datasets |
| --- | ---: | ---: |
| `coeff_late` | `-5.1506%` | `1` |
| `memory_pool` | `-32.0345%` | `1` |
| `memory_plus_coeff` | `-36.3672%` | `2` |

## Code-Theory Consistency

[Intended theory] 如果 target set 信息对 A6 basis-coeff interface 有稳定价值，那么 memory-level
target-set-aware readout 应该在稳定控制下超过 no-target pooled control。

[Code realization] 脚本把 intervention point 从 `coeff` 前移到 encoder memory feature，并加入
rank-truncated row-space target、validation alpha、wrong-target control 和 shrinkage target-set readout。

[Observed boundary] 当前 offline ridge readout route 仍不稳定。rank64 下 ETTh2/Weather 出现 pathology；
rank16 减轻但未消除问题，且 memory-level target-set readout 仍输给 pooled controls。

## Decision

`B10-TSI-D` 不支持把当前 offline readout 设计推进为 method，也不能用于否定更大的
target-set-aware architecture 方向。

下一步不能继续用 frozen ridge oracle 反复否定方向。若 B10 继续，必须进入 Step 4-6，设计真正的
native trainable target-query memory readout，并在 implementation gate 中保留 no-target query control；
若该 narrative gate 不能成立，则 StageB 应回到 Step 2/3。
