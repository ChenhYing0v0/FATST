# Phase4-S Predictability Scheduling

## 目的

[Decision] `predictability_downweight` 是 Phase4-S 在 S1 small gate 失败后的最小回退实现。
它不改变 model architecture，只改变 training loss 和 supervision trace。

研究动机：

- S1 `conditioned_future_unit_scheduling` 相比 `full_time_mse` 有收益；
- 但 Weather 相对 R.3 全面退化；
- offline diagnostic 显示 Weather 的 high-novelty selected blocks 同时是 high-variation
  blocks，更接近 low-predictability / noisy-hard；
- 因此下一版不应继续给所有 hard blocks 加压，而应区分 learnable-hard 与 noisy-hard。

## Forward / Training Flow

入口仍是 `baselines/patch_encoder_target_set_decoder/train.py`。

1. `horizon_mixed_training` 只对 `horizon_mixed` 和 `r3_prefix_risk` 为 true。
2. `predictability_downweight` 不在 mixed set 中，因此训练 loader 只使用
   `train_horizons_effective=[supervision_pred_len]`。
3. 当前 small-gate 设计中 `supervision_pred_len=720`，所以每个 train step 都生成
   `pred.shape = [B, 720, C]`，`true.shape = [B, 720, C]`。
4. `predictability_downweight_loss()` 把 720 future steps 按 `block_size=48` 切成 15 个
   blocks。
5. 每个 block 计算两个 train-label-only scores：
   - `novelty`: block 相对最后一个 history step 的 MSE；
   - `variation`: block 内部一阶差分能量。
6. top novelty blocks 表示 hard candidates；top variation blocks 表示 noisy candidates。
7. `noisy_blocks = top_novelty ∩ top_variation`。
8. `learnable_blocks = top_novelty - top_variation`；如果为空，则回退为 top novelty blocks。

## Loss

令 $u$ 表示 720 内的 block。

基础 loss 是 predictability-aware weighted dense loss：

$$
\mathcal{L}_{time}
=
\frac{1}{BTC}
\sum_{b,t,c}
w_t
(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

其中：

- noisy-hard blocks 的 $w_t$ 设为 `supervision_predictability_floor_weight`；
- 其他 blocks 的 $w_t$ 设为 `1`；
- step weights 会按均值归一化，避免整体 loss scale 大幅变化。

auxiliary loss 只作用在 learnable-hard blocks：

$$
\mathcal{L}_{unit}
=
\operatorname{MSE}(\hat{Y}_{learnable},Y_{learnable}).
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{time}
+
\lambda\mathcal{L}_{unit}.
$$

其中 $\lambda$ 是 `supervision_aux_weight`。

## Trace

`supervision_trace.csv` 增加：

| Column | Meaning |
| --- | --- |
| `predictability_learnable_blocks` | 当前 step 中被 auxiliary emphasis 的 block 数 |
| `predictability_noisy_blocks` | 当前 step 中被 floor weight 降权的 block 数 |
| `predictability_mean_weight` | 归一化后 step weight 均值，应接近 1 |
| `predictability_floor_weight` | noisy-hard block 的原始 floor weight |

已有字段仍保留：

- `unit_type=predictability_downweight`;
- `condition_type=novelty_x_variation`;
- `condition_top_blocks`;
- `condition_mean_score`;
- `loss_time`;
- `loss_unit`;
- `loss_total`。

## Code-Theory Consistency

[Theory] 如果 high novelty block 是 learnable-hard，应给予额外监督；如果 high novelty block
同时 high variation，则它更可能是 noisy-hard，强加压会污染 shared representation。

[Code Realization] 当前实现用 top novelty 与 top variation 的交集作为 noisy-hard proxy，
用 top novelty 去掉 high variation 后的 blocks 作为 learnable-hard proxy。

[Proxy Limit] `variation` 只是低可预测性的局部 proxy，不等价于严格 aleatoric uncertainty。
它不能证明某个 block 不可预测，只能作为 train-only、无泄漏的 first gate。

[Falsification] 如果 small gate 中 `predictability_downweight` 不能保留 CFUS 相对
`full_time_mse` 的收益，或者 Weather 仍相对 R.3 全面退化，则该 proxy 不足以支撑
paper-core，需要回退到更强的 predictability estimator 或 SRP-inspired isolation path。
