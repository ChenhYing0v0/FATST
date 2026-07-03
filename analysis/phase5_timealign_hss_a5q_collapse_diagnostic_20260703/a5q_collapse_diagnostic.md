# Phase5-A5Q Collapse Diagnostic

## 当前问题

[Question] `A5-Q_elastic_causal_target_query_decoder` 的理论叙事很自然：TimeAlign encoder 输出
patch-wise representations，future query 通过 cross-attention 主动选择预测每个 future segment 所需
的历史信息，再用 causal target self-attention 保证 prefix consistency。但远程结果显示 A5-Q 是本轮
A5 中 collapse 最严重的 family：

- `a5q_seg48_small` 相对 `best_stage_control` 平均 MSE `+42.41%`，wins `0/12`；
- `a5q_seg24_wide` 相对 `best_stage_control` 平均 MSE `+55.62%`，wins `0/12`；
- `seg24_wide` 比 `seg48_small` 更差，说明简单加密 target segments / 加宽 FFN 没有解决问题。

本报告只诊断 A5-Q，不提出新的 remote candidate。

## 实际代码路径

### 共同 TimeAlign encoder

```text
batch_x: [B, L, C]
Normalize
PatchEmbed -> [B*C, patch_num, d_model]
TimeAlign encoder layers
reshape -> x_tokens: [B, C, patch_num, d_model]
```

### A5-Q head

代码实现为：

```text
memory = x_tokens.reshape(B*C, patch_num, d_model)
segment_count = ceil(H / S)
features = [segment_center / 720, segment_width / 720]
query = target_query_embed(features): [segment_count, d_model]
query = expand(query): [B*C, segment_count, d_model]
query = LayerNorm(query + cross_attn(query, memory, memory))
target = LayerNorm(query + causal_self_attn(query, query, query))
target = LayerNorm(target + FFN(target))
segments = target_segment_out(target): [B*C, segment_count, S]
output = reshape/trim/permute -> [B, H, C]
```

[Fact] A5-Q 的 direct output shape 与 prefix-invariance smoke 成立：

- `decode(96)` vs `decode(720)[:, :96]` mismatch 约 `4.77e-07`；
- 因此 collapse 不是明显 shape bug，也不是 eval-time prefix contract 破坏。

## 关键错位 1：部分数据集没有 patch-wise memory

[Fact] A5-Q 的核心叙事依赖 `memory: [B*C, patch_num, d_model]` 中存在多个 history patch tokens。
但本轮 official preset 下：

| Dataset | `patch_num` | `d_model` | `dropout` | A5-Q implication |
| --- | ---: | ---: | ---: | --- |
| ETTh2 | 48 | 32 | 0.1 | 有 patch-wise memory，但 token dim 很小 |
| ETTm1 | 1 | 256 | 0.9 | cross-attention 退化为单 token 读取 |
| Weather | 48 | 128 | 0.1 | 有 patch-wise memory |

[Strong Evidence] 对 `ETTm1`，`patch_num=1` 时 cross-attention 没有可选择的 patch。softmax 只有一个
key，attention weight 恒为 1；cross output 对所有 future queries 基本是同一个 memory value 的变换。
因此“future query 选择不同历史 patch 信息”的核心机制在 ETTm1 上不成立。

[Interpretation] 这解释了为什么 ETTm1 上 A5-Q collapse 极端严重：

- `a5q_seg48_small` 相对 official unified：H96 `+112.54%`，H720 `+57.05%`；
- `a5q_seg24_wide` 相对 official unified：H96 `+151.20%`，H720 `+79.27%`。

## 关键错位 2：dropout 继承 official preset，对 query decoder 过强

[Fact] A5-Q 把 `configs.dropout` 同时用于：

- `target_cross_attn`；
- `target_self_attn`；
- `target_query_ffn`。

[Strong Evidence] `ETTm1` official preset 在 unified 720 下使用 `dropout=0.9`。这个 dropout 对原
TimeAlign dense head 主要影响 encoder/autoencoder block；但对 A5-Q，它直接作用在新的 attention
decoder 内部。也就是说，A5-Q 在 ETTm1 上不仅 cross-attention 退化为单 token，而且 decoder 内部还有
极强随机失活。

[Hypothesis] `seg24_wide` 比 `seg48_small` 更差，很可能与此有关：更宽 FFN 与更多 target segments
在 `dropout=0.9` 下只增加了高噪声训练路径，没有增加有效可用容量。

## 关键错位 3：A5-Q 对 ETTh2/Weather 的 head capacity 明显低于 dense head

参数量诊断：

| Dataset | dense `proj_x` params | `a5q_seg48_small` head params | ratio | `a5q_seg24_wide` head params | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 1,106,640 | 19,728 | 0.018 | 43,896 | 0.040 |
| ETTm1 | 185,040 | 672,688 | 3.635 | 863,512 | 4.667 |
| Weather | 4,424,400 | 188,976 | 0.043 | 284,568 | 0.064 |

[Strong Evidence] 对 ETTh2/Weather，A5-Q head 参数只有 dense head 的约 `1.8%~6.4%`。它虽然有
更强的结构归纳偏置，但没有足够参数自由度替代 dense full-head rows。

[Interpretation] 这与结果吻合：ETTh2/Weather 上 A5-Q 没有 ETTm1 那么灾难，但仍显著弱于 controls。
该结果更像 under-capacity / under-conditioned decoder，而不是单纯初始化坏点。

## 关键错位 4：segment query 太弱，不能表达 step-specific readout

[Fact] 每个 target segment 的 query 只由两个 scalar 构造：

```text
[segment_center / 720, segment_width / 720]
```

这两个 scalar 经过 `target_query_embed` 得到 `d_model` query。query 没有：

- calendar/time features；
- horizon-specific observed context；
- local trend / last value anchor；
- channel identity embedding；
- dense-row prior；
- learned step row function；
- teacher/student preservation path。

[Hypothesis] 这导致 query 在早期训练中只能学习非常弱的 coordinate-conditioned readout。相比
`proj_x(hidden)` 直接为每个 future step 学一行权重，A5-Q 要先学习 query embedding、cross-attention、
self-attention 和 segment output 的组合函数，优化路径更长。

## 不是主要原因的项

### 不是 eval-time prefix consistency bug

[Fact] smoke mismatch 接近 numerical zero。

### 不像单纯“query 初始化问题”

[Evidence] 若只是初始化不足，`seg24_wide` 或更大 FFN 至少应在部分 setting 上改善。但结果是
`seg24_wide` 对 `seg48_small` 为 `0/12` wins，ALL 相对变差 `+7.76%`。

[Interpretation] 初始化可能放大早期训练难度，但主要问题更像 architecture/configuration mismatch：
memory token 数、dropout、query feature 弱、dense capacity 丢失共同造成 collapse。

### 不像普通 overfitting

[Evidence] A5-Q 的 train loss 从 epoch 1 到 epoch 10 有下降，但 val MSE 仍高，尤其 ETTm1 基本维持在
`0.92~0.99` 量级。它没有表现为 train 很低而 val 崩，更像有效拟合能力不足或优化路径噪声太大。

## 结论

[Decision] A5-Q 的理论叙事成立，但当前实现并没有真正实现“future query 选择 patch-wise 历史信息”
这一机制在所有数据集上的有效条件。

更具体地说：

1. 对 ETTm1，`patch_num=1` 使 cross-attention 机制退化，且 `dropout=0.9` 严重破坏 query decoder；
2. 对 ETTh2/Weather，虽有 `patch_num=48`，但 A5-Q head 参数远少于 dense head，且 query feature 太弱；
3. `seg24_wide` 变差说明不能通过简单加密 target query 或加宽 FFN 修复；
4. 本轮 collapse 更像 implementation-theory mismatch + capacity path 缺失，而不是单一 shape bug 或初始化 bug。

## 后续诊断建议

[Next Diagnostic Only] 若继续研究 A5-Q，应先做离线/小规模 diagnostic，而不是远程完整 gate：

1. 对 `patch_num=1` 数据集禁用 A5-Q，或强制使用更细 patch_num，验证 cross-attention 是否恢复有效；
2. 固定 A5-Q decoder dropout 为独立小值，例如 `0.0/0.1`，不要继承 official preset 的 `0.9`；
3. 导出 cross-attention weights，检查 query 是否真的选择不同 history patches；
4. 加一个 dense-head teacher projection diagnostic，只比较 A5-Q 能否拟合 trained dense head 输出，而不是直接拟合 labels；
5. 检查 per-segment output variance 与 target variance，判断 collapse 是过平滑、过噪声还是 segment bias。

[Rule] 上述建议目前只能作为 diagnostic。不能把它们直接升级为 paper-core candidate；若某项诊断显示机制真实存在，再回 Step 4/5 重写 A5-Q 的 narrative gate。
