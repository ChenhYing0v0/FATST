# Phase5-A5 Unified Heads Code Explanation

本文档说明 `baselines/timealign_official/models/TimeAlign.py` 中新增的 A5 unified head modes：

- `continuous-forecast-basis-operator`
- `elastic-causal-target-query-decoder`

## 共同入口

两个 mode 都复用 official TimeAlign backbone：

```text
batch_x: [B, L, C]
patch_emb_x -> x: [B*C, patch_num, d_model]
encoder layers -> x
reshape -> x_tokens: [B, C, patch_num, d_model]
```

与旧 `official` head 的关键区别是：A5 mode 不先生成 fixed `[B, 720, C]` 再 crop，而是在
`target_prefix=H` 下直接返回 `[B, H, C]`。因此训练循环中的 multi-prefix loss 会对每个 requested
prefix 调用一次 model，并直接监督对应 prefix。

## A5-B: Continuous Forecast-Basis Operator

### 参数

```text
basis_rank: K
basis_coeff: [B, C, d_model * patch_num] -> [B, C, K]
```

### Forward flow

```text
x_tokens: [B, C, patch_num, d_model]
hidden = flatten(x_tokens): [B, C, R]
coeff = basis_coeff(hidden): [B, C, K]
tau_H = [1 / 720, ..., H / 720]
basis(tau_H): [H, K]
output = einsum("hk,bck->bch", basis, coeff): [B, C, H]
permute -> [B, H, C]
```

`basis(tau_H)` 是 deterministic multi-resolution coordinate features，包含 constant /
polynomial terms 与 sinusoidal terms。它没有 pretrained dense row table，也不从 `proj_x` 拷贝权重。

### Prefix consistency

同一个 future coordinate `t / pred_len` 在 `H=96` 与 `H=720` 请求中生成同一 basis row。因此
在 eval/no-dropout 下，`decode(96)` 与 `decode(720)[:, :96]` 应接近 numerical zero mismatch。

## A5-Q: Elastic Causal Target-Query Decoder

### 参数

```text
target_query_segment_len: S
target_query_embed: [segment_center, segment_width] -> [d_model]
target_cross_attn: query attends to TimeAlign history tokens
target_self_attn: causal target-target attention
target_segment_out: [d_model] -> [S]
```

### Forward flow

```text
x_tokens: [B, C, patch_num, d_model]
memory = reshape(x_tokens): [B*C, patch_num, d_model]
segment_count = ceil(H / S)
segment_features: [segment_count, 2]
query = target_query_embed(segment_features): [segment_count, d_model]
query = expand(query): [B*C, segment_count, d_model]
cross = target_cross_attn(query, memory, memory)
query = LayerNorm(query + cross)
target = target_self_attn(query, query, query, causal_mask)
target = LayerNorm(query + target)
target = LayerNorm(target + target_query_ffn(target))
segments = target_segment_out(target): [B*C, segment_count, S]
trim to H -> [B, C, H]
permute -> [B, H, C]
```

这里的 target query 使用 absolute segment coordinate，不使用 global horizon id。causal mask 保证
较短 prefix 中的 target segments 不会受到较长 horizon 的 future segments 影响。

### Prefix consistency

若 `H=96` 与 `H=720` 的重叠 segments 使用同一 coordinate，并且 self-attention 是 causal 的，
则 `H=720` 中前几个 segments 的可见 target graph 与 `H=96` 相同。segment length 选择 48 或 24
时，当前 evaluation horizons 都是 segment boundary 的整数倍。

## Code-Theory 一致性评估

### Intended theory

A5 的目标是提供 first-principles unified prediction architecture：requested prefix 进入主生成路径，
输出是 direct `[B,H,C]`，prefix consistency 是 architecture contract，而不是事后 regularizer。

### Code realization

- A5-B 通过 coordinate basis/operator 实现 forecast function；
- A5-Q 通过 target query graph 与 causal mask 实现 prefix-elastic decoder；
- 两者都不读取 trained dense rows，不使用 `base + residual/correction`，也不要求 teacher checkpoint。

### 仍是 proxy 的部分

- A5-B 的 capacity preservation 目前由 `basis_rank` 控制，是否足够替代 dense head 需要远程结果验证；
- A5-Q 的 capacity preservation 由 query segment length、decoder width 和 attention layers 近似控制，
  不是数学意义的 full-head function preservation；
- 当前 smoke 只能证明 shape 与 prefix-invariance，不证明 forecasting performance。

### 可证伪证据

- 若本地 smoke 中 `decode(96)` 与 `decode(720)[:, :96]` mismatch 明显大于 numerical tolerance，
  说明 architecture contract 实现错误；
- 若 A5-B-r128 明显强于 A5-B-r64，说明 basis bottleneck 真实存在；
- 若 A5-Q-seg24-wide 明显强于 A5-Q-seg48-small，说明 target-query family 受 capacity 限制；
- 若四个 arms 都弱于 H1/H1C/A3D 且无 segment-level 修复信号，A5-Q/A5-B 应回 Step 4/5，而不是继续堆机制。
