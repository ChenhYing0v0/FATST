# Phase1 Segment Query Head

## 目的

`PatchEncoderSegmentQueryHead` 是 Phase1-A 的最小 future-side decoder。它保留
`PatchEncoderFixedHead` 的 patch encoder，只替换 output head，用 future segment queries
生成 segment-level hidden states。

该实现用于验证：

```text
fixed flatten head -> future segment query decoder
```

是否能在 one-to-one horizon training 下改善 forecast quality 或 segment-level error profile。

## Forward Flow

Input:

```text
x: [B, L, C]
```

RevIN normalization:

```text
x_norm: [B, L, C]
```

Channel-independent patching:

```text
x_norm.permute -> [B, C, L]
reshape -> [B*C, 1, L]
ReplicationPad1d -> [B*C, 1, L + stride]
unfold -> patches: [B*C, N, patch_len]
```

Patch encoder:

```text
patch_embedding(patches) + pos_embedding -> z: [B*C, N, d_model]
TransformerEncoder(z) -> z: [B*C, N, d_model]
```

Future segment decoder:

```text
segment_queries: [1, J, d_model]
expand -> query: [B*C, J, d_model]
CrossAttention(query, z, z) -> U: [B*C, J, d_model]
FFN(U) -> U: [B*C, J, d_model]
```

Here:

```text
J = ceil(pred_len / segment_len)
```

Segment output:

```text
segment_head(U) -> [B*C, J, segment_len]
reshape/crop -> [B*C, pred_len]
reshape/permute -> y: [B, pred_len, C]
```

RevIN denormalization returns the final prediction:

```text
y: [B, pred_len, C]
```

## 与 FixedHead 的关键差异

`PatchEncoderFixedHead` uses:

```text
Flatten(z): [B*C, N*d_model]
Linear(N*d_model, pred_len)
```

`PatchEncoderSegmentQueryHead` uses:

```text
learnable segment queries -> cross-attention over z -> shared segment head
```

[Inference] 这使 output side 出现显式 future segment states `U`。后续 Phase2 可以对 `U`
做 future-aware alignment，Phase3 可以在 `U` 上做 segment-level routing。

## 当前实现边界

- 仍然是 one-to-one horizon training。
- 不支持 mixed-horizon training。
- 不引入 future teacher branch。
- 不引入 MoE。
- `--max-train-batches` 与 `--max-eval-batches` 只用于 smoke verification；正式实验保持默认
  `None`，遍历完整 train/val/test loader。
- `segment_query_similarity.csv` 只记录 learnable query parameters 的 cosine similarity，
  不是完整 hidden-state diagnostic；后续应补充 batch-level `U` similarity。

## Code-Theory Consistency

Intended theory:

> fixed head 缺少 future-side states；segment query decoder 为不同 future segments 提供
> 显式 readout states。

Code realization:

- `segment_queries` 是每个 future segment 的 learnable query。
- `SegmentDecoderBlock` 通过 cross-attention 从 encoded patch states `z` 读取信息。
- `segment_head` 将每个 segment state 映射为 `segment_len` 个 future values。

Proxy boundary:

- 当前 `segment_queries` 只按 segment index 区分，不包含真实 timestamp 或 horizon condition。
- 当前 decoder 的 segment readout 是 shared linear head，可能 underfit fixed head 的大参数矩阵。

Falsification evidence:

- 如果 MSE/MAE 全面弱于 fixed head，且增加合理参数控制后仍失败，则 simple segment query
  decoder 不足以作为论文核心。
- 如果 segment query similarity 接近完全同质，且 segment-level metrics 无改善，则 future segment
  states 没有形成有效分化。

## Remote Runner

`scripts/remote/run_phase1_segment_decoder_gate.sh` runs the Phase1-A gate:

```text
PatchEncoderFixedHead
PatchEncoderSegmentQueryHead
× {ETTh2, ETTm1, Weather}
× {96, 192, 336, 720}
```

Default output root:

```text
/home/yingch/exp_outputs/r-2026-fatst/phase1_future_segment
```

The runner accepts:

- `GPU_IDS="1"` for single-GPU sequential execution.
- `GPU_IDS="1 2"` for two concurrent jobs, one per listed GPU slot.

[Important] The project rule still requires checking `nvidia-smi` before launching remote
experiments. The runner prints a GPU snapshot at start, but it does not replace the pre-run
occupancy decision.
