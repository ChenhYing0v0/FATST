# Phase5 B14 Prerequisite Contextual Patch Encoder Code Explanation

## Scope

本次更新不修改 A6 learned-basis forecast operator 的数学形式，只替换可选 history encoder。legacy
`timealign-token-mlp` 保持默认，用于加载历史 checkpoints 与复现 accepted A6；新
`contextual-patch-transformer` 是等待 effectiveness gate 的 carrier candidate。

入口：

- `baselines/timealign_official/models/TimeAlign.py`；
- `baselines/timealign_official/train_repo.py`；
- `scripts/remote/run_phase5_stage_b_b14_contextual_patch_encoder_gate.sh`。

## Forward Computation

### 1. Normalize

原始输入：

```text
x: [B,720,C]
```

`Normalize(..., "norm")` 得到 `x_norm: [B,720,C]`。denormalization statistics 与 legacy A6 相同。

### 2. Overlapping Patches

`ContextualPatchEncoder.forward` 接收：

```text
x_norm.permute(0,2,1): [B,C,720]
```

右侧 replication padding `stride=S` 后，`unfold(K,S)`：

```text
patches: [B,C,P,K]
```

- `K=16,S=8 -> P=90`；
- `K=48,S=24 -> P=30`。

共享 `Linear(K,D)` 后：

```text
tokens: [B,C,P,D]
```

patch projection 跨 variables 与 patch positions 共享，保留 channel-independent contract。

### 3. Contextual Patch Mixing

channels 被临时并入 batch：

```text
[B,C,P,D] -> [B*C,P,D]
```

加入 learnable positional embedding `[1,P,D]`。每层执行：

```text
Q,K,V: [B*C,heads,P,d_head]
scores: [B*C,heads,P,P]
scores_l = QK^T / sqrt(d_head) + scores_(l-1)
attention output: [B*C,P,D]
residual + post BatchNorm
FFN(D -> d_ff -> D)
residual + post BatchNorm
```

`history_res_attention=False` 时不累积 previous scores，但默认 gate 使用 `True`。最后恢复：

```text
memory: [B,C,P,D]
```

### 4. A6 Readout

只有 A6 readout 前才 flatten：

```text
hidden = memory.flatten(-2): [B,C,P*D]
coeff = learned_basis_coeff(hidden): [B,C,256]
output = basis[:H] @ coeff + bias[:H]: [B,H,C]
```

因此 requested horizon `H` 仍只决定 basis prefix；history encoder 不读取 horizon IDs。

## Public History-Memory Interface

`Model.encode_history(x)` 统一返回 `[B,C,P,D]`：

- legacy A6：把 TimeAlign tokens reshape 成显式 memory；
- contextual candidate：直接返回 cross-patch contextual memory。

B14 analyzer 必须使用该接口，不再手工复制 `PatchEmbed/encoder` path。这避免 model forward 更新后
diagnostic 静默读取错误 intervention point。

## Training Configuration

新增 CLI：

- `--encoder-mode`；
- `--history-patch-len/--history-patch-stride`；
- `--history-d-model/--history-n-heads/--history-d-ff`；
- `--history-e-layers`；
- `--history-dropout/--history-attn-dropout`；
- `--history-res-attention`；
- `--learning-rate`。

contextual encoder 当前只允许：

```text
mode=unified
pred_len=720
readout_mode=learned-basis-forecast-operator
```

该约束防止它被误接到 TimeAlign future reconstruction/alignment 或已关闭的 StageB heads。

## Source-Derived Invariants

从 PatchTST adopted：

- end replication padding；
- overlapping patch projection；
- channel independence；
- learnable positional encoding；
- pre-softmax residual attention；
- post BatchNorm residual blocks。

没有追求 upstream output parity，因为本地输出 head、lookback 与 multi-prefix training contract 不同。

## Code-Theory Consistency

### Intended Theory

统一 history encoder 应生成跨 datasets 都存在的 contextual patch memory，使后续 future units 可以选择不同
history evidence，同时不把 benchmark horizons 写入 history representation。

### Code Realization

- 同一 patch rule 保证 `P>1`；
- self-attention 是显式 cross-patch information path；
- positional embedding 保留 patch order；
- `[B,C,P,D]` 在 A6 flatten 前公开；
- A6 prefix-native basis operator保持不变。

### Remaining Proxies

- contextual token 不等于 causal local segment；attention 也不等于 retrieval explanation；
- dataset-specific width 仍是 capacity hyperparameter；
- performance 尚未验证，legacy A6 仍是 active carrier。

### Falsification

设计被以下证据否定或要求回 Step 5/6：

- 3-dataset gate 明显低于 clean A6，且不是 numeric/optimization pathology；
- patch memory gradients non-finite 或 checkpoint reload 不稳定；
- token representations在 patch axis collapse，无法形成 B14 可区分的 demand/sensitivity profiles；
- 性能仅依赖某个 dataset-specific topology，而不是相同 patch-wise computation graph。

## Verification

本地 checker验证：

- legacy forward 与更新前的显式 computation path exact equal；
- `P16-S8`/`P48-S24` memory shape 为 `90/30`；
- prefix output shape、finite gradients、state-dict reload；
- contextual encoder 拒绝 official readout。

## Post-Gate Repair：Hierarchical Patch Memory

full contextual replacement未通过 performance gate。新增 `encoder_mode=hierarchical-patch-memory`，其
forecast computation与 `timealign-token-mlp` 完全相同；只新增无参数的 `CanonicalPatchMemory`：

```text
x_norm [B,720,C]
  -> permute [B,C,720]
  -> ReplicationPad1d(0,24)
  -> unfold(K=48,S=24)
  -> local_memory [B,C,30,48]
```

`Model.encode_history(x)` 仍返回 carrier memory；新增 `Model.encode_retrieval_memory(x)` 返回 B14 canonical
local memory。`CanonicalPatchMemory` 没有 parameters/buffers，因此 hierarchical model与 legacy A6：

- state-dict keys相同；
- parameter count相同；
- forecast forward逐元素相同；
- clean A6 checkpoints可 `strict=True` 加载。

这条 repair没有让 local memory进入预测，也没有加入 residual correction。B14通过 problem diagnostic 后，
才允许设计可训练 projection/retrieval及其 exact no-retrieval controls。
