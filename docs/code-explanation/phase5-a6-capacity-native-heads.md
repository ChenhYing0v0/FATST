# Phase5-A6 Capacity-Native Heads Code Explanation

本文档说明 `baselines/timealign_official/models/TimeAlign.py` 中新增的 A6 readout modes：

- `prefix-native-dense-equivalent-row-bank`
- `learned-basis-forecast-operator`

## 共同 backbone flow

两个 mode 都复用 official TimeAlign backbone：

```text
batch_x: [B, L, C]
patch_emb_x -> x: [B*C, patch_num, d_model]
encoder layers -> x
reshape -> x_tokens: [B, C, patch_num, d_model]
flatten -> hidden: [B, C, R], where R = patch_num * d_model
```

A6 的目标是保留 direct requested-prefix output：给定 `target_prefix=H`，head 直接返回
`[B, H, C]`，而不是先生成 `[B, 720, C]` 后再 crop。

## A6-DER: Prefix-Native Dense-Equivalent Row Bank

### Parameters

```text
proj_x.weight: [720, R]
proj_x.bias: [720]
```

### Forward flow

```text
hidden: [B, C, R]
H = target_prefix or pred_len
weight_H = proj_x.weight[:H]: [H, R]
bias_H = proj_x.bias[:H]: [H]
output = linear(hidden, weight_H, bias_H): [B, C, H]
permute -> [B, H, C]
```

A6-DER 与 official dense head 有相同 row capacity，但调用方式是 prefix-native。它是
capacity ceiling / control，不是单独的 paper-core method。

### Prefix consistency

`decode(H)` 与 `decode(720)[:, :H]` 使用完全相同的 `proj_x.weight[:H]` 与 `proj_x.bias[:H]`。
因此 eval/no-dropout 下重叠 prefix 应达到 numerical zero mismatch。

## A6-LBF: Learned-Basis Forecast Operator

### Parameters

```text
basis_rank: K
learned_basis_coeff: [R] -> [K]
learned_temporal_basis: [720, K]
learned_temporal_bias: [720]
```

### Forward flow

```text
hidden: [B, C, R]
coeff = learned_basis_coeff(hidden): [B, C, K]
basis_H = learned_temporal_basis[:H]: [H, K]
bias_H = learned_temporal_bias[:H]: [H]
output = einsum("hk,bck->bch", basis_H, coeff) + bias_H: [B, C, H]
permute -> [B, H, C]
```

当 `K` 足够大时，`basis_H @ learned_basis_coeff.weight` 可表达 dense linear readout rows。
因此 A6-LBF 是 A5-B deterministic basis 的 capacity-native replacement：它仍是 prefix-consistent
forecast operator，但 temporal basis 不再被固定 Fourier/polynomial features 限制。

### Prefix consistency

每个 future row `t` 对应同一个 `learned_temporal_basis[t]` 和 `learned_temporal_bias[t]`。因此
不同 requested prefixes 的重叠 rows 使用同一参数路径。

## Code-Theory 一致性评估

### Intended theory

A5-Q/A5-B 失败说明 prefix consistency 本身不够；unified head 还必须保留 dense-level forecasting
capacity。A6 将 capacity preservation 放进 operator class：A6-DER 给出 dense-equivalent ceiling，
A6-LBF 给出 rank-controlled learned-basis approximation。

### Code realization

- A6-DER 使用 `proj_x.weight[:H]` 和 `proj_x.bias[:H]` 直接生成 requested prefix；
- A6-LBF 使用 learned temporal basis 与 linear coefficient 生成 requested prefix；
- 两者都不使用 pretrained checkpoint、teacher anchor 或 residual correction。

### 仍是 proxy 的部分

- A6-DER 是 control，不能单独构成 SCI contribution；
- A6-LBF 的实际 capacity 由 `basis_rank` 决定；`K=256/512` 是否足够必须由 remote gate 判断；
- 当前 smoke 只能证明 tensor contract 与 prefix consistency，不证明 forecasting effectiveness。

### 可证伪证据

- 若 A6-DER 仍明显弱于 best stage controls，说明 Stage A bottleneck 不只是 head operator capacity；
- 若 A6-LBF 明显弱于 A6-DER，说明 learned-basis rank 或 linear factorization 仍不足；
- 若 A6-LBF 接近 A6-DER 并超过 A5-B，说明 A5-B 主要失败点是 fixed basis under-capacity。
