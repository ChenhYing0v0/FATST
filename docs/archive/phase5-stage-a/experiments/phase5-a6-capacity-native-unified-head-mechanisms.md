# Phase5-A6: Capacity-Native Prefix-Consistent Unified Head Mechanisms

本文档接在 A5-Q/A5-B effectiveness failure 与 A5-Q collapse diagnostic 之后，推进 Phase5-A 的
Step 4/5。目标不是继续修补 A5-Q，而是重新提出能同时满足 prefix-native generation 与 dense-level
forecasting capacity 的 unified head mechanism。

## 11-step 状态

| Field | Content |
| --- | --- |
| `current_step` | Phase5-A6：Step 4/5，重新提出 capacity-native unified head mechanism |
| `problem` | A5-Q 的 target-query graph 解释性强、prefix contract 成立，但缺少能承接 TimeAlign dense/full-head forecasting capacity 的路径；A5-B 的 deterministic basis/operator 也显示 rank/operator class 上限不足。 |
| `existence_evidence` | A5-Q diagnostic 证明 ETTm1 dropout 错位只是 collapse amplifier：修复后仍 `0` win；A5-B rank 128 比 rank 64 好但仍相对 best stage control `+14.19%`；A3D teacher preservation partial pass 说明 function/capacity preservation 是真实关键变量。 |
| `idea` | 从 “query / basis 是否 prefix-consistent” 转为 “prefix-native head 是否具备 dense-equivalent capacity”。优先设计 anchor-free、非 residual、可直接请求 `[B,H,C]` 的 capacity-native readout family。 |
| `theory_check` | 一个 learned dense row table 的线性 head 本质是矩阵 $W \in \mathbb{R}^{720 \times R}$。若新 head 的 operator class 不能近似该矩阵族，就会重复 A5-Q/A5-B 的 capacity collapse。 |
| `design` | 提出 A6-DER dense-equivalent control 与 A6-LBF learned-basis forecast operator；A6-QBR 保留 target-query 叙事但只作为第二阶段。 |
| `narrative_gate` | A6-DER：control_only；A6-LBF：conditional_pass，需与 A6-DER 一起验证，不能单独声称 SCI contribution。 |
| `effectiveness_gate` | pending；必须先做 A6-DER/A6-LBF local smoke，再进入最小 remote gate。 |
| `artifacts` | 本文档；后续若实现，需同步 `docs/code-explanation/` 与 stage ledger。 |
| `decision` | A6-LBF 是下一轮优先实现候选；A6-DER 是必需 capacity ceiling/control。A6-QBR 暂缓，除非 A6-LBF 证明 learned dense-equivalent capacity 有效。 |

## 设计原则

[Rule] A6 不继承以下 A5 失败模式：

- 不再对 A5-Q 做 `dropout / patch_num / width` sweep；
- 不使用 pretrained dense rows 作为 anchor；
- 不使用 `base + residual/correction` 作为 paper-core 主体；
- 不把 teacher checkpoint 作为方法成立的必要条件；
- 不把 fixed 720 output 后 crop 伪装成 prefix-native generation。

[Required Contract] A6 head 必须满足：

1. requested prefix `H` 决定本次实际生成的 rows/tokens；
2. forward 直接返回 `[B,H,C]`；
3. `decode(96)` 与 `decode(720)[:, :96]` 在 eval/no-dropout 下由 architecture 保证一致；
4. capacity preservation 来自 operator class 本身，而不是 trained checkpoint；
5. 可解释为什么它比 A5-Q/A5-B 更接近 dense head 的 function class。

## Candidate A6-DER：Prefix-Native Dense-Equivalent Row Bank

### 核心 idea

[Idea] 保留 dense head 的完整 row-level capacity，但改成 prefix-native invocation：

```text
hidden: [B, C, R]
W: [720, R]
b: [720]
requested prefix H
y_hat_H = linear(hidden, W[:H], b[:H]): [B, C, H] -> [B, H, C]
```

它与 official full dense head 的参数容量等价，但不会在 forward 中先生成 `[B,720,C]` 再 crop。
训练时 multi-prefix loss 直接监督 requested prefix rows。

### 理论意义

[Fact] A6-DER 是 dense-equivalent upper bound/control，不是强 paper-core mechanism。它回答一个关键
诊断问题：若完全保留 dense row capacity 且只改变 prefix-native invocation，是否能接近或超过 H1/H1C/A3D controls？

若 A6-DER 仍失败，则 Stage A 的主要瓶颈可能不在 head capacity，而在 multi-prefix objective /
backbone representation conflict。若 A6-DER 成功，则说明 A5-Q/A5-B 的失败确实来自 operator class
不足，下一步才有资格研究压缩或结构化版本。

### Narrative gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | strong：直接测试 dense capacity 是否是必要条件 |
| mechanism novelty | weak-medium：更像 necessary control than final method |
| tensor/gradient path | strong：每个 prefix 直接训练对应 rows |
| contribution boundary | control_only：不能单独作为 SCI paper-core |
| implementation priority | required before A6-LBF remote interpretation |

## Candidate A6-LBF：Learned-Basis Dense-Equivalent Forecast Operator

### 核心 idea

[Idea] 将 A5-B 的 deterministic coordinate basis 替换为 learned temporal basis，并显式给出 dense-equivalent
capacity 上界：

```text
hidden: [B, C, R]
coeff = A(hidden): [B, C, K]
temporal_basis = B[:H]: [H, K]
y_hat_H = einsum("hk,bck->bch", temporal_basis, coeff) + bias[:H]
```

当 $K \ge \operatorname{rank}(W)$ 时，该结构可表达 dense linear head $W h$；当 $K < 720$ 时，它是
rank-controlled dense approximation。和 A5-B 的差异是：A5-B 使用 fixed polynomial/sinusoidal basis，
capacity 由 handcrafted basis 限制；A6-LBF 让 temporal basis 自身可学习，从而把失败点从
“basis 形状不够”转为可控的 rank-capacity tradeoff。

### Prefix consistency

同一个 future row `t` 总是使用同一个 `temporal_basis[t]` 与 `bias[t]`。因此不同 requested prefix
之间的重叠输出由同一组 row basis 生成，prefix consistency 是 architecture-level property。

### 与旧路线的区别

- 不依赖 pretrained dense rows；
- 不使用 residual/correction；
- 不使用 teacher checkpoint；
- 不把 benchmark horizon id 当离散类别；
- 比 A5-B 更接近 dense head function class，因为 basis 是 learned matrix，而不是固定 temporal features；
- 比 A5-Q 更直接保留 forecasting capacity，因为每个 future row 有 learned row-basis participation。

### Narrative gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | strong：针对 A5-B deterministic basis under-capacity 与 A5-Q readout capacity collapse |
| mechanism novelty | medium：rank-controlled prefix-native learned forecast operator |
| tensor/gradient path | strong：prefix rows 直接参与 loss，basis/coeff 均获得 prefix-specific gradient |
| capacity preservation | medium-high：$K=720$ 可 dense-equivalent；较小 $K$ 可测试 intrinsic rank |
| contribution boundary | conditional_pass：必须与 A6-DER ceiling 一起报告，避免被审稿人理解为普通 low-rank head |
| implementation priority | first method candidate |

### 风险与反证

- 若 `K=256/512` 仍明显弱于 A6-DER，说明 dense capacity 需要更高 rank 或非线性 row interaction；
- 若 `K=720` 才有效，A6-LBF 可能退化成 dense head reparameterization，SCI contribution 变弱；
- 若 A6-DER 也失败，则不能继续 head-only route，应回 Step 2/3 重审 multi-prefix objective 是否才是核心问题。

## Candidate A6-QBR：Query-Bilinear Readout

### 核心 idea

[Idea] 保留 A5-Q 的 future-query 表达，但把 query decoder 的 final readout 改成 dense-equivalent
bilinear operator：

```text
future_query_t: [Dq]
feature = P(hidden): [B, C, K]
row_key_t = G(future_query_t): [K]
y_hat_t = dot(row_key_t, feature) + b_t
```

这等价于 A6-LBF 的 query-parameterized form：query 不再承担 cross-attention 主体，而是承担
row-basis generation / indexing。它保留 target-query 可解释性，但 capacity path 来自 bilinear
readout，而不是低容量 attention decoder。

### Narrative gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | medium-strong：修复 A5-Q 缺少 dense-level readout 的问题 |
| mechanism novelty | medium：target-query semantics + dense-equivalent bilinear readout |
| tensor/gradient path | strong if implemented as direct row-bilinear operator |
| capacity preservation | medium-high，取决于 `K` 与 query-to-row mapping |
| contribution boundary | deferred：应等 A6-LBF 证明 learned basis path 有效后再实现 |

## 推荐执行顺序

| Priority | Candidate | Role | Reason |
| ---: | --- | --- | --- |
| 1 | `A6-DER_prefix_native_dense_equivalent_row_bank` | capacity ceiling / control | 必须先知道 dense-equivalent prefix-native head 是否能恢复 capacity。 |
| 2 | `A6-LBF_learned_basis_forecast_operator` | primary method candidate | 直接修复 A5-B 的 fixed-basis under-capacity，同时保持 anchor-free 和 prefix-native。 |
| 3 | `A6-QBR_query_bilinear_readout` | deferred method candidate | 只有当 A6-LBF 有效后，才值得把 target-query 叙事重新接入 dense-equivalent readout。 |

## 最小实验建议

[Design] 第一轮不做大 sweep，只做 capacity gate：

- `A6-DER`：dense-equivalent prefix-native row bank；
- `A6-LBF-r256`：learned basis rank 256；
- `A6-LBF-r512`：learned basis rank 512。

Gate universe 维持 `ETTh2 + ETTm1 + Weather` 和 `96/192/336/720`，与 A5-Q/A5-B 可比。

### 判定规则

1. 若 `A6-DER` 仍相对 `best_stage_control` 明显失败：回 Step 2/3，重审 Stage A 是否应从 head
   design 转向 multi-prefix objective / representation conflict。
2. 若 `A6-DER` 通过但 `A6-LBF` 不通过：说明 dense capacity 必要但低秩 learned basis 不足，
   后续可考虑 rank-adaptive 或 nonlinear row interaction。
3. 若 `A6-LBF-r512` 接近或超过 A6-DER / best controls：A6-LBF 进入 paper-core candidate gate，
   同步做 rank/basis diagnostics。
4. 若 `A6-LBF-r256` 与 `r512` 差距很小且均优于 A5-B：说明 A5-B 失败主要来自 fixed basis，
   A6-LBF 具备清晰 narrative。

## 当前决策

[Decision] 下一步应实现 A6-DER 与 A6-LBF 的 local smoke，不启动 A6-QBR。远程前必须更新
`docs/code-explanation/`，完成 shape/prefix-invariance smoke，并提交推送。
