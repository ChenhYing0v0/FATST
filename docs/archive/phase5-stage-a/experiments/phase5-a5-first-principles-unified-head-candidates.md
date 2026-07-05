# Phase5-A5: First-Principles Unified Head Candidates

## 当前步骤

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5：提出多个 first-principles unified head idea，并进行理论与 narrative 预评估 |
| `stage` | Phase5-A5 |
| `active_carrier` | official-source TimeAlign |
| `status` | `candidate_proposal` |
| `previous_rejection` | `A5_pcf_prefix_consistent_function_preserving_decoder` 已被判定为 `narrative_rejected_after_review` |

## 设计边界

[Rule] 本轮只提出候选，不允许直接实现。任何候选进入 Step 7 前，必须单独通过 narrative gate。

[Decision] A5 的 unified head 不能再满足于以下形式：

- 从 trained full-head rows 中选择或拷贝参数；
- `base + residual/correction` 作为 paper-core 主体；
- teacher/distillation 是方法成立的必要条件；
- 在 H1/H1C/A2/A3C/A3D/A3E existing paths 之间做 selector；
- 把 benchmark horizon id 作为离散类别输入。

[Required Contract] 新候选必须满足：

1. requested prefix 进入主生成路径；
2. output 是 direct `[B, H, C]`，不是先生成 720 再 crop；
3. prefix consistency 来自 architecture 或主生成 contract，而不是事后 regularizer alone；
4. capacity preservation 来自参数化或 operator class，而不是 pretrained dense anchor；
5. 能解释为什么它比 A2/A3D/A3E 更根本，而不是旧机制组合。

## 调研依据

[Strong Evidence] 本轮候选来自项目内 Zotero notes 与历史 diagnostics：

- ElasTST：varied-horizon inference 的关键是 horizon-invariance、future placeholders 和 structured
  mask；这支持“prefix consistency 应进入 architecture”。
- TIMEPERCEIVER：target query 可以显式指定任意 target timestamps；这支持“requested future
  positions 应进入 decoder query”。
- SRP++：step-invariant representation 可能存在 expressiveness bottleneck；这支持“unified head
  不应只有一个 shared latent + dense rows”。
- TransDF：forecast horizon 变长会增加 step-wise task conflict；这支持“training units 不应等同于
  evaluation horizons”。
- output/error-process diagnosis：当前问题有明显 step-region error-process 结构；这支持“head 应
  生成 trajectory/process，而不是独立输出点”。

## Candidate A5-Q：Elastic Causal Target-Query Decoder

### 核心 idea

[Idea] 用 future position queries 作为主生成单元。对于 requested prefix `H`，构造
`q_1, ..., q_H`，每个 query 对应一个 future step 或小 segment。query 通过 cross-attention 从
TimeAlign history tokens 中读取信息，再通过 causal / prefix-structured self-attention 建模 target
positions 之间的依赖。

### Tensor contract

```text
TimeAlign encoder output:
  x_tokens: [B, C, patch_num, d_model]

reshape by channel:
  memory: [B*C, patch_num, d_model]

requested prefix:
  H -> target_queries: [B*C, H_or_segments, d_model]

decoder:
  target_queries cross-attend memory
  target_queries self-attend with prefix-causal mask

readout:
  y_hat_H: [B, H, C]
```

### Prefix consistency

若 query self-attention 使用 causal mask，且 query embedding 使用 absolute future position
`t / 720`，那么请求 `H=96` 与请求 `H=720` 时，前 96 steps 的 query 可见信息相同。这样 prefix
consistency 是 architecture-level property，而不是只靠 `L_cons` 修补。

### 与旧路线的区别

- 不使用 dense 720 rows；
- 不使用 pretrained anchor；
- 不用 residual correction；
- 和 H1B `prefix-token-decoder` 的区别是：H1B 是较弱的 per-prefix readout，缺少 target-query
  interaction / structured mask / prefix-invariance contract；A5-Q 的核心是 target set 作为一个
  masked output graph。

### Narrative gate 预评估

| Gate Item | 预评估 |
| --- | --- |
| direct multi-prefix generation | strong |
| prefix consistency | strong，若 mask 与 position encoding 设计正确 |
| capacity preservation | medium，需要证明 target-query decoder 的容量足以替代 dense rows |
| target-prefix awareness | strong，prefix request 通过 query set 与 mask 进入主路径 |
| SCI story | strong：从 full-head crop 转为 prefix-elastic target-query decoding |

### 风险

- 参数量和训练成本可能高于现有 head；
- 若 query decoder 太弱，会重复 H1B capacity collapse；
- 若 query 使用全局 `H` embedding 影响每个 step，可能破坏 prefix invariance。

### 最小验证

1. local shape smoke：`H=96/192/336/720` 输出均为 `[B,H,C]`；
2. architecture prefix-invariance check：同一 batch 下 `decode(96)` 与 `decode(720)[:, :96]`
   在无 dropout eval mode 下应接近；
3. remote gate 前必须有参数量对齐 control，例如 `A5-Q-small` 与 `A5-Q-wide`。

## Candidate A5-B：Continuous Forecast-Basis Operator

### 核心 idea

[Idea] 把 unified head 写成一个从 future coordinate 到 forecast value 的 continuous operator。
TimeAlign hidden state 生成一组 basis coefficients；requested prefix 只决定在哪些 future coordinates
上 evaluation。

### Tensor contract

```text
z: [B, C, R]
coeff = f_theta(z): [B, C, K]
tau_H = [1/720, 2/720, ..., H/720]
basis(tau_H): [H, K]
y_hat_H = basis(tau_H) @ coeff: [B, H, C]
```

其中 `basis` 可以是 learnable Fourier / spline / multi-resolution temporal basis，也可以是
coordinate MLP 的隐式 basis。

### Prefix consistency

同一个 absolute coordinate `tau=t/720` 在不同 requested prefixes 下使用同一个 `basis(tau)`。
因此 `decode(96)` 与 `decode(720)[:, :96]` 在 architecture 上天然一致。

### 与旧路线的区别

- 不需要 dense rows；
- 不需要 pretrained anchor；
- 不是 segment residual；
- requested prefix 通过 coordinate grid 进入主生成路径；
- capacity preservation 来自 basis rank / multi-resolution basis，而不是 checkpoint。

### Narrative gate 预评估

| Gate Item | 预评估 |
| --- | --- |
| direct multi-prefix generation | strong |
| prefix consistency | strong |
| capacity preservation | medium，取决于 basis rank 和 multi-resolution capacity |
| target-prefix awareness | medium：prefix 通过 coordinate grid 进入，不是通过离散 horizon id |
| SCI story | strong：把 unified head 从 row table 改为 forecast function/operator |

### 风险

- 低秩 basis 可能损伤 high-frequency / short-horizon 细节；
- basis rank 若过大，可能退化成 disguised dense head；
- coordinate MLP 若没有结构约束，解释性会弱。

### 最小验证

1. 先做 small/medium rank 两档，而不是宽 sweep；
2. 必须报告 component/basis energy 与 per-step error；
3. gate 重点看 h96 是否不因平滑 basis 被损伤，以及 h720 是否不发生 late drift。

## Candidate A5-I：Cumulative Innovation Process Decoder

### 核心 idea

[Idea] 不直接独立预测每个 future value，而是预测 future trajectory 的 innovation process：
`delta_1, ..., delta_H`。最终输出由 cumulative operator 得到：

```text
y_hat_t = y_base + sum_{i=1}^{t} delta_i
```

这里的 `delta_i` 可以是 step-level，也可以是 segment-level 后再 upsample。它把 output/error-process
diagnosis 中的 step-region 结构纳入 head 本身。

### Tensor contract

```text
z: [B, C, R]
innovation_queries_H: [H_or_segments, d_model]
delta_H = decoder(z, innovation_queries_H): [B, H, C]
y_hat_H = cumulative(delta_H): [B, H, C]
```

`y_base` 可以来自 last observed value、normalized zero baseline 或 learned initial state，但不能来自
pretrained full-head prediction。

### Prefix consistency

cumulative operator 对 prefix 天然闭合：`H=96` 的 trajectory 就是 `H=720` trajectory 的前 96 个
innovations 的积分，只要 innovation decoder 不使用破坏 prefix-invariance 的 global `H` embedding。

### 与旧路线的区别

- 不是 dense-row readout；
- 不是 full prediction residual；
- 直接把 forecast trajectory / error process 作为解码对象；
- capacity 来自 innovation process 的 temporal structure，而不是 full-head rows。

### Narrative gate 预评估

| Gate Item | 预评估 |
| --- | --- |
| direct multi-prefix generation | strong |
| prefix consistency | strong |
| capacity preservation | medium-low，需要证明 cumulative parameterization 不会导致 drift |
| target-prefix awareness | medium，prefix 通过 innovation length 和 mask 进入 |
| SCI story | medium-strong：与 error-process evidence 对齐，但要避免变成普通差分预测 |

### 风险

- cumulative sum 可能放大 bias，尤其 h720；
- 若 delta 预测太平滑，short horizon 可能受损；
- 若 y_base 设计不当，会引入新的 confounder。

### 最小验证

1. 必须输出 segment-level drift diagnostics；
2. 与 direct value query decoder 做 control；
3. 只有在 h720 late segment 不系统性恶化时才进入远程完整 gate。

## Candidate A5-S：Step-Specific Hypernetwork Head

### 核心 idea

[Idea] 使用一个小型 hypernetwork 根据 future coordinate `tau=t/720` 生成 readout weights：

```text
W_t, b_t = hypernet(phi(t/720))
y_hat_t = linear(z, W_t, b_t)
```

它保留 direct readout 的强表达形式，但权重不是 dense table rows，而是由共享 hypernetwork 连续生成。

### Prefix consistency

同一个 `t/720` 总是生成同一组 readout weights，因此不同 requested prefixes 的重叠部分一致。

### 与旧路线的区别

- 不复制或依赖 pretrained dense rows；
- 不使用 residual；
- step-specificity 来自共享函数 `hypernet(phi(t))`，不是 720 个独立 rows。

### Narrative gate 预评估

| Gate Item | 预评估 |
| --- | --- |
| direct multi-prefix generation | strong |
| prefix consistency | strong |
| capacity preservation | medium-high，接近 dense row capacity 但有共享生成约束 |
| target-prefix awareness | medium，prefix 通过 coordinate set 进入 |
| SCI story | medium：容易被认为是 generated dense head，需要强调 continuous/shared row function |

### 风险

- 如果 hypernetwork 过大，会被审稿人看成 dense rows 的重参数化；
- 如果过小，可能 capacity collapse；
- 和 SRP++ 的 step-specific representation 接近，但缺少 representation-side adaptation。

### 最小验证

1. 参数量必须严格控制，避免 disguised dense table；
2. 检查 generated weights 的 smoothness / rank；
3. 与 A5-B basis operator 对照：若两者类似性能，优先 A5-B，因为 narrative 更干净。

## Candidate A5-M：Masked Future Placeholder Head

### 核心 idea

[Idea] 受 ElasTST 启发，将 future placeholders 加入 decoder 输入。requested prefix `H` 决定 placeholder
tokens 数量与 structured mask；model 只生成 visible prefix placeholders。

### Tensor contract

```text
history memory: [B*C, patch_num, d_model]
future placeholders: [B*C, H_patches, d_model]
structured mask: placeholder cannot leak from unavailable future tokens
decoder output placeholders -> y_hat_H: [B, H, C]
```

### Prefix consistency

structured mask 保证额外的 longer-horizon placeholders 不改变 already requested prefix 的信息流。

### Narrative gate 预评估

| Gate Item | 预评估 |
| --- | --- |
| direct multi-prefix generation | strong |
| prefix consistency | strong if mask is correct |
| capacity preservation | medium |
| target-prefix awareness | strong |
| SCI story | medium-strong，但与 ElasTST 过近，需要明确本项目只抽取 head contract |

### 风险

- 与 ElasTST 机制相似度高，需避免变成复刻；
- TimeAlign already has training-only future branch，placeholder decoder 可能与 future reconstruction branch
  的职责冲突；
- 实现成本较高。

### 最小验证

先作为 `diagnostic_only` 检查 mask/invariance，不建议作为第一实现候选。

## 候选排序

| Rank | Candidate | 建议角色 | 理由 |
| ---: | --- | --- | --- |
| 1 | `A5-Q_elastic_causal_target_query_decoder` | first paper-core candidate | 与 TIMEPERCEIVER/ElasTST/SRP++ 都有机制连接；requested prefix 进入主 query graph；最像真正 unified head |
| 2 | `A5-B_continuous_forecast_basis_operator` | parallel paper-core candidate | 叙事最干净：forecast function/operator；实现相对可控；需防 underfit |
| 3 | `A5-S_step_specific_hypernetwork_head` | capacity-oriented candidate/control | 最接近 dense capacity 但不依赖 pretrained rows；需防被视作 dense row reparameterization |
| 4 | `A5-I_cumulative_innovation_process_decoder` | mechanism candidate/control | 与 output-process 诊断吻合；风险是 long-horizon drift |
| 5 | `A5-M_masked_future_placeholder_head` | diagnostic/backlog | narrative 强但与 ElasTST 太近且实现成本高，暂不作为第一实现 |

## 推荐下一步

[Decision] 不直接实现任何候选。下一步应先对 Rank 1 和 Rank 2 分别写 narrative gate mini-note：

1. `A5-Q_elastic_causal_target_query_decoder`
2. `A5-B_continuous_forecast_basis_operator`

选择标准：

- 若优先追求 SCI 机制新颖性与 prefix-native architecture，优先 A5-Q；
- 若优先追求实现可控、快速 gate、避免再次 capacity collapse，优先 A5-B；
- 若二者 narrative gate 都通过，先实现 A5-B local smoke，再实现 A5-Q；因为 A5-B 更容易做最小
  shape/invariance 验证。

## 不应做的下一步

- 不实现 PCF；
- 不重新引入 pretrained dense anchor；
- 不做 teacher weight sweep；
- 不用 A4S signal 继续做 existing-path routing；
- 不把 Stage B routing 提前为主方法。
