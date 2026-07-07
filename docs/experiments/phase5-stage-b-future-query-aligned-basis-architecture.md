# Phase5 StageB B8 Future-Query Aligned Basis Architecture

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B8-FQA` |
| `current_step` | Step 1-2：文献调研与 architecture problem proposal |
| `problem` | StageA A6-LBF-r256 已经统一 decoder/head，但 sample-specific coefficient 对 horizon/position 不变；future positions 只通过全局 learned basis 区分，缺少 target-position-aware representation |
| `existence_evidence` | 目前只有 code-theory evidence 与文献动机，仍需 `B8-OCD` coefficient-space oracle diagnostic |
| `idea` | 引入 future-position query/placeholder tokens，使其 attend 到 history tokens，并在 learned-basis operator 前生成 target-position-aware coefficient modulation |
| `theory_check` | 这是 architecture-level 路线，直接深化 unified prediction；不恢复 generic TimeAlign auxiliary loss，也不转向 channel-correlation modeling |
| `design` | 仅为候选设计，尚未实现 |
| `narrative_gate` | `promising_not_passed`：有论文叙事潜力，但必须先通过 `B8-OCD` |
| `effectiveness_gate` | 未评估 |
| `artifacts` | 本 protocol；`analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` |
| `decision` | `proposed_architecture_candidate`；`B7-UPO` 降级为 small objective contribution candidate |

## 为什么 StageB 不应停在 B7

[Inference] `B7-UPO` 的问题是真实且有用的：当前 nested multi-prefix objective 会造成 prefix/segment 权重不均。但它本质上是 objective/training refinement，较难支撑第二个论文主创新点。

[Decision] StageB 应先寻找 architecture-level 第二贡献。因为 StageA 已经改变 forecast operator/readout，StageB 更合理的叙事是改变 feeding this operator 的 representation interface。

## 外部网络调研范围

[Fact] 本轮调研没有只依赖 Zotero 或本地 notes。外部网络调研使用了以下 primary / near-primary sources：

- TimeAlign arXiv HTML：<https://arxiv.org/html/2509.14181v3>
- TimeAlign arXiv abstract：<https://arxiv.org/abs/2509.14181>
- TimeAlign official code repository：<https://github.com/TROUBADOUR000/TimeAlign>
- ElasTST arXiv abstract：<https://arxiv.org/abs/2411.01842>
- TimePerceiver arXiv abstract / HTML：<https://arxiv.org/abs/2512.22550>，<https://arxiv.org/html/2512.22550v1>
- TimePerceiver official code repository：<https://github.com/efficient-learning-lab/TimePerceiver>

[Fact] SRP++ 证据来自本地 paper note `Papers/srp-step-specific-representation.md`。本轮访问 OpenReview 时遇到浏览器验证阻挡，因此 SRP++ 不作为外部网络已核验来源，只作为本地 note 辅助证据。

## 文献证据

### TimeAlign

[Fact] TimeAlign 认为 history-only forecasting 存在 past/future distribution mismatch，并通过 training-time future reconstruction branch 与 local/global representation alignment 来弥合预测表示和未来目标之间的差异。其 arXiv 摘要还明确强调 reconstruction-based alignment、frequency mismatch correction 与 mutual-information motivation。

[Implication] 对本项目有用的是问题意识：prediction representation 需要更 future-aware。但原始机制不适合作为 StageB 主线，因为 B4 已经显示 inherited align/recon branch 对 clean A6-LBF 性能不是必要条件；直接恢复 `w_align * align_loss` 会削弱本论文的 contribution boundary。

### ElasTST

[Fact] ElasTST 针对 varied-horizon forecasting 使用 future placeholders 与 structured self-attention masks，使 extending inference horizon 不改变已有 future outputs，并通过 horizon reweighting 训练多 horizon 能力。

[Implication] future positions 可以作为 tokens/placeholders 进入模型，且不需要泄漏 future values。若 B8 后续实现，必须保留 prefix/horizon invariance，不能让更长 horizon 的 query 改写较短 prefix 的输出。

### TimePerceiver

[Fact] TimePerceiver 将 forecasting 泛化到 extrapolation、interpolation、imputation 等不同 target segments，并在 decoder 中使用对应 target timestamps 的 learnable queries 来 retrieve input information。官方仓库也将其描述为 unified encoder-decoder forecasting framework。

[Implication] target-position-aware queries 是可信的 architecture mechanism。但 B8 不应复制完整 TimePerceiver，而应把 query mechanism 限定在 A6 learned-basis coefficient interface 上。

### SRP++

[Fact] 本地 note 记录的 SRP++ 主张是：multi-step forecasting 可能存在 step-invariant representation bottleneck，因此需要 step/segment-specific representations。

[Implication] 这与 A6-LBF 的潜在瓶颈一致：`coeff[c]` 是 sample-specific，但对所有 future positions 共享。StageB 可以引入 future-position-specific representation，同时保留一个 unified model/operator。

## A6 的 Code-Theory 问题

当前 A6-LBF-r256 的核心计算是：

```text
hidden = encoder(history)                     # [B, C, R]
coeff = learned_basis_coeff(hidden)           # [B, C, K]
prediction[t, c] = learned_temporal_basis[t] @ coeff[c] + bias[t]
```

因此：

- `coeff[c]` 是 sample-specific 和 channel-specific；
- `learned_temporal_basis[t]` 是 target-position-specific，但对所有 samples 共享；
- final dot product 前没有 sample-specific future-position representation。

[Hypothesis] 这形成了 StageB 的 architecture problem：

> unified forecast operator 不只需要 prefix-native basis，还需要 target-position-aware predictive representations，使 sample-specific coefficient state 能针对不同 future regions 自适应。

这不同于 B6-PLO。B6 研究 label/residual 是否需要 basis/frequency objective；B8 研究 architecture 是否应该在 prediction 前显式暴露 future positions as query states。

## 候选架构

名称：`Future-Query Aligned Basis Operator`。

最小 tensor path：

```text
history tokens:
  x_tokens = encoder(history)                 # [B*C, Nx, D]

future queries:
  q_pos = future_position_embedding           # [Nf, D]
  q_tokens = repeat(q_pos, B*C)               # [B*C, Nf, D]

future-query alignment:
  z_f = CrossAttention(q_tokens, x_tokens)    # [B*C, Nf, D]

A6 base coefficient:
  c_base = learned_basis_coeff(pool(x_tokens)) # [B, C, K]

target-position modulation:
  delta_c_s = zero_init_mlp(z_f[s])           # [B, C, K] per future segment/query
  c_s = c_base + gate_s * delta_c_s

prediction:
  for t in segment s:
      y_t = learned_temporal_basis[t] @ c_s + bias[t]
```

设计约束：

- inference 时不能使用 future values；
- future queries 只能包含 target positions 或 segment IDs；
- 保留 StageA learned-basis forecast operator；
- `gate_s` 或 modulation head 零初始化，使初始 forward 等价于 A6-LBF-r256；
- 使用 structured masking 或 independent future queries 保持 prefix/horizon invariance。

## 与 TimeAlign 的关系

不应原样恢复 TimeAlign：

- 原始 TimeAlign 将 history branch 对齐到 target reconstruction branch；
- B4 已经显示 inherited align/recon 不是 A6 性能必要条件；
- 重新使用 `w_align * align_loss` 会让论文边界退回到 TimeAlign variant。

可以继承的部分是：

- TimeAlign 的问题 framing：history-only representation 可能不够 future-aligned；
- training-only future branch 可以作为 diagnostic teacher，而不是第一优先的正式方法；
- alignment 应评估在 final predictor 真正使用的 representation space 中。

B8 的机制变化是：

- 从 target-value reconstruction alignment 转为 future-position query alignment；
- 从 generic hidden-state alignment 转为 A6 basis coefficient-space modulation；
- 从 auxiliary-loss-first 转为 architecture-first。

## 与相关架构工作的差异

与 ElasTST 相比：

- 二者都使用 future placeholders/queries；
- ElasTST 是完整 varied-horizon Transformer architecture；
- B8 是附着在 A6 learned-basis operator 前的轻量 future-query module。

与 TimePerceiver 相比：

- 二者都使用 target-position-aware queries；
- TimePerceiver 构建 generalized forecasting encoder-decoder；
- B8 保留标准 LTSF task，并研究 target queries 是否应该调制 basis coefficients。

与 SRP++ 相比：

- 二者都关注 step/segment-specific representation；
- SRP++ 采用 adapter/expert specialization；
- B8 使用 future-position query states，并保留 single unified model/operator。

## Narrative Gate 评估

当前判断：`promising_not_passed`。

优势：

- 直接深化 StageA：StageA 统一 decoder/head；StageB 使 representation-to-operator interface 具备 future-position awareness；
- 属于 architecture-level contribution，不只是 objective optimization；
- 可以相对 A6 做 function-preserving initialization；
- 避开 channel-correlation route 与 generic frequency auxiliary losses；
- 比原样恢复 TimeAlign future alignment 更能形成独立贡献边界。

风险：

- 如果替换而不是调制 A6 capacity，可能重演旧 A5 target-query failure；
- 若 basis-coefficient interface 不够中心，容易与 TimePerceiver/ElasTST 重叠；
- 必须先证明 segment-specific coefficient modulation 有 residual headroom；
- cross-attention 需要保持轻量，不能把收益变成单纯参数量收益。

## 实现前必须完成的诊断

`B8-OCD`：coefficient-space oracle capacity diagnostic。

目标：

> 检验在同一个 A6 learned temporal basis 下，允许 future-segment-specific coefficients 是否能降低 A6 errors。

所需 artifacts：

- clean A6 checkpoint，或至少包含 `learned_temporal_basis` 的 checkpoint-equivalent state；
- ETTh2、ETTm1、Weather 的 predictions/targets。

诊断流程：

1. 加载 A6 learned basis `B`。
2. 对每个 sample/channel/未来 segment，用该 segment 的 true values 求解 ridge least-squares coefficient `c_s^*`。
3. 比较：
   - global A6 prediction；
   - 同一 basis 下的 oracle segment-specific coefficient reconstruction；
   - 必要时加入 DCT/low-rank control。
4. 如果 oracle segment-specific coefficients 显著降低 tail/segment residuals，B8 才有真实 architecture target。
5. 如果 oracle gains 很小，或可由 generic DCT/frequency control 解释，则不实现 B8。

只有 `B8-OCD` 通过后，才能进入 Step 4-6 method design 与 remote implementation。
