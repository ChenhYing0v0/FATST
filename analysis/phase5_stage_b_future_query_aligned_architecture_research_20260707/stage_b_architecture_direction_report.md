# StageB Architecture Direction Research Report

## 决策

[Decision] `B8-FQA` 曾是优先推进的 StageB architecture candidate，但已被后续 `B8-OCD` negative control
否决。

完整名称：`Future-Query Aligned Basis Operator`。

当前状态：`rejected_by_ocd_control`，不是 method-ready。

`B8-OCD` 已完成，结果不支持实现 B8-FQA。本文保留原 architecture research 的推理过程，但 post-diagnostic
decision 已改为：StageB 回到 Step 2/3，重新寻找 architecture-level 第二贡献问题。

## Post-Diagnostic Addendum

[Fact] `B8-OCD` 使用 clean A6 checkpoint 与 predictions，固定 `learned_temporal_basis`，比较 learned basis
和 DCT control 的 global/segment residual correction。完整报告见
`analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/b8_ocd_report.md`。

[Fact] learned basis 的 segment-specific correction 相比 global correction 有明显额外 headroom。Rank 64 的
segment-minus-global reduction 为 ETTh2 `16.85%`、ETTm1 `28.19%`、Weather `22.01%`。

[Counter-Evidence] DCT control 的绝对 residual reduction 更强。Rank 64 的 segment reduction 中，learned
basis 为 ETTh2 `79.05%`、ETTm1 `72.77%`、Weather `61.91%`，DCT control 为 ETTh2 `87.61%`、ETTm1
`91.85%`、Weather `91.18%`。

[Decision] B8 的叙事逻辑仍然通顺，但当前问题证据被 generic low-frequency control 混淆。`B8-FQA` 不应进入
Step 4-6 method design，也不应实现。

## 用户约束

[Fact] 用户明确指出：objective optimization 是后续想研究的问题，但更适合作为小贡献点；StageB 应优先建立第二个主创新点，最好是模型架构层面的研究，因为 StageA 主要创新在 decoder/head。

[Decision] 本轮 StageB architecture search 不考虑 channel-correlation modeling。

## 外部网络调研说明

[Fact] 本轮调研不是只参考 Zotero 或本地 paper notes。外部网络调研使用了以下来源：

| 来源 | 类型 | 本轮用途 |
| --- | --- | --- |
| <https://arxiv.org/html/2509.14181v3> | TimeAlign arXiv HTML | 核验 past/future distribution mismatch、reconstruction branch、local/global alignment、frequency mismatch 叙事 |
| <https://arxiv.org/abs/2509.14181> | TimeAlign arXiv abstract | 核验论文摘要、code link、贡献边界 |
| <https://github.com/TROUBADOUR000/TimeAlign> | TimeAlign official code repository | 核验其 plug-and-play reconstruction-based alignment 定位 |
| <https://arxiv.org/abs/2411.01842> | ElasTST arXiv abstract | 核验 future placeholders、structured self-attention masks、horizon-invariant varied-horizon forecasting |
| <https://arxiv.org/abs/2512.22550> | TimePerceiver arXiv abstract | 核验 generalized forecasting、target timestamps learnable queries |
| <https://arxiv.org/html/2512.22550v1> | TimePerceiver arXiv HTML | 核验 encoder-decoder target-query architecture 叙事 |
| <https://github.com/efficient-learning-lab/TimePerceiver> | TimePerceiver official code repository | 核验 official implementation 与 unified encoder-decoder framing |

[Fact] SRP++ 证据来自本地 `Papers/srp-step-specific-representation.md`。本轮 OpenReview 页面访问被浏览器验证阻挡，因此 SRP++ 不计入外部网络已核验证据，只作为本地 note evidence。

## 为什么不能简单恢复 TimeAlign Future Align

原始 TimeAlign 与 StageB 相关，但不能直接作为本项目 StageB 主线：

- TimeAlign 的核心贡献是 training-time future reconstruction 与 representation alignment；
- B4 dependency ablation 已经说明 A6-LBF 不实质依赖 inherited align/recon；
- 恢复 generic `w_align * align_loss` 会使论文重新像 TimeAlign variant；
- TimeAlign alignment 并不是围绕 unified multi-horizon basis coefficients 设计的。

因此 StageB 不应变成：

```text
A6-LBF + TimeAlign future branch + align loss
```

真正可继承的是更抽象的问题意识：

> history-only representations may not be sufficiently aligned with future target positions.

B8 将这个问题转化为 architecture problem，而不是 auxiliary-loss problem。

## 文献综合

### TimeAlign

[Fact] TimeAlign 诊断的是 history-only forecasting 的结构限制：历史表示直接映射到未来目标，容易产生 past/future distribution mismatch 和 low-frequency smoothing。其方法是引入 training-only future reconstruction branch，并进行 global/local representation alignment。

[Implication] 对本项目来说，应保留“prediction representation 需要 future alignment”的问题 framing，但拒绝原样继承机制。原因是它不针对 unified-horizon coefficient space，且 B4 已证明它不是 A6-LBF 的必要性能来源。

### ElasTST

[Fact] ElasTST 使用 future placeholders 与 structured self-attention masks，使 varied-horizon inference 中扩展 horizon 不改变已有 future outputs。

[Implication] future positions 可以作为 tokens/placeholders 进入 architecture，且不需要 future-value leakage。B8 若实现，应明确保证 prefix/horizon invariance。

### TimePerceiver

[Fact] TimePerceiver 使用 target-position-aware decoder queries，并将 forecasting 泛化到任意 target segments 的 extrapolation、interpolation、imputation。

[Implication] target queries 是可信架构机制，但 B8 不应复制完整 generalized forecasting framework，而应把 query mechanism 约束在 A6 learned-basis coefficient interface 上。

### SRP++

[Fact] 本地 note 记录 SRP++ 认为 multi-step forecasting 可能需要 step/segment-specific representations，而不是所有 future steps 共用同一表示。

[Implication] 这支持 A6-LBF 的一个潜在瓶颈：global coefficient vector 可能不足以同时服务不同 future regions。

## A6 的核心 architecture problem

A6-LBF 当前预测路径为：

```text
hidden = encoder(history)            # [B, C, R]
coeff = W(hidden)                    # [B, C, K]
y[t, c] = basis[t] @ coeff[c] + b[t]
```

该设计干净且强，但存在一个明确的 architecture limitation：

- `basis[t]` 是 future-position-specific，但对所有 samples 全局共享；
- `coeff[c]` 是 sample-specific，但对所有 future positions 共享；
- 进入 final basis dot product 前，没有 `coeff[t, c]` 或 target-position-aware representation。

[Hypothesis] 第二个主创新可以瞄准这个缺口：

> unified multi-horizon model 不应只在 history-only encoder 后接 prefix-native basis decoder，还应在 basis prediction 前将 history representation 对齐到 future positions。

## 原推荐候选：B8-FQA

最小架构：

```text
history tokens:
  x_tokens = encoder(history)                     # [B*C, Nx, D]

future queries:
  q_pos = future position / segment embeddings    # [Nf, D]

future-query alignment:
  z_f = CrossAttention(q_pos, x_tokens, x_tokens) # [B*C, Nf, D]

A6 base path:
  c_base = learned_basis_coeff(pool(x_tokens))    # [B, C, K]

position-aware coefficient modulation:
  c_s = c_base + alpha_s * DeltaCoeff(z_f[s])     # [B, C, K]

prediction:
  y_t = basis[t] @ c_s + bias[t], for t in segment s
```

capacity preservation 规则：

- `alpha_s` 零初始化；
- 初始 forward 与 clean A6-LBF-r256 完全等价；
- 这符合项目规则：capacity-preserving claim 必须有 code-theory check，不能把随机初始化权重复制误称为保留已学能力。

## 候选路线比较

| 候选 | 核心想法 | 叙事强度 | 可行性 | 主要风险 |
| --- | --- | --- | --- | --- |
| 恢复 TimeAlign align | 重新加入 future reconstruction/align branch | 弱 | 高 | 像 inherited TimeAlign variant，不够 A6-specific |
| Basis-aware align | 对齐 history/future coefficients | 中 | 中 | B6 已显示 learned basis top32 不强于 DCT，需新证据 |
| ElasTST-style placeholders | 在 encoder 中加入 future placeholders | 中高 | 中 | 容易变成完整 ElasTST adaptation |
| TimePerceiver-style target queries | 用 target queries 从 history 中取信息 | 高 | 中 | 若不保留 A6 path，可能重演旧 A5 target-query collapse |
| `B8-FQA` | 用 future queries 调制 A6 basis coefficients | 最高 | 中 | 需要 oracle evidence；实现必须轻量 |

## B8 如何衔接 StageA

StageA 的贡献是：

> 用 prefix-native learned-basis forecast operator 替换 dense/fixed prediction head。

StageB B8 的贡献候选是：

> 使 feeding this operator 的 representation interface 具备 future-position awareness，让 unified operator 对不同 future regions 接收不同的 sample-conditioned states。

因此论文可以形成两个连续的 architecture contribution：

1. unified forecast operator；
2. future-query aligned representation interface。

## 与已有工作的区别

不同于 TimeAlign：

- 不把 future reconstruction/alignment loss 作为主机制；
- 不做 generic hidden alignment；
- alignment target 是 future position/query state 到 A6 coefficient space 的调制路径。

不同于 ElasTST：

- 不构建完整 placeholder Transformer；
- 保留 A6 learned-basis operator；
- future queries 只服务 coefficient modulation。

不同于 TimePerceiver：

- 不转向 generalized interpolation/imputation framework；
- target queries 不是完整 decoder；
- queries 的作用被限制在 learned-basis coefficient interface。

不同于 B7：

- B7 改 objective/training weighting；
- B8 改 architecture。

## 原计划必须做的诊断

`B8-OCD`：coefficient-space oracle capacity diagnostic。

目的：

> 判断在同一 learned basis 下，future-segment-specific coefficients 是否能显著降低 A6 residuals。

流程：

1. 获取 clean A6 checkpoint，或至少获取包含 `learned_temporal_basis` 的等价 state。
2. 使用 ETTh2、ETTm1、Weather 的 predictions/targets。
3. 对每个 sample/channel/segment，在相同 learned basis rows 上求 ridge least-squares coefficients。
4. 比较 A6 global coefficient prediction 与 oracle segment-specific coefficient reconstruction。
5. 必要时加入 DCT/low-rank control，排除 generic frequency/basis explanation。

Gate：

- 若 oracle segment-specific coefficients 在 ETTh2 和 ETTm1 至少稳定降低 tail/segment residuals，且不能被 generic DCT control 解释，则 B8 进入 Step 4-6 method design；
- 若收益很小、只在 Weather 出现、或完全由 generic basis control 解释，则不实现 B8。

## 原下一步研究动作

当前不要实现 B8。以下原计划已经由 `B8-OCD` 执行并返回负向控制结果，仅作为历史记录保留。

下一步：

1. 定位或同步 clean A6 checkpoint，确认包含 `learned_temporal_basis`；
2. 编写 `B8-OCD` diagnostic protocol 与 analyzer；
3. 只有 `B8-OCD` 通过，才进入 Step 4-6 method design；本次没有通过，因此不进入方法设计。
