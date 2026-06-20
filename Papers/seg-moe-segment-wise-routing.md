# Seg-MoE: Segment-Wise Routing

## 来源

- Title: `Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers`
- Zotero key: `PY6VZSMM`
- DOI/arXiv: `10.48550/ARXIV.2601.21641`
- Authors: Evandro S. Ortigossa, Eran Segal
- Year/status: 2026 preprint in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, ablations, and limitations.

## 机制主张

[Strong Evidence] Seg-MoE 认为 token-wise MoE routing 不符合 time series 的连续局部结构，可能把同一局部趋势的相邻 tokens 分给不同 experts。它把 contiguous time-step / patch segments 作为 routing unit，使 expert 在 segment 内处理局部结构。

## Tensor / 模块语义

- Backbone: encoder-only Transformer，patching 后有 $M$ patch tokens。
- Segment construction: normalized hidden states $H^b \in \mathbb{R}^{M \times d_{model}}$ 被切成 $C=\lceil M/\omega_b \rceil$ 个 segments $u_c^b \in \mathbb{R}^{\omega_b \times d_{model}}$。
- Router: flatten segment 得到 $\tilde{u}_c^b \in \mathbb{R}^{\omega_b d_{model}}$，线性 gating 后 Top-K 选择 experts。
- Shared expert: 每个 segment 总是经过 shared expert，再加 routed experts。
- Multi-resolution: 不同 Transformer blocks 可使用不同 segment length $\omega_b$。

## 关键默认

- $\omega=1$ 退化为 token-wise MoE。
- segment-wise router 是核心 inductive bias。
- 使用 shared fallback expert 增加稳定 dense path。
- 代码和实验配置显示不同 dataset 使用不同 segment schedule。

## 对本项目的意义

- one model for multi-horizon: [Inference] segment routing 可对应 horizon segments，但原文主要处理 input segments。
- future-aware architecture: [Speculative] future horizon segment-wise routing 是本项目可探索方向。
- MoE: [Strong Evidence] 直接相关，是 MoE granularity 的重要证据。

## 可采用

- MoE routing unit 不应默认是单 token；候选包括 input segment、future segment、state segment。
- 增加 $\omega=1$ token-wise vs $\omega>1$ segment-wise ablation。
- 使用 shared expert 防止 sparse routing 不稳定。

## 暂不采用

- 不直接照搬 multi-resolution schedule；先从单一 segment size 做机制 gate。
- 不默认使用其 backbone 细节，如 RMSNorm/GQA/FlashAttention，除非本仓库需要。

## 风险与需复查点

- segment length 是关键超参数，可能带来调参成本。
- non-overlapping segment 可能存在边界效应，原文也把 overlapping segment 作为 future work。
