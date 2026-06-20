# MoHETS: Mixture-of-Heterogeneous-Experts

## 来源

- Title: `MoHETS: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts`
- Zotero key: `WSKUSM6X`
- DOI/arXiv: `10.48550/ARXIV.2601.21866`
- Authors: Evandro S. Ortigossa, Guy Lutsker, Eran Segal
- Year/status: 2026 preprint in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, ablations, and discussion.

## 机制主张

[Strong Evidence] MoHETS 认为标准 MoE 的 homogeneous MLP experts 不适合 time series 中 global trend、local periodicity、non-stationary regimes 的异质结构。它用 heterogeneous experts：shared depthwise-convolution expert 维持 sequence continuity，routed Fourier-based experts 处理 patch-level periodic structures，并通过 cross-attention 引入 exogenous covariates。

## Tensor / 模块语义

- Input $X \in \mathbb{R}^{D \times T}$ 经 InstanceNorm、patchify、GroupNorm、PatchEmbed 得到 patch embeddings。
- Transformer block: self-attention、cross-attention、MoHE。
- Exogenous embedding: endogenous 与 covariates 线性投影、concat/fuse、patch embedding，作为 cross-attention 的 key/value。
- MoHE:
  - shared expert: depthwise convolution，用于连续性/局部趋势；
  - routed experts: Fourier-based experts，用于周期结构；
  - Top-K router 选择 routed experts，并与 shared path 组合。
- Output head: convolutional patch decoder，替代参数重的线性 projection head。

## 关键默认

- Experts 不是同构 MLP，而是带不同 inductive bias 的 operators。
- Cross-attention 假设 future covariates 可用。
- Output horizon $H_o$ 是效率/精度折中，原文观察 $H_o=24$ 较稳。

## 对本项目的意义

- one model for multi-horizon: [Inference] convolutional patch decoder 和 output horizon 可为 multi-horizon head 提供参考。
- future-aware architecture: [Inference] future covariates cross-attention 是外部 future-aware 路线。
- MoE: [Strong Evidence] 直接相关，支持 heterogeneous expert architecture 而非同构专家堆叠。

## 可采用

- 本项目 MoE expert 可以有不同 operator bias，例如 convolution、Fourier、linear/state operator。
- shared continuity expert + routed specialized experts 是稳健结构候选。
- 输出 head 不一定是大 linear projection，可考虑 patch decoder。

## 暂不采用

- 不把 future covariates 作为标准任务依赖，除非 dataset protocol 支持。
- 不直接导入 MoHETS 代码或旧 baseline 结果；先用其机制指导本地实现。

## 风险与需复查点

- heterogeneous experts 可能增加 attribution 难度：性能来自 expert bias、routing、还是 cross-attention。
- output horizon $H_o$ 与 benchmark horizon 的关系需要单独 ablation。
