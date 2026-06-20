# TFPS: Pattern-Specific Experts

## 来源

- Title: `Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift`
- Zotero key: `PXCHMY4H`
- Authors: Yanru Sun, Zongxia Xie, Emadeldeen Eldele, Dongyue Chen, Qinghua Hu, Min Wu
- Year/status: 2024/10/13, NeurIPS 2025 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, appendices; some OCR/math layout is noisy.

## 机制主张

[Strong Evidence] TFPS 认为 patch-level distribution shift 使 single uniform model 难以泛化。它用 time/frequency dual-domain encoder 表征 patches，再用 subspace clustering 识别 latent patterns，最后用 pattern-specific experts 建模不同 patch pattern。

## Tensor / 模块语义

- Input $X \in \mathbb{R}^{L \times C}$ 被切成 patches，得到 $X_{PE} \in \mathbb{R}^{C \times N \times D}$。
- DDE: time-domain Transformer branch 输出 $z_t$；frequency branch 用 Fourier sublayer 输出 $z_f$。
- PI: learnable subspace bases $D$ 与 embedded representation $z$ 计算 subspace affinity $S$，再 sharpen 成 $\hat{S}$。
- MoPE: 使用 $S$ 作为 cluster assignment / gate，Top-K 选择 pattern experts，对 $z$ 做 patch-wise expert processing。
- Loss: forecasting MSE 加 time/frequency PI clustering regularization。

## 关键默认

- pattern grouping 来自 subspace clustering，不是普通 softmax router。
- time 和 frequency domain 各自有 PI 和 MoPE。
- 关注 patch-level drift，而不是 whole-series label。

## 对本项目的意义

- one model for multi-horizon: [Inference] 可把 horizon segment 的 drift 作为 pattern-specific routing 对象。
- future-aware architecture: [Speculative] 如果 PI 作用于 future query 或 predicted future states，可转为 future-aware routing。
- MoE: [Strong Evidence] 直接相关，是 pattern-specific MoE 设计证据。

## 可采用

- 用 patch-level distribution shift 作为 MoE 必要性的机制论证。
- 采用 subspace affinity 或 cluster separability 作为 routing diagnostic。
- 先实现轻量 pattern identifier，再决定是否需要 full dual-domain branch。

## 暂不采用

- 不先引入复杂 subspace regularization；首轮可用更简单的 routing probe 验证机制。
- 不默认 time/frequency 双分支都进入主模型，避免过宽。

## 风险与需复查点

- subspace clustering 超参数和 expert 数量可能 dataset-sensitive。
- PI 学到的 cluster 是否等价于 forecast-relevant state，需要额外 diagnostic。
