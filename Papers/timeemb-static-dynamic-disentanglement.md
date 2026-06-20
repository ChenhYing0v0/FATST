# TimeEmb: Static-Dynamic Disentanglement

## 来源

- Title: `TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting`
- Zotero key: `VFRBDA4N`
- Authors: Mingyuan Xia, Chunxu Zhang, Zijian Zhang, Hao Miao, Qidong Liu, Yuanshao Zhu, Bo Yang
- Year/status: NeurIPS 2025 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, limitations, and checklist.

## 机制主张

[Strong Evidence] TimeEmb 把时间序列分为 time-invariant static component 与 time-varying dynamic component。它用 learnable embedding bank 捕捉全局稳定周期模式，用 frequency-domain filter 处理动态扰动，在轻量结构下提升 non-stationary forecasting。

## Tensor / 模块语义

- Input: $X \in \mathbb{R}^{L \times D}$。
- rFFT: $X$ 转为 frequency representation $\mathcal{X} \in \mathbb{C}^{F \times D}$。
- Embedding bank: $E \in \mathbb{R}^{M \times F \times D}$，按 last timestamp 查表得到 static component $X^s$。
- Dynamic component: $X^d = \mathcal{X} - X^s$。
- Frequency filter: complex spectral modulation $\omega \in \mathbb{C}^{F \times 1}$，得到 $H_\omega(X^d)$。
- Fusion: $\dot{X} = H_\omega(X^d) + X^s$，再 IFFT、MLP prediction、inverse normalization。
- Loss: time-domain MSE 加 frequency-domain MAE。

## 关键默认

- embedding bank 的 key 来自 timestamp slot，例如 hour 或 15-min slot。
- static component 是 dataset-level learnable bank，不是当前样本平滑趋势。
- filter 共享跨 channels，主要复杂度来自 FFT。

## 对本项目的意义

- one model for multi-horizon: [Inference] 不直接解决，但 static/dynamic decomposition 可作为 multi-horizon shared state。
- future-aware architecture: [Speculative] static bank 可扩展为 future slot embedding。
- MoE: [Speculative] static/dynamic residual 可作为专家分工依据。

## 可采用

- 使用 timestamp-indexed embedding bank 作为 lightweight future/time prior 的参考。
- 使用 frequency-domain residual diagnostics 评估模型是否只学静态周期。
- 对比 static bank 和 future-query embedding 的差异。

## 暂不采用

- 不直接依赖 timestamp proxy 作为核心创新，避免机制被解释成 calendar lookup。
- 不把固定 $M$ 作为普适周期假设。

## 风险与需复查点

- embedding bank 需要可靠时间戳；标准 benchmark 的时间戳可用性与粒度要检查。
- 对不规则或周期弱数据可能失效。
