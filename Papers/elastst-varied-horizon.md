# ElasTST: Varied-Horizon Forecasting

## 来源

- Title: `ElasTST: Towards Robust Varied-Horizon Forecasting with Elastic Time-Series Transformer`
- Zotero key: `MXLVX75Z`
- Authors: Jiawen Zhang, Shun Zheng, Xumeng Wen, Xiaofang Zhou, Jiang Bian, Jia Li
- Year/status: 2024/11/04, NeurIPS 2024 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction is readable and includes method, experiments, and appendices.

## 机制主张

[Strong Evidence] ElasTST 解决 varied-horizon inference：一个模型在不同 inference horizon 下输出应保持 horizon-invariant，即扩展预测长度不应改变已经存在的 future positions 输出。它使用 placeholders、structured self-attention mask、tunable RoPE、multi-scale patch 与 horizon reweighting。

## Tensor / 模块语义

- Input: history 与 future placeholders 拼接成 $X \in \mathbb{R}^{L+T}$。
- Patching: $X_p \in \mathbb{R}^{N \times P}$，encoder 得到 $H \in \mathbb{R}^{N \times D}$。
- Structured mask: placeholder-only patches 不参与影响其他 placeholder，避免 future placeholders 之间互相泄漏。
- TRoPE: 用可调 period coefficients $P_j$ 替代 NLP 默认 RoPE 频率。
- Multi-patch assembly: 多 patch size 共享 Transformer backbone，融合短期细粒度与长期粗粒度信息。
- Horizon reweighting: 固定最长 horizon 训练，用权重近似随机 horizon sampling。

## 关键默认

- Non-autoregressive placeholder generation。
- single model covers varied horizons。
- structured mask 是 horizon-invariance 的核心。
- multi-patch 配置常用 `{8, 16, 32}`。

## 对本项目的意义

- one model for multi-horizon: [Strong Evidence] 直接相关，是最重要种子之一。
- future-aware architecture: [Strong Evidence] future placeholders 与 target positions 明确进入模型结构。
- MoE: [Speculative] structured mask 和 multi-scale patch 可以与 expert routing 结合。

## 可采用

- horizon-invariance 作为本项目 multi-horizon 设计的硬性测试。
- placeholder + mask 作为 future-aware architecture 的最小可实现路线。
- horizon reweighting 可作为单模型训练 baseline。

## 暂不采用

- 不直接完整复刻 ElasTST；先抽取 invariance test、masking policy、horizon reweighting。
- 不把 foundation-model zero-shot 作为早期目标。

## 风险与需复查点

- Tunable RoPE 和 multi-patch 超参数对 dataset/horizon 可能敏感。
- NMAE/NRMSE 指标与本仓库后续 MSE/MAE protocol 需要对齐。
