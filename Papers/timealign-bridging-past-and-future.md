# TimeAlign: Bridging Past and Future

## 来源

- Title: `BRIDGING PAST AND FUTURE: DISTRIBUTION-AWARE ALIGNMENT FOR TIME SERIES FORECASTING`
- Zotero key: `9JK37FWJ`
- Authors: Yifan Hu, Jie Yang, Tian Zhou, Peiyuan Liu, Yujin Tang, Rong Jin, Liang Sun
- Year/status: 2026, ICLR conference paper in Zotero extraction
- Full text status: [Strong Evidence] Zotero full text is readable and covers method, theory, experiments, and appendices.

## 机制主张

[Strong Evidence] TimeAlign 认为常规 forecasting paradigm 从 history representation 直接映射到 future target，会产生 past/future distribution mismatch，并倾向低频平滑。它通过 training-only reconstruction branch 编码 ground-truth future $Y$，再把 prediction branch hidden state $H_x$ 对齐到 reconstruction hidden state $H_y$，让预测表征更接近 future distribution。

## Tensor / 模块语义

- Input: history $X \in \mathbb{R}^{C \times L}$，target $Y \in \mathbb{R}^{C \times T}$。
- Predict branch: patch $X$ 得到 $X_p \in \mathbb{R}^{(C \cdot N) \times D}$，经过 M 层 encoder 得到 $H_x$，prediction head 输出 $\hat{Y}_{pred}$。
- Reconstruct branch: patch $Y$ 得到 $Y_p$，经过轻量 encoder 得到 $H_y$，reconstruction head 输出 $\hat{Y}_{recon}$。
- Alignment: 对 $H_x$ 先过额外 Linear，再与 stop-grad 的 $H_y$ 做 local patch-wise cosine alignment 和 global relation/distance alignment。
- Loss: $L = L_{pred} + L_{recon} + \lambda L_{align}$。

## 关键默认

- Reconstruct branch 只在 training 使用。
- Reconstruct branch 提供 future-side representation anchor。
- Local/global alignment 的梯度主要约束 predict branch，避免 reconstruction anchor 被 alignment 拉偏。
- 主要 benchmark 使用 ETT、Weather、Electricity、Traffic、Solar 等，horizon 为 `{96, 192, 336, 720}`。

## 对本项目的意义

- one model for multi-horizon: [Inference] 它不是直接解决 multi-horizon single model，但 alignment loss 可扩展为 horizon-aware target representation。
- future-aware architecture: [Strong Evidence] 直接相关。它把未来 $Y$ 在 training 中作为可学习监督信号，属于明确 future-aware training signal。
- MoE: [Speculative] 可把 $H_y$ 或 alignment residual 用作 routing signal，但原文不是 MoE 方法。

## 可采用

- 采用 training-only future branch 作为 future representation teacher。
- 采用 local/global alignment diagnostics 检查预测表征是否过度低频化。
- 采用 high-frequency error 或 frequency similarity 作为本项目后续机制诊断。

## 暂不采用

- 不直接照搬 dual-branch 结构作为主模型；它更像 future-aware supervision scaffold。
- 不把 reconstruction branch 当推理依赖，避免推理时 future leakage。

## 风险与需复查点

- [Hypothesis] Reconstruction branch 接近 target distribution，但可能学习到过强的 target autoencoding shortcut。
- alignment weight $\lambda$ 可能 dataset-sensitive。
- 需要在本仓库用同一 backbone 做 w/o alignment gate，才能判断机制是否真实贡献。
