# QDF / MetaDF: Quadratic Direct Forecast

## 来源

- Title: `Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models`
- Zotero key: `R8YQ4UWY`
- Authors: Hao Wang, Licheng Pan, Yuan Lu, Zi Ciu Chan, Tianqiao Liu, Shuting He, Zhixuan Chu, Qingsong Wen, Haoxuan Li, Zhouchen Lin
- Year/status: 2025/10/08, ICLR 2026 conference paper in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes theory, algorithms, experiments, and appendices.

## 机制主张

[Strong Evidence] QDF 指出标准 MSE 把 future steps 当作独立等权任务，忽略 label autocorrelation 和 heterogeneous task weights。它用 quadratic-form weighted objective 学习未来步骤之间的 inverse covariance / weighting matrix，以改进 direct multi-step forecasting 的训练目标。

## Tensor / 模块语义

- Forecast model: $g_\theta: \mathbb{R}^{H \times D} \rightarrow \mathbb{R}^{T \times D}$。
- Standard loss: $\|Y - g_\theta(X)\|^2$，等价于 identity weighting matrix。
- QDF loss: $(Y - g_\theta(X))^\top \bar{\Sigma}(Y - g_\theta(X))$。
- $\bar{\Sigma}$ 通过 train split 内的 bilevel / meta-style procedure 学习，目标是提升 holdout generalization。
- 约束：通过 Cholesky factorization 保证 covariance PSD。

## 关键默认

- 只用 training set 拆分学习 weighting matrix，不用 validation/test，避免 leakage。
- 额外计算只发生在 training objective 中，不影响 inference。
- 方法 model-agnostic，可插到不同 direct forecasting backbones。

## 对本项目的意义

- one model for multi-horizon: [Strong Evidence] 直接相关，提供 multi-step objective 维度的证据。
- future-aware architecture: [Inference] 它不改 architecture，但把 future step dependency 显式写入 loss。
- MoE: [Speculative] 可用 learned covariance 或 step weights 作为 horizon-aware routing/diagnostic。

## 可采用

- 将 future-step covariance 作为 multi-horizon 训练的 objective baseline。
- 建立 step correlation / conditional variance diagnostics。
- 首轮先实现简单 diagonal/non-diagonal horizon weighting ablation，再决定是否做完整 bilevel。

## 暂不采用

- 不在初始化阶段引入完整 meta-learning loop，成本和复杂度偏高。
- 不把 QDF 性能直接当 architecture 改进证据。

## 风险与需复查点

- learned matrix 是否跨 dataset/horizon 稳定需要验证。
- bilevel procedure 的实现细节可能影响公平性。
