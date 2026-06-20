# ProtoTS: Hierarchical Prototypes

## 来源

- Title: `ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting`
- Zotero key: `SLZEMUSJ`
- Authors: Ziheng Peng, Shijie Ren, Xinyue Gu, Linxiao Yang, Xiting Wang, Liang Sun
- Year/status: 2025/10/08, ICLR 2026 conference paper in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, limitations, and appendices, but PDF extraction has some layout noise.

## 机制主张

[Strong Evidence] ProtoTS 用 learnable prototypes 表示典型 future temporal patterns，并通过 hierarchical prototype learning 在 coarse prototypes 与 detailed prototypes 之间折中 interpretability 和 accuracy。其核心不是 MoE，而是 prototype-weighted prediction。

## Tensor / 模块语义

- 输入包含 historical endogenous variables $Y_{1:L}$、historical exogenous variables $X_{1:L}$ 和 future exogenous variables $X_{L+1:L+H}$。
- Multi-channel embedding: endogenous、discrete exogenous、continuous exogenous 分通道编码后融合。
- Bottleneck channel fusion: 通过 feature/time MLP bottleneck 过滤无关 covariate 信息。
- Prototype: 每个 prototype 有 embedding $\mu \in \mathbb{R}^d$ 和 temporal pattern $p \in \mathbb{R}^T$。
- Prediction: query representation 与 prototypes 的距离经 softmax 得到权重，再组合 prototype temporal patterns。

## 关键默认

- 明确使用 future exogenous variables。
- prototypes 是 output-side temporal curves，不只是 latent cluster centers。
- hierarchical prototypes 支持 coarse-to-fine split 和 expert steering。

## 对本项目的意义

- one model for multi-horizon: [Inference] prototype pattern 可天然对应不同 horizon segment 或 temporal regime，但原文侧重 exogenous forecasting。
- future-aware architecture: [Strong Evidence] 通过 future exogenous variables 和 output-pattern prototypes 形成 future-aware prediction。
- MoE: [Inference] prototype similarity 与 expert routing 结构相近，可作为 interpretable routing 的参考。

## 可采用

- 学习 output-side pattern basis/prototypes 作为解释性 future state。
- 在 routing diagnostics 中使用 prototype similarity，而不是只看 top-k expert id。
- 将 prototype hierarchy 作为论文中 interpretability/diagnostics 辅助，不必作为主干。

## 暂不采用

- 不默认采用人工 expert editing；本项目首要目标是 forecasting mechanism 和 performance。
- 不把 exogenous variables 作为必要输入，除非 dataset protocol 明确支持。

## 风险与需复查点

- 对没有 strong exogenous covariates 的标准 LTSF benchmark，prototype优势可能下降。
- hierarchical split 的选择和解释质量可能依赖 domain expertise。
