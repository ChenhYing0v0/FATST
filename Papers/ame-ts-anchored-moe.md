# AME-TS: Anchored Mixture-of-Experts

## 来源

- Title: `AME-TS: Anchored Mixture-of-Experts for Time Series Forecasting`
- Zotero key: `IZMRGTIG`
- DOI/arXiv: `10.48550/ARXIV.2605.25166`
- Authors: Rui Wang, Renhao Xue, Ray Razi, Huan Song, Hannah R. Marlowe
- Year/status: 2026 preprint in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, and appendices.

## 机制主张

[Strong Evidence] AME-TS 认为 standard MoE expert identities permutation-symmetric，specialization 不稳定且难解释。它用 forecastability、seasonality、trend、sparsity 等 temporal descriptors 构造 series-level structural prior，并用 KL prior-alignment loss 约束 token-level MoE routing。

## Tensor / 模块语义

- Regime predictor: $g_\phi(X) = [r_f, r_s, r_t, r_{sp}] \in [0,1]^4$。
- Expert prior: descriptor scores 映射到 specialized experts，弱结构或高不确定性分给 shared experts。
- Token router: layer/token-level distribution $p_l(e | z_t^{(l)})$。
- Prior alignment: $KL(p_l(e | z_t^{(l)}) \| q(e | X))$。
- Layer-wise prior weight: deeper layers 用更强结构约束。
- Forecasting backbone: encoder-only Transformer with MoE layers，masked prediction loss。

## 关键默认

- structural prior 只用于 training regularization；inference 只用 learned router。
- regime predictor frozen。
- descriptors 是 soft profile，不做 hard assignment。
- shared experts 处理结构弱、混合或不确定 series。

## 对本项目的意义

- one model for multi-horizon: [Inference] 本身偏 foundation model routing，不直接处理 horizon。
- future-aware architecture: [Speculative] 可把 descriptor 扩展为 future-state descriptors。
- MoE: [Strong Evidence] 非常相关，提供“结构先验约束 routing”的明确路径。

## 可采用

- 本项目 MoE 不应只做无约束 softmax router；应考虑可解释结构先验或 routing diagnostic。
- forecastability/seasonality/trend/sparsity 可作为 offline routing audit，不必先作为主模型输入。
- shared expert + specialized expert 的划分值得保留为设计候选。

## 暂不采用

- 不直接引入 forecastability 等 handcrafted descriptors 进入主路径，除非用户批准；本仓库保持干净机制路线。
- 不先做 foundation-scale pretraining。

## 风险与需复查点

- descriptor prior 可能把模型推向 handcrafted signal，和本项目 future-aware hypothesis 可能冲突。
- regime predictor 的训练数据、归一化和泛化需要独立审计。
