# SRP++: Step-Specific Representation

## 来源

- Title: `On the Necessity of Step-Specific Representation Learning for Multi-Step Forecasting`
- Zotero key: `YRL4AHYC`
- Authors: Licheng Pan, Zhijian Xu, Zi Ciu Chan, Haoxuan Li, Shuting He, Xiaoxi Li, Hao Wang, Yuan Lu, Qingsong Wen
- Year/status: 2025/10/08, under review as ICLR 2026 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes motivation, theorem, method, experiments; status is under review.

## 机制主张

[Strong Evidence] SRP 认为 direct multi-step forecasting 中的 step-invariant representation 存在 expressiveness bottleneck。不同 future steps 可能需要不同 representations；单一 shared representation 加 linear head 会产生不可由更强 encoder 消除的误差。SRP++ 用 segment-specific LoRA experts 产生 step/segment-specific representations。

## Tensor / 模块语义

- Standard direct forecast: encoder 输出 shared $R$，linear decoder 用不同 row $W_t$ 预测 $\hat{Y}_t$。
- Theorem: 当 output steps 多于 representation/decoder rank 能覆盖的范围时，存在 independent of encoder 的 residual error。
- SRP: 先训练 one-step model；再为每个 future step 注入 step-specific LoRA，冻结 foundation model。
- SRP++: 把 horizon 分为 $K$ 个 segments，每个 segment 输出 $S=T/K$ steps；每个 segment 用 mixture of LoRA experts：
  $M^{(k)} = M + \sum_p \Delta_k^{(p)} B^{(p)}A^{(p)}$。
- Inference: 各 segment-specific adapters 生成 representations 后拼接输出。

## 关键默认

- Direct forecasting paradigm。
- Pretrained/frozen base model 加 parameter-efficient adaptation。
- Segment size 要满足 expressiveness condition，且利用 step dependency 降低成本。
- MoE 在这里作用于 LoRA adapter sharing，不是普通 token expert。

## 对本项目的意义

- one model for multi-horizon: [Strong Evidence] 核心相关，挑战 “一个 shared representation 覆盖所有 horizon” 的常见假设。
- future-aware architecture: [Strong Evidence] 强调 future step-specific representations。
- MoE: [Strong Evidence] 用 MoE-style LoRA expert sharing 做 step-specific adaptation。

## 可采用

- 作为本项目理论入口：multi-horizon 不能只靠一个 shared latent + linear output。
- 实现 horizon/segment-specific lightweight adapters 或 FiLM，而不是每个 horizon 独立模型。
- 把 representation step-specificity 作为 diagnostic：不同 horizon 的 hidden states 是否可分。

## 暂不采用

- 不直接采用两阶段 one-step pretrain + LoRA adaptation，首轮成本高且可能偏离主模型。
- 不把 anonymous under-review 结果作为最终论文强证据，需后续跟踪状态。

## 风险与需复查点

- 理论条件与常见 deep forecaster 的实际 tensor shape 是否完全吻合需要复核。
- adapter-based route 可能更像 fine-tuning framework，而非统一 architecture。
