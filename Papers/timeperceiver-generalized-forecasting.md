# TIMEPERCEIVER: Generalized Forecasting

## 来源

- Title: `TIMEPERCEIVER: An Encoder-Decoder Framework for Generalized Time-Series Forecasting`
- Zotero key: `34AMEC37`
- Authors: Jaebin Lee, Hankook Lee
- Year/status: NeurIPS 2025 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction is readable and includes method, ablations, and full results.

## 机制主张

[Strong Evidence] TIMEPERCEIVER 把 standard forecasting 扩展为 generalized temporal prediction：输入和 target 可以是任意时间位置的 segments，目标包括 extrapolation、interpolation、imputation。其核心是 encoder-decoder 与 training formulation 对齐，而不是只改 encoder。

## Tensor / 模块语义

- Standard task: $X_{past}$ 预测 $X_{future}$。
- Generalized task: 给定任意 observed index set $I$，预测 target index set $J$。
- Encoder: arbitrary patch embedding 后，input tokens 与 latent bottleneck tokens 通过 cross-attention 交互，再在 latent tokens 上 self-attention。
- Decoder: target timestamp 对应 learnable query tokens，通过 cross-attention 从 encoded input representations 中取信息。
- Positional encoding: 使用 temporal positional embedding 与 channel positional embedding 支撑 arbitrary target query。

## 关键默认

- 通过 generalized formulation 做 end-to-end training，避免单独 pretrain/finetune。
- latent bottleneck 是固定容量的 auxiliary memory，用于捕捉 temporal 和 cross-channel dependencies。
- decoder query 显式携带 target timestamp 信息。

## 对本项目的意义

- one model for multi-horizon: [Strong Evidence] 直接相关。target query 可以指定不同 horizon，天然支持 multi-horizon。
- future-aware architecture: [Strong Evidence] decoder query 是 future position-aware 的预测接口。
- MoE: [Speculative] latent bottleneck 或 target queries 可作为 future-aware routing keys。

## 可采用

- 采用 “target query / future query” 思想，把 horizon 从 output head 的隐式维度变成显式输入。
- 用 generalized masking task 做辅助训练，但先保持本仓库任务边界清楚。
- 用 latent bottleneck 作为未来状态路由的候选 memory。

## 暂不采用

- 不直接复制 Perceiver-style 全架构；先抽取 target-query 机制。
- 不在首轮实现 interpolation/imputation，避免任务面过宽。

## 风险与需复查点

- generalized task 与标准 LTSF benchmark 的训练成本、采样策略需要重新定义。
- 如果 target query 表达不足，可能退化为普通 horizon embedding。
