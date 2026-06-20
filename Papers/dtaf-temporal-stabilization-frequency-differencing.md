# DTAF: Temporal Stabilization and Frequency Differencing

## 来源

- Title: `Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing`
- Zotero key: `LAKF59WZ`
- Authors: Junkai Lu, Peng Chen, Chenjuan Guo, Yang Shu, Meng Wang, Bin Yang
- Year/status: AAAI-26 in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes method, experiments, ablations, and analysis.

## 机制主张

[Strong Evidence] DTAF 认为 non-stationarity 同时存在 temporal domain 和 frequency domain。它用 Temporal Stabilizing Fusion 抽取并减去 temporal non-stationary patterns，再用 Frequency Wave Modeling 对 patch spectra 做 differencing，捕捉动态频率漂移。

## Tensor / 模块语义

- Input 经 InstanceNorm、patching、embedding 得到 patch embeddings。
- TFS / Non-stationary MoE Filter:
  - router 为每个 patch 分配 expert weights；
  - experts 提取 non-stationary pattern $X_i^{patterns}$；
  - stable residual $X_i^{stable} = X_i^{patch} - X_i^{patterns}$。
- Stable loss: 用 KL 约束 patch residual distributions 更稳定。
- Temporal fusion: 对 stable patches 进行 trend/seasonal decomposition 与 historical weighted aggregation。
- FWM: 对 temporal branch representation 做 rFFT，计算 adjacent patch spectrum difference，Top-K 选择变化显著的 frequencies，再 iFFT 回 time domain。
- Dual-branch attention: temporal/frequency features 分支 attention 后 concat，再 predictor 输出。

## 关键默认

- MoE experts 被定义为 non-stationary pattern extractors，而不是直接预测 experts。
- frequency differencing 是 patch-to-patch spectrum change，不是全局 FFT。
- 使用 TFB benchmark，结果平均多次运行。

## 对本项目的意义

- one model for multi-horizon: [Inference] 不直接处理 single-model multi-horizon。
- future-aware architecture: [Speculative] frequency/temporal drift 可作为 future-state 变化的 proxy。
- MoE: [Strong Evidence] 直接相关，提供“专家抽取扰动，再建模 residual”的另类 MoE 路线。

## 可采用

- 将 MoE 专家定义为 residual/stabilization operator，而不一定是 prediction head。
- 用 patch-to-patch spectrum difference 做 non-stationarity diagnostic。
- 对未来模型增加 temporal/frequency branch diagnostic，而非一开始加完整双分支。

## 暂不采用

- 不直接采用 KL residual-stability loss，需先验证其统计意义。
- 不先引入完整 DTAF 双分支，避免与主线 future-aware 设计纠缠。

## 风险与需复查点

- experts “学到 non-stationary pattern” 的因果解释需要更严格验证。
- Top-K frequency differencing 可能对 patch size 敏感。
