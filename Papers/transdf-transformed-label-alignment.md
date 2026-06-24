# TransDF: Transformed Label Alignment

## 来源

- Title: `TransDF: Time-Series Forecasting Needs Transformed Label Alignment`
- Zotero key: `BK8HKCXT`
- Authors: Hao Wang, Licheng Pan, Zhichao Chen, Xu Chen, Qingyang Dai, Lei Wang, Haoxuan Li, Zhouchen Lin
- Year/status: 2025/05/23, arXiv preprint in Zotero extraction
- Full text status: [Strong Evidence] Zotero extraction includes theory, method, experiments, and appendices.

## 机制主张

[Strong Evidence] TransDF 认为 time-series forecasting 的 learning objective 有两个核心问题：

1. temporal MSE 忽略 label sequence autocorrelation，因此相对 true label likelihood 有 bias；
2. forecast horizon 变长会把预测拆成过多 step-wise tasks，增加 multi-task optimization difficulty
   和 task conflict。

TransDF 的做法不是改变模型结构，而是把 label sequence 投影到 decorrelated components，并只对
最重要的 components 施加强监督，从而同时降低 autocorrelation bias 和 task amount。

## Tensor / 模块语义

- Forecast model:
  $g_\theta: \mathbb{R}^{H \times D} \rightarrow \mathbb{R}^{T \times D}$。
- Standard temporal MSE:
  $\sum_{t=1}^{T} \|Y_t-\hat{Y}_t\|^2$。
- Label matrix:
  $Y \in \mathbb{R}^{m \times T}$ for normalized label sequences。
- Projection matrix:
  $P^\star$ from SVD / PCA-style projection。
- Transformed components:
  $Z = YP^\star$ and $\hat{Z}=\hat{Y}P^\star$。
- Training loss:
  $L_{\alpha,\gamma}=\alpha L_{\mathrm{trans},\gamma}+(1-\alpha)L_{\mathrm{tmp}}$，
  where only the top $K=\mathrm{round}(\gamma T)$ significant components are aligned.

## 关键默认

- Projection is computed from training labels only.
- Method is model-agnostic and inference-free; inference still outputs the original time-domain forecast.
- Significant component selection is a training supervision choice, not an evaluation-horizon choice.
- The paper reports sensitivity over $\alpha$ and rank ratio $\gamma$ and finds broad effective ranges.

## 对本项目的意义

- horizon-agnostic supervision: [Strong Evidence] 直接相关。TransDF 明确支持“不以 evaluation
  horizons 作为训练监督单位”的方向。
- objective / multi-step loss: [Strong Evidence] 与 QDF 同源，进一步把 future-step dependency
  从 covariance weighting 扩展到 decorrelated low-rank components。
- one model for multi-horizon: [Inference] 可作为 unified forecaster 的 training objective；
  evaluation 仍可在 `96,192,336,720` 上进行。
- future-aware architecture / MoE: [Speculative] 只有在 component-level supervision 诱导出稳定
  representation heterogeneity 后，才值得作为二级机制。

## 可采用

- 把 existing horizon-set 实验降级为 diagnostic evidence，而不是直接决定方案。
- 将 training supervision units 从 horizon subsets 推广为 future components、random target
  intervals 或 low-rank label projections。
- 先做 label-side diagnostics：future label covariance、effective rank、top-component energy、
  component-to-horizon contribution。

## 暂不采用

- 不直接复刻完整 TransDF 作为最终方法。
- 不把 TransDF reported SOTA 当成本项目贡献证据。
- 不在 Step 1-3 尚未完成前启动大规模 loss implementation。

## 风险与需复查点

- Component supervision 是否能改善本仓库的 target-set carrier，需要本地 gate。
- Projection basis 是否跨 dataset 稳定，是否会偏向 high-variance variables，需要诊断。
- 如果只优化 top components，可能改善 aggregate trend 但损伤 short-term high-frequency steps。
