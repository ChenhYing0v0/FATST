# Phase5 StageB Step 2/3 Problem Redefinition Report

`current_step`: StageB Step 2/3 problem redefinition after B1 distance-confounded result.

## Scope

[Fact] 本报告基于三类证据：repo 内 B1 full diagnostic、Phase4 历史诊断、Zotero 与外部网络文献调研。

[Boundary] 本报告只判断 StageB 是否还有更强 problem candidate；不授权 model code、training objective 或 remote experiment。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3：重新定义 problem，并判断是否仍值得进入下一轮 diagnostic |
| `problem` | B1 已证明 raw future-unit reliability 被 forecast step/distance 主导；StageB 需要一个能控制 distance confounder 的 reliability problem |
| `existence_evidence` | B1 detrended proxy table、Phase4 label autocorrelation / gradient conflict / residual stability 诊断、外部 TransDF/FreDF/autocorrelation/objective-weighting 文献 |
| `idea` | 将 reliability 从 raw hard future units 改写为 distance-normalized structural residual difficulty，优先测试 train-only `seasonal_residual` 是否预测 detrended residual difficulty |
| `theory_check` | 如果 unit MSE 可分解为 distance trend + structural residual，那么 raw weighting 会退化为 horizon weighting；只有 structural residual 可由 train-only proxy 稳定解释时，StageB 才有独立问题 |
| `design` | 提出 `B3-DSR` diagnostic-only candidate；先做 robustness diagnostic，不做 method implementation |
| `narrative_gate` | partial：问题更强于 B2-RAS，但尚未证明稳定、可训练、可贡献 |
| `effectiveness_gate` | not applicable；当前没有训练新方法 |
| `artifacts` | this report；`docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` |
| `decision` | 找到更强 problem candidate `B3-DSR`；StageB 不关闭，但必须继续停留在 Step 2/3 diagnostic |

## Research Inputs

### Internal Evidence

[Fact] B1 full diagnostic 显示 A6-LBF-r256 的 48-step future units 存在 material heterogeneity：ETTh2 max/min MSE gap `195.16%`，ETTm1 `99.09%`，Weather `248.48%`。

[Counter-Evidence] 同一张表也显示 `Spearman(step, MSE)` 为 ETTh2 `0.99`、ETTm1 `0.99`、Weather `1.00`。这使 raw hard/easy unit 不能直接成为 reliability-aware supervision 的问题定义。

[Fact] B1 的 detrended proxy table 中，`seasonal_residual` 对 detrended MSE 的 Spearman 相关为 ETTh2 `0.35`、ETTm1 `0.59`、Weather `0.81`；detrended top-quartile overlap 均为 `0.50`。相反，它对 raw MSE 的相关为 ETTh2 `-0.74`、ETTm1 `-0.79`、Weather `-0.10`。

[Inference] 这说明 `seasonal_residual` 不是简单跟随 forecast distance 增长的 proxy；它更像是在 step trend 被移除后，捕捉 label-side structural residual difficulty。

[Fact] Phase4 label-basis audit 已显示 train labels 在 `pred_len=720` 下存在强 label covariance / low-rank structure：effective rank 为 ETTh2 `11.47`、ETTm1 `9.74`、Weather `24.18`，top-16 variance 约 `79%-88%`。

[Fact] Phase4 gradient conflict diagnostic 显示 late-vs-early conflict 在 Weather 上明显：`late_337_720` vs `early_1_96` 的 all_shared cosine 为 `-0.0149`，readout cosine 为 `-0.0219`。但旧 adapter-only stabilized routing gate 后续失败，因此不能直接恢复为 StageB method。

### External Literature Evidence

[Fact] TransDF 指出 temporal MSE 面临两个 objective-level 问题：label autocorrelation 和随 forecast horizon 增长的 excessive tasks；其路线是将 label sequence 转到 decorrelated components，并优先对齐 significant components。来源：<https://arxiv.org/abs/2505.17847>。

[Fact] FreDF 将问题定位为历史序列和 label sequence 都存在 autocorrelation，认为 direct forecast 的 conditional-independence 假设忽略了 label autocorrelation，并用 frequency-domain alignment 绕开该 bias。来源：<https://arxiv.org/abs/2402.02399>。

[Fact] 2026 autocorrelation survey 将 deep TSF 的两个核心挑战概括为 history autocorrelation 建模与 label autocorrelation objective 建模，并指出 learning objectives 是近年被补充强调的方向。来源：<https://arxiv.org/html/2603.19899v1>。

[Fact] Forecast stability 的 dynamic loss weighting 文献把 accuracy 与 stability 视作 main/auxiliary objective，并讨论 dynamic loss weights 可能改善 rolling-origin forecast instability，但也指出 gradient-based weighting 会增加 training complexity。来源：<https://arxiv.org/html/2409.18267v2>。

[Fact] Multi-task uncertainty weighting 证明不同任务 loss 的相对权重会显著影响 multi-task performance，但这是通用 MTL 理论，不自动证明 time-series future-unit weighting 有效。来源：<https://arxiv.org/abs/1705.07115>。

## Candidate Problems Evaluated

| Candidate | Verdict | Reason |
| --- | --- | --- |
| Raw future-unit reliability weighting | reject | B1 raw unit difficulty 几乎完全被 forecast step/distance 解释；直接 weighting 会退化为 horizon-distance weighting |
| B2 reliability-aware supervision allocation | reject before implementation | B1 未证明稳定的 non-distance train-only proxy；继续实现会违反 narrative gate |
| Noisy-hard / learnable-hard routing | defer | Phase4 residual stability 支持该概念，但 A6-LBF-specific 证据不足；旧 adapter-only route 已失败 |
| Gradient-conflict routing | defer | Phase4 有 conflict evidence，但它是 method complexity 很高的 routing 问题，不是当前 Step 2/3 的最小重定义 |
| Transformed-label / frequency-domain alignment | related but not current StageB core | 文献支持 label autocorrelation problem，但更接近 StageA objective/head story；若直接采用会偏离 A6-LBF reliability extension |
| Distance-normalized seasonal residual reliability (`B3-DSR`) | propose diagnostic | 它显式控制 step-distance trend，并且 B1 中 `seasonal_residual` 对 detrended MSE 在三数据集均为正相关 |

## Stronger Problem Definition

新的 StageB problem 不应再问：

> 哪些 future units 的 raw MSE 更高？

而应改问：

> 在已固定的 A6-LBF-r256 unified operator 上，是否存在一个 train-only、distance-normalized 的 structural reliability signal，能解释 forecast-distance trend 之外的 residual difficulty？

形式上，令 unit error 为：

$$
E_{d,u} = T_d(s_u) + R_{d,u},
$$

其中 $T_d(s_u)$ 是由 unit start step 或 forecast distance 解释的 monotonic trend，$R_{d,u}$ 是 distance-normalized residual difficulty。

`B3-DSR` 的问题是：

$$
\operatorname{rank}(P^{seasonal}_{train}(d,u))
\;\approx\;
\operatorname{rank}(R_{d,u}),
$$

而不是：

$$
\operatorname{rank}(P_{train}(d,u))
\;\approx\;
\operatorname{rank}(E_{d,u}).
$$

## Why This Gets Past The Step-Distance Confounder

[Strong Evidence] `seasonal_residual` 与 raw MSE 的相关在 ETTh2/ETTm1 为负，在 Weather 近零；因此它不是一个简单的 increasing-with-step proxy。

[Moderate Evidence] 同一 proxy 与 detrended MSE 的相关在三个数据集均为正，且 Weather 最强。这符合“label-side structural residual 在去除 step trend 后解释剩余困难”的假设。

[Theoretical Argument] 若某个 proxy 只通过 future distance 起作用，那么在控制 step trend 后，它与 residual difficulty 的相关应显著衰减到无方向或不稳定。`label_novelty` 符合这个失败模式：raw rho 在 ETTh2/ETTm1 接近 `0.99`，但 detrended rho 变为 `0.05/-0.17`。`seasonal_residual` 的模式相反，因此值得作为 B3 diagnostic candidate。

[Self-Critique] 当前 B3 evidence 仍然弱。每个 dataset 只有 15 个 48-step units，Spearman 估计很粗；`seasonal_residual` 的 period assumption 可能偏向 ETT/Weather 的 known seasonality；top-quartile overlap 只有 `0.50`，不足以直接设计 loss weights。

## Why Other Problems Are Not Stronger Now

1. Raw reliability 的证明失败：B1 的 `Spearman(step, MSE)=0.99/0.99/1.00` 足以说明 raw future-unit hard/easy rank 不是独立 problem。
2. Gradient routing 的证据不够当前化：Phase4 conflict 在旧路径上成立，但 A6-LBF-r256 的 learned-basis operator 可能已经改变 gradient geometry。
3. Dynamic loss weighting 的外部文献支持 objective weighting 作为通用路线，但它不解决 “weight what” 的问题；没有 non-distance proxy 时，仍会变成 engineering patch。
4. TransDF/FreDF 支持 label autocorrelation 是真实问题，但直接转向 transform/frequency loss 会把 StageB 从 reliability extension 改成新的 StageA-style objective contribution。

## Decision

[Decision] 找到了一个比 B2-RAS 更强的问题候选：`B3-DSR`，即 distance-normalized seasonal residual reliability。

[Gate] `B3-DSR` 只通过 Step 2/3 的 problem-candidate gate，尚未通过 Step 4-6 narrative gate。

[Next Required Diagnostic] 在不改模型代码的前提下，先验证：

- detrending 形式是否稳健：linear step residual、rank residual、prefix-normalized residual；
- `seasonal_residual` 的信号是否在不同 block size 下稳定：`24/48/96`；
- 是否能在 dataset 内 bootstrap 后仍保持正相关；
- 是否能区分 structural residual 与 raw high-frequency noise；
- 是否能给出 future method 的 leakage-free train-only computation path。

[Rollback Point] 如果 B3 diagnostic 不能稳定证明 distance-normalized signal，则 StageB 应关闭或转入 label-transform/objective 方向的全新 Stage，而不是继续堆叠 reliability allocation。

## Postscript: B3 Diagnostic Completed

[Fact] B3 diagnostic has been completed at
`analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/`.

[Decision] Result is `partial_pass_needs_stronger_proxy_or_method_boundary`.

[Interpretation] The redefined problem was stronger than raw reliability/B2, but the actual diagnostic did not prove enough robustness for method implementation. `seasonal_residual` aligns well with `linear_step_residual`, yet stricter `rank_step_residual` / `prefix_normalized_residual` and bootstrap checks remain unstable.

[Next] Do not implement B3 loss weighting. The next StageB decision must either find a stronger train-only structural proxy or close StageB and move to a broader label-autocorrelation objective route.
