# Phase5 StageB: Distance-Normalized Seasonal Residual Diagnostic

`current_step`: StageB Step 2/3 problem-existence diagnostic for `B3-DSR`.

本文档定义下一步 diagnostic protocol。它不授权 model implementation、loss implementation 或 remote training。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3：在 B1 distance-confounded 后重定义 reliability problem |
| `problem` | raw future-unit error 主要由 forecast distance 主导；需要测试是否存在 distance-normalized structural residual difficulty |
| `existence_evidence` | B1 中 `seasonal_residual` 对 detrended MSE 在 ETTh2/ETTm1/Weather 上均为正相关 |
| `idea` | 用 train-only seasonal residual 预测去除 step-distance trend 后的 held-out residual difficulty |
| `theory_check` | 如果 proxy 预测的是 structural residual 而非 horizon distance，则其 raw correlation 可弱或为负，但 detrended correlation 应稳定为正 |
| `design` | post-hoc diagnostic only；复用 A6-LBF-r256 predictions 和 local train split labels |
| `narrative_gate` | `partial_pass_needs_stronger_proxy_or_method_boundary` |
| `effectiveness_gate` | not applicable；无新方法训练 |
| `artifacts` | `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` |
| `decision` | B3 diagnostic completed；不能实现 reliability-aware supervision，除非后续找到更强 train-only structural proxy 或更清晰 method boundary |

## What We Plan To Test

[Hypothesis] A6-LBF-r256 的 unit error 可以拆成 forecast-distance trend 与 structural residual difficulty：

$$
E_{d,u}=T_d(s_u)+R_{d,u}.
$$

B3 测试的是：

$$
P^{seasonal}_{train}(d,u)\rightarrow R_{d,u},
$$

而不是：

$$
P_{train}(d,u)\rightarrow E_{d,u}.
$$

## Why It Matters

[Fact] B1 已拒绝 B2-RAS，因为 raw unit reliability 与 step-distance 高度混合。

[Inference] 如果 B3 能证明 distance-normalized residual difficulty 可由 train-only seasonal residual 捕捉，StageB 才能重新进入 Step 4-6，讨论 supervision allocation 是否有 paper-level novelty。

[Boundary] B3 不允许：

- 用 validation/test error 作为训练信号；
- 直接按 `96/192/336/720` horizon identity 加权；
- 在未证明 proxy 稳定前实现 loss weighting；
- 恢复 teacher/EMA/QBR/nested/residual adapter 旧代码。

## Data And Artifacts

Input:

| Source | Role |
| --- | --- |
| A6-LBF-r256 `predictions_test.npz` | held-out diagnostic labels only |
| A6-LBF-r256 `training_log.csv` | optional trajectory context |
| local train split labels | compute train-only seasonal residual proxy |

Planned output:

| Artifact | Meaning |
| --- | --- |
| `stage_b_b3_detrending_robustness.csv` | proxy vs residual difficulty under multiple detrending methods |
| `stage_b_b3_blocksize_robustness.csv` | block size `24/48/96` stability |
| `stage_b_b3_bootstrap_stability.csv` | bootstrap confidence / sign stability |
| `stage_b_b3_report.md` | B3 pass/fail decision and rollback point |

## Diagnostic Design

### Unit Definitions

Evaluate three horizon-free block sizes:

$$
\mathcal{U}_b = \{[1,b],[b+1,2b],\dots,[720-b+1,720]\},
\quad b\in\{24,48,96\}.
$$

### Residual Difficulty Labels

For each dataset $d$ and unit $u$:

$$
E_{d,u}=\operatorname{mean}_{i,t,c}(\hat{Y}_{i,t,c}-Y_{i,t,c})^2.
$$

Construct distance-normalized residual labels with at least three variants:

| Label | Definition | Purpose |
| --- | --- | --- |
| `linear_step_residual` | residual after fitting $E_{d,u}=a_d+b_d s_u+\epsilon_{d,u}$ | simplest distance control |
| `rank_step_residual` | residual rank after removing monotonic step-rank trend | robust to nonlinear monotonic trends |
| `prefix_normalized_residual` | unit error divided by prefix-local or neighboring trend baseline | tests local relative difficulty |

### Train-Only Proxy

Primary proxy:

$$
P^{seasonal}_{train}(d,u)
=
\operatorname{mean}_{train,t,c}
\left|Y_{t,c}-Y_{t-p_d,c}\right|,
\quad t\in u.
$$

`p_d` must be selected from train-side dataset periodicity only. Any dataset-specific period assumption must be recorded in the report.

Secondary controls:

- `label_novelty`;
- `local_variation`;
- shuffled-unit proxy;
- pure step-index proxy.

## Narrative Gate

B3 can advance from Step 2/3 to Step 4-6 only if all conditions hold:

1. `seasonal_residual` has positive alignment with distance-normalized residual labels on at least two datasets and no strong contradiction on the third.
2. The sign and approximate rank are stable under at least two detrending variants.
3. The signal is not reproduced by pure step-index proxy after detrending.
4. Bootstrap or leave-unit-out checks do not collapse the finding to one outlier block.
5. The future method can compute the proxy from train-side information only.

B3 fails if:

- correlations disappear after stricter detrending;
- the signal works only for Weather or only for ETT;
- the proxy mainly identifies noisy high-frequency labels without learnable residual structure;
- any plausible method would need held-out residual leakage.

## Effectiveness Gate For Future Method

Not active yet. If B3 passes and a later B4 method is designed, it must still pass:

- A6-LBF-r256 as baseline;
- full `96/192/336/720` MSE/MAE matrix;
- no early-prefix damage;
- stability / last-vs-best drift audit;
- trace evidence that allocation followed train-only structural residual proxy.

## Decision

[Decision] B3 diagnostic completed with `partial_pass_needs_stronger_proxy_or_method_boundary`.

[Evidence] `seasonal_residual` 对 `linear_step_residual` 的 Spearman 在三数据集和三种 block size 下均为正，约为 ETTh2 `0.31-0.38`、ETTm1 `0.43-0.62`、Weather `0.81-0.83`。

[Counter-Evidence] 该信号在 stricter residual labels 下不稳定：ETTh2/ETTm1 的 `rank_step_residual` 出现负相关，Weather 的 `prefix_normalized_residual` 在 block `24/96` 为负；若只依赖 linear detrending，会把 B3 夸大为 method-ready。

[Decision] B3 is stronger than B2-RAS because it explicitly controls the forecast-distance confounder, but it is not method-ready.

[Rollback Point] If B3 fails, StageB should close or restart from a broader label-autocorrelation objective problem, not implement reliability-aware weighting.
