# Phase4: Component-Balanced Objective Design

[Status] 已暂停。本文仅保留为 `Horizon Supervision Scheduling` 路线下的历史候选设计。
除非后续 HSS decision report 重新激活 component-basis supervision，否则它不是当前
Phase4 implementation target。

`current_step`: 11-step loop 的 Step 4-6。

## Context

[Fact] Label Basis Audit 已通过：`pred_len=720` 的 train-label sequence 有强 off-diagonal
correlation 和低 effective rank。

[Fact] Existing-Residual Projection Audit 显示 R.3 residual 也有 component-space structure：

- gap mean residual top16 energy share: `0.789`;
- non-gap mean residual top16 energy share: `0.828`;
- gap minus non-gap top16-over-label ratio: `-0.054`;
- H720 segment gap minus non-gap top16 reconstruction share: `+0.002`。

[Decision] 这支持 component-space supervision 继续推进，但不支持 top-only TransDF-style loss。
Known gaps 没有更集中在 dominant top components；只强化 top components 可能改善趋势拟合，却继续
忽略 lower-variance detail / local error。

## Step 4: Idea

候选 idea 命名为 `Component-Balanced Objective`。

核心思想：

> 用 train-label component basis 替代 evaluation horizons 作为辅助 supervision basis；
> 但不只监督 top components，而是在 time-domain MSE 之外加入 variance-balanced component
> residual penalty，使 dominant components 与 detail components 都有受控梯度。

## Step 5: Theory Check

标准 time-domain loss：

$$
\mathcal{L}_{time}
=
\frac{1}{BHD}\lVert \hat{Y}-Y\rVert_2^2
$$

将 residual reshape:

$$
E = \hat{Y}-Y,\qquad E' \in \mathbb{R}^{(BD)\times H}
$$

使用 train-label eigenbasis:

$$
P_H \in \mathbb{R}^{H\times H},\qquad \lambda_1\ge\dots\ge\lambda_H
$$

Component residual:

$$
C = E'P_H
$$

候选 component loss：

$$
\mathcal{L}_{comp}
=
\frac{1}{H}\sum_{k=1}^{H} w_k \rho(C_k)
$$

其中 $\rho$ 可先用 squared error 或 smooth L1；权重使用 variance-balanced form：

$$
w_k
=
\operatorname{clip}\left((\lambda_k+\epsilon)^{-\beta}, w_{\min}, w_{\max}\right)
$$

并归一化到平均权重为 1。最终 loss:

$$
\mathcal{L}
=
(1-\alpha)\mathcal{L}_{time}
+
\alpha\mathcal{L}_{comp}
$$

### Why not top-only

[Fact] Residual Projection Audit 中 gap rows 的 top16-over-label ratio 低于 non-gap rows。
因此 top-only supervision 不一定修复 specialist gaps。

[Inference] 更合理的做法是保留 time-domain MSE，同时通过 component balancing 防止训练只关注
高方差 dominant components。

## Step 6: Minimal Design

### Candidate Variants

1. `component_top16_l1`
   - diagnostic baseline；
   - 只监督 top16 components；
   - 预期不作为最终候选，用于验证 top-only 风险。

2. `component_balanced_beta025`
   - $w_k=(\lambda_k+\epsilon)^{-0.25}$；
   - mild balancing。

3. `component_balanced_beta050`
   - $w_k=(\lambda_k+\epsilon)^{-0.5}$；
   - stronger balancing，风险是放大噪声/detail components。

4. `component_group_balanced`
   - 将 components 按 cumulative variance 分组，例如 `0-80%`, `80-95%`, `95-100%`；
   - 每组平均 pressure 相同；
   - 更稳定，但实现稍复杂。

### First Implementation Scope

[Decision] 第一版只实现 `component_top16_l1` 和 `component_balanced_beta025/beta050`。
不实现 group-balanced，避免过早扩大变量。

训练仍使用 R.3 carrier：

- `model_variant=target_set`;
- `step_loss_weighting=prefix_risk`;
- `target_horizons=96,192,336,720`;
- evaluation remains `96,192,336,720`。

### Required Artifacts

- per-dataset component basis metadata；
- effective config with component loss parameters；
- training log columns:
  - `train_component_loss`;
  - `train_time_loss`;
  - `train_component_top16_energy_share` if cheap；
- standard metrics by horizon and segment；
- existing prefix consistency outputs。

### Gate

First local/remote gate passes only if:

1. mean MSE vs R.3 improves or remains within `+0.2%`;
2. no dataset average degrades above `+1.0%`;
3. H96 and H720 specialist gaps do not worsen;
4. prefix mismatch remains near numerical zero;
5. component loss does not simply reduce top-component error while increasing detail/segment error.

### Rollback

- If top-only helps but balanced variants fail, the result is likely regularization and not a stable paper story.
- If balanced variants improve only Weather or only ETT, inspect dataset-specific basis before continuing.
- If all component variants fail, return to Candidate B: random target-interval supervision.

## 11-Step Record

- `current_step`: Step 4-6。
- `problem`: training supervision units should not be fixed to evaluation horizons.
- `existence_evidence`: label basis audit and residual projection audit.
- `idea`: component-balanced objective over train-label basis.
- `theory_check`: component basis decorrelates future labels; variance balancing avoids top-only bias.
- `design`: implement three minimal loss variants on R.3 carrier.
- `gate`: improve or preserve R.3 while reducing known horizon/segment gaps.
- `artifacts`: this document; residual projection report.
- `decision`: proceed to implementation only for minimal component objective variants.
