# Phase5 StageB: A6-LBF Reliability-Aware Supervision Redesign

`current_step`: StageB Step 2/3 problem redefinition；B1 diagnostic candidate
进入 Step 4-6 narrative gate。本文档只定义 StageB 的问题、诊断和候选设计；不启动代码实现或远程训练。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3：重新定义基于 `A6-LBF-r256` 的 future-aware reliability problem；B1 diagnostic 进入 Step 4-6 |
| `problem` | `A6-LBF-r256` 已解决 unified forecast operator 的主要 architecture 问题，但当前 training loss 仍对 future units 使用全局预测压力、reconstruction pressure 和 alignment pressure；它没有区分 learnable-hard、noisy-hard、easy/shared prefix 或 dataset-specific unreliable units |
| `existence_evidence` | A6-LBF-r256 已成为 clean carrier；其相对 fixed-horizon TimeAlign 的收益跨 dataset 不均衡；A6 capacity-native gate 曾提示 official-last trajectory drift；当前仍缺少 A6-LBF-specific future-unit reliability diagnostic |
| `idea` | 先用 diagnostic-only B1 验证 A6-LBF 上 future-unit reliability heterogeneity 是否真实、是否可由 train-only proxy 捕捉；只有 B1 通过后，才设计 reliability-aware supervision allocation |
| `theory_check` | 如果 future units 的 prediction error、error volatility 或 optimization drift 存在结构性差异，则统一 dense pressure 可能把 noisy units、learnable-hard units 和 easy units 混在一起优化；但如果差异只是 horizon length 或 dataset scale，则 StageB 不具备独立贡献 |
| `design` | B1 post-hoc reliability diagnostic completed；B2 reliability-aware supervision allocation rejected before implementation |
| `narrative_gate` | B1: `partial_pass_distance_confounded`；B2: rejected because the reliability signal is still distance-confounded |
| `effectiveness_gate` | B1 不以 MSE improvement 过关；它必须证明 problem existence。B2 才以 full evaluation horizons 的 MSE/MAE、segment behavior、stability 和 trace consistency 过关 |
| `artifacts` | planned: `analysis/phase5_stage_b_reliability_diagnostic_20260706/` |
| `decision` | 当前推进 B1 diagnostic，不实现 StageB method；若 B1 失败，rollback 到 StageB Step 2/3，并保留 A6-LBF-r256 作为独立 contribution |

## What We Plan To Test

[Hypothesis] 在 `A6-LBF-r256` 上，future prediction units 的 reliability 不是均匀的。某些 units
属于 learnable-hard：当前 error 高，但 train-side signal 稳定、可被额外 pressure 改善；另一些
units 属于 noisy-hard：当前 error 高，但 signal volatility 高、额外 pressure 可能带来 overfit 或
early-prefix collateral damage。

B1 要测试的不是新方法是否提升 MSE，而是 StageB 的问题是否真实：

> A6-LBF 的剩余 gap 是否来自 future-unit reliability heterogeneity，而不是来自 head capacity、
> rank 不足、旧 StageA interface failure 或 evaluation-horizon artifact？

## Why It Matters

[Fact] StageA 的论文叙事已经由 `A6-LBF-r256` 承担：它是 prefix-native learned-basis forecast
operator，用一个 unified model 覆盖 `96/192/336/720`。

[Inference] StageB 若要成为第二贡献，必须站在 StageA 之上，而不是替代 StageA。合理衔接是：

1. StageA 解决 unified prediction operator。
2. StageB 研究 unified operator 的 supervision reliability。
3. StageB 的机制只改变训练压力或监督 allocation，不改变 A6-LBF 的 inference interface。

[Boundary] StageB 不能写成以下内容：

- 旧 B0 `w_recon/w_align` ablation 的正式升级；
- teacher/self-teacher/EMA 稳定化；
- target-query、QBR、nested decoder 或 residual adapter 的复活；
- 用 validation/test residual 直接作为 training signal；
- 手工按 `96/192/336/720` evaluation horizon routing。

## Problem Definition

设 A6-LBF unified forecaster 为：

$$
\hat{Y}_{1:H}=f_\theta(X,H),
$$

其中 `hidden: [B, C, R]` 经过 `learned_basis_coeff` 得到 coefficients，再与
`learned_temporal_basis[:H]` 组合得到 prefix-native prediction。

当前训练目标可以概括为：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+ w_{recon}\mathcal{L}_{recon}
+ w_{align}\mathcal{L}_{align}.
$$

其中 $\mathcal{L}_{pred}$ 可以是 `full` 或 `multi-prefix`，但它没有显式区分 future unit
$u=[s:e]$ 的 reliability：

$$
\mathcal{L}_{pred}
=
\sum_{u\in\mathcal{U}} \alpha_u \mathcal{L}_u,
$$

当前 $\alpha_u$ 主要由 loss mode 或 prefix choice 隐式决定，而不是由 train-side reliability
证据决定。

StageB 的核心问题是：

> 是否存在一个 horizon-free、train-side 可估计的 reliability signal，使 A6-LBF 的 supervision
> pressure 能区分 learnable-hard 与 noisy-hard future units，并在不改变 inference interface 的前提下
> 提升 unified forecasting stability？

## Existence Evidence

[Strong Evidence] A6-LBF-r256 已经是可用 carrier：主线文档记录其相对 fixed-horizon per-horizon
TimeAlign 达到 `9/12` MSE wins，overall MSE `-4.82%`；相对 official unified TimeAlign 达到
`11/12` MSE wins，overall MSE `-1.92%`。

[Moderate Evidence] A6-LBF 的收益不是完全均匀：相对 fixed-horizon baseline，ETTh2 为 `4/4`
wins 且 mean MSE `-10.89%`，ETTm1 为 `3/4` wins 且 `-1.46%`，Weather 为 `2/4` wins 且
`-0.36%`。这提示 StageB 应关注 stability 和 allocation，而不是继续扩大 head rank。

[Moderate Evidence] A6 capacity-native gate 曾显示 official-last trajectory drift，尤其 ETTh2
best epoch 早于 last epoch。这不是改变 protocol 的理由，但支持继续诊断 optimization trajectory
与 future-unit reliability。

[Missing Evidence] 目前还没有 A6-LBF-specific 的 future-unit reliability diagnostic。旧 StageA 的
A4/A4S、H1C 或 B0 证据不能直接升级为 StageB method；最多作为历史提示。B1 必须重新在 A6-LBF-r256
carrier 上验证。

## B1 Diagnostic Design

### Candidate

| Field | Content |
| --- | --- |
| `id` | `B1-RED` |
| `status` | `diagnostic_only` |
| `name` | A6-LBF reliability evidence diagnostic |
| `role` | Prove or reject StageB problem existence before method implementation |
| `carrier` | `A6-LBF-r256` on official-source TimeAlign |
| `method_change` | none |

### Data And Artifacts

B1 优先复用 A6-LBF-r256 existing artifacts：

- `metrics_by_target_horizon.csv`;
- `segment_metrics.csv` if present in run directories;
- `predictions_test.npz` if present in run directories;
- `training_log.csv`.

若 local synced raw artifacts 缺少 `predictions_test.npz` 或 segment metrics，则下一步只补 diagnostic
export，不改变 model 或 training objective。

Planned output:

| Artifact | Meaning |
| --- | --- |
| `stage_b_a6_lbf_unit_reliability.csv` | per dataset / future unit 的 MSE、MAE、error volatility、sample count |
| `stage_b_a6_lbf_proxy_alignment.csv` | train-only proxy 与 held-out unit difficulty 的相关性 |
| `stage_b_a6_lbf_trajectory_drift.csv` | per dataset 的 first/best/last validation drift |
| `stage_b_a6_lbf_reliability_report.md` | B1 完整结论和 StageB rollback/pass decision |

Diagnostic status:

- `analysis/phase5_stage_b_reliability_diagnostic_20260706/` 已用现有
  `predictions_test.npz`、`training_log.csv` 和 local train split labels 生成 full B1 report；
- 该 report 证明 48-step future unit heterogeneity material；
- 但 `Spearman(step, MSE)` 在三个数据集上分别为 `0.99/0.99/1.00`，说明 heterogeneity 主要被
  forecast-distance confounder 解释；
- train-only proxy 对 detrended MSE 的信号不稳定，不能支撑 B2 method；
- 因此 B1 decision 为 `partial_pass_distance_confounded`，B2 仍不得进入实现。

### Unit Definition

第一版使用 horizon-free future blocks：

$$
\mathcal{U}=\{[1,48],[49,96],\dots,[673,720]\}.
$$

`block_size=48` 的理由：

- 不等于 benchmark horizons；
- 能细分 early/middle/late future regions；
- 仍可聚合到 `96/192/336/720` evaluation prefix 做 sanity check。

### Diagnostic Metrics

For each dataset $d$ and unit $u$:

$$
E(d,u)=\operatorname{mean}_{i,t,c}\left(\hat{Y}_{i,t,c}-Y_{i,t,c}\right)^2,
\quad t\in u.
$$

Reliability heterogeneity:

$$
G(d)=\frac{\operatorname{mean}_{u\in TopHard(d)}E(d,u)}
{\operatorname{mean}_{u\in Easy(d)}E(d,u)}.
$$

Error volatility:

$$
V(d,u)=\operatorname{std}_{i}\left(
\operatorname{mean}_{t,c}(\hat{Y}_{i,t,c}-Y_{i,t,c})^2
\right).
$$

Train-only proxy candidates:

| Proxy | Source | Allowed For Future Training? | Risk |
| --- | --- | --- | --- |
| `label_novelty` | train batch history + future label | yes | may over-emphasize shocks |
| `local_variation` | train future label differences | yes | may equal high-frequency noise |
| `seasonal_residual` | train label vs seasonal naive reference | yes | dataset-specific period assumption |
| `online_loss_ema` | training trace only | yes, after implementation | can become unstable feedback |
| held-out error | validation/test prediction artifacts | diagnostic only | leakage if used for training |

The key check is whether a train-only proxy ranks difficult units similarly to held-out prediction difficulty:

$$
\rho_d = \operatorname{SpearmanRankCorr}(P_{train}(d,u), E_{heldout}(d,u)).
$$

## B1 Narrative Gate

B1 passes Step 2/3 only if all are true:

1. [Problem existence] At least two datasets show material future-unit heterogeneity, not only a monotonic
   horizon-length effect.
2. [Train-side access] At least one train-only proxy has non-trivial alignment with held-out hard units
   (`Spearman rho >= 0.25` as a first audit threshold, not a publishable claim).
3. [Carrier specificity] The diagnostic is computed on `A6-LBF-r256`, not on archived StageA variants.
4. [No leakage boundary] Any validation/test error is used only to judge the proxy, not as a training signal.
5. [Paper relevance] The resulting problem can be stated as reliability-aware supervision for unified
   forecasting, not as post-hoc dataset tuning.

B1 fails if any of the following happens:

- hard units are explained almost entirely by fixed step index or evaluation horizon identity;
- train-only proxies cannot identify hard units beyond noise;
- heterogeneity is present only on one dataset with no stable cross-dataset pattern;
- the only plausible method would require validation/test residual routing.

Current B1 result:

- problem existence: partial；
- train-side access: partial and dataset-dependent；
- carrier specificity: pass，诊断对象是 `A6-LBF-r256`；
- leakage boundary: pass，validation/test prediction errors 只作为 diagnostic labels；
- paper relevance: fail for immediate B2，因为当前信号仍被 step-distance confounder 主导。

## B2 Method Candidate Boundary

`B2-RAS` is rejected for the current B1 evidence. The formulas below are kept only as the
previous candidate boundary; they are not authorized for implementation unless a new Step 2/3
problem definition removes the distance confounder.

If B1 passes, the minimal method candidate is:

$$
\mathcal{L}_{B2}
=
\mathcal{L}_{full}
+
\lambda
\sum_{u\in\mathcal{U}} a_t(u\mid p_{train})\mathcal{L}_u.
$$

Where:

- $\mathcal{L}_{full}$ preserves dense A6-LBF coverage;
- $a_t(u\mid p_{train})$ is computed only from train-side proxy;
- $\lambda$ is small in the first gate to avoid destroying StageA carrier capacity;
- inference remains exactly A6-LBF prefix-native readout.

B2 is not allowed to:

- change `learned_temporal_basis` interface;
- route by `target_horizon` identity;
- use validation/test residuals during training;
- restore teacher/EMA/target-query/QBR/nested paths.

## Effectiveness Gate For Future B2

B2 can only become a paper-core candidate if it satisfies:

1. Main MSE/MAE: improves over A6-LBF-r256 on at least `8/12` dataset-horizon settings or produces a
   clear average gain without Weather collapse.
2. No early-prefix damage: h96 and early blocks cannot regress enough to erase StageA unified benefit.
3. Stability: last-val drift is reduced or not worsened relative to A6-LBF-r256.
4. Trace consistency: supervision trace shows the intended reliability allocation actually occurred.
5. Mechanism diagnostics: gains concentrate in units predicted by the train-only reliability proxy; if gains
   appear elsewhere, the method story must be revised.

## Immediate Next Action

Historical B1 action was completed:

```text
input: A6-LBF-r256 predictions_test.npz / segment metrics / training_log.csv
output: analysis/phase5_stage_b_reliability_diagnostic_20260706/
decision: B1 pass/fail for StageB problem existence
```

Current decision:

- B1 result is `partial_pass_distance_confounded`;
- B2-RAS is rejected before implementation;
- StageB has returned to Step 2/3 and proposed `B3-DSR` as a diagnostic-only problem candidate;
- the active next protocol is
  `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md`.

No remote experiment should be launched before B3 diagnostic passes its problem-existence gate and the Stage Ledger is updated again.
