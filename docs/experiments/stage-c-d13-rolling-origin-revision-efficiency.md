# StageC D13 Rolling-Origin Revision Efficiency Diagnostic

## Status

| Field | Value |
| --- | --- |
| stage | future independent forecast-revision project |
| current_step | deferred Step 2-3 problem verification |
| role | diagnostic_only |
| method_training | false |
| test_access | false |
| current_project_status | `deferred_next_paper`；not active |
| restart_entry | root `New-idea.md` |
| D13-A | not currently authorized |
| D13-B | conditional on future D13-A pass |
| rollback | D13-A fail -> Step 2；D13-B fail -> NIFRO patch-direct hypothesis Step 2 |

[Decision] 2026-07-15起，该protocol随forecast-revision surface转为下一篇独立SCI idea。当前
`R_2026_FATST`主线已回到fixed-past unified multi-horizon generation，active protocol为D14。本文件只作未来
restart artifact，不得由当前ledger直接启动。

## What We Plan To Test

D13不训练NIFRO或IARL。它只用已冻结的A6-LBF-natural checkpoints回答：

1. 对同一target，new-origin forecast是否通常比old-origin forecast更准确；
2. forecast revision的squared energy是否换回了相应的MSE improvement；
3. train-only scalar revision calibration能否在validation稳定改善A6；
4. 只有前三项通过后，newly arrived patch是否包含可预测的ideal correction信息。

## Why It Matters

新主线把unified multi-horizon forecasting定义为causal forecast-revision surface：

$$
F(o,\tau)=E[Y_\tau\mid\mathcal F_o].
$$

NIFRO负责表示surface，IARL负责约束revision。若A6本身不存在跨dataset revision inefficiency，
两项机制都缺少practical necessity；因此D13必须发生在method Step 4-7之前。

## Data And Artifact Construction

### Checkpoints

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- seeds：2021、2022、2023；
- profiles：configs/stage_c_five_dataset_natural_profiles.json；
- checkpoint：各profile既有best-validation A6 checkpoint；
- frozen replacement：false；每个origin独立运行完整A6；
- train：只拟合scalar/probe controls；
- validation：唯一final gate；
- test：不得读取。

### Same-Target Pairing

共同origin gaps：

$$
\delta\in\{15,30,60\}.
$$

new-origin horizons：

$$
h\in\{48,96,144,192,288,336,512\},
$$

只保留$h+\delta\le720$。对target $\tau=o+\delta+h$：

$$
\hat y_{old}=\hat y_{o,\tau},
\qquad
\hat y_{new}=\hat y_{o+\delta,\tau}.
$$

old/new origins各自使用其时点之前的720-step history，禁止把new-origin history复用于old forecast。

### Fixed-Window Caveat

A6使用rolling 720-step input。窗口前移时既加入new block，也移除expired block，所以effective model
inputs并非严格nested。full-information conditional-mean relation只作为ideal reference，不能作为A6
必然满足的定理。实现必须保存added、expired与shared-middle blocks，并完成window-expiry attribution。
若主要signal来自expired block，D13-A不得支持NIFRO/IARL nested-information claim。

## Statistics

对每个dataset、seed、gap、horizon、channel group：

$$
\Delta=\hat y_{new}-\hat y_{old},
$$

$$
R=E[\Delta^2],
\qquad
G=E[(y-\hat y_{old})^2-(y-\hat y_{new})^2],
$$

$$
C=E[(y-\hat y_{new})\Delta].
$$

必须验证：

$$
G-R-2C=0
$$

只允许floating-point tolerance内误差。

输出统计：

| Statistic | Definition | Meaning |
| --- | --- | --- |
| revision_energy | $R$ | forecast update幅度 |
| accuracy_gain | $G$ | new origin带来的MSE改善 |
| revision_error_moment | $C$ | new error与revision的相关moment |
| revision_efficiency | $G/(R+\epsilon)$ | revision energy被accuracy gain解释的比例 |
| harmful_revision_fraction | point error在revision后增大的cell比例 | update造成伤害的频率 |
| alpha_star | $E[(y-\hat y_{old})\Delta]/(R+\epsilon)$ | scalar-optimal revision strength |
| calibrated_delta_mse | train-fit $\alpha$在validation相对raw A6的MSE变化 | 最小可恢复headroom |
| vertical_volatility | same-target revisions的aggregate magnitude | 与stability literature对齐的secondary metric |

## Mandatory Controls

1. no-revision：$\alpha=0$；
2. raw A6 revision：$\alpha=1$；
3. train-fit scalar calibration：$\hat y_{old}+\alpha\Delta$；
4. origin-shuffled：破坏nested information pairing；
5. target-shifted：破坏same-target alignment；
6. added-only / expired-only / shared-middle attribution；
7. long-context/no-expiry control若接口可行；不可行时必须降低conclusion confidence；
8. pooled与per-gap结果同时报告；
9. MSE primary，L1只作replication。

参数$\alpha$只能在chronological train pairs拟合。validation上不得重新选择或裁剪。

## D13-A Gate

单dataset pass需同时满足：

1. $G>0$；
2. 至少2/3 seeds的revision efficiency同方向偏离1，且$|\eta-1|\ge0.10$；
3. train-fit scalar calibration在validation MSE改善至少0.3%，至少2/3 seeds为正；
4. no-revision的new-origin MSE差于raw A6；
5. aligned effect强于origin-shuffled与target-shifted controls；
6. revision inefficiency不能主要由expired block解释。

总体pass：

1. 至少3/5 datasets pass；
2. five-dataset macro validation MSE calibration为正；
3. 不存在任一dataset超过5%的严重退化；
4. 所有identity、pairing、split与finite-value invariants通过。

## D13-B Conditional Gate

D13-A通过后才允许。

### Probe

train-only ridge/linear probe预测ideal correction $y-\hat y_{old}$，比较：

1. old-state-only；
2. new-patch-only；
3. old-state + new-patch；
4. old-state + time-shifted-new-patch。

### Pass

至少3/5 datasets中：

1. old-state + new-patch优于old-state-only；
2. 优于time-shifted control；
3. 至少2/3 seeds方向一致；
4. validation improvement超过预注册small-effect threshold。

threshold必须在实现前基于metric scale统一冻结，不允许看validation结果后修改。

## Decision Matrix

| D13-A | D13-B | Decision |
| --- | --- | --- |
| fail | not run | NIFRO/IARL joint route closed；rollback Step 2 |
| pass | fail | patch-direct NIFRO unsupported；IARL只保留为Step 2 question |
| pass | pass | NIFRO/IARL只进入formal Step 4-6 |
| invalid | not judged | repair diagnostic；不得拒绝方向 |

## Required Artifacts

实现后必须生成：

1. effective protocol/config JSON；
2. checkpoint/profile/split hashes；
3. pair counts by dataset/seed/gap/horizon；
4. raw cell or sufficient-statistic artifacts；
5. dataset/seed summary CSV；
6. control comparison CSV；
7. identity/invariant report；
8. plots：revision energy vs accuracy gain、alpha、harmful fraction；
9. Chinese research interpretation；
10. failure attribution与11-step decision。

## Source Boundary

D13使用外部工作来定义mandatory comparisons，不用Zotero coverage判断novelty：

- MQ-RNN / forking-sequences：https://arxiv.org/abs/1711.11053
- Forking-Sequences：https://arxiv.org/abs/2510.04487
- N-BEATS-S：https://doi.org/10.1016/j.ijforecast.2022.06.007
- On forecast stability：https://doi.org/10.1016/j.ijforecast.2025.01.006
- Forecast AC：https://arxiv.org/abs/2601.10863
- Multi-horizon rationality bounds：https://doi.org/10.1080/07350015.2012.634337

完整mainline与novelty审计见
analysis/stage_c_post_d12_revision_surface_mainline_20260715/systematic_review_and_mainline_redesign.md。
