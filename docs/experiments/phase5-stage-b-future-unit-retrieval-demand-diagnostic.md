# Phase5 StageB B14 Future-Unit Retrieval Demand Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B14-FURD` |
| `current_step` | paused before Step 3；blocked by hierarchical patch-memory equivalence gate |
| `problem` | A6 可能用高度共享的 input sensitivity 服务不同 large future units，而各 unit error gradients 实际要求不同 raw-history evidence |
| `existence_evidence` | B13-A large-unit gradient pressure 在 `12/12` settings 稳定；B13-B1/B2 关闭 recurrent transition，但未否定 non-recurrent region-specific retrieval |
| `idea` | 先比较 label-weighted retrieval demand 与 target-independent A6 input sensitivity；只有 mismatch 成立才设计 trainable retrieval mechanism |
| `theory_check` | 如果 demand 比 existing sensitivity 显著更 unit-specific，则 unit-aware retrieval 有 carrier-specific problem basis；若 sensitivity 已同样分化，则新 retrieval 可能重复 A6 已有 path |
| `design` | after prerequisite pass：contextual-token loss demand vs token-to-output sensitivity；raw-input statistics become robustness controls |
| `narrative_gate` | pending；target-query/retrieval 本身是 prior art，A 只能提供 problem evidence |
| `effectiveness_gate` | not applicable before Step 4-6 |
| `artifacts` | literature audit ready；analyzer/report pending |
| `decision` | full CPE replacement failed；do not run until exact A6-preserving hierarchical memory passes |

## Prerequisite Hold

[Decision] 本 protocol 暂停执行。active A6 已不再依赖 TimeAlign future alignment，因此项目先重构统一的
contextual patch-wise history encoder，而不是继续用 raw-history interface绕过 ETTm1 `P=1`。

前置协议：`docs/experiments/phase5-stage-b-b14-prerequisite-patchwise-encoder.md`。

full contextual replacement已失败并关闭。repair前置 gate通过后，本 diagnostic 的 main interface改为
parameter-free normalized local patches `[B,C,30,48]`；current A6 sensitivity仍从 forecast carrier path对 raw
history求导，再按同一 overlapping patch support聚合。本文需在正式启动前重新同步 gate 与 analyzer。

## Why This Is Not A Retrieval Model Yet

TimePerceiver 已使用 target queries cross-attend input representations；ElasTST 已使用 varied-count future
placeholders；MQ-RNN 已使用 horizon-specific contexts。因此直接实现 query cross-attention 不能通过
SCI narrative gate。

B14-A 只验证 A6 carrier 是否存在一个更精确的 contradiction：

```text
task error demands unit-specific history evidence
but
current A6 input sensitivity remains shared across units
```

不存在该 contradiction 时，不实现 future-unit retrieval。

## Common Raw-History Interface

active A6 presets 的 hidden patch count 为 ETTh2 `48`、ETTm1 `1`、Weather `48`。因此 hidden-patch
attention 不能作为跨数据集 diagnostic。B14-A 对共同的 raw input 使用：

```text
batch_x: [B,720,C]
```

所有 main metrics 在 `720` history positions 上定义。ETTh2/Weather 的 hidden-patch analysis 可后续作为
supplemental，但没有否定权限。

## Unit Sizes

- `U=180`：四个 benchmark-independent large future units；
- `U=240`：三个更大 units，保留用户提出的 information-carrying prior；
- 不使用 `24/48/96`；
- 不使用 `360`，因为只有两个 units，无法稳定区分 adjacent/far structure。

## A6 Forward Path

```text
x_raw [B,720,C]
  -> A6 Normalize/PatchEmbed/Encoder
  -> hidden [B,C,P,D]
  -> coeff [B,C,256]
  -> prediction [B,720,C]
```

model 与 checkpoint frozen；只保留 autograd，不更新参数。

## Metric 1：Label-Weighted Retrieval Demand

对 future unit $m$：

$$
\mathcal{L}_m
=
\operatorname{MSE}(\hat y_{mU:(m+1)U},y_{mU:(m+1)U}),
$$

$$
d_m(t)
\propto
\operatorname{mean}_{b,c}
\left|\frac{\partial \mathcal{L}_m}{\partial x_{b,t,c}}\right|.
$$

$d_m$ 在 history dimension 上做 L1 normalization，形成 `720` 维 non-negative distribution。它描述当前
error signal 对不同 history positions 的需求，不等同于 causal feature attribution。

## Metric 2：Target-Independent Existing Sensitivity

为了避免把 residual magnitude 直接当 retrieval，使用 Hutchinson estimator 估计每个 future unit output
Jacobian 对 raw history positions 的 squared sensitivity。

对 draw $r$ 的 Rademacher vector $v_{m,r}$：

$$
g_{m,r}
=
\frac{\partial \langle v_{m,r},\hat y_m\rangle}{\partial x},
$$

$$
s_m(t)
\propto
\sqrt{
\operatorname{mean}_{r,b,c} g_{m,r}(b,t,c)^2
}.
$$

同一 unit-size、batch、draw 对所有 units 复用同形状 Rademacher pattern，降低 pair comparison 的 Monte
Carlo noise。默认 `4` draws。

## Pairwise Statistics

对 demand profiles $d_i,d_j$ 与 sensitivity profiles $s_i,s_j$ 分别计算：

- cosine similarity；
- Jensen-Shannon divergence；
- adjacent/far means；
- first-last pair；
- pair-distance Spearman；
- demand-pair 与 sensitivity-pair matrix Spearman；
- per-unit demand-vs-sensitivity cosine；
- profile entropy 与 temporal centroid。

核心 mismatch：

$$
\Delta_{cos}
=
\operatorname{mean}_{i<j}\cos(s_i,s_j)
-
\operatorname{mean}_{i<j}\cos(d_i,d_j),
$$

$$
\Delta_{JS}
=
\operatorname{mean}_{i<j}JS(d_i,d_j)
-
\operatorname{mean}_{i<j}JS(s_i,s_j).
$$

正 gap 表示 task demand 比 current model sensitivity 更 unit-specific。

## Controls And Attribution

1. `target-independent sensitivity control`：排除所有差异都来自 A6 output Jacobian；
2. `same Rademacher draws`：减少 unit pairs 之间的 estimator noise；
3. `coeff-gradient pair control`：记录 unit loss 对 A6 coeff 的 signed gradient cosine，判断 raw-history demand
   是否只是 B13 coefficient conflict 的机械重述；
4. `raw-history common interface`：避免 ETTm1 single hidden patch 造成 false negative；
5. `basis geometry cross-reference`：与 B13-A basis-pair artifacts 对照，不把 basis rows 的 geometry 当 retrieval；
6. deterministic bootstrap：对 batch-level gaps 做 `1000` 次 bootstrap。

## Pre-Registered Gate

单个 dataset/unit-size setting 为 `retrieval_demand_mismatch_support`，当同时满足：

1. bootstrap `p05(Delta_cos) > 0.05`；
2. bootstrap `p05(Delta_JS) > 0.01`；
3. mean sensitivity pairwise cosine `>= 0.80`。

整体进入 B14-B 需要至少两个 datasets 的 U180 与 U240 均支持，即至少 `4/6` settings，并且每个支持
dataset 两个 sizes 同向。

通过标签：

```text
partial_pass_retrieval_demand_mismatch
```

这只允许设计 exact parameter-matched B14-B probe：

```text
late_coordinate_control
vs
unit_coordinate_history_retrieval
```

不允许直接实现 paper-core model。

若 demand 与 sensitivity 同样分化：

```text
current_a6_sensitivity_already_unit_specific
```

说明新 retrieval 可能重复 A6 已有 path，回滚 Step 2。

若 demand 本身也高度共享：

```text
retrieval_demand_problem_not_supported
```

若 Hutchinson variance、non-finite gradients、zero-norm profiles 或跨 batch instability 破坏比较：

```text
diagnostic_invalid_for_direction_rejection
```

只修复 diagnostic，不否定 broader future-unit direction。

## Expected Artifacts

- `b14_future_unit_retrieval_batches.csv`；
- `b14_future_unit_retrieval_pairs.csv`；
- `b14_future_unit_retrieval_profiles.csv`（batch-mean profiles）；
- `b14_future_unit_retrieval_summary.csv`；
- `b14_future_unit_retrieval_bootstrap.csv`；
- `b14_future_unit_retrieval_report.md`。

## Rollback Boundary

B14-A 只能决定 carrier-specific retrieval problem 是否成立：

- A 失败：回滚 Step 2，不实现 cross-attention/query retrieval；
- A 通过：进入 B14-B probe design，仍不进入 Step 4-6；
- B14-B 通过后，才允许重新做 fresh primary-source audit 与 narrative novelty gate；
- 任一正结果都不能复活 B13 recurrence、B9 hard stage、B8 late coefficient correction 或 B11 basis field。
