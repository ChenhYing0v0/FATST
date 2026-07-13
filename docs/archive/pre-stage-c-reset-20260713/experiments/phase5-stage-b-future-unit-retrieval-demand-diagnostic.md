# Phase5 StageB B14 Future-Unit Retrieval Demand Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B14-FURD` |
| `current_step` | Step 3 decision completed；rollback Step 2/3 |
| `problem` | A6 可能用高度共享的 input sensitivity 服务不同 large future units，而各 unit error gradients 实际要求不同 raw-history evidence |
| `existence_evidence` | B13-A large-unit gradient pressure 在 `12/12` settings 稳定；B13-B1/B2 关闭 recurrent transition，但未否定 non-recurrent region-specific retrieval |
| `idea` | 比较 model-independent label-patch dependence 与 target-independent A6 sensitivity；只有 mismatch 成立才设计 trainable retrieval mechanism |
| `theory_check` | label-history dependence必须绕开 A6 Jacobian；否则 error-gradient demand与 sensitivity共享同一 computation path，负结果没有方向否定权 |
| `design` | DCT-8 linear CKA label dependence vs A6 sensitivity on 29 valid `K48-S24` supports；shuffle + coverage controls |
| `narrative_gate` | pending；target-query/retrieval 本身是 prior art，A 只能提供 problem evidence |
| `effectiveness_gate` | not applicable before Step 4-6 |
| `artifacts` | A1 + A2 3-dataset artifacts and cross-dataset report complete |
| `decision` | `blocked_by_nonrobust_label_patch_evidence`；no trainable retrieval；rollback Step 2/3 |

## Prerequisite Hold

[Decision] 本 protocol 暂停执行。active A6 已不再依赖 TimeAlign future alignment，因此项目先重构统一的
contextual patch-wise history encoder，而不是继续用 raw-history interface绕过 ETTm1 `P=1`。

前置协议：`docs/experiments/phase5-stage-b-b14-prerequisite-patchwise-encoder.md`。

full contextual replacement已失败并关闭。repair前置 gate通过后，本 diagnostic 的 main interface改为
parameter-free normalized local patches。初始 30-token interface含 right replication padding；为避免末端 evidence
重复，Step 3改用 29 个 valid `[B,C,29,48]` patches。

[Result] valid interface本地 checker已证明 manual slice与 overlap-add reconstruction exact，且 A6 output exact
不变。Step 3 analyzer已实现；远程运行时每个 batch都重新验证该 contract，任何失败均使 diagnostic invalid。

## Returned A1 And Failure Attribution

A1在 ETTh2、ETTm1、Weather 的 U180/U240 `0/6` settings通过：`Delta_cos p05`为
`-0.0315...0.0051`，`Delta_JS p05`为 `-0.00153...0.00153`，远低于预注册 gate。24/24 batch evidence
contract exact，mass error最多 `1.19e-7`，因此不是 numeric或 side-path错误。

但 A1只能关闭精确命题 `current A6 error gradient reveals unit-specific patches while A6 Jacobian remains shared`。
error demand $J_{A6}^T r_m$ 与 sensitivity都受同一 $J_{A6}$ 约束，因此该负结果不能否定 A6尚未利用的
label-history evidence。failure attribution为 `diagnostic_valid_for_exact_carrier_contradiction`，同时
`diagnostic_invalid_for_direction_rejection`。不实现 retrieval；允许一次 A2 model-independent Step 3 repair。

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

## Common Patch-Evidence Interface

active A6 presets 的 hidden patch count 为 ETTh2 `48`、ETTm1 `1`、Weather `48`。因此 hidden-patch
attention 不能作为跨数据集 diagnostic。B14-A 从共同的 normalized history构造：

```text
x_norm [B,720,C]
  -> valid unfold(K=48,S=24)
  -> local_memory [B,C,29,48]
```

attribution先在 720 positions上计算，再映射到 29 patches。对每个 position $t$，其 attribution除以覆盖该
position的 patch数，再分配给相应 patches；因此 overlapping不会重复计数，patch mass之和严格等于 raw-position
mass。ETTh2/Weather legacy hidden-patch analysis只可作为 supplemental control。

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

## Metric 1：Model-Independent Label-Patch Dependence（A2 Main）

history patches与 future unit都投影到固定 rank-8 orthonormal DCT descriptors。将 batch与 channels合并为
observations，对每个 history patch和 future unit计算 centered linear CKA。CKA在 29 patches上 L1 normalize为
label-dependence profile；同 batch使用 4 次 shuffled target CKA作为 finite-sample control。

该 profile不经过 A6 encoder/readout，因此可以检验 current model尚未表达的 predictive dependence。A2 main
mismatch用 `A6 sensitivity cosine - label CKA cosine` 与 `label CKA JS - sensitivity JS`。

## Metric 2：Error-Conditioned Demand（A1 Control）

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
\left|\frac{\partial \mathcal{L}_m}{\partial x^{norm}_{b,t,c}}\right|.
$$

$d_m$ 先形成 720-position non-negative distribution，再 coverage-corrected聚合成 29-patch distribution。它是
model-conditioned error demand，不等同于 causal feature attribution或 label-only oracle。

## Metric 3：Target-Independent Existing Sensitivity

为了避免把 residual magnitude 直接当 retrieval，使用 Hutchinson estimator 估计每个 future unit output
Jacobian 对 normalized history positions 的 squared sensitivity。

对 draw $r$ 的 Rademacher vector $v_{m,r}$：

$$
g_{m,r}
=
\frac{\partial \langle v_{m,r},\hat y_m\rangle}{\partial x^{norm}},
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

历史 A1 mismatch：

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

A2把上述 $d$ 替换为 model-independent label-patch CKA profile，并额外要求 true CKA相对 shuffled CKA的
batch-mean gap具有正 bootstrap lower bound。

## Controls And Attribution

1. `target-independent sensitivity control`：排除所有差异都来自 A6 output Jacobian；
2. `same Rademacher draws`：减少 unit pairs 之间的 estimator noise；
3. `coeff-gradient pair control`：记录 unit loss 对 A6 coeff 的 signed gradient cosine，判断 raw-history demand
   是否只是 B13 coefficient conflict 的机械重述；
4. `valid-patch evidence audit`：manual slice、overlap-add reconstruction、forecast equivalence逐 batch验证；
5. `basis geometry cross-reference`：与 B13-A basis-pair artifacts 对照，不把 basis rows 的 geometry 当 retrieval；
6. deterministic bootstrap：对 batch-level gaps 做 `1000` 次 bootstrap；
7. `mass-conservation control`：position-to-patch aggregation error必须 `<=1e-6`。

## Pre-Registered Gate

单个 dataset/unit-size setting 为 `retrieval_demand_mismatch_support`，当同时满足：

1. bootstrap `p05(Delta_label_cos) > 0.05`；
2. bootstrap `p05(Delta_label_JS) > 0.01`；
3. bootstrap `p05(mean_true_CKA - mean_shuffled_CKA) > 0`；
4. mean sensitivity pairwise cosine `>= 0.80`。

[Returned] A2仅 Weather-U180通过，整体 `1/6`；没有任何 dataset的 U180/U240同时通过。ETTm1虽有
positive CKA-shuffle gap，但 profile mismatch低于 gate；ETTh2 CKA未超过 shuffle。整体 gate失败。

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
- `b14_history_patch_evidence_audit.csv`；
- `b14_future_unit_retrieval_report.md`。

## Rollback Boundary

B14-A 只能决定 carrier-specific retrieval problem 是否成立：

- A 失败：回滚 Step 2，不实现 cross-attention/query retrieval；
- A 通过：进入 B14-B probe design，仍不进入 Step 4-6；
- B14-B 通过后，才允许重新做 fresh primary-source audit 与 narrative novelty gate；
- 任一正结果都不能复活 B13 recurrence、B9 hard stage、B8 late coefficient correction 或 B11 basis field。

[Final Rollback] A1 exact contradiction `0/6`、A2 independent repair `1/6`，且没有 numeric/evidence pathology。
关闭当前 B14-FURD route，回 Step 2/3；下一问题只允许审计 minimal `patch_num=1 -> patch_num>1` carrier
tokenization，不允许继续 retrieval/CKA/unit-size sweep。
