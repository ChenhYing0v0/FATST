# ISCF-v0 Step4 Scope-Relation Problem, Narrative Gate and D1 Protocol

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | ISCF-v0 Step4 problem/narrative audit；D1 validation diagnostic pre-result |
| `problem` | independent future-output coupling scopes是否学得了超越shared target difficulty与architecture prior的stable local response relation？ |
| `existence_evidence` | ISCF-v0 vs A6_FULL post-hoc `+1.3584%` MSE；test function audit有complementarity/topology，但residual common被target混淆且aligned low-rank 0/15 |
| `idea` | 把五个scope视为同一history state上的五个output-coupling operators，先测量label-free hidden-response geometry，再决定是否存在relation-aware pre-synthesis problem |
| `theory_check` | fixed-past Bayes boundary仍成立；不使用requested H。$J_s(h)u$只测局部operator relation，不把oracle或residual correlation当causal evidence |
| `design` | 15 frozen checkpoints；validation histories；32 hidden rows × 16 directions；independent-direction null + 16 matched random-init readouts；no targets/training/test |
| `narrative_gate` | `conditional_pass_for_diagnostic_only`；paper method gate pending D1 result与Step5 theory |
| `effectiveness_gate` | not applicable |
| `artifacts` | `configs/stage_c_iscf_v0_scope_response_d1.json`、diagnostic script、code explanation |
| `decision` | authorize ISCF-SRA-D1 only；method implementation/remote training/formal test remain false |

## 2. Why the previous common-residual result is insufficient

上一轮定义$E_s=f_s-y$。若所有forecasts都远小于target variation，即使$f_s$之间没有可利用的relation，$-y$也会让
$E_s$高度相关。因此`median residual correlation=0.9151`和`common residual energy=0.9320`不能区分：

1. shared target difficulty；
2. shared encoder带来的representation compatibility；
3. 五个independent readouts真正学得的operator relation。

[Decision] residual common只保留为confounded clue，不再作为Step4 existence gate。oracle headroom也只说明预测误差互补，
不能推出router或relation module。

## 3. Narrowed problem definition

定义scope operator $F_s(h)$，其中$h$是共同history representation，$s\in\{1,48,144,360,720\}$只表示future-output
coupling scope。对小扰动方向$u$，局部响应为：

$$
R_s(h,u)=\frac{F_s(h+\epsilon u)-F_s(h-\epsilon u)}{2\epsilon}.
$$

研究问题不是“scopes是否有不同forecast”，而是：

> 在不使用future labels、requested H或人工scale order时，五个independent scope operators对同一history-state
> perturbation是否呈现跨seed稳定、超越architecture-only control的common/scope-specific response geometry？

若答案为否，relation-aware ISCF redesign没有existence basis；若答案为是，只允许进入Step5，研究一个unified、non-ordered
pre-synthesis relation operator，仍不能直接进入implementation。

## 4. Primary-source audit and novelty boundary

检索日期：`2026-07-21`。来源优先conference proceedings/OpenReview/arXiv primary pages。检索覆盖multi-scale forecasting
experts、output fusion、representation-guided sharing、gradient/task relation和expert specialization。

| Work | Covered chain | Boundary for ISCF |
| --- | --- | --- |
| [MoLE, AISTATS 2024](https://proceedings.mlr.press/v238/ni24a.html) | 多个forecast experts、end-to-end router与output mixture | multiple predictors + learned fusion不新 |
| [TimeMixer, ICLR 2024](https://openreview.net/pdf?id=7oLshfEIC2) | input multiscale decomposition与Future-Multipredictor-Mixing | multi-scale future mixing不新 |
| [DMSC, arXiv 2025](https://arxiv.org/abs/2508.02753) | dynamic multi-scale patches、cross-scale coordination与adaptive scale-routing MoE | scale coordination/router叙事已有直接近邻；当前仅preprint，证据权重低于正式会议 |
| [Cross-Stitch, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html) | learned shared/task-specific representation mixing | generic common/private mixing不新 |
| [Similarity-guided MTL, CoLLAs 2022](https://proceedings.mlr.press/v199/gurulingan22a.html) | representation similarity引导feature sharing | similarity-to-sharing逻辑不新 |
| [Bayesian gradient aggregation, ICML 2024](https://proceedings.mlr.press/v235/achituve24a.html) | task-specific gradient uncertainty与aggregation | gradient relation/aggregation本身不新 |
| [Moirai-MoE, ICML 2025](https://proceedings.mlr.press/v267/liu25an.html) | data-driven time-series experts，反人工frequency specialization | data-driven specialization不新 |

[Decision] 可能保留的contribution-level空隙必须是完整链：

```text
future-output coupling scope
-> label-free response-relation necessity
-> non-ordered pre-synthesis interaction
-> one unified jointly trained operator
-> independent/ordered/Q1/random/capacity matched attribution
```

当前尚未证明第二项，所以不得命名新method或设计第二loss/router。

## 5. ISCF-SRA-D1 design

### 5.1 Role and split

- role=`diagnostic_only_validation_label_free`；
- 5 datasets × seeds2021/2022/2023，共15 frozen checkpoints；
- 只调用validation histories；loader虽返回batch tuple，但代码不读取`batch_y`值；
- 不保存或计算MSE/MAE，不访问test，不改变checkpoint。

### 5.2 Statistics

对每个scope的`response_bank [16,32,5,720]`沿direction、row、future coordinate展平，并分别中心化、unit-RMS。

- `synchronized_common_energy`：五个normalized responses均值的energy；
- `private_response_energy=1-common`；
- `direction_null_common_p95`：每个scope独立置换direction identities后的64次null；
- `random_init_common_p95`：16个同architecture random-init readouts在相同hidden/directions上的95%分位数；
- `pairwise_response_distance`：五个scope的10维function-response topology；
- `cross_seed_topology_spearman`：同dataset三对seeds的topology rank stability。

### 5.3 Frozen gates

| Gate | Pass rule | Failure attribution |
| --- | --- | --- |
| synchronized relation | 至少12/15高于direction-null p95 | 否则`hypothesis_false_or_local_probe_weak` |
| learned beyond structure | 至少12/15高于random-init p95 | 否则`capacity_control_explains` |
| noncollapse | median private≥0.05且pairwise distance≥0.05 | 否则common-only/Q1足以解释 |
| topology stability | 至少4/5 datasets median seed rho≥0.5 | 否则`relation_seed_unstable` |

全过才得到`scope_response_relation_supported_for_step5_design`。这只授权Step5 theory/design，不授权method implementation。

## 6. Narrative self-critique

即使D1通过，response geometry也可能来自scope pooling的固定数学结构，而非训练所需relation；random-init control正是为此设置。
此外frozen checkpoint与hidden representation存在co-adaptation，因此D1不能估计新增relation path的end-to-end gain。若D1失败，
只能在数值稳定且controls有效时拒绝当前local-response problem；若finite difference对epsilon敏感或responses退化，应标记
`diagnostic_invalid_for_direction_rejection`，而不是关闭ISCF carrier。

## 7. Authorization boundary and next cursor

当前授权：

```text
run ISCF-SRA-D1 on frozen validation checkpoints only
```

仍未授权：

- new method implementation；
- checkpoint training/fine-tuning；
- remote formal test；
- router、second loss、requested-H conditioning；
- 根据validation/test结果做per-dataset scope selection。

若local smoke与prelaunch checks通过，可commit/push后运行remote frozen diagnostic；结果必须回写本目录并重新执行Step4
narrative decision。
