# Phase5 StageB B11 Emergent Subspace Aggregation

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B11-ESA` |
| `current_step` | Step 4-6：native basis-conditioned aggregation design gate pending |
| `problem` | A6 不应依赖显式 stage / horizon encoding；需要利用 `learned_temporal_basis` 自发形成的 continuous future geometry，让架构更自然地聚合 history information |
| `existence_evidence` | B10-TSI-A/B 显示 basis row spaces 已有分化、coeff 在不同 subspaces 上被差异化使用；B11 进一步证明 sliding-window basis subspaces 沿未来时间轴连续变化，coeff projection 也随窗口距离变化 |
| `idea` | 用 basis-induced continuous subspace descriptors 驱动 history memory aggregation，而不是把人工 stage token 加到 coeff |
| `theory_check` | `B11-ESA` 只在 A6 primary prediction path 内工作；目标是更有效利用 basis geometry，不改变 unified/prefix-consistent 立场 |
| `design` | 待 Step 4-6 设计；候选方向是 continuous basis-conditioned subspace aggregation |
| `narrative_gate` | `pending`; diagnostic supports entering Step 4-6 |
| `effectiveness_gate` | `not_evaluated` |
| `artifacts` | `analysis/phase5_stage_b_b11_esa_basis_coeff_diagnostic_20260708/b11_esa_basis_coeff_report.md`; `docs/code-explanation/phase5-stage-b-b11-esa-basis-coeff-diagnostic.md` |
| `decision` | `problem_candidate_passed`; design gate must include no-basis / shuffled-basis / constant-slot controls |

## Motivation

B9/B10 的失败暴露了一个叙事风险：显式 `stage token` 或 `target-set head` 会让模型看起来像
horizon-conditioned / stage-conditioned forecaster，而不是 unified multi-horizon model。

B11 的问题不是：

```text
stage_id -> coeff
```

而是：

```text
learned_basis geometry -> subspace descriptors -> history aggregation -> coeff/state
```

也就是说，future-region 信息不作为外部标签输入，而是来自 A6 自己学出的 basis geometry。

## Diagnostic Result

`B11-ESA` Step 2/3 诊断读取 clean A6 checkpoints，在不训练模型的情况下测试两个问题：

1. `learned_temporal_basis[720,256]` 是否自发形成 future geometry；
2. 真实 forward 中的 `coeff[B,C,256]` 是否沿这些 subspaces 差异化使用。

### Hard Row Clustering

KMeans row clustering 的结果并不充分稳定：

| Dataset | K=4 stage NMI | Projection cosine |
| --- | ---: | ---: |
| ETTh2 | `0.5325` | `0.2708` |
| ETTm1 | `0.0057` | `0.0483` |
| Weather | `0.0068` | `0.0146` |

[Interpretation] 这说明不能把 B11 简化成“basis rows 自发聚成四个 hard stages”。ETTm1/Weather 的
cluster 不是时间局部的，继续做 hard clustering 会重新落入 stage-like 分段叙事。

### Sliding-Window Subspace Geometry

更符合 unified 叙事的是 sliding-window subspace diagnostic。用 window length `96`、stride `48`、
rank `16` 得到：

| Dataset | Adjacent overlap | Far overlap | Distance-overlap Spearman | Adjacent proj cosine | Far proj cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.3900` | `0.0649` | `-0.7016` | `0.5585` | `0.1504` |
| ETTm1 | `0.4021` | `0.0811` | `-0.5472` | `0.5391` | `0.2379` |
| Weather | `0.3810` | `0.0700` | `-0.2786` | `0.4071` | `0.0484` |

[Interpretation] A6 basis 沿未来时间轴形成连续变化的 subspace geometry：相邻 windows 的 subspace
overlap 高，远距离 windows 的 overlap 低。真实 `coeff` 在这些 subspaces 上的投影方向也随时间距离降低。

## Narrative Gate Implication

B11 与 B9/B10 的区别：

| Route | Mechanism | Narrative issue |
| --- | --- | --- |
| B9 | 人工 stage token 调制 `coeff` | 过于 stage-conditioned；no-stage control 阻断 |
| B10-C/D | frozen/offline target-set readout | readout/head 病态；不是 trainable native path |
| B11 | basis-induced continuous subspace aggregation | 与 unified model 更一致；仍需 Step 4-6 gate |

## Step 4-6 Requirements

B11 若进入 method design，必须满足：

1. 不输入 hard `stage_id` 或 `horizon_id`；
2. basis/subspace descriptors 来自 `learned_temporal_basis` 或其 smooth window/subspace representation；
3. history aggregation 发生在 primary prediction path 内，而不是 residual correction；
4. prefix consistency 必须可检查；
5. 必须设置 controls：
   - `no-basis`: 使用 learned constant slots，不看 basis geometry；
   - `shuffled-basis`: 打乱 basis-window/order 后生成 descriptors；
   - `constant-slot`: 同参数量但不随 future basis 改变；
   - `A6 fallback`: 初始化或 gate 关闭时回到 A6。

## Decision

`B11-ESA` 通过 Step 2/3 problem diagnostic，可进入 Step 4-6 design gate。

当前不得直接实现远程实验。下一步先写清楚 continuous basis-conditioned aggregation 的 tensor path、
fallback、controls 和最小 small gate。
