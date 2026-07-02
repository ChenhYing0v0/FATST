# Phase5 A4S Validation-Prefix Signal Diagnostic 深入分析

## 诊断目标

A4 和 A4R 后的关键问题是：

> 现有 unified interface paths 的 best-path 差异，是否能被训练期或 validation-time 可观测信号解释？

A4S 不训练新模型，也不提出 routing 方法。它只加载已有 checkpoints，在 validation split 上导出
prefix-wise diagnostics，并判断这些 signals 是否能解释 A4 中的 `relative_vs_setting_best_pct`。

## 主要结果

| Scope | strongest signal | Pearson | Spearman | 判断 |
| --- | --- | --- | --- | --- |
| ALL | `teacher_student_mae` | `0.586` | `0.388` | 未通过 |
| ETTh2 | `teacher_student_mae` | `0.567` | `0.629` | 局部强，但只在 ETTh2 成立 |
| ETTm1 | `teacher_student_mse` | `-0.408` | `-0.596` | 方向与 ETTh2 相反 |
| Weather | `full_context_prefix_mse` | `-0.838` | `-0.851` | 局部强，但 signal 语义不同 |

## 机制解释

[Strong Evidence] A4S 没有通过 signal-existence gate。ALL-level 最强 signal 的 Spearman 只有
`0.388`，低于预设的 `0.55`。这说明 prefix-wise validation signal 虽然比 A4R 的 run-level log 更细，
但仍不足以支持一个跨 dataset 的 reliability selector。

[Fact] dataset 内部存在局部强信号，但方向不一致：

- ETTh2：teacher-student disagreement 越大，gap-to-best 越大，符合“偏离 H1 teacher 会变差”的解释；
- ETTm1：teacher-student MSE 与 gap-to-best 负相关，说明更接近 H1 teacher 不一定更好；
- Weather：full-context prefix MSE / residual 类 signal 最强，且方向与 ETTh2 的 teacher signal 不同。

[Decision] 因此不能继续把 Stage A 写成 `existing path reliability routing`。如果方法需要按 dataset
或 horizon 切换解释规则，它会退化成 post-hoc oracle，而不是高水平 SCI 可接受的统一机制。

## 对 Stage A 的影响

A4S 只否定以下路线：

- 在 H1/H1C/A2/A3C/A3D/A3E 这些 existing paths 之间做 learned selector；
- 把 validation-prefix MSE、teacher-student disagreement 或 residual 直接写成通用 reliability score；
- 继续通过叠加 A3F 或更多 path mixture 寻找小幅 metric repair。

A4S 不直接否定更宽泛的 interface 贡献，但它触发 Step 2/3 rollback：

- 若保留 interface 贡献，必须重新定义问题，而不是继续沿用 existing-path routing；
- 若放弃 interface 贡献，必须重构 paper-mainline，保证论文仍有足够的创新性、工作量和逻辑闭环。

## 下一步建议

下一步不应继续做 A4T/A4U 之类的 selector sweep。更合理的是做 Stage A reviewer-style re-evaluation：

1. 判断 `Capacity-Preserving Prefix-Aware Interface` 是否还能作为主贡献存在；
2. 若能存在，重新提出一个不是 existing-path selection 的机制，例如更根本的 prefix-consistent output contract；
3. 若不能存在，重构论文为以 `Reliability-Aware Future Supervision Routing` 为主贡献，同时补足 problem formulation、carrier evidence 和 method evidence，避免论文创新点缩水。
