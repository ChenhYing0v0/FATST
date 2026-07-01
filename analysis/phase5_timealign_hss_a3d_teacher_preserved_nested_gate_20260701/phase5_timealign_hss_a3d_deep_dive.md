# Phase5 A3D Teacher-Preserved Nested Deep Dive

## 1. Artifact Audit

[Fact] 远端 `/home/yingch/exp_outputs/r-2026-fatst` 当前最新完整 Phase5 结果仍是
`phase5_timealign_hss_a3d_teacher_preserved_nested_gate`。未发现
`phase5_timealign_hss_a3e_target_conditioned_nested_gate` 目录、launcher log 或
`target_conditioned_nested_*` metrics。

[Decision] 因此本次可分析的新结果仍应按 A3D 处理；A3E 仍是 pending remote gate，而不是已完成实验。

## 2. What A3D Actually Proves

A3D 的关键证据不是单纯的 ALL mean improvement，而是三条机制证据同时成立：

1. `teacher_preserved_nested_w03` 相对 A3C warm-started nested 为 `-0.73%`，相对 A2 nested 为
   `-0.66%`。这说明 A3C 的失败确实不只是 nested structure 本身的问题，训练期 teacher
   preservation 能修复一部分 function-preservation gap。
2. `teacher_preserved_nested_w03` 相对 H1 target-set 为 `-0.06%`，相对 H1C row-gated 为
   `-0.48%`，并且 `8/12` settings 赢 H1C。它已经接近强 baseline，不是无效补丁。
3. teacher L1 在所有 dataset/arm 上下降，说明 teacher consistency 被模型实际吸收，而不是
   logging artifact。

因此 A3D 应标记为 `partial_pass`，不能标记为 failure。

## 3. Why It Still Cannot Be Paper-Core

A3D 的失败集中在 ETTm2：

| Arm | ETTm2 vs A3C | ETTm2 vs H1C | ETTm2 vs Fixed |
| --- | ---: | ---: | ---: |
| `w03` | `+0.18%` | `+0.22%` | `+1.87%` |
| `w10` | `+0.24%` | `+0.29%` | `+1.94%` |

更细看 horizon：

- ETTm2 `96/192`：A3D 相对 A3C 变差最明显，`w03` 分别为 `+0.80%`、`+0.34%`；
- ETTm2 `336/720`：A3D 接近或略优于 A3C，但仍输给 H1C；
- ETTh2：A3D 全 horizons 明显赢 A3C/H1C；
- Weather：`w03` 小幅赢 A3C/H1C，但幅度很小。

[Inference] 这说明 teacher preservation 主要解决了 capacity/function transfer，但没有解决
target-prefix specialization。ETTm2 的短中 horizon 对 prefix-specific behavior 更敏感；单纯把
nested student 拉近 H1 teacher，可能会把它拉回 teacher 的平均行为，而不是学到更适合 requested
target set 的 decoder 行为。

## 4. Teacher Weight Diagnosis

`w10` 的 teacher L1 下降更强：

- ETTh2：`-66.73%`，强于 `w03` 的 `-49.46%`；
- ETTm2：`-78.27%`，强于 `w03` 的 `-70.23%`；
- Weather：`-71.60%`，强于 `w03` 的 `-68.17%`。

但 `w10` 的 overall 不优于 `w03`：

- ALL vs H1C：`w03=-0.48%`，`w10=-0.44%`；
- Weather：`w03=-0.08%`，`w10=+0.03%`；
- ETTm2：`w10` 比 `w03` 更差。

[Inference] 更强 teacher preservation 不等于更好 forecasting。过强的 teacher constraint 可能
压制 nested primary head 自己学习 prefix-specific correction 的空间。因此后续不应继续扫
teacher weight；这会变成调参，而不是 SCI 级机制设计。

## 5. Validation And Training Dynamics

best epoch 分布：

| Dataset | w03 Best Epoch | w10 Best Epoch | Note |
| --- | ---: | ---: | --- |
| ETTh2 | 4 | 3 | 早期达到最优，后续 last gap 约 `1.5%` |
| ETTm2 | 5 | 5 | 训练较稳定，last gap 小于 `0.4%` |
| Weather | 9 | 1 | `w10` 第一轮最优，后续 teacher 吸收不带来 val 改善 |

[Inference] A3D 的问题不是简单 early stopping 或训练不足。ETTm2 的 last gap 很小，但仍输给
H1C；Weather `w10` 甚至在 epoch 1 最优，说明更强 teacher constraint 可能很快把模型锁在
teacher-like 解附近。

## 6. Decision

A3D 的结论应写成：

- `teacher preservation` 是有效的 capacity-preservation support mechanism；
- 但它不是最终 `Capacity-Preserving Prefix-Aware Interface`，因为它没有显式处理 requested
  target set 与 primary nested head 的结构耦合；
- 继续沿 teacher weight、warm-start 或 residual patch 调参没有足够论文叙事价值；
- 下一步应进入 A3E：target-conditioned primary nested interface。

A3E 的必要边界：

- `target_conditioned_nested_warm` 只作为和 A3C 对齐的 initialization control；
- `target_conditioned_nested_scratch` 是 dependency diagnostic；
- 若 warm arm 优于 A3C/A3D，而 scratch arm 不成立，结论是 target conditioning 有增量，但需要
  capacity initialization；
- 若 warm arm 也失败，应评估 A3F teacher-preserved + target-conditioned 的最小组合，而不是回到
  warm-start 或 teacher-weight sweep。

