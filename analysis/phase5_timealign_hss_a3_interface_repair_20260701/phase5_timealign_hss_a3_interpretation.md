# Phase5 A3 Interface Repair Interpretation

## 结论

[Decision] A3-1 `dense_initialized_nested_segment_decoder_multiprefix` 不通过
paper-core gate。它相对 A2 nested 有极小改善，但改善幅度不足以说明 capacity gap 被修复；
相对 H1 `target_set_decoder_multiprefix` 与 H1C `row_gated_dense_head_multiprefix` 仍未过
核心 gate。

## Gate Summary

| Dataset | vs A2 nested | vs H1 target-set | vs H1C row-gated | vs fixed | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | `-0.06%` | `+1.88%` | `+0.43%` | `-11.05%` | partial no-op |
| ETTm2 | `-0.15%` | `-0.16%` | `+0.00%` | `+1.65%` | weak repair |
| Weather | `+0.03%` | `-0.06%` | `-0.06%` | `-0.17%` | no repair over A2 |
| ALL | `-0.06%` | `+0.55%` | `+0.12%` | `-3.19%` | fail as core |

## Mechanism Reading

[Fact] A3-1 保留了 A2 nested 的 forward：按 `[0:96]`、`[96:192]`、`[192:336]`、
`[336:720]` segment heads 生成 prefix。它只改变 initialization：每个 segment head 从
`proj_x.weight/bias` 的对应 row slice 复制初值。

[Critical Limit] 这不是严格意义上的 trained capacity preservation。当前实现中 `proj_x` 在模型
初始化时是随机参数，A3-1 复制的是同一次初始化里的 random dense rows；它不是从已训练的 full
head、H1 target-set head 或 H1C row-gated head 继承预测能力。因此 A3-1 只能检验
`nested segment + dense-like initialization`，不能检验真正的 `nested segment + learned capacity
preservation`。

[Strong Evidence] 结果与这个机制边界一致：ALL 相对 A2 nested 只有 `-0.06%`，ETTh2 为
`-0.06%`，ETTm2 为 `-0.15%`，Weather 反而 `+0.03%`。这说明 shallow initialization 不是
A2 nested 的主要 bottleneck。

[Counter-Evidence] A3-1 仍然保留 A2 nested 的部分正向信号：ALL 相对 fixed 为 `-3.19%`，
ETTh2 相对 fixed 为 `-11.05%`，Weather 相对 H1C 为 `-0.06%`。因此结果不能否定 nested /
prefix-composition interface，只能否定当前 shallow initialization repair。

## Training Stability

- ETTh2 best epoch 为 `1`，last gap to best 为 `+8.98%`，与 A2 nested 的 early-best drift
  类似；
- ETTm2 best epoch 为 `6`，last gap 为 `+0.72%`；
- Weather best epoch 为 `6`，last gap 为 `+0.36%`。

训练轨迹没有显示 A3-1 带来稳定性层面的明显改善。ETTh2 仍然最早达到 best checkpoint，说明
初始化修复没有解决该 dataset 上的 over-training / validation drift。

## Decision

A3-1 decision: `shallow_dense_initialization_no_capacity_repair`。

下一步不应继续调 segment initialization，也不应把 A3-1 写成 capacity-preserving contribution。
合理 rollback 是 Step 5/6：设计真正的 capacity / teacher preservation：

1. `teacher_preserved_nested_segment_decoder`：用 H1 target-set 或 H1C row-gated 作为 teacher，
   对 nested outputs 加 prediction consistency / distillation；
2. `target_conditioned_nested_segment_decoder`：把 target-set condition 注入 nested segment heads，
   测试 H1 的 condition-before-readout 信号能否与 nested composition 结合；
3. `warm_started_nested_segment_decoder`：若要声明 trained capacity preservation，必须从已训练
   full/H1/H1C checkpoint 初始化 nested heads，而不是复制同一模型里未训练的 `proj_x` rows。

按照论文主线原则，如果 A3-2 与现有 H1/H1C 性能接近，应优先选择机制叙事更强、贡献边界更清晰的
teacher-preserved 或 target-conditioned nested route，而不是只追求微小 metric 修补。
