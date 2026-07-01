# Phase5 A2 Unified Interface Gate Interpretation

## 结论

[Decision] A2 不通过 paper-core gate，但 `nested_segment_decoder_multiprefix` 是明确的
partial pass。它没有显著超过 H1 `target_set_decoder_multiprefix` 或 H1C
`row_gated_dense_head_multiprefix`，因此不能作为最终 interface 贡献；但它证明 nested /
prefix-composition 方向比 dense-row prefix readout 更有潜力。

## Gate Summary

| Arm | ALL vs H1 target-set | ALL vs H1C row-gated | ALL vs fixed | H1 wins | H1C wins | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_row_initialized_prefix_decoder_multiprefix` | `+5.39%` | `+4.92%` | `+1.27%` | `0/12` | `0/12` | fail |
| `nested_segment_decoder_multiprefix` | `+0.61%` | `+0.18%` | `-3.13%` | `6/12` | `6/12` | partial pass |

## Dataset-Level Reading

- ETTh2: `nested_segment_decoder_multiprefix` 保留 strong unified benefit，相对 fixed 为
  `-11.00%`，但弱于 H1 `+1.94%`、弱于 H1C row-gated `+0.48%`。
- ETTm2: `nested_segment_decoder_multiprefix` 基本追平 H1，相对 H1 为 `-0.01%`，
  相对 H1C row-gated 为 `+0.15%`，相对 fixed 仍为 `+1.81%`，没有明显降低 H1 的 fixed gap。
- Weather: `nested_segment_decoder_multiprefix` 是正向信号，相对 H1 为 `-0.08%`，
  相对 H1C row-gated 为 `-0.09%`，相对 fixed 为 `-0.20%`，并在 `4/4` horizon 上赢 H1C。

## Mechanism Judgment

[Fact] `dense_row_initialized_prefix_decoder_multiprefix` 失败，说明仅复用 dense head rows 并不能
形成有效 interface。它缺少 H1 的 target-set hidden conditioning，也没有 nested composition；
因此虽然避免了 random variable head，但没有带来新的结构优势。

[Strong Evidence] `nested_segment_decoder_multiprefix` 是 A2 中唯一值得保留的 interface 方向。
它在 Weather 全 horizon 优于 H1C，并在 ETTm2 基本追平 H1。这个结果支持 prefix-consistent /
nested composition 比 post-hoc gate 或 dense-row prefix readout 更有叙事潜力。

[Limit] `nested_segment_decoder_multiprefix` 仍未过 gate：ALL 相对 H1 为 `+0.61%`，相对 H1C 为
`+0.18%`；ETTm2 fixed gap 仍为 `+1.81%`。因此它只能作为 A3 的 substrate，而不能成为最终
paper-core interface。

## Decision

A2 decision: `nested_interface_partial_pass_capacity_gap_remains`。

下一步不应直接转向 future reliability routing，也不应继续泛化 head sweep。合理 rollback 是
Step 5/6：围绕 nested composition 设计 A3，加入 capacity/teacher preservation。优先候选：

1. `dense_initialized_nested_segment_decoder`：用 `proj_x.weight` 的对应 row slices 初始化
   nested segment heads，测试 nested composition + dense capacity preservation；
2. `teacher_preserved_nested_segment_decoder`：在 H1 target-set 或 row-gated control 上加入
   distillation / consistency loss，约束 nested interface 不丢失当前 best carrier 的输出能力；
3. `target_conditioned_nested_segment_decoder`：给 nested segments 加 target-set condition，
   测试 H1 的 condition-before-projection 信号能否和 nested composition 结合。

Stage B / D1 reliability diagnostic 可以并行准备，但方法主线仍应先完成 A3 interface gate。
