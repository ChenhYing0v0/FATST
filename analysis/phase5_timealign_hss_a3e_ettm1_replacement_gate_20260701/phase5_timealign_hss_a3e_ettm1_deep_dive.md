# Phase5 A3E ETTm1 Replacement Deep Dive

## 1. Artifact Scope

[Fact] 本次远程实验已完成，A3E replacement gate 的 dataset universe 是：

```text
ETTh2 + ETTm1 + Weather
```

其中 `ETTm1` 替换 `ETTm2`。由于 ETTm1 之前没有 Phase5 TimeAlign references，本次同步并分析了：

- ETTm1 official fixed references；
- ETTm1 H1 `target_set_decoder_multiprefix`；
- ETTm1 H1C `row_gated_dense_head_multiprefix`；
- ETTm1 A2 `nested_segment_decoder_multiprefix`；
- ETTm1 A3C `checkpoint_initialized_nested_segment_decoder_multiprefix`；
- ETTm1 A3D `teacher_preserved_nested_w03`；
- A3E `target_conditioned_nested_warm/scratch` on `ETTh2 + ETTm1 + Weather`。

## 2. A3E Gate Result

| Arm | ALL vs A2 | ALL vs A3C | ALL vs A3D | ALL vs H1 | ALL vs H1C | ALL vs Fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_conditioned_nested_warm` | `-0.36%` | `-0.25%` | `+0.25%` | `+0.20%` | `-0.25%` | `-4.52%` |
| `target_conditioned_nested_scratch` | `-0.38%` | `-0.26%` | `+0.24%` | `+0.18%` | `-0.27%` | `-4.51%` |

[Decision] A3E 不通过 paper-core gate。

原因：

1. 相对 A3C 的增量只有约 `-0.25%`，不足以支撑一个新的核心 interface 贡献；
2. 相对 A3D 和 H1 仍为正 gap，说明 target-conditioned nested 并没有成为更强 primary interface；
3. warm 与 scratch 几乎持平，甚至 scratch 的 ALL 略好。这说明 A3E 的效果不像是清晰的
   capacity-initialized target conditioning 机制；
4. ETTm1 上 warm/scratch 都输给 A3C，未解决新增 minute-level dataset 的关键问题。

## 3. A3E By Dataset

| Dataset | Warm vs A3C | Scratch vs A3C | Warm vs H1C | Scratch vs H1C | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | `-0.80%` | `-1.25%` | `-0.29%` | `-0.76%` | target-conditioned nested 有正向，但主要来自 ETTh2 |
| ETTm1 | `+0.18%` | `+0.73%` | `-0.53%` | `+0.02%` | warm 能超过 H1C，但不能超过 A3C；scratch 近似无效 |
| Weather | `-0.13%` | `-0.26%` | `+0.06%` | `-0.06%` | 只有很小的边际变化 |

[Inference] A3E 的正向信号不是来自一个稳定的 target-conditioning mechanism，而是 dataset-specific
微小波动。它没有证明“requested target set 直接进入 primary nested head”是当前 Stage A 的
有效主机制。

## 4. ETTm1 Chain Analysis

ETTm1 相对 fixed specialist 的链条如下：

| Method | Mean vs Fixed | Wins vs Fixed | h96 | h192 | h336 | h720 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official unified | `-0.44%` | `1/4` | `+0.93%` | `-2.72%` | `+0.03%` | `+0.00%` |
| H1 target-set | `-1.21%` | `3/4` | `-1.27%` | `-3.51%` | `-0.34%` | `+0.28%` |
| H1C row-gated | `-1.30%` | `3/4` | `-1.25%` | `-3.58%` | `-0.46%` | `+0.08%` |
| A2 nested | `-1.36%` | `3/4` | `-1.37%` | `-3.66%` | `-0.50%` | `+0.08%` |
| A3C warm-started nested | `-1.99%` | `3/4` | `-2.87%` | `-4.33%` | `-0.90%` | `+0.14%` |
| A3D teacher-preserved w03 | `-1.13%` | `3/4` | `-1.26%` | `-3.45%` | `-0.25%` | `+0.45%` |
| A3E warm | `-1.82%` | `3/4` | `-2.61%` | `-4.13%` | `-0.76%` | `+0.23%` |
| A3E scratch | `-1.28%` | `3/4` | `-1.31%` | `-3.54%` | `-0.45%` | `+0.18%` |

ETTm1 的主要结论：

1. ETTm1 并不像 ETTm2 那样简单表现为 severe unified decrease。official unified already has
   mean vs fixed `-0.44%`，但只有 `1/4` wins，说明 improvement 被 h192 主导，h96/h336/h720
   仍有 specialist gap。
2. H1/H1C/A2 都能把 ETTm1 改成 `3/4` wins vs fixed，说明 prefix-aware / nested interface
   对 ETTm1 有真实价值。
3. A3C 是 ETTm1 上最强的当前候选，mean vs fixed `-1.99%`。A3E warm 没有超过 A3C，而 A3D
   teacher preservation 反而明显弱于 A3C。
4. 所有方法在 h720 仍弱于 fixed 或接近 fixed，说明 ETTm1 的主要收益来自短中 horizon，不来自
   长 horizon readout。

## 5. Training Dynamics

A3E best epoch：

| Dataset | Warm Best Epoch | Warm Last Gap | Scratch Best Epoch | Scratch Last Gap |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 2 | `+7.73%` | 2 | `+9.97%` |
| ETTm1 | 10 | `0.00%` | 10 | `0.00%` |
| Weather | 6 | `+0.10%` | 9 | `+0.06%` |

[Inference] A3E 的 ETTh2 official-last 结果受后期退化影响较大；ETTm1 则不是 early stopping
问题，因为 best epoch 就是 last epoch。换言之，A3E 在 ETTm1 上失败不是因为 checkpoint
selector，而是结构本身没有超过 A3C。

## 6. Mechanism Judgment

[Decision] A3E 应标记为 `failed_as_core_candidate`。

它能作为 diagnostic evidence 保留：

- target conditioning 直接进入 primary nested head 并没有带来稳定 SCI-level 增益；
- warm-start 仍不能作为机制贡献；
- ETTm1 暴露出一个重要事实：teacher preservation 与 target conditioning 都可能削弱 A3C
  warm-started nested 在 minute-level dataset 上的优势。

但它不能作为论文贡献：

- 增量太小；
- 机制归因不清晰；
- 没有解决新增的 ETTm1 gate；
- scratch 与 warm 太接近，削弱了“capacity initialization + target conditioning”的叙事。

## 7. Next Research Decision

不建议直接进入 A3F `teacher_preserved + target_conditioned`。

原因是 A3F 的两个组成机制在 ETTm1 上并没有同时显示正向：

- A3D teacher preservation 弱于 A3C；
- A3E target conditioning 也弱于 A3C；
- 直接叠加会违反 narrative gate，变成在失败机制上堆叠。

更合理的 rollback point 是 Step 2/3/4：

1. 重新定义 Stage A 的核心问题：不是单纯“target prefix 是否进入 head”，而是“不同数据集/不同
   horizon 下，哪类 capacity-preserving path 是 reliable 的”。
2. 用 ETTm1/ETTm2 分裂作为 evidence：minute-level datasets 上，teacher/target conditioning
   可能不是可靠修复；A3C-style learned row capacity 反而更强。
3. 下一步优先做 `interface reliability diagnostic`，而不是继续新增 head：
   - 比较 A3C/A3D/A3E 在每个 dataset/horizon 上相对 H1/H1C 的 gain map；
   - 检查 h720 长 horizon 是否系统性阻碍 unified interface；
   - 判断是否应把 Stage A 与 Stage B 合并为 reliability-aware interface routing，而不是单独追求
     一个 universal prefix-aware head。

