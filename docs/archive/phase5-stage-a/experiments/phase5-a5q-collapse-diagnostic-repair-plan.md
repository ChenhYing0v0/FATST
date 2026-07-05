# Phase5-A5-Q Collapse Diagnostic Repair Plan

本文档记录 A5-Q collapse 之后的下一步修复性诊断。该工作是 `diagnostic_only`，不是新的
paper-core method candidate；只有当诊断证据支持新的机制边界时，才允许回到 Step 4-6 重新提出
method candidate。

## 11-step 状态

| Field | Content |
| --- | --- |
| `current_step` | Phase5-A5：Step 4/5/6 diagnostic repair design |
| `problem` | A5-Q 的 target-query 叙事清晰，但 returned gate 出现最严重 collapse；需要区分理论路线失败、实现变量错位、以及 dataset preset 导致的退化。 |
| `existence_evidence` | A5-Q prefix smoke 成立，但 `a5q_seg48_small` 相对 `best_stage_control` 平均 MSE `+42.41%`，`a5q_seg24_wide` 为 `+55.62%`，wins `0/12`。 |
| `idea` | 不继续加宽 A5-Q；先隔离两个已定位的实现变量：A5-Q decoder dropout 与 ETTm1 `patch_num=1` memory-token 退化。 |
| `theory_check` | 若 cross-attention memory 长度为 1，query 无法选择 patch-wise history information；若 decoder dropout 继承 ETTm1 preset 的 `0.9`，target-query path 的有效容量会被严重破坏。 |
| `design` | 新增 `target_query_dropout` 与 `patch_num_override`，保持默认行为不变；用小矩阵验证 dropout 与 memory-token 假设。 |
| `narrative_gate` | not_required：本轮是 diagnostic-only，不可因为局部指标改善直接升级为 paper-core。 |
| `effectiveness_gate` | 诊断门：若 dropout 或 patch override 明显修复 A5-Q，再回 Step 4/5 重写 capacity mechanism；若仍 collapse，则 A5-Q family 继续保持 failed core candidate。 |
| `artifacts` | 预期输出根目录：`/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a5q_diagnostic_gate` |
| `decision` | pending；等待 diagnostic artifacts 返回后写入 `analysis/` 并同步 ledger。 |

## 待验证假设

### H-Dropout

[Hypothesis] A5-Q 的 decoder dropout 不应直接继承 official TimeAlign backbone preset。

[Evidence] ETTm1 unified 720 preset 的 `dropout=0.9`，而 A5-Q 将该值用于 `target_cross_attn`、
`target_self_attn` 和 `target_query_ffn`。这会使 target-query decoder 的信息流比 dense head 更直接地
受到随机破坏。

[Falsification] 若 `target_query_dropout=0.1/0.0` 后 ETTm1 与其他数据集仍无系统改善，则 collapse
不能主要归因于 dropout。

### H-PatchMemory

[Hypothesis] ETTm1 的 `patch_num=1` 让 A5-Q cross-attention 退化。

[Evidence] A5-Q 的理论叙事依赖 query 对 patch-wise history tokens 选择信息；当 `memory=[B*C,1,D]`
时，cross-attention softmax 只有一个 key，query 无法表达“未来位置需要哪些历史 patch”。

[Falsification] 若 ETTm1 使用 `patch_num_override=48` 后仍无明显改善，则 memory-token 退化不是
主要瓶颈，或者 backbone preset 与训练路径存在更深的 capacity gap。

## 远程矩阵

| Arm | Datasets | Purpose |
| --- | --- | --- |
| `a5q_seg48_dropout01` | `Weather`, `ETTm1`, `ETTh2` | 将 A5-Q decoder dropout 固定为 `0.1`，验证是否修复 high-dropout collapse。 |
| `a5q_seg48_dropout00` | `Weather`, `ETTm1`, `ETTh2` | 去掉 A5-Q decoder dropout，验证是否存在更强的 optimization/capacity recovery。 |
| `a5q_ettm1_patch48_dropout01` | `ETTm1` | 在 ETTm1 上把 `patch_num` 从 1 改为 48，并使用 `target_query_dropout=0.1`。 |
| `a5q_ettm1_patch48_dropout00` | `ETTm1` | 同时移除 decoder dropout，用于判断 patch memory 与 dropout 的叠加影响。 |

## 解释边界

- 本轮结果只回答“为什么 A5-Q collapse”以及“是否存在实现错位”。
- 若修复后只接近原 A5-Q，但仍显著弱于 H1/H1C/A3D，A5-Q 仍不能恢复为 paper-core。
- 若修复后显著改善，需要重新做 narrative gate：新的贡献不能写成“调小 dropout”或“改 patch_num”，而必须解释为
  capacity-aware target-query interface 的必要条件。
