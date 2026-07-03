# Phase5-A5-Q Collapse Diagnostic Gate Report

本文档分析 A5-Q diagnostic-only gate。该实验只用于解释 collapse 原因，不恢复 A5-Q 的 paper-core candidate 身份。

## 结论摘要

[Strong Evidence] A5-Q 的 ETTm1 严重 collapse 主要混入了 decoder dropout 实现错位：`target_query_dropout=0.1/0.0` 后，ETTm1 mean MSE 相对旧 A5-Q 分别改善 -34.09% / -36.81%。

[Strong Evidence] 但修复 dropout 后仍没有通过 effectiveness gate：最佳单 setting 是 ETTm1 h96 的 `a5q_seg48_dropout01`，相对 best stage control 仍为 2.16%，本轮所有 diagnostic arms 对 best stage control 的 wins 仍为 0。

[Fact] ETTm1 `patch_num_override=48` 没有修复问题，反而弱于保留 `patch_num=1` 的 dropout 修复：`a5q_ettm1_patch48_dropout00` 相对 best stage control 平均仍差 41.02%。

[Hypothesis] A5-Q 当前失败不再应解释为简单实现 bug，而应解释为 target-query decoder 的 capacity / optimization path 不足：它能产生 prefix-consistent graph，但不能替代 dense/time-specific readout 的 forecasting capacity。

## Overall Summary

| Arm | n | mean MSE | vs old A5-Q | vs official unified | vs best stage control | wins vs best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a5q_ettm1_patch48_dropout00` | 4 | 0.4761 | -23.16% | +38.96% | +41.02% | 0 |
| `a5q_ettm1_patch48_dropout01` | 4 | 0.5365 | -13.36% | +57.09% | +59.41% | 0 |
| `a5q_seg48_dropout00` | 12 | 0.3327 | -13.35% | +13.49% | +16.76% | 0 |
| `a5q_seg48_dropout01` | 12 | 0.3346 | -13.57% | +13.72% | +16.95% | 0 |

## Hypothesis Tests

### H-Dropout

ETTm1 旧 A5-Q 使用 official preset `dropout=0.9` 进入 target-query decoder。本轮将 decoder dropout 固定为 `0.1/0.0` 后，h96/h192 明显恢复，其中 `a5q_seg48_dropout01` 在 ETTm1 h96 只比 best stage control 差 `+2.16%`，并比 official unified 低 `-1.68%`。

但长 horizon 仍失败：ETTm1 h336/h720 在 `a5q_seg48_dropout01` 下相对 best stage control 仍为 `+33.65%/+40.86%`。因此 H-Dropout 被支持为 collapse amplifier，但不能解释 A5-Q family 的全部 capacity gap。

### H-PatchMemory

将 ETTm1 `patch_num` 从 1 改为 48 后，没有带来预期修复。`a5q_ettm1_patch48_dropout00` 虽相对旧 A5-Q 平均改善 `-23.16%`，但弱于 `a5q_seg48_dropout00` 的 `-36.81%`，且相对 best stage control 平均仍差 `+41.02%`。

因此 H-PatchMemory 作为理论错位仍成立：`patch_num=1` 确实破坏了 query-select-patch 叙事；但简单提高 memory token count 不是有效修复，可能引入 backbone preset shift、optimization 难度和过大的 readout search space。

### Replication / Variance Note

ETTh2 的 `a5q_seg48_dropout01` 与旧 A5-Q 完全一致，符合 preset dropout 已为 0.1 的预期。Weather 的 `a5q_seg48_dropout01` 相对旧 A5-Q 有约 `-6.64%` mean improvement；由于机制配置理论上等价，本文将其标记为 run variance / nondeterminism 信号，不作为机制证据。

## Decision

`A5-Q_collapse_diagnostic_repair` 判定为 `diagnostic_only_completed_failed_as_repair`：它解释了 ETTm1 collapse 的重要实现因素，但没有产生足够的 effectiveness recovery。

下一步不应继续对 A5-Q 做简单 dropout/patch/width sweep。若要保留 target-query 叙事，必须回 Step 4/5 重新设计 capacity mechanism，例如让 query decoder 具备 function-preserving path、time-specific readout capacity 或 teacher-preserved initialization，并重新过 narrative gate。

## Artifacts

- `phase5_timealign_hss_a5q_diagnostic_metrics.csv`
- `phase5_timealign_hss_a5q_diagnostic_comparison.csv`
- `phase5_timealign_hss_a5q_diagnostic_summary.csv`
- `phase5_timealign_hss_a5q_diagnostic_training_summary.csv`
- raw metrics/logs under ignored `raw/` directory
