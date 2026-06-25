# Phase4 Gradient Conflict Diagnostic Decision

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10 for diagnostic; Step 5/6 for next method design |
| `problem` | S1/S2 的 loss-only HSS 对 Weather 失败，疑似 difficult/late future units 污染 shared representation |
| `existence_evidence` | SRP code audit, Phase4 S1/S2 failure, gradient conflict diagnostic |
| `idea` | 用 per-unit gradient cosine 判断是否需要 gradient-routing / representation-aware HSS |
| `theory_check` | 若 conflict 主要来自 late/noisy units 与 early/predictable units，则 adapter/detach path 有必要 |
| `design` | `pred_len=720`, ETTh2/Weather, 40 warmup steps, 16 diagnostic batches, group-wise backward |
| `gate` | Weather conflict should be stable and stronger than ETTh2 on shared parameter paths |
| `artifacts` | `analysis/phase4_gradient_conflict_diagnostic_20260625` |
| `decision` | Partial pass: 进入 method design，但不能只做 noisy-hard isolation；应做 late/conflict-aware gradient routing |

## Key Findings

[Fact] block bucket trace 稳定复现 Weather 的 noisy-hard 存在：

| Dataset | Mean learnable blocks | Mean noisy blocks | Mean predictable blocks |
| --- | ---: | ---: | ---: |
| `ETTh2` | `2.38` | `1.62` | `11.00` |
| `Weather` | `1.88` | `2.12` | `11.00` |

[Fact] Weather 的 `noisy_hard` vs `predictable_easy` 在 shared output/readout path 上冲突明显：

| Pair | Parameter group | ETTh2 mean cosine | Weather mean cosine | Interpretation |
| --- | --- | ---: | ---: | --- |
| `noisy_hard` vs `predictable_easy` | `readout_head` | `0.6455` | `0.1190` | Weather noisy-hard 与 predictable path 明显更冲突 |
| `noisy_hard` vs `predictable_easy` | `all_shared` | `0.6482` | `0.1403` | Weather shared gradient direction 更接近冲突/弱一致 |

[Fact] 但 Weather 的 `noisy_hard` vs `early_1_96` 不比 ETTh2 更糟：

| Pair | Parameter group | ETTh2 mean cosine | Weather mean cosine |
| --- | --- | ---: | ---: |
| `noisy_hard` vs `early_1_96` | `all_shared` | `0.3040` | `0.3392` |
| `noisy_hard` vs `early_1_96` | `encoder` | `0.2281` | `0.3117` |

[Strong Evidence] 最强冲突不是单纯 noisy-hard，而是 late vs early：

| Pair | Parameter group | ETTh2 mean cosine | Weather mean cosine | Weather negative share |
| --- | --- | ---: | ---: | ---: |
| `late_337_720` vs `early_1_96` | `all_shared` | `0.1912` | `-0.0149` | `0.5625` |
| `late_337_720` vs `early_1_96` | `readout_head` | `0.2379` | `-0.0219` | `0.5625` |
| `late_337_720` vs `early_1_96` | `encoder` | `0.0213` | `0.0128` | `0.5000` |

## Interpretation

[Decision] 诊断支持 gradient-routing HSS 的主线升级，但不支持“只隔离 noisy-hard blocks”的下一步。

[Inference] S2 失败的原因更可能是：Weather 的 late future units 与 early/predictable units 在
shared readout/head path 上产生强冲突；noisy-hard 只是 late/conflict 的一个子现象。简单
`floor_weight` 不能改变 gradient destination，因此不能修复冲突。

[Counter-Evidence] ETTh2 的 `learnable_hard` vs `early_1_96` 也有明显低 cosine，尤其 encoder
mean cosine 为 `-0.1222`。如果把所有 hard units 都隔离，可能会抹掉 S1 在 ETTh2 上的收益。

## Next Method Direction

[Decision] 下一步设计应从 `noisy-hard isolation` 改为
`late/conflict-aware adapter routing`：

1. dense anchor 继续训练 shared base；
2. early/predictable units 保持 shared path；
3. late/conflict units 的 auxiliary pressure 不直接更新 shared readout/head；
4. adapter path zero-init，确保初始预测等价于 base；
5. adapter 更新范围优先限制在 readout/head 或 small residual branch，而不是全 encoder。

[Gate] 最小方法 gate：

1. Weather vs R.3 不能再 `0/4` collapse；
2. Weather `late_337_720` segment relative MSE 必须改善；
3. ETTh2 相对 R.3 的 `3/4` 正信号不能丢失；
4. trace 必须记录 routed units、adapter contribution 与 shared/adapter loss；
5. prefix consistency 保持 numerical-zero。

[Rollback] 若 adapter-routing 方法不能改善 Weather late conflict，则回退到 Step 5：
重新定义 predictability proxy，使用 residual stability 或 seasonal baseline residual，而不是继续堆
更多 adapter/expert。
