# Phase5 Stage B Problem Redefinition And Narrative Gate

本文档在 Stage A architecture exhaustion audit 之后启动 Stage B 的 Step 2/3/4/5。它不是方法实现文档，
也不是 remote launch plan；它先回答一个更基础的问题：在 Stage A standalone unified head route 暂停后，
Stage B 是否还能成为 paper-core，若可以，必须怎样改写问题边界。

## 11-Step State

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3 -> Step 4/5：重新定义 Stage B problem 并做 narrative gate |
| `problem` | 旧 Stage B 假设建立在“Stage A unified architecture 先成立”之上；但 Stage A head route 已暂停，不能直接把 routing 当替代品 |
| `existence_evidence` | A6-LBF/A6-DER 恢复 capacity 仍不足，A7DG 证明 stability/selective pressure 有局部信号，A4/A4R/A4S 证明 existing-path reliability signal 不足 |
| `idea` | 将 Stage B 从“在 final head 之上做 routing”改写为“future-aware carrier 中 future supervision pressure 的 reliability / allocation 问题” |
| `theory_check` | TimeAlign 有 training-time future branch、alignment/reconstruction pressure 与 official-last behavior；若这些 future supervision signals 对不同 future units 有利有害不同，就可能成为 paper-core problem |
| `design` | 先做 `B0_future_supervision_pressure_audit` diagnostic：不提出 routing method，只验证 future supervision pressure 是否有可观测 harmful/useful structure |
| `narrative_gate` | conditional pass for diagnostic：Stage B 可以重新进入，但必须避开 manual routing、head-selector 和 early-stop protocol |
| `effectiveness_gate` | pending；B0 是 diagnostic-only，不能直接作为 paper-core method |
| `artifacts` | Stage A exhaustion audit、A4/A4R/A4S diagnostics、A6/A7/A8TAG/QBR reports |
| `decision` | Stage B 进入 problem redefinition；下一步只设计 B0 diagnostic，不启动 method remote |

## Why Old Stage B Cannot Be Resumed Directly

[Fact] 旧 Stage B 写法要求 A5 unified prediction architecture 至少成立，否则 reliability-aware future
supervision routing 会被审稿人质疑为绕过 unfair full-720 crop / interface mismatch。

[Fact] 当前 Stage A 并没有给出 final architecture；它给出的是 route-level negative evidence：

- A5-Q/A5-B 说明 prefix-native contract without capacity 会 collapse；
- A6-LBF/A6-DER 说明 dense-equivalent capacity 仍不足；
- A7DG 说明 official-last stability 有局部信号；
- A8TAG/QBR 说明 teacher advantage 与 target-query row-key 都不能独立解决问题。

[Decision] 因此 Stage B 的新问题不能写成“在已解决的 unified head 上加 routing”。它必须写成：

> TimeAlign-like future-aware carriers expose a supervision-allocation problem:
> future reconstruction/alignment pressure is not uniformly reliable across future units, datasets,
> and training trajectories. Stage A evidence shows that prediction-head redesign alone is not enough;
> the next question is whether the future supervision path itself should be diagnosed and controlled.

## Reframed Stage B Problem

新的 Stage B 问题是：future-aware training branch 的 supervision pressure 是否存在可观测的
useful/harmful structure？

这里的 supervision pressure 包括：

- future reconstruction loss；
- history-future representation alignment loss；
- future-unit / prefix-region validation behavior；
- official-last trajectory drift；
- teacher/student disagreement 只作为 diagnostic signal，不作为默认 method。

## Boundary Conditions

[Rule] Stage B 不能做：

- manual dataset/horizon routing；
- existing-path selector；
- 以 early-stop / best-val 作为主协议；
- 简单降低 TimeAlign loss 后声称机制成立；
- 把 A7DG/A8TAG/QBR 混合成新 gate；
- 在未证明 future supervision pressure problem 真实存在前实现 MoE 或 routing module。

[Rule] Stage B 可以做：

- diagnostic-only pressure audit；
- loss-only ablation as control；
- gradient-path diagnostic；
- source-informed analysis of TimeAlign future branch；
- 若 B0 证明 problem exists，再做 Step 4/5 method narrative gate。

## Candidate B0: Future Supervision Pressure Audit

[Idea] 不先提出 routing method，而是审计 TimeAlign 的 future branch pressure 是否可解释：

1. 训练日志层面：记录 `w_recon * recon_loss`、`w_align * align_loss`、prediction loss 与 validation drift；
2. prefix/future-unit 层面：导出不同 target prefix 下的 prediction residual、future reconstruction residual、
   alignment proxy；
3. intervention 层面：只做最小 control，例如 `w_recon=0`、`w_align=0` 或 reduced pressure，判断是 future branch
   pressure 本身造成 harm，还是 head/capacity 已经饱和。

## Narrative Gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | conditional pass：Stage A 失败证明 head-only route 不足，TimeAlign future branch 是自然下一问题 |
| novelty | medium：从 unified head 转向 future supervision allocation，但必须避免 generic loss weighting |
| tensor/gradient path | pending：B0 必须明确 future branch loss 如何进入 shared encoder / readout |
| contribution boundary | conditional：B0 只验证问题；method gate 另写 |
| risk | high：若 B0 只证明降低 loss 有效，创新性不足；若 signal 不稳定，则 Stage B 失败 |

## Next Action

[Decision] 下一步写 `phase5-stage-b0-future-supervision-pressure-audit.md`，先完成 source/code audit：

- TimeAlign `recon_loss` 与 `align_loss` 的 tensor path；
- 当前 `train_repo.py` 是否能分别控制 `w_recon` 与 `w_align`；
- 需要新增哪些 logging/export 才能判断 pressure useful/harmful；
- 最小 diagnostic matrix，必须先声明 diagnostic-only。

[Decision] 未完成 B0 source/code audit 前，不启动 remote experiment。

