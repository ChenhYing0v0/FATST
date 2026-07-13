# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | Step 2-3 |
| `active_question` | nested future representation 与 projective risk 是否是跨 dataset 的真实问题？ |
| `active_candidates` | `SC1-PMFO`, `SC2-PIR` |
| `active_protocol` | `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md` |
| `method_implementation` | unauthorized |
| `rollback_point` | problem evidence fails -> Step 2 |

## Completed Foundation

### SC0 natural carrier

[Decision] dataset 可有自然结构偏好，但不得为每个新机制重新精调。使用 validation-only 两阶段小 grid
一次性冻结：Weather=P12/D64/ff128、ETTm1=P24/D32/ff64、ETTh2=P12/D64/ff128。params 差异只报告，
不参与选择。9 profile-seed validation stability gate 已通过。

### Natural baseline test reference

[Fact] 2026-07-13 完成 3 datasets × 3 seeds × 8 horizons，72/72 test metrics；checkpoint/profile 均在
test 前冻结，`selection_used_test=false`。该 reference 只用于后续对比，不允许反向修改 protocol。

[Risk] ETTh2 H48 test MSE CV=`5.30%`，后续必须报告三 seed；这与训练期 validation best-vs-last
`31.63%-44.95%` 恶化不是同一统计。

### Research reset and archive

[Decision] StageB 不再是 active cursor。旧 scripts、local candidates、configs 与 protocol/code docs 已移入
archive；`analysis/` 作为不可变 evidence store 保留。活动入口只保留 natural A6 carrier、baseline test 与
PMFO/PIR diagnostic。

## Step 1: Prior-Art Audit

已确认的 novelty pressure：

- ElasTST：horizon-invariant placeholders 与 horizon reweighting；
- TimePerceiver：target timestamp queries；
- FlowState：functional basis + dynamic horizon/resolution；
- Implicit Forecaster：implicit future waves；
- TransDF/QDF：label decorrelation与task covariance weighting。

[Decision] explicit horizon conditioning、continuous coordinate query、simple functional basis、simple harmonic
step weighting 都不能单独成为 paper core。Step 4-6 前仍需补做 wavelet/refinement/neural-operator 专项审计。

## Step 2-3: Active Problem Diagnostics

### SC1-PMFO

问题：A6 exact consistent但始终生成 H720；是否存在稳定的 nested coarse-to-fine future structure，使模型能
在不读取 horizon ID 的前提下按输出域限制计算并增量细化？

Gate：至少 2/3 datasets、3 seeds 支持 stable increment-energy structure；fixed nested basis必须优于 random
orthogonal/no-refinement controls。若失败，rollback Step 2，重新审计 semigroup operator；不修 PMFO head。

### SC2-PIR

问题：deployment horizon measure 的变化是否产生跨 dataset 的非平凡 gradient/risk差异，并且 nested
increments 是否提供 raw step reweighting之外的解释量？

Gate：至少 2/3 datasets 显示稳定 gradient direction变化；projected risk必须超越 ElasTST-style harmonic
weights 的必然结果。若失败，关闭 PIR；horizon measure 只保留为 protocol/evaluation定义。

## Step 4-6: Conditional Design Gate

只有 Step 2-3通过后才执行：

1. 给出 nested space、refinement identity、restriction proof 与 complexity；
2. 定义 PIR 对 L2/Huber 的 exact/approximate边界；
3. 完成 prior-art matrix 与 contribution boundary；
4. 预注册 parameter/FLOP、no-refinement、fixed-basis、raw-weight controls；
5. 明确每项 mechanism 的 falsification condition。

## Step 7-10: Conditional Experiment Path

1. 单 dataset/seed最小 gate；
2. PMFO 与 PIR 分开过 effectiveness gate；
3. `2x2` factorial：A6/PMFO × full/PIR；
4. 3 datasets × 3 seeds × dense horizons；
5. 第二 backbone generality；
6. official native baseline reproduction 后再横向比较。

禁止在最小 gate 前加入 Encoder innovation、MoE、router、auxiliary reconstruction 或 per-horizon tuning。

## Next Concrete Action

实现 `SC1/SC2-D1` offline diagnostic analyzer 与 code explanation：从 frozen train batches、labels 和
natural-baseline residuals 构造 nested projections，输出 energy、reconstruction、measure risk 与 module
gradient tables。先做本地 semantic smoke；该阶段不需要 remote training。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
