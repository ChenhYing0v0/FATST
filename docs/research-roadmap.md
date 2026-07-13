# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | Step 4-6 |
| `active_question` | 如何把已确认的nested structure与measure-conditional risk转化为novel、可证明的operator/objective？ |
| `active_candidates` | `SC1-PMFO`, `SC2-PIR` |
| `active_protocol` | `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md` |
| `method_implementation` | unauthorized pending narrative gate |
| `rollback_point` | novelty/theory gate fails -> Step 2 |

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

## Step 2-3: Completed Problem Diagnostics

[Decision] D1-v2已完成：PMFO structure与frozen ordered-memory gate均3/3；PIR aggregate gate 3/3。
SC1通过problem gate；SC2以measure-conditional形式通过。以下内容转为已完成problem record。

### SC1-PMFO

问题：A6 已按`basis[:H]`直接计算H步输出，但只提供single dense rank-256 future subspace。是否存在稳定
的nested coarse-to-fine future structure，A6 `memory: [B,C,P,D]`是否保留该信息，以及新的operator能否在
不读取horizon ID的前提下提供refinement/local-support computation？

Gate：至少2/3 datasets、3 seeds支持evaluation-space future deviation与baseline residual的stable increment
structure；frozen A6必须优于zero-deviation baseline，且patch shuffle/collapse必须产生至少1%的SSE恶化。
Linear probe只作辅助量，negative R2之间的差值不得形成pass。
learned basis geometry用于区分“容量足够但缺层次”与“subspace本身不足”。若失败，rollback Step 2；
不得用同步更换Encoder与decoder掩盖归因。

### SC2-PIR

问题：deployment horizon measure 的变化是否产生跨 dataset 的非平凡 gradient/risk差异，并且 nested
increments 是否提供 raw step reweighting之外的解释量？

Gate：至少 2/3 datasets 显示稳定 gradient direction变化；projected risk必须超越 ElasTST-style harmonic
weights 的必然结果。若失败，关闭 PIR；horizon measure 只保留为 protocol/evaluation定义。

## Step 4-6: Active Design Gate

当前执行：

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

先完成SC1的multiresolution/lifting/multiwavelet/neural-operator专项prior-art与refinement/restriction proof；
并行完成SC2的deployment measure、L2 exact decomposition、Huber/L1 boundary。实现顺序保持串行：operator
contract稳定后才实现PIR，并预注册raw uniform/log与benchmark weighting controls。D1-v2解释见
`analysis/stage_c_d1_pmfo_pir_offline_v2_20260713/research_interpretation.md`。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
