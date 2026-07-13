# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | Step 7B remote architecture screening running |
| `active_question` | PMFO-RCT的refinement-conservative mechanism能否在matched controls下产生独立effectiveness？ |
| `active_candidates` | `SC1-PMFO-RCT`, `SC2-MIPR` |
| `active_protocol` | `docs/experiments/stage-c-pmfo-rct-step7-protocol.md` |
| `method_implementation` | PMFO-RCT与controls已实现；15-run matrix running on 3090 GPU 0/1/2 |
| `rollback_point` | capacity/no-transition control explains -> Step 4；numeric/readout fault -> Step 6 |

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
step weighting 都不能单独成为 paper core。wavelet/refinement/neural-operator专项审计已在2026-07-13
Step 4-6完成，并进一步排除了generic hierarchical interpolation与learnable lifting claim。

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

## Step 4-6: Completed Design Gate

2026-07-13已完成：

1. external primary-source matrix表明arbitrary horizon、functional basis、hierarchical interpolation、
   learned lifting与raw harmonic weighting均不能单独成文；
2. SC1收紧为`PMFO-RCT`：future interval tree的detail位于父尺度正交补，H只做domain pruning；
3. mixed-radix `(90,30,10,5,1)` orthogonality/refinement/prefix invariants均在`1.33e-15`内通过；
4. SC2收紧为`MIPR`：$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，是L2 measure-induced
   block-diagonal surrogate，不是exact raw risk；
5. 预注册dense/no-transition/no-conservation与raw/random-projector controls；
6. SC1/SC2均标记`narrative_ready`，但SC2实现必须等待SC1 operator contract。

## Step 7-10: Active Conditional Path

1. Step 7A已完成：90/90 shape-prefix cases及refinement/conservation/locality gate通过，不训练；
2. Step 7B固定在ETTm1+ETTh2+Weather、seed2021比较A6、dense matched、no-transition、
   no-conservation、PMFO-RCT；
3. SC1通过后才实现MIPR，对比same-measure raw与random-projector control；
4. `2x2` factorial：A6/PMFO × raw/MIPR；
5. 3 datasets × 3 seeds × dense horizons；
6. 第二 backbone generality；
7. official native baseline reproduction 后再横向比较。

禁止在最小 gate 前加入 Encoder innovation、MoE、router、auxiliary reconstruction 或 per-horizon tuning。

## Next Concrete Action

以长间隔监控Step 7B的15 runs；报告dataset/run位置、epoch和ETA。全部完成后轻量同步并在本地重算gate，
按capacity/no-transition/interface/numeric规则作failure attribution。运行期间不加入Encoder innovation、
MIPR、MoE或per-horizon tuning。launch记录见
`analysis/stage_c_step7b_pmfo_rct_20260713/launch_record.md`。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
