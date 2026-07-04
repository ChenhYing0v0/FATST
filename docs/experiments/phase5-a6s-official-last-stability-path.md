# Phase5-A6S Official-Last Stability Path Design

本文档记录 A6OD 之后的 Step 4/5 研究设计。目标不是改用 early stopping，而是在
`official-last` / without early stop 协议下设计稳定的 prefix-native unified head 训练路径。

## External Protocol Evidence

[Fact] TimeAlign GitHub issue #1 讨论了为什么测试使用最终训练权重而不是 validation-best 权重。
作者回复的核心解释是：长时序预测中 validation 与 test 可能存在分布偏移，基于 validation 的早停可能导致
训练不充分；固定训练轮数是其遵循的公开实现协议之一。

Source: https://github.com/TROUBADOUR000/TimeAlign/issues/1#issuecomment-4780784543

[Decision] 这条证据与本项目规则一致：`best-val` 只能作为 diagnostic-only upper-bound audit，
不能替代主协议。A6 的 `official-last trajectory drift` 不应被解释为“应该改用 early stop”，而应解释为
“需要提升 final checkpoint 在可能存在 validation-test shift 下的稳定性”。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5：提出 explicit stability path，并做理论可行性筛选 |
| `problem` | A6-LBF 已恢复 dense-capacity path，但 final checkpoint 仍未超过 best controls；ETTh2 存在明显 last-vs-best validation drift |
| `existence_evidence` | A6 partial-pass diagnostic：ETTh2 三个 A6 arms 的 last-vs-best validation MSE 平均漂移 `+11.81%`；A6OD 最佳 `lbf_r256_stochastic_p1` 仍相对 best control `+1.79%`、wins `0/4` |
| `idea` | 不追逐 validation-best checkpoint，而是设计 official-last-compatible stability mechanism |
| `theory_check` | 若机制只依赖 validation selection，则违反 TimeAlign protocol；若机制在训练中改变 optimization geometry 或 final-weight stability，则可作为 diagnostic/control 或 method candidate |
| `design` | 先做 ETTh2-only 最小 gate：`A6S-EMA` control、`A6S-HeadStability` candidate、二者组合、`DER-EMA` control |
| `narrative_gate` | conditional：`A6S-EMA` 只能 control；`A6S-HeadStability` 可作为 A6-LBF 内生稳定化候选；`A6S-SelfTeacher` 暂缓 |
| `effectiveness_gate` | pending remote：以 official-last final test MSE、last-vs-best drift、best-control gap、prefix-wise behavior 为主 |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a6s_stability_gate` |
| `decision` | ready_for_minimal_remote_gate |

## Candidate Stability Paths

| ID | Candidate | Role | Rationale | Narrative Gate |
| --- | --- | --- | --- | --- |
| `A6S-EMA` | Exponential Moving Average final weights | diagnostic/control-first | 不使用 validation selection，用训练轨迹的 moving average 降低 final checkpoint 方差；直接检验 A6 drift 是否是 weight trajectory instability | conditional：机制简单、paper novelty 弱，先作为 control，不直接 paper-core |
| `A6S-SelfTeacher` | Online EMA teacher consistency for prefix outputs | method-candidate-if-passes | 用当前模型的 EMA teacher 约束 prefix-native head 的 late-epoch prediction drift；不依赖外部 pretrained dense anchor，也不改变 checkpoint selector | stronger：有明确 stability mechanism，但需证明不是普通 KD trick |
| `A6S-HeadStability` | Learned-basis operator temporal smoothness / coefficient norm regularization | diagnostic/method candidate | 直接约束 `learned_temporal_basis @ learned_basis_coeff.weight` 的 operator geometry，降低 dense-row-dictionary 式 late drift | conditional：若 regularizer 与 prefix-native operator 绑定清楚，可进入 method gate |
| `A6S-ExternalTeacher` | Distill from H1/A3D-style stable controls | diagnostic-only | 检查 best controls 是否主要来自 teacher/nested stability advantage | failed_for_core_by_default：依赖外部 trained controls，贡献边界弱，不能直接作为 paper-core |

## Code-Theory Check

### A6S-EMA

[Fact] `A6S-EMA` 不使用 validation selection。训练过程中维护参数 EMA shadow，最后使用 final EMA
weights 进行 `official-last` evaluation。它回答的问题是：A6 drift 是否来自最后若干 optimization steps
的 weight variance。

[Decision] `A6S-EMA` 只能作为 control-first，因为 EMA 是通用训练稳定化技巧，SCI contribution 边界弱。
若它显著改善 A6-LBF，则说明下一步应设计更机制化的 stability path，而不是把 EMA 本身作为主方法。

### A6S-HeadStability

[Fact] `A6S-HeadStability` 只作用于 `learned-basis-forecast-operator`。其正则项约束 induced operator
`learned_temporal_basis @ learned_basis_coeff.weight` 的相邻 future rows 差分，目标是降低 dense-row
dictionary 式 late-epoch operator drift。

[Decision] 该候选与 A6-LBF 的 prefix-native learned operator 绑定更紧，具备 method-candidate 潜力；
但第一轮只能作为 ETTh2 diagnostic gate。若它改善 ETTh2 但损害 ETTm1/Weather 或 prefix behavior，
不能升级为 paper-core。

## Minimal Remote Gate

| Variant | Role | Readout | Extra mechanism |
| --- | --- | --- | --- |
| `lbf_r256_base` | same-root baseline | `learned-basis-forecast-operator` | none |
| `lbf_r256_ema099` | EMA control | `learned-basis-forecast-operator` | final EMA weights, decay `0.99` |
| `lbf_r256_smooth1e3` | HeadStability candidate | `learned-basis-forecast-operator` | operator smoothness weight `0.001` |
| `lbf_r256_ema099_smooth1e3` | interaction diagnostic | `learned-basis-forecast-operator` | EMA + operator smoothness |
| `der_ema099` | dense-equivalent EMA control | `prefix-native-dense-equivalent-row-bank` | final EMA weights, decay `0.99` |

## Immediate Decision

[Decision] 下一步不再做 objective-sampling sweep，也不做 rank-only sweep。优先启动上述 ETTh2-only
minimal remote gate：

1. `A6S-EMA` 作为最小 control，回答 final checkpoint 是否主要是 trajectory variance 问题；
2. `A6S-HeadStability` 作为更接近 A6-LBF 机制的候选，回答 learned-basis operator 是否需要显式几何稳定化；
3. `A6S-SelfTeacher` 保留为若前两者不足时的 method-candidate；
4. `A6S-ExternalTeacher` 只做 diagnostic，不升级为 paper-core。
