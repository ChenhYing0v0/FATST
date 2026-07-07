# Phase5 StageB B7 Unified Prefix Optimization Diagnostic

## Stage Record

| Field | Content |
| --- | --- |
| `candidate_id` | `B7-UPO` |
| `current_step` | Step 1-3: literature research, problem proposal, problem-existence diagnostic |
| `problem` | A6-LBF-r256 已经统一了 prediction operator，但当前 multi-prefix training 可能没有统一地优化各 forecast regions；nested prefix losses 会重复覆盖 short steps 并欠覆盖 long-tail steps |
| `existence_evidence` | 当前 loss 的 closed-form step weight 显示 `0-96` segment 的平均训练权重是 `336-720` tail segment 的 `14.39x`；现有 segment artifacts 显示 A6 相对 fixed 的 mean MSE gain 在 tail bucket 收窄 |
| `idea` | 先诊断 unified horizon-task imbalance / interference；若成立，再设计 horizon-balanced 或 conflict-aware prediction loss |
| `theory_check` | 与 B6 不同，B7 不引入 frequency/basis auxiliary objective，而是分析同一 prediction loss 在 nested horizons 下的 task weighting 和 gradient path |
| `design` | 先做 offline segment-level diagnostic；下一步做 gradient/task-interference diagnostic，仍不训练新模型 |
| `narrative_gate` | partial：与 StageA unified prediction narrative 连贯；但 Weather 反例说明不能直接进入 method |
| `effectiveness_gate` | not evaluated |
| `artifacts` | `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` |
| `decision` | `prefix_imbalance_problem_candidate`; rollback point is Step 2/3 if gradient diagnostics fail |

## Literature Reading

[Fact] Multi-step forecasting literature把 long-horizon prediction 的核心分歧放在 recursive、direct、
MIMO/multiple-output 等 strategy 上。Ben Taieb 等指出 Direct strategy 会把不同 horizon 近似独立处理，
而 MIMO 试图保持 forecast sequence 内部的 stochastic dependency，但会降低 predictor flexibility：
<https://souhaib-bentaieb.com/publications/long-term-prediction/>。

[Fact] Multiple-output modeling paper 进一步把 direct 和 MIMO 看成同一 multi-step prediction 轴上的两端：
direct 是多个 single-output tasks，MIMO 是一个 multiple-output task，中间存在 MISMO-style trade-off：
<https://www.sciencedirect.com/science/article/abs/pii/S0925231210001013>。

[Fact] 深度 multi-horizon 工作也在强调 direct multi-horizon / one-forward prediction。MQRNN 使用
Direct Multi-Horizon Forecasting 并提出 forking-sequences 训练方案以提升 stability：
<https://arxiv.org/abs/1711.11053>。Informer 的 generative decoder 也强调 one-forward long sequence
prediction：
<https://arxiv.org/abs/2012.07436>。

[Fact] 从 optimization 角度看，多任务共享结构的平均 loss 可能遭遇 task imbalance 或 gradient conflict。
CAGrad 的问题定义指出，standard average loss 在任务梯度不一致时可能损害单个任务：
<https://arxiv.org/abs/2110.14048>。

[Inference] A6-LBF-r256 已经在 architecture 上选择了 MIMO-like unified forecast operator，但 training
objective 仍是多个 nested prefix tasks 的简单平均。因此 StageB 可以从 “unified architecture 是否被
unified optimization 正确训练” 继续深化，而不是转向通道相关性或 generic frequency loss。

## Code-Grounded Problem

当前 A6-LBF training path 在 `readout_mode=learned-basis-forecast-operator` 下对每个 selected horizon
分别 forward：

```text
for horizon in [96, 192, 336, 720]:
    horizon_loss = L1(outputs[:, :horizon, :], target_y[:, :horizon, :])
pred_loss = mean(prefix_losses)
```

因此第 `t` 个 future step 的 effective scalar weight 是：

$$
w(t)=\frac{1}{|\mathcal{H}|}\sum_{H \in \mathcal{H}, t \le H}\frac{1}{H},
\quad \mathcal{H}=\{96,192,336,720\}.
$$

这个权重不均匀：

| Future range | Mean weight relative to tail |
| --- | ---: |
| `0-96` | `14.39x` |
| `96-192` | `6.89x` |
| `192-336` | `3.14x` |
| `336-720` | `1.00x` |

## Offline Diagnostic Result

Report:
`analysis/phase5_stage_b_unified_prefix_optimization_20260707/stage_b_b7_unified_prefix_optimization_report.md`.

Segment-level comparison uses clean A6-LBF-r256 versus fixed-horizon TimeAlign. Positive relative MSE means A6 is
worse than fixed.

| Dataset | Early `0-96` mean MSE vs fixed | Tail `336-720` mean MSE vs fixed | Reading |
| --- | ---: | ---: | --- |
| ETTh2 | `-7.17%` | `-0.69%` | tail gain collapses |
| ETTm1 | `-2.87%` | `+1.47%` | tail turns negative |
| Weather | `-0.66%` | `-1.26%` | counterexample |
| ALL | `-3.57%` | `-0.16%` | tail-minus-early gap `+3.41%` |

[Moderate Evidence] ETTh2/ETTm1 support the problem: the under-weighted tail is exactly where A6 advantage weakens.

[Counter-Evidence] Weather does not support the same monotonic reading. This prevents direct method implementation.

## Narrative Gate

Passes partially:

- It follows StageA directly: StageA contributes a unified forecast operator; B7 asks whether the unified operator is
  trained by a genuinely unified horizon objective.
- It is not B6-PLO: no learned-basis/frequency auxiliary target is introduced.
- It is not channel modeling: all evidence and proposed diagnostics are along horizon/prefix/task axes.
- It is not a generic loss-weighting patch if follow-up gradient diagnostics prove conflict or imbalance inside A6's
  nested prefix tasks.

Fails to become method-ready now:

- Evidence is segment-level and post-hoc; it does not yet prove training-time gradient conflict.
- Weather is a material counterexample.
- The current result may reflect horizon difficulty, not only objective imbalance.

## Next Diagnostic

`B7-GTD`: gradient/task decomposition diagnostic.

Required before implementation:

1. Sample a small fixed number of train batches from each dataset.
2. Compute gradient vectors for `h96/h192/h336/h720` prefix losses on shared A6 parameters.
3. Report cosine similarity, norm ratio, and worst-conflict pairs.
4. Compare observed gradient conflict with segment-level tail weakness.
5. Only if conflict/imbalance is stable, design Step 4-6 method candidates:
   - horizon-balanced prefix loss;
   - tail-preserving nested-prefix weighting;
   - conflict-aware prefix gradient aggregation.

Rollback: if `B7-GTD` is weak or dataset-specific only, do not implement B7. Keep StageB paused and consolidate the
paper around Contribution 1.
