# Phase5 TimeAlign-HSS H0 Interpretation

## Decision

[Decision] `prefix_scheduling_pass_with_stochastic_candidate`.

H0 通过作为 TimeAlign-HSS 的第一层 carrier：所有 prefix-supervision variants 都优于
official `full` unified loss；其中 `stochastic-prefix` 几乎追平 `multi-prefix`，说明 D0 的收益
不必绑定为每个 batch 同时优化固定 benchmark horizons，可以被解释为 train-time supervision
scheduling。

## Main Result

| loss_mode | mean MSE vs full | mean MSE vs multi-prefix | wins vs fixed | mean MSE vs fixed |
| --- | ---: | ---: | ---: | ---: |
| `full` | 0.00% | +2.09% | 3/12 | -1.08% |
| `multi-prefix` | -2.03% | 0.00% | 7/12 | -3.04% |
| `balanced-step` | -1.22% | +0.83% | 6/12 | -2.26% |
| `stochastic-prefix` | -1.90% | +0.13% | 7/12 | -2.90% |
| `continuous-prefix` | -1.67% | +0.37% | 7/12 | -2.69% |

[Strong Evidence] `stochastic-prefix` 是最重要的 H0 结果：它只在每个 batch 采样一个 prefix，
但整体平均 MSE 只比 `multi-prefix` 高 `+0.13%`，并且相对 fixed 的平均 gap 为 `-2.90%`，
接近 `multi-prefix` 的 `-3.04%`。

[Strong Evidence] `balanced-step` 全部优于 `full`，但明显弱于 `multi-prefix` 与
`stochastic-prefix`。这说明 D0/H0 的收益不只是“不重叠 future regions 都监督一下”或简单
region reweight；prefix form 本身有额外价值。

[Evidence] `continuous-prefix` 也优于 `full`，但弱于 `stochastic-prefix`。它说明脱离 benchmark
horizon id 有潜力，但当前单 prefix sample、`32` step 粒度的 pool 还不是最优 schedule。

## Dataset Reading

| dataset | best schedule reading |
| --- | --- |
| ETTh2 | `stochastic-prefix` 在 h96/h192/h336 最好，`continuous-prefix` 在 h720 最好；schedule-like modes 不损伤 ETTh2 unified benefit |
| ETTm2 | `multi-prefix` 仍最强；`stochastic-prefix` 明显优于 full，但仍弱于 fixed specialist，说明 ETTm2 可能还需要更稳定或更强的 prefix schedule |
| Weather | `multi-prefix` 最强于 short/mid horizons，`stochastic-prefix` 最强于 h720；`stochastic-prefix` 保持接近 fixed 的平均表现 |

[Fact] `stochastic-prefix` 的 dataset-level mean MSE vs full：
ETTh2 `-3.39%`、ETTm2 `-1.25%`、Weather `-1.07%`。

[Fact] `continuous-prefix` 的 dataset-level mean MSE vs full：
ETTh2 `-3.01%`、ETTm2 `-1.07%`、Weather `-0.94%`。

## Training Dynamics

[Fact] ETTh2 的 post-best drift 显著降低：
`full` 为 `+20.76%`，`multi-prefix` 为 `+13.64%`，`stochastic-prefix` 为 `+6.41%`，
`continuous-prefix` 为 `+7.77%`。

[Fact] Weather 的 `stochastic-prefix` best epoch 为 `10`，last-best gap 为 `0.00%`；
这比 `full` 的 best epoch `5` 与 `+0.25%` drift 更稳定。

[Counter-Evidence] ETTm2 上 `multi-prefix` 仍是最强，`stochastic-prefix` 与 `continuous-prefix`
没有超过它；因此还不能把最终方法直接定为 stochastic/continuous schedule。需要下一步做
schedule strength / sample count / seed robustness。

## Research Implication

[Decision] H0 支持把 TimeAlign-HSS 主线升级为：

> Prediction-prefix supervision scheduling first, future-alignment supervision scheduling second.

这比直接进入 D1/M1 更合理。当前真正有叙事潜力的候选不是 `multi-prefix` 本身，因为它仍直接绑定
benchmark horizons；而是 `stochastic-prefix`，因为它证明了 train-time prefix scheduling 可以在
每个 batch 只监督一个 prefix 的情况下接近 full prefix-set objective。

## Next Plan

进入 Phase5-H0B：schedule robustness / horizon-agnostic refinement。

优先实验：

1. `stochastic-prefix_k2`：每个 batch 采样 2 个 prefix，检查是否超过 `multi-prefix` 或稳定提升 ETTm2。
2. `continuous-prefix_k2`：每个 batch 从 continuous pool 采样 2 个 prefix，检查 horizon-agnostic schedule 是否能追上 benchmark-specific schedule。
3. `continuous-prefix_pool96`：从 `96,192,...,720` 或 `64,128,...,704,720` 采样，判断当前 continuous 弱势是否来自过短 prefix 噪声。
4. 若 H0B 通过，再做 seed/checkpoint sensitivity；若 H0B 失败，保留 `multi-prefix` 作为 strong interface control，但 HSS 主线需要回到 target-set / prefix-aware readout，而不是继续调 random schedule。
