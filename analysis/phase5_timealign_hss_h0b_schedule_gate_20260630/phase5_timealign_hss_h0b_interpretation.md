# Phase5-H0B Schedule Robustness Interpretation

## 结论

[Decision] H0B 是 `prefix_scheduling_robust_but_saturated`。

`stochastic_prefix_k2` 通过 robustness gate：ALL mean MSE 相对 `full` 为 `-2.03%`，
相对 `multi-prefix` 为 `+0.00%`，相对 fixed specialist 为 `-3.04%`，并且在
`12` 个 dataset-horizon setting 中有 `11` 个是 H0B 内部 best arm。

但 H0B 没有证明继续调 schedule 参数能形成新的性能突破。`stochastic_prefix_k2`
只是基本追平 `multi-prefix`，没有显著超过；`continuous_prefix_k2` 与
`continuous_prefix_pool96` 都弱于 `stochastic_prefix_k2`。因此下一步不应继续扩大
random prefix schedule sweep，而应回到 Step 6，设计 unified head / readout 层面的
prefix-aware 或 target-set-aware carrier。

## 关键结果

| Arm | Mean MSE vs full | Mean MSE vs multi-prefix | Wins vs fixed | Mean MSE vs fixed |
| --- | ---: | ---: | ---: | ---: |
| `stochastic_prefix_k2` | -2.03% | +0.00% | 7/12 | -3.04% |
| `continuous_prefix_k2` | -1.78% | +0.26% | 7/12 | -2.79% |
| `continuous_prefix_pool96` | -1.48% | +0.57% | 6/12 | -2.49% |

Dataset-level gate:

| Dataset | Best H0B arm | Interpretation |
| --- | --- | --- |
| ETTh2 | `stochastic_prefix_k2` except h720 | unified benefit is preserved; all arms beat fixed |
| ETTm2 | `stochastic_prefix_k2` on all horizons | residual fixed gap remains; schedule strength does not solve ETTm2 |
| Weather | `stochastic_prefix_k2` on all horizons | no-harm is preserved; continuous variants are weaker |

## 机制判断

[Strong Evidence] prefix supervision 是有效 carrier。H0B 中所有 arms 都在 `12/12`
settings 上优于 `full`，说明 unified 720 head 只用 full-horizon prediction loss 会低估短
prefix 的监督需求。

[Strong Evidence] `k=2` 的 stochastic prefix schedule 足够稳健，但已经接近上限。
它相对 H0 single-sample `stochastic-prefix` 的 ALL mean MSE 改善为 `-0.13%`，相对
`multi-prefix` 为 `+0.00%`。这更像是 variance reduction，而不是新的机制突破。

[Strong Evidence] continuous horizon-agnostic schedule 目前不是更强路线。`continuous_prefix_k2`
虽然比 H0 `continuous-prefix` 稍好，但仍比 `multi-prefix` 高 `+0.26%`；`pool96`
反而更弱，说明 H0 continuous 的问题并不是“过短 prefix 噪声”这么简单。

[Hypothesis] 当前瓶颈已经从 loss scheduling 转移到 readout/interface。TimeAlign official
unified 仍使用固定 `Linear(d_model * patch_num, 720)` head，再裁剪 prefix 做 evaluation。
即使 train-time prefix loss 改善了监督覆盖，模型结构仍没有显式表达“同一 unified 模型按
不同 target length 生成一致预测”的能力。

## Gate 判定

| Gate | Result | Evidence |
| --- | --- | --- |
| 至少一个 schedule arm 接近或超过 `multi-prefix` | Pass | `stochastic_prefix_k2` ALL mean vs `multi-prefix` 为 `+0.00%` |
| ETTm2 residual gap 缩小 | Fail | `stochastic_prefix_k2` vs fixed 为 `+2.08%`，基本等同 H0 `multi-prefix` 的 `+2.06%` |
| Weather no-harm | Pass | `stochastic_prefix_k2` vs fixed 为 `-0.12%` |
| ETTh2 gain preservation | Pass | `stochastic_prefix_k2` vs fixed 为 `-11.07%` |
| horizon-agnostic continuous schedule stronger | Fail | `continuous_prefix_k2`/`pool96` 均弱于 `stochastic_prefix_k2` |

## 下一步

回到 11-step loop 的 Step 6：设计 `H1 Prefix-Aware / Target-Set-Aware Readout`。

最小实验不应再扩大 schedule 超参，而应保留 TimeAlign backbone 与 future alignment
mechanism，替换或扩展 prediction head，使 unified model 显式接收 target-set / prefix
request。建议优先做两个 arms：

1. `prefix_conditioned_head`: 共享 backbone，prediction head 接收 horizon/prefix embedding，
   对请求的 prefix 输出预测。
2. `target_set_decoder`: 共享 backbone，按 target set 生成多 prefix outputs，并用
   `stochastic_prefix_k2` 作为训练监督。

Gate：在不损伤 ETTh2 和 Weather 的前提下，ETTm2 相对 fixed 的 residual gap 必须明显缩小；
否则说明 TimeAlign carrier 对 ETTm2 的问题不是 unified interface，而可能是 dataset-specific
representation 或 official hyperparameter mismatch。
