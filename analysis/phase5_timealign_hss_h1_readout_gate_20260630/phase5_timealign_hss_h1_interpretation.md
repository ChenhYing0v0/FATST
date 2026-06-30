# Phase5-H1 Readout Gate Interpretation

## 结论

[Decision] H1 是 `readout_route_weak_pass_with_target_set_candidate`。

`target_set_decoder_multiprefix` 是当前更强 arm：ALL mean MSE 相对 H0 `full` 为
`-2.69%`，相对 H0 `multi-prefix` 为 `-0.68%`，相对 H0B `stochastic_prefix_k2` 为
`-0.69%`，相对 fixed specialist 为 `-3.65%`。它在 `12` 个 dataset-horizon setting 中拿到
`10` 个 H1 内部 best。

但 H1 还不是 full pass。核心 gate 是缩小 ETTm2 residual fixed gap，而
`target_set_decoder_multiprefix` 在 ETTm2 上相对 fixed 仍为 `+1.81%`。相比 H0B
`stochastic_prefix_k2` 的 `+2.08%` 只是小幅改善，没有达到“明显缩小”。

## 关键结果

| Arm | Mean MSE vs full | Mean MSE vs multi-prefix | Mean MSE vs H0B stochastic-k2 | Wins vs fixed | Mean MSE vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefix_conditioned_stochastic_k2` | -2.45% | -0.43% | -0.43% | 7/12 | -3.42% |
| `target_set_decoder_multiprefix` | -2.69% | -0.68% | -0.69% | 7/12 | -3.65% |

Dataset-level summary:

| Dataset | Best H1 arm | Result |
| --- | --- | --- |
| ETTh2 | `target_set_decoder_multiprefix` | strong pass, vs fixed `-12.65%` |
| ETTm2 | `target_set_decoder_multiprefix` | weak improvement, vs fixed `+1.81%` |
| Weather | `target_set_decoder_multiprefix` | no-harm, vs fixed `-0.12%` |

MAE 方向与 MSE 一致：`target_set_decoder_multiprefix` 的 ALL mean MAE 相对 H0B
`stochastic_prefix_k2` 为 `-0.64%`，相对 fixed 为 `-1.79%`。

## 机制判断

[Strong Evidence] H1 证明 readout/interface 不是伪问题。两个 H1 arms 都超过 H0B
`stochastic_prefix_k2`，说明把 requested prefix 显式放入 prediction readout，确实比继续只调
loss schedule 更有价值。

[Strong Evidence] target-set supervision 强于单纯 stochastic prefix condition。
`target_set_decoder_multiprefix` 在 `10/12` 个 setting 中是 H1 内部 best，说明 unified
multi-horizon 模型需要显式 target-set readout，而不只是每 batch 采样两个 prefix。

[Limit] 当前 H1 仍然保留 `proj_x: Linear(..., 720)`，只是对 projection 前 hidden 加了
prefix condition。因此它没有真正做到 variable-length prediction head；每个 requested prefix
仍通过一个 720-step projection 后再 crop。这可能解释了为什么 ETTm2 residual gap 只小幅缩小。

[Hypothesis] 下一步的瓶颈不是“是否知道 prefix”，而是 readout 的生成结构仍然过粗。
H1B 应测试更直接的 variable-prefix / prefix-token readout，让模型按 target set 生成对应
prefix，而不是始终生成 720 后裁剪。

## Gate 判定

| Gate | Result | Evidence |
| --- | --- | --- |
| ETTm2 residual fixed gap 明显缩小 | Weak Fail | `+2.08% -> +1.81%`，方向正确但幅度不足 |
| ETTh2 unified benefit 不丢失 | Pass | `target_set_decoder_multiprefix` vs fixed `-12.65%` |
| Weather no-harm | Pass | `target_set_decoder_multiprefix` vs fixed `-0.12%` |
| H1 超过 H0B stochastic-k2 | Pass | ALL mean MSE `-0.69%`，wins `9/12` |
| paper-story potential | Weak Pass | target-set readout 比 random schedule 更贴近 unified multi-horizon forecasting |

## 下一步

回到 11-step loop 的 Step 6，进入 `H1B Variable-Prefix Readout`，不要直接进入
future reliability scheduling。

建议的最小实验：

1. `target_set_prefix_head`: 用 shared hidden 生成 requested prefix length 的 output，再 pad 或
   pack 成 evaluation artifact；不再强制每个 request 都走 720-step projection。
2. `prefix_token_decoder`: 把 target prefix/token embedding 作为 query，从 hidden patch states
   中读出对应未来 steps；这是更像 decoder 的实现，paper story 更自然。

Gate：ETTm2 相对 fixed 的 gap 至少应从 `+1.81%` 明显压低，目标是接近 `+1.0%` 或更低；
同时 ETTh2 保持至少 `-11%` 级别 fixed gain，Weather 不退化。
