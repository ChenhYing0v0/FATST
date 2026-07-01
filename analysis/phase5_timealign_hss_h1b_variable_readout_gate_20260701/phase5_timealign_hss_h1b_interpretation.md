# Phase5-H1B Variable Readout Gate Interpretation

## 结论

[Decision] H1B 是 `variable_readout_fail_capacity_collapse`。

两个真正 variable-prefix prediction heads 都失败。`target_set_prefix_head_multiprefix`
虽然比 `prefix_token_decoder_multiprefix` 稳定，但 ALL mean MSE 仍然相对 H1
`target_set_decoder_multiprefix` 高 `+14.41%`，相对 H0B `stochastic_prefix_k2` 高
`+13.62%`，相对 fixed specialist 高 `+10.26%`。`prefix_token_decoder_multiprefix`
更差，ALL mean MSE 相对 H1 高 `+25.52%`。

这说明 H1B 的问题不是训练 schedule，而是 readout capacity collapse：直接移除 TimeAlign 的
dense `Linear(...,720)` projection 后，简单 dynamic head / token attention decoder 不能承载
原 head 的 step-specific dense mapping。

## 关键结果

| Arm | Mean MSE vs H0 full | Mean MSE vs H0B stochastic-k2 | Mean MSE vs H1 target-set | Wins vs fixed | Mean MSE vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `target_set_prefix_head_multiprefix` | +11.31% | +13.62% | +14.41% | 2/12 | +10.26% |
| `prefix_token_decoder_multiprefix` | +22.08% | +24.64% | +25.52% | 0/12 | +20.68% |

Dataset-level summary:

| Dataset | Best H1B arm | Result |
| --- | --- | --- |
| ETTh2 | `target_set_prefix_head_multiprefix` | only 2/4 horizons beat fixed, far worse than H1 |
| ETTm2 | `target_set_prefix_head_multiprefix` | fails all horizons, vs fixed `+17.91%` |
| Weather | `target_set_prefix_head_multiprefix` | fails all horizons, vs fixed `+13.12%` |

MAE 方向一致：`target_set_prefix_head_multiprefix` 的 ALL mean MAE 相对 H1 target-set 为
`+11.03%`；`prefix_token_decoder_multiprefix` 为 `+17.22%`。

## 机制判断

[Strong Evidence] 真正 variable-prefix output 并不会自动改善 unified multi-horizon forecasting。
H1B 去掉了固定 720 projection 后，模型失去了强 step-specific output capacity。

[Strong Evidence] `prefix_token_decoder` 的轻量 attention 读出不适合直接替代 TimeAlign head。
它在 `12/12` settings 上都不如 H1 target-set conditioned 720 projection，并且对 Weather
早期 segment 退化非常明显。

[Inference] TimeAlign 的强 baseline 很大程度依赖 dense output projection 的容量。
H1 的 positive signal 不是“去掉 720 projection”，而是在保留 dense projection 的前提下，
让 requested-prefix/target-set 信息调制 readout。

[Hypothesis] 下一步若继续 decoder/head route，应设计 capacity-preserving decoder：
保留 `Linear(...,720)` 的 dense base path，再加入 prefix/target-set conditioned residual、
low-rank adapter 或 row-wise gate，而不是用小 MLP / token attention 直接替换 dense head。

## Gate 判定

| Gate | Result | Evidence |
| --- | --- | --- |
| ETTm2 fixed gap 明显低于 H1 `+1.81%` | Fail | best H1B arm vs fixed 为 `+17.91%` |
| ETTh2 保持强收益 | Fail | best H1B arm vs fixed 仅 `-0.24%`，H1 为 `-12.65%` |
| Weather no-harm | Fail | best H1B arm vs fixed 为 `+13.12%` |
| H1B 超过 H1 target-set conditioned 720 projection | Fail | best H1B arm ALL mean `+14.41%` |

## 下一步

回到 11-step loop 的 Step 5/6，设计 `H1C Capacity-Preserving Prefix Decoder`。

建议不要继续扩展当前两个 variable-prefix heads。下一步候选应满足：

1. 保留 dense 720 base projection，避免 readout capacity collapse；
2. prefix/target-set 信息只控制 residual adapter、low-rank delta 或 row-wise gate；
3. adapter zero-init / near-zero init，初始行为等价 H1 target-set conditioned base；
4. gate 以 H1 `target_set_decoder_multiprefix` 为主 reference，而不是只对比 H0 full。

候选 arms：

- `dense_prefix_residual_adapter`: `base_720 + prefix-conditioned low-rank residual`；
- `row_gated_dense_head`: 对 720 projection rows 做 prefix-conditioned gating；
- `prefix_adapter_shared_dense`: 共享 dense projection，prefix condition 只调制 hidden 的低秩子空间。

如果 H1C 仍不能缩小 ETTm2 gap，则应回 Step 2/3 重新判断 TimeAlign 是否适合作为 HSS 主 carrier。
