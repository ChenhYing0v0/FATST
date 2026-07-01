# Phase5 A3B Nested Residual Gate Interpretation

## 结论

[Decision] A3B 不通过 effectiveness gate，也不能作为 paper-core interface 候选。它只保留为
diagnostic/control evidence：把 nested structure 放在 dense base 的 residual path 上，会削弱
A2 nested 的正向信号。

## Gate Summary

| Comparison | ALL mean relative MSE | Wins |
| --- | ---: | ---: |
| vs A2 nested | `+4.42%` | `0/12` |
| vs A3-1 shallow | `+4.48%` | `0/12` |
| vs H1 target-set | `+5.09%` | `0/12` |
| vs H1C row-gated | `+4.61%` | `0/12` |
| vs fixed | `+0.99%` | `3/12` |

## Dataset Reading

- ETTh2: A3B 仍相对 fixed 为 `-2.91%`，但相对 A2 nested 为 `+9.01%`，
  相对 H1 为 `+11.12%`。它保留不了 ETTh2 上 nested primary 的 strong benefit。
- ETTm2: 相对 fixed 为 `+4.54%`，相对 A2 nested 为 `+2.71%`，说明 residual path 没有修复
  ETTm2 fixed gap。
- Weather: 相对 fixed 为 `+1.33%`，相对 A2 nested 为 `+1.53%`，A2 nested 在 Weather 上的
  partial gain 消失。

## Mechanism Judgment

[Fact] A3B 的初始函数等价于 `proj_x(hidden)[:, :, :H]`，因此它修复了 A3-1 的 shallow
initialization code-theory 错误。

[Counter-Evidence] 但 A3B 把 nested 放在 residual correction path，而不是 primary
prediction interface。实验结果支持用户提出的担忧：A2 nested 的正向信号来自 primary nested
output contract，而不是“任意位置加入 nested structure”。

[Decision] 停止 residual/nested correction route。下一步如果继续 Stage A，必须回到 primary
nested interface，并用真正 learned capacity preservation 支撑它。合理方向是
`checkpoint-initialized-nested-segment-decoder`：从已训练 H1 target-set checkpoint warm-start
shared carrier，并把 H1 `proj_x` learned rows 转换为 nested segment heads。
