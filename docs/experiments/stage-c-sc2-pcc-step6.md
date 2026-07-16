# StageC SC2-PCC-v1-TI Step 6 Protocol

## Frozen Position

| Field | Value |
| --- | --- |
| `candidate` | `SC2-PCC-v1-TI` |
| `current_step` | Step6 pass；Step7A local implementation next |
| `test_informed` | true；SC-D15-T1 exposed |
| `model carrier` | PCSD-CF shared coupling field；exact v1 architecture已effectiveness fail |
| `method role` | nested-prefix risk到target-coordinate的same-forward credit transport |
| `local implementation` | authorized |
| `remote/test` | false / false |

## Non-Negotiable Contract

- full $T=720$ forecast一次forward产生arms、policy与fused prediction；
- requested $H$不进入model、loss target或router feature；
- no offline/EMA teacher、no frozen replacement、no second forward、no separate fine-tuning stage；
- capability与transport target全部stop-gradient；
- inference graph与plain PCSD完全相同；
- coefficients全datasets共享，不做test/dataset/horizon反向调参。

具体公式、controls、metrics与gates见
`analysis/stage_c_sc2_pcc_step6_design_20260716/step6_source_design_audit.md`，machine-readable freeze见
`configs/stage_c_sc2_pcc_step6.json`。

## Step 7A Required Gates

1. vectorized prefix-risk/transport与direct nested loops float64一致；
2. plain/equal/pointwise/transport九arms的loss decomposition严格匹配config；
3. identical arms产生uniform capability，transported floor不低于$\epsilon/S$；
4. `q/c`无gradient，no-stopgrad conditional ablation只改变该路径；
5. continuous schedule endpoints与monotonicity；
6. real PCSD one-batch gradients finite，所有scope output均获非零auxiliary gradient；
7. plain forward、parameter count、inference output与原PCSD一致；
8. arbitrary prefix仍只crop同一full output；
9. diagnostics列、来源tensor、计算与meaning全部登记；
10. test loader、remote runner与checkpoint mutation均不可达。

Step7A任一失败返回Step5/6修复。全部通过后仍需单独做remote resource smoke与prelaunch gate。
