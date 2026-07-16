# StageC SC2-PCC-v1-TI Step 7A Protocol

## Current Position

| Field | Value |
| --- | --- |
| `candidate` | `SC2-PCC-v1-TI` |
| `current_step` | Step7A pass；Step7B prelaunch audit next |
| `local gate` | 35/35 |
| `effectiveness` | unknown |
| `remote_authorized` | false |
| `test_accessed/authorized` | false / false |
| `rollback` | Step7A failure -> Step5/6；当前未触发 |

## Frozen Implementation Contract

- one full-$T=720$ forward同时产生fused forecast、five arms与policy；
- raw-scale L1与full dense-prefix measure；requested $H$不进入model、credit或router feature；
- capability与transported credit stop-gradient；
- no offline/EMA teacher、no second forward、no frozen replacement、no second-stage fine-tuning；
- default inference signature、parameters与prefix crop behavior不变；
- nine Phase-A objective modes与Step6 config完全一致；
- coefficients跨datasets固定，只有frozen natural carrier profile允许dataset-aware差异。

## Local Command

```bash
conda run -n r2026-fsa python scripts/check_stage_c_sc2_pcc_step7a.py
```

输出：

- `analysis/stage_c_sc2_pcc_step7a_local_20260716/step7a_cases.csv`
- `analysis/stage_c_sc2_pcc_step7a_local_20260716/local_gate.json`
- `analysis/stage_c_sc2_pcc_step7a_local_20260716/step7a_local_gate_report.md`

## Step 7A Gate Result

1. vectorized prefix risk/transport vs direct loops：pass；
2. nine-mode decomposition：9/9 pass；
3. identical-arm uniformity与skill floor：pass；
4. stop-gradient/no-stopgrad path boundary：pass；
5. schedule endpoints、ramp与monotonicity：pass；
6. real PCSD one-batch finite gradients与5/5 scope auxiliary gradients：pass；
7. default output、parameter count与raw-scale fusion identity：pass；
8. arbitrary prefix projectivity：pass；
9. 20个diagnostics fields：complete；
10. adapter smoke只访问train/val，test/remote/checkpoint mutation：false。
11. production CLI接受PCSD+PCC，并拒绝PCC与非PCSD readout组合。

## Next Gate

Step7B prelaunch audit必须先构造并静态验证`9 modes × 5 datasets × seed2021 = 45 runs`，包括profile/hash、
initialization pairing、validation-only、best-val-H720、resume/status、GPU workload ordering、result completeness与analyzer
dry-run。通过后才可单独授权3090 remote screen。conditional Phase B、test与confirmation seeds继续held。
