# StageC D14-A0 Research Interpretation

## Decision

- `decision`: `d14_a0_neutral_rrr_fail_requires_failure_attribution`
- `diagnostic_valid`: `true`
- `problem_evidence_pass`: `false`
- `rollback_step`: Step 2-3 failure-attribution audit
- 本诊断没有使用test split、没有训练forecast model，正结果最多授权返回Step 4-6。

## Gate Summary

- carrier skill datasets: 4/5
- stable crossing datasets: 0/5
- canonical oracle macro gain: 0.000586
- contiguity datasets: 0/5
- contiguity macro gain: -0.001427

## Dataset Summary

| Dataset | Carrier gain | Oracle gain | Contiguity gain | Stable crossing |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | 13.2350% | 0.0515% | -0.1724% | False |
| ETTh2 | 14.3564% | 0.0740% | -0.1960% | False |
| ETTm1 | 22.1330% | 0.0424% | -0.1403% | False |
| ETTm2 | 15.8844% | 0.1250% | -0.2047% | False |
| Weather | -0.0162% | 0.0000% | -0.0001% | False |

## Failure Attribution Boundary

carrier或numeric invariant失败只能判定诊断无效；问题gate失败只否定当前neutral PCA64 + linear RRR证据，不能自动否定所有nonlinear E2E coupling机制。
