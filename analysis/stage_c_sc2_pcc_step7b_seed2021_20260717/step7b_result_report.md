# SC2-PCC-v1-TI Step7B Validation Result

| Field | Value |
| --- | --- |
| `expected_runs` | `45` |
| `valid_runs` | `45` |
| `valid_reference_runs` | `15` |
| `complete` | `true` |
| `method_pass` | `false` |
| `decision` | `generic_or_pointwise_control_explains_return_step4` |
| `test_used` | `false` |

## Frozen Gates

| Gate | Pass |
| --- | --- |
| `pcc_over_a6` | `true` |
| `pcc_over_plain` | `true` |
| `pcc_over_pointwise_pcc` | `true` |
| `pcc_over_prior_composed` | `false` |
| `arm_degradation_recovery` | `true` |
| `pairwise_nrmse_retention` | `false` |
| `policy_not_collapsed` | `true` |

该报告由validation-only artifacts自动生成。performance、arm recovery、diversity、policy与shared-gradient statistics
分别保存在同目录CSV/JSON中；在45/45 complete前不得作partial method judgment。
