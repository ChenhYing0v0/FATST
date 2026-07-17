# SIFF/MCCA Step 7B Prelaunch Gate

## Frozen Matrix

- five datasets：`Weather, ETTm1, ETTm2, ETTh1, ETTh2`；
- seed：`2021`；
- 11 new arms × 5 datasets = `55` runs；
- training/evaluation：from-scratch E2E、best-validation-H720 checkpoint、dense validation H1-H720；
- test access：`false`；confirmation seeds：`false`。

核心factorial为`PCSD/SIFF × EQUAL/PCC/MCCA`。其中`PCSD_EQUAL`与`PCSD_PCC`复用未变的历史
artifacts，其余11个arm新训练。controls包括SIFF constant/permuted、Q1-wide、independent-scope、dense matched，
以及pointwise MCCA和uniform balanced OT。

## Gate Result

| Category | Pass |
| --- | --- |
| frozen hashes | true |
| Step7A gate | true |
| 55-run matrix | true |
| dataset-major workload order | true |
| 55 CLI contracts | true |
| all model constructors | true |
| validation-only authorization | true |
| runner/evaluator/analyzer tooling | true |

Decision：`step7b_prelaunch_pass_remote_seed2021_authorized`。下一步只允许启动该冻结matrix；remote resource
smoke失败则回Step7A修复implementation，不得临时改method或dataset profile。
