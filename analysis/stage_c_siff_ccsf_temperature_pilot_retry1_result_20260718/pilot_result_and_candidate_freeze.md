# CCSF temperature pilot retry1结果与formal candidate冻结

## 1. 结果结论

[Fact] retry1完整通过9/9 result-audit categories：15/15 runs、60/60 validation cells、15个checkpoint hashes、
training/metrics finite、全部`val/best-val/official_test_mode=false`。pilot checkpoints不复用。

按预注册的五dataset × 四horizon macro validation MSE，选择：

> `tau=0.25`，formal candidate=`SC1-SIFF-v2-CCSF-v1-tau25`。

| Temperature | Macro MSE | Macro MAE | tau0.25 MSE gain |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.569870 | 0.455082 | +0.2991% |
| 0.10 | 0.568970 | 0.454326 | +0.1415% |
| 0.25 | **0.568165** | **0.453679** | selected |

## 2. 稳定性与限制

tau0.25在17/20 dataset-horizon cells、4/5 dataset means、4/4 horizon means上最优。唯一成组例外是ETTm1：
H192偏好tau0.1，H336/H720偏好tau0.05，导致ETTm1 dataset mean中tau0.25比tau0.05差约0.45%。协议禁止
per-dataset tuning，所以不为ETTm1单独改temperature。

[Interpretation] 选择是一致且可执行的，但margin较小。temperature只是teacher geometry的普通共享超参数，不是论文
创新或机制证据；不能把+0.14%–0.30%的validation差异写成CCSF优越性。真正的effectiveness仍要看from-scratch
10-arm official-test Phase A。

## 3. Formal candidate freeze

`configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json`冻结：

- architecture/objective/control matrix继承Step6；
- runtime repair固定为`sqrt(mean_square + 1e-6)`；
- relative/standardized calibration统一使用tau0.25；
- 10 arms × 5 datasets × seed2021 = 50 from-scratch runs；
- H96/H192/H336/H720 validation选checkpoint，official test只做最终四层评估；
- pilot checkpoints与weights一律不复用；
- 10 hard comparisons和internal mechanism health仍全部保留。

## 4. Decision

`decision=freeze_tau025_formal_candidate_prelaunch_next`。本轮只授权实现Step7B formal prelaunch tooling，包括50-job
runner、test evaluator、internal-artifact completeness与四层analyzer；remote Phase A、official test和confirmation
仍为false。不得因为pilot完整或tau0.25占优而直接写成paper-core pass。

11-step cursor：pilot是Step8 hyperparameter selection，结果审计完成后返回Step6/7B冻结formal candidate与正式实验
合同；下一步完成formal prelaunch gate，失败则回Step7A tooling/contract repair，不改变已选temperature。
