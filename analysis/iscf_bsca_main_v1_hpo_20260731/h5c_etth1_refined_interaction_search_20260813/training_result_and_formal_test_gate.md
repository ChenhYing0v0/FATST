# H5C Training Result and Formal-Test Gate

## 1. Decision

H5C已完成全部54个ETTh1 train/validation jobs。Training artifact、validation selector、
numeric health、provenance与checkpoint hash审计通过；用户于2026-08-13明确要求继续
formal test，因此当前decision为：

```text
H5C_training_complete_54_checkpoint_manifest_frozen_formal_test_authorized
```

该授权只覆盖一次完整54-checkpoint × four-Horizon=`216` standard-row official-test
audit。禁止partial execution、validation预筛选、per-H/per-metric/per-cell selection和自动
Main I/Main II mutation。

## 2. Training completion audit

| Item | Result |
| --- | --- |
| Exact training commit | `fe9ac10b49779b46e5d1e1aaba2566af796cb8e4` |
| Training interval | `2026-08-13 11:43:04--13:17:47 +08:00` |
| Expected/completed jobs | `54/54` |
| Validation four-H rows | `216/216` |
| Training-stage official-test artifacts | `0/54` |
| Failure-token logs | `0` |
| Unique checkpoint SHA256 | `54/54` |
| Best epoch range | `1--3` |
| GPU final state | GPU0/1/2 idle，18 MiB，0% utilization |
| Quota | `188G/200G soft/220G hard` |

`best_epoch=1--3`说明120-epoch budget没有被耗尽；checkpoint仍严格由validation four-H
mean MSE选择，formal test不得重新选择epoch。Validation-best diagnostic为
`ETTh1__h5c_ctx589_p19`、mean MSE=`1.097401`，仅用于artifact audit，不据此缩小54-profile
official-test matrix。

## 3. Immutable checkpoint manifest

- Manifest: `analysis/iscf_bsca_main_v1_hpo_20260731/h5c_checkpoint_manifest.csv`
- Rows: `54`
- Unique trial IDs: `54`
- Unique checkpoint hashes: `54`
- Manifest SHA256: `e94f95a67c748f95d72e1aab6ced4aaae498982105da5234d411d4c5c0c8379f`
- Test output root: `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5c/test_audit`

每一行固定phase、dataset、trial/profile、seed、best epoch、validation mean MSE、parameter
count、pre-test checkpoint SHA256、read-only training artifact目录和独立test artifact目录。
Manifest生成后不允许retrain或修改checkpoint。

## 4. Formal-test contract

- Config: `configs/iscf_bsca_main_v1_hpo_etth1_h5c_test_audit.json`
- Candidate: `ISCF-BSCA-MAIN-v1-etth1-h5c-test-informed-20260813`
- Matrix: 54 checkpoints × `{96,192,336,720}` = 216 standard rows；MSE/MAE完整报告。
- Atomic publication：每个trial先写独立temporary目录，通过720-row/invariant/NPZ/hash检查后才原子移动到target目录。
- ABORT gate：任一evaluation、hash或artifact invariant失败即停止worker领取后续任务。
- Initial launch：target files=`0`、temporary files=`0`、ABORT absent；否则fail closed。
- Selection：H5B current profile与54个H5C profiles共同使用冻结Main I/Main II comparison surfaces；一个profile服务全部four H。
- Eligibility：mean MSE与MAE均不超过H5B selected profile的`1.002×`。
- Success：Main II ETTh1 best cells从4/8提高到至少5/8；stretch=6/8。

## 5. Evidence and authorization boundary

- 本轮是single-seed、test-tuned HPO，不能表述为untouched-holdout或strictly confirmatory evidence。
- Complete formal test只判断paper-facing performance；不提供BSCA mechanism attribution。
- H5B `h5b_seq640_p20`在H5C selector完成前继续作为frozen fallback。
- Extra seeds、selected-profile confirmation、H5D、architecture/objective redesign及paper-table mutation均未授权。
- Formal test完成后必须先做54/54 artifact/hash audit和完整negative-trial reporting，再运行冻结selector。
