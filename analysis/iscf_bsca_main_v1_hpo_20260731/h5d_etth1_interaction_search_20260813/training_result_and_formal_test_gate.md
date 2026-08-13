# H5D Training Result and Formal-Test Gate

## 1. Decision

H5D已完成48/48个ETTh1 train/validation jobs。Artifact completeness、four-H
validation selector、numeric health、provenance与checkpoint hash审计全部通过；用户于
2026-08-13明确要求继续formal test并分析结果。当前decision为：

```text
H5D_training_complete_48_checkpoint_manifest_frozen_formal_test_authorized
```

该授权仅覆盖一次完整48-checkpoint × four-Horizon=`192` standard-row official-test
audit。禁止partial execution、per-H/per-metric/per-cell/per-seed selection和自动修改
Main I/Main II。

## 2. Training completion audit

| Item | Result |
| --- | --- |
| Exact training commit | `21df4c80cd484350ea4ae777d6453bae94d8512c` |
| Training interval | `2026-08-13 14:43:23--16:09:53 +08:00` |
| Expected/completed jobs | `48/48` |
| Validation four-H rows | `192/192` |
| Training-stage official-test artifacts | `0/48` |
| Failure-token logs | `0` |
| Unique checkpoint SHA256 | `48/48` |
| Best epoch range | `1--3` |
| Final GPU state | GPU0/1/2 idle，18 MiB，0% utilization |
| Quota | `187G/200G soft/220G hard` |

每个checkpoint由validation mean MSE over `{96,192,336,720}`选择，formal test不得
重选epoch。Validation-best diagnostic为`ETTh1__h5d_p20_do0_r152`、mean
MSE=`1.0938418797902867`；它仅证明selector artifact可审计，不用于减少formal-test matrix。

## 3. Immutable checkpoint manifest

- Manifest: `analysis/iscf_bsca_main_v1_hpo_20260731/h5d_checkpoint_manifest.csv`
- Rows / unique trial IDs / unique checkpoint hashes: `48 / 48 / 48`
- Manifest SHA256: `480180333de60c3f53d98c894b8854e4169401edcf7ca378d20f1b213e233a9e`
- Test root: `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5d/test_audit`

Manifest固定phase、trial/profile、seed、best epoch、validation mean MSE、parameter count、
pre-test checkpoint SHA256、read-only training artifact目录与独立test artifact目录。生成后
禁止retrain或修改checkpoint。

## 4. Formal-test contract

- Config: `configs/iscf_bsca_main_v1_hpo_etth1_h5d_test_audit.json`
- Candidate: `ISCF-BSCA-MAIN-v1-etth1-h5d-test-informed-20260813`
- Matrix: `48 × {96,192,336,720}`，MSE/MAE全部报告。
- Initial gate: target/temp artifacts均须为0，remote commit精确匹配，三GPU resource pass。
- Publication: 每trial先写temporary目录，通过dense 720-row、provenance、NPZ与checkpoint
  hash检查后原子发布；任一错误触发ABORT。
- Selection: 一个ETTh1 profile服务全部four H；先过相对H5B mean MSE/MAE各`1.002×`
  双guard，再按Main II best、Main I best、Main II top-2及预注册tie-break排序。
- Success: Main II ETTh1 best cells从`4/8`严格提高到至少`5/8`；stretch=`6/8`。

## 5. Evidence boundary

本轮为single-seed、test-tuned paper-facing HPO，不能描述为untouched holdout或严格
confirmatory evidence，也不提供BSCA mechanism attribution。H5B `h5b_seq640_p20`在完整
selector完成前继续作为fallback。H5E、extra seeds、selected-profile confirmation、
architecture/objective redesign与paper-table mutation均未授权。
