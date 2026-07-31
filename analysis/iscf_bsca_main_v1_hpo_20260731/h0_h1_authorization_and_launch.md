# ISCF-BSCA-MAIN-v1 H0/H1 Authorization and Launch

## Status

| Field | Value |
| --- | --- |
| `date` | `2026-07-31` |
| `candidate` | `ISCF-BSCA-MAIN-v1` |
| `protocol` | `ISCF-BSCA-MAIN-v1-HPO` |
| `launch_commit` | `7361d9e9545e32295fe220f8c59641d5aa61107f` |
| `config_sha256` | `819a9870d87a6f0bd9c9293b12ffbe2d8dc96b62c69f61fc981fca40844a71b0` |
| `search_space_sha256` | `9708fb711ead7f3c1ac386848e294d5f60b6150df85240c2d2c1b9644aeaeefb` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `conda_env` | `moe` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h1` |
| `primary_seed` | `2021` |
| `H1_matrix` | `8 datasets × 2 anchors = 16 jobs` |
| `official_test_jobs` | `0` |
| `status` | `H1_train_validation_running` |

## Authorization

用户于2026-07-31明确要求开始按冻结计划推进最重要的
`ISCF-BSCA-MAIN-v1`超参数调优。当前解释并记录为：

- Tier A local HPO protocol/source patch：authorized；
- Tier B1 H0/H1 remote audit/smoke/training：authorized；
- bounded Tier B2 HPO和完整H2后的official-test aggregate profile ranking：
  authorized；
- Tier B3 selected-profile confirmation：not authorized；
- Tier C final paper reporting audit：not authorized。

## Version Sync

Local commits：

1. `83ab4d4 feat(stage-c): prepare ISCF-BSCA main HPO`；
2. `7361d9e fix(stage-c): resolve Solar dataset path`。

两次commit均已push至`origin/main`。Remote到GitHub的SSH fetch持续超时，因此未
手工复制源码，而是使用包含相同Git objects的bundle，通过remote
`git fetch <bundle>`和`git merge --ff-only`同步。Remote最终HEAD与
`7361d9e9545e32295fe220f8c59641d5aa61107f`完全一致。

Remote原有三处unrelated dirty analysis CSV已在merge前核对与incoming paths无
交集，并完整保留：

- `analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/local_gate/parameter_and_theory_cases.csv`；
- `analysis/stage_c_pcsd_cf_step7b_prelaunch_20260716/arm_matrix_checks.csv`；
- `analysis/stage_c_sc2_pcc_step7b_prelaunch_20260717/initialization_pairing.csv`。

## H0 Dataset Audit

H0只构造train/validation loader，`test_loader_constructed=false`。

| Dataset | Shape identity | SHA256 | Cadence | Finite | Loader |
| --- | --- | --- | --- | --- | --- |
| ECL | `26304 × 321` | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` | hourly, monotonic, no duplicates | pass | train/val pass |
| Solar | `52560 × 137` | `230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2` | source-declared 10-minute；no timestamp in file | pass | train/val pass |
| Exchange | `7588 × 8` | `48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842` | daily, monotonic, no duplicates | pass | train/val pass |

所有dataset channel count、expected rows（如冻结）、NaN/Inf、constant-channel、
date/OT columns、split boundaries和train-only scaler fit contract通过。

## GPU and Storage Preflight

Launch前：

| GPU | Model | Used MiB | Free MiB | Utilization |
| ---: | --- | ---: | ---: | ---: |
| 0 | RTX 3090 | 18 | 24107 | 0% |
| 1 | RTX 3090 | 18 | 24107 | 0% |
| 2 | RTX 3090 | 18 | 24107 | 0% |

无compute processes。`/home`可用约889 GiB。Environment：
Python `3.12.13`、PyTorch `2.9.0+cu128`、CUDA `12.8`。

## Smoke Gates

1. Remote contract checker：pass；
2. 16-job dry-run：pass，`test_jobs=0`；
3. 6-job new-dataset canary：6/6 pass；
4. 16-job full construction/resource smoke：16/16 pass；
5. failure scan：无Traceback、OOM、NaN或Inf。

## H1 Launch

Launch time=`2026-07-31T14:20:27+08:00`。

```text
MODE=train bash scripts/remote/run_iscf_bsca_main_v1_hpo.sh
```

后台orchestrator PID=`545400`。最初三个jobs：

- GPU0：`ECL__h1_timealign`；
- GPU1：`ECL__h1_conservative`；
- GPU2：`Solar__h1_timealign`。

首次live snapshot分别占用约2107、4542、3960 MiB，利用率为75%、91%、89%。
三个job均已进入epoch 1且loss finite。Runner采用global workload-aware queue，
任何GPU完成当前job后领取剩余最长job。

## Next Gate

H1必须满足16/16 training artifacts、four-H validation MSE/MAE、checkpoint hash与
numeric health完整。H1完成后：

1. 用实测epoch time、peak memory与validation health冻结H2 additional profiles；
2. H2总trial上限仍为每dataset 5个（含H1两个）；
3. H2完整训练前不访问official test；
4. H2 test ranking只使用每dataset four-H mean official-test MSE。
