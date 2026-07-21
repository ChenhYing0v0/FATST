# SIFF-v2 FCC-v1 A6_FULL Scope Prelaunch Report

## 1. Status

| Field | Record |
| --- | --- |
| `candidate_version` | `SC1-SIFF-v2-FCC-v1` |
| `current_step` | Step7B prelaunch complete；Step8 remote/test authorized |
| `method_identity` | immutable `SIFF-v2-EQ-ATTR-v1`；evaluation-only confirmation |
| `user_change` | remove `A6_MEASURE` from FCC；use `A6_FULL` |
| `new_matrix` | 3 arms × 5 datasets × seeds2022/2023 = 30 runs |
| `historical_reference` | same 3 arms × 5 datasets × seed2021 = 15 runs |
| `effective_matrix` | 45 runs；180 official-test cells |
| `prelaunch_gate` | 25/25 checks pass；30/30 jobs frozen |
| `remote_training` | authorized；not started at this report |
| `official_test` | authorized once after 30/30 training；not started |
| `decision` | `step7b_prelaunch_pass_proceed_commit_remote_preflight` |

## 2. Comparator and claim boundary

[Decision] FCC只包含：

1. `siff_equal`：immutable candidate；
2. `a6_full`：用户指定的method-package performance baseline；
3. `siff_independent_equal`：same-objective、parameter-matched ordered-field attribution control。

`A6_MEASURE`没有出现在config arms、launch jobs、comparisons或machine gates中。其历史上优于SIFF的negative result
继续保留在mainline和limitations中，但不会影响本次FCC machine decision。

[Claim Boundary] `SIFF_EQUAL vs A6_FULL`同时改变readout architecture与objective，因此只回答完整method package
相对source carrier是否稳定提升。它不能单独证明ordered field贡献。ordered-field attribution必须依赖
`SIFF_EQUAL vs SIFF_INDEPENDENT_EQUAL`。

## 3. Frozen matrix

| Arm | Readout | Objective | Rank |
| --- | --- | --- | --- |
| `siff_equal` | `siff-coupling-field` | `equal_skill` | 256 |
| `a6_full` | `learned-basis-forecast-operator` | `off` | 256 |
| `siff_independent_equal` | `siff-independent-scope-control` | `equal_skill` | ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/116/116/106/116` |

所有new runs均from-scratch joint encoder-decoder training。五个natural profiles、learning rate `1e-4`、batch size
32、最多20 epochs、patience 5、four-horizon validation selector和full-crop evaluation保持与seed2021协议一致。

launch order按workload排序：先完成两seeds的Weather，再运行ETTm1、ETTh1、ETTm2与ETTh2。runner同时调度
GPU0/1/2，且training与formal test为两个独立execution modes。

## 4. Gates

对three-seed pooled evidence，两项comparison都必须满足：

- MSE macro gain至少`+0.3%`；
- MAE macro gain严格为正；
- MSE dataset wins至少3/5；
- MSE horizon wins至少3/4；
- MSE seed wins至少2/3。

两个comparison与internal health同时通过才得到
`passed_core_candidate_pending_modern_baselines`。若A6_FULL gate失败，停止SIFF paper-core rescue；若只败给
independent，标记`performance_pass_attribution_blocked_stop_fcc_promotion`。

## 5. Prelaunch evidence

`prelaunch_gate.json`给出25/25：

- candidate/status/authorization contract通过；
- exact arms、seeds、datasets、30-job launch order通过；
- `A6_MEASURE`在FCC config/runner中为0；
- matrix为30 new + 15 historical = 45 effective runs；
- seed2021 15/15 references完整、protocol pass、checkpoint hashes unique；
- seed2021逐dataset encoder initialization pairing通过；
- `+0.3%` margin未因用户更换comparator而降低；
- runner dry-run 30/30与two-seed coverage通过；
- three-seed analyzer synthetic smoke通过；
- official test禁止per-dataset/horizon/cell tuning。

## 6. Execution boundary

1. 本地prelaunch不是performance result；
2. remote前必须commit/push、remote fast-forward pull和`nvidia-smi`；
3. resource smoke固定为Weather-SIFF seed2022与ETTm2-independent seed2023的two-batch smoke；
4. smoke finite且无OOM后才启动30-run training；
5. 30/30 training前`FORMAL_TEST_ONLY=1`必须拒绝执行；
6. formal test完成后analyzer必须联合seed2021/2022/2023，禁止只报告new seeds。

Artifacts：

- `configs/stage_c_siff_v2_fcc_v1.json`；
- `scripts/remote/run_stage_c_siff_v2_fcc_v1.sh`；
- `scripts/analyze_stage_c_siff_v2_fcc.py`；
- `scripts/check_stage_c_siff_v2_fcc_prelaunch.py`；
- `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/prelaunch_gate.json`；
- `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/jobs.csv`；
- `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/historical_reference_audit.csv`。

最终decision：

```text
step7b_prelaunch_pass_proceed_commit_remote_preflight
```
